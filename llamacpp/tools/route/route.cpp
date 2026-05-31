// llama-route: token-level cascade routing.
//
// A small local model (e.g. Qwen3-0.6B) generates token-by-token. At each step an
// MLP router reads the small model's last-layer hidden state and decides whether the
// NEXT span should instead be produced by a large cloud model (Qwen3-32B via SGLang).
// When routed, the cloud model generates a span; the span tokens are fed back into the
// local model's KV cache (shared Qwen3 vocab) and local generation resumes.
//
// This is milestone 1: real plumbing, placeholder router (heuristic / random / loadable MLP).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include "router.h"
#include "sglang_client.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ----- custom (non-common) CLI options ------------------------------------------------

struct route_options {
    std::string url        = "http://127.0.0.1:30000";
    std::string weights;            // --router-weights : MLP weights file (empty => placeholder)
    std::string dump_path;          // --dump-hidden    : write per-step hidden states here
    float       threshold  = 0.5f;  // --route-threshold
    int         span       = 32;    // --route-span     : max cloud tokens per route event
    float       random     = -1.0f; // --route-random P : force routing with prob P (debug)
    float       cloud_temp = 0.7f;  // --route-temp     : cloud sampling temperature
};

// Pull our flags out of argv (each takes one value), leaving the rest for common_params_parse.
static std::vector<char *> extract_route_options(int argc, char ** argv, route_options & ro) {
    std::vector<char *> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](float & dst) { if (i + 1 < argc) dst = std::stof(argv[++i]); };
        auto takei = [&](int & dst)   { if (i + 1 < argc) dst = std::stoi(argv[++i]); };
        auto takes = [&](std::string & dst) { if (i + 1 < argc) dst = argv[++i]; };
        if      (a == "--route-url")       takes(ro.url);
        else if (a == "--router-weights")  takes(ro.weights);
        else if (a == "--dump-hidden")     takes(ro.dump_path);
        else if (a == "--route-threshold") take(ro.threshold);
        else if (a == "--route-span")      takei(ro.span);
        else if (a == "--route-random")    take(ro.random);
        else if (a == "--route-temp")      take(ro.cloud_temp);
        else rest.push_back(argv[i]);
    }
    return rest;
}

// ----- hidden-state dump (substrate for later router training) ------------------------

struct hidden_dump {
    std::ofstream f;
    int32_t n_embd = 0;
    bool open(const std::string & path, int32_t n_embd_) {
        if (path.empty()) return false;
        f.open(path, std::ios::binary);
        if (!f.good()) return false;
        n_embd = n_embd_;
        const char magic[4] = {'R','H','I','D'};
        int32_t version = 1;
        f.write(magic, 4);
        f.write(reinterpret_cast<char *>(&version), sizeof(version));
        f.write(reinterpret_cast<char *>(&n_embd), sizeof(n_embd));
        return true;
    }
    // record: int32 pos, int32 token, uint8 routed, float score, float hidden[n_embd]
    void write(int32_t pos, int32_t token, bool routed, float score, const float * h) {
        if (!f.is_open()) return;
        uint8_t r = routed ? 1 : 0;
        f.write(reinterpret_cast<char *>(&pos),   sizeof(pos));
        f.write(reinterpret_cast<char *>(&token), sizeof(token));
        f.write(reinterpret_cast<char *>(&r),     sizeof(r));
        f.write(reinterpret_cast<char *>(&score), sizeof(score));
        if (h) f.write(reinterpret_cast<const char *>(h), (size_t) n_embd * sizeof(float));
    }
};

// decode a sequence of tokens; logits+embeddings are produced for the final token only.
static bool decode_seq(llama_context * ctx, llama_batch & batch, int n_batch,
                       const std::vector<llama_token> & toks, int & n_past) {
    const int n = (int) toks.size();
    for (int i = 0; i < n; ) {
        const int cur = std::min(n_batch, n - i);
        common_batch_clear(batch);
        for (int j = 0; j < cur; ++j) {
            const bool last = (i + j == n - 1);
            common_batch_add(batch, toks[i + j], n_past, {0}, last);
            n_past++;
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed\n", __func__);
            return false;
        }
        i += cur;
    }
    return true;
}

static const char * COL_CLOUD = "\033[36m"; // cyan
static const char * COL_RESET = "\033[0m";

int main(int argc, char ** argv) {
    route_options ropts;
    std::vector<char *> rest = extract_route_options(argc, argv, ropts);

    common_params params;
    common_init();

    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMPLETION)) {
        return 1;
    }

    // enable per-token last-layer hidden states alongside logits
    params.embedding    = true;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_context  * ctx   = llama_init->context();
    llama_model    * model = llama_init->model();
    common_sampler * smpl  = llama_init->sampler(0);
    if (!ctx || !model) {
        LOG_ERR("%s: failed to load model / create context\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // router
    router rt;
    rt.cfg.threshold    = ropts.threshold;
    rt.cfg.route_random = ropts.random;
    if (!ropts.weights.empty()) {
        std::string err;
        if (!rt.load_mlp(ropts.weights, err)) {
            LOG_ERR("%s: failed to load router weights: %s\n", __func__, err.c_str());
            return 1;
        }
        if (rt.n_embd != n_embd) {
            LOG_ERR("%s: router n_embd=%d != model n_embd=%d\n", __func__, rt.n_embd, n_embd);
            return 1;
        }
        LOG_INF("%s: loaded MLP router (n_embd=%d, n_hidden=%d)\n", __func__, rt.n_embd, rt.n_hidden);
    } else {
        LOG_INF("%s: placeholder router (%s), threshold=%.3f\n", __func__,
                ropts.random >= 0.0f ? "random" : "confidence-heuristic", ropts.threshold);
    }

    // cloud client
    sglang_client cloud;
    cloud.base_url = ropts.url;
    const bool cloud_up = cloud.health();
    LOG_INF("%s: SGLang @ %s : %s\n", __func__, ropts.url.c_str(), cloud_up ? "up" : "DOWN");

    hidden_dump dump;
    const bool dumping = dump.open(ropts.dump_path, n_embd);
    if (!ropts.dump_path.empty()) {
        LOG_INF("%s: hidden-state dump -> %s (%s)\n", __func__, ropts.dump_path.c_str(),
                dumping ? "ok" : "FAILED to open");
    }

    // tokenize prompt
    if (params.prompt.empty()) {
        LOG_ERR("%s: empty prompt; pass -p \"...\"\n", __func__);
        return 1;
    }
    std::vector<llama_token> all_tokens = common_tokenize(ctx, params.prompt, true, true);
    if (all_tokens.empty()) {
        LOG_ERR("%s: prompt tokenized to nothing\n", __func__);
        return 1;
    }

    const int n_batch = params.n_batch;
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int n_past = 0;

    // echo prompt
    printf("%s", params.prompt.c_str());
    fflush(stdout);

    if (!decode_seq(ctx, batch, n_batch, all_tokens, n_past)) return 1;

    const int  n_predict = params.n_predict > 0 ? params.n_predict : 256;
    int   n_local = 0, n_cloud = 0, n_route_events = 0;
    double cloud_ms = 0.0;
    int   produced = 0;

    const auto t_gen0 = std::chrono::steady_clock::now();

    while (produced < n_predict) {
        const float * hidden = llama_get_embeddings_ith(ctx, -1);
        const float * logits = llama_get_logits_ith(ctx, -1);

        router_decision d = rt.decide(hidden, n_embd, logits, n_vocab);

        if (d.route && cloud_up) {
            // hand a span to the cloud model
            n_route_events++;
            const std::string prefix = common_detokenize(ctx, all_tokens, false);
            const int budget = std::min(ropts.span, n_predict - produced);
            sglang_result r = cloud.generate(prefix, budget, ropts.cloud_temp);
            cloud_ms += r.latency_ms;

            std::vector<llama_token> span;
            if (r.ok && !r.text.empty()) {
                span = common_tokenize(ctx, r.text, false, true);
            }
            if (span.empty()) {
                // cloud failed or produced nothing -> fall back to one local token this step
                LOG_WRN("%s: cloud span empty (%s); falling back to local\n", __func__,
                        r.ok ? "no tokens" : r.error.c_str());
                d.route = false;
            } else {
                if ((int) span.size() > budget) span.resize(budget);
                if (dumping) dump.write(n_past - 1, span.front(), true, d.score, hidden);
                for (llama_token t : span) common_sampler_accept(smpl, t, false);
                printf("%s", COL_CLOUD);
                for (llama_token t : span) printf("%s", common_token_to_piece(ctx, t).c_str());
                printf("%s", COL_RESET);
                fflush(stdout);
                all_tokens.insert(all_tokens.end(), span.begin(), span.end());
                if (!decode_seq(ctx, batch, n_batch, span, n_past)) break;
                n_cloud  += (int) span.size();
                produced += (int) span.size();
                if (llama_vocab_is_eog(vocab, span.back())) break;
                continue;
            }
        }

        // local: sample one token from the small model
        const llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        if (dumping) dump.write(n_past - 1, id, false, d.score, hidden);
        if (llama_vocab_is_eog(vocab, id)) break;

        printf("%s", common_token_to_piece(ctx, id).c_str());
        fflush(stdout);
        all_tokens.push_back(id);
        if (!decode_seq(ctx, batch, n_batch, {id}, n_past)) break;
        n_local++;
        produced++;
    }

    const auto t_gen1 = std::chrono::steady_clock::now();
    const double gen_s = std::chrono::duration<double>(t_gen1 - t_gen0).count();

    printf("\n\n");
    LOG_INF("---- routing summary ----\n");
    LOG_INF("tokens: %d local + %d cloud = %d (route events: %d)\n",
            n_local, n_cloud, produced, n_route_events);
    LOG_INF("route rate: %.1f%% of tokens from cloud\n",
            produced ? 100.0 * n_cloud / produced : 0.0);
    LOG_INF("wall: %.2fs total, %.0fms in cloud (%d calls)\n", gen_s, cloud_ms, n_route_events);
    LOG_INF("local throughput (excl. cloud wait): %.1f tok/s\n",
            (gen_s - cloud_ms / 1000.0) > 0 ? n_local / (gen_s - cloud_ms / 1000.0) : 0.0);

    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
