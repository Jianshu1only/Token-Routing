#include "router.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

// Binary weight file format (little-endian):
//   char    magic[4]  = "RMLP"
//   int32   version   = 1
//   int32   n_embd
//   int32   n_hidden
//   float   w1[n_hidden * n_embd]   (row-major: w1[h*n_embd + e])
//   float   b1[n_hidden]
//   float   w2[n_hidden]
//   float   b2[1]
bool router::load_mlp(const std::string & path, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) { err = "cannot open weights file: " + path; return false; }

    char magic[4] = {0};
    f.read(magic, 4);
    if (std::memcmp(magic, "RMLP", 4) != 0) { err = "bad magic (expected RMLP)"; return false; }

    int32_t version = 0;
    f.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != 1) { err = "unsupported router weights version"; return false; }

    f.read(reinterpret_cast<char *>(&n_embd),   sizeof(n_embd));
    f.read(reinterpret_cast<char *>(&n_hidden), sizeof(n_hidden));
    if (n_embd <= 0 || n_hidden <= 0 || n_embd > (1 << 20) || n_hidden > (1 << 20)) {
        err = "invalid dimensions in weights file"; return false;
    }

    w1.resize((size_t) n_hidden * n_embd);
    b1.resize(n_hidden);
    w2.resize(n_hidden);

    f.read(reinterpret_cast<char *>(w1.data()), w1.size() * sizeof(float));
    f.read(reinterpret_cast<char *>(b1.data()), b1.size() * sizeof(float));
    f.read(reinterpret_cast<char *>(w2.data()), w2.size() * sizeof(float));
    f.read(reinterpret_cast<char *>(&b2), sizeof(b2));

    if (!f) { err = "weights file truncated"; return false; }

    has_mlp = true;
    return true;
}

static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// xorshift64*; deterministic, no global state
static inline float next_uniform(uint64_t & s) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    uint64_t r = s * 0x2545F4914F6CDD1DULL;
    return (float) ((r >> 11) * (1.0 / 9007199254740992.0)); // [0,1)
}

// top-1 softmax probability over logits, computed stably
static float top_prob(const float * logits, int n_vocab) {
    if (!logits || n_vocab <= 0) return 1.0f;
    float maxl = logits[0];
    for (int i = 1; i < n_vocab; ++i) maxl = logits[i] > maxl ? logits[i] : maxl;
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum += std::exp((double)(logits[i] - maxl));
    return (float) (1.0 / sum); // exp(max-max)=1 over sum
}

router_decision router::decide(const float * hidden, int n_embd_in,
                               const float * logits, int n_vocab) {
    router_decision d;

    // 1. forced random mode
    if (cfg.route_random >= 0.0f) {
        float u = next_uniform(rng_state);
        d.score  = cfg.route_random;
        d.route  = u < cfg.route_random;
        d.reason = "random";
        return d;
    }

    // 2. trained MLP gate
    if (has_mlp && hidden && n_embd_in == n_embd) {
        float acc = b2;
        for (int h = 0; h < n_hidden; ++h) {
            const float * wrow = &w1[(size_t) h * n_embd];
            float a = b1[h];
            for (int e = 0; e < n_embd; ++e) a += wrow[e] * hidden[e];
            if (a < 0.0f) a = 0.0f; // ReLU
            acc += w2[h] * a;
        }
        d.score  = sigmoidf(acc);
        d.route  = d.score > cfg.threshold;
        d.reason = "mlp";
        return d;
    }

    // 3. placeholder confidence heuristic: route when the small model is unsure.
    //    score = 1 - top_prob  (high when uncertain). route when score > threshold.
    float p  = top_prob(logits, n_vocab);
    d.score  = 1.0f - p;
    d.route  = d.score > cfg.threshold;
    d.reason = "heuristic";
    return d;
}
