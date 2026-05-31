#include "sglang_client.h"

#include "http.h"

#include <nlohmann/json.hpp>

#include <chrono>

using json = nlohmann::ordered_json;

bool sglang_client::health() const {
    try {
        auto [cli, parts] = common_http_client(base_url);
        cli.set_read_timeout(5, 0);
        cli.set_connection_timeout(5, 0);
        auto res = cli.Get("/health");
        return res && res->status == 200;
    } catch (const std::exception &) {
        return false;
    }
}

sglang_result sglang_client::generate(const std::string & prefix_text,
                                      int   max_new_tokens,
                                      float temperature) const {
    sglang_result out;
    const auto t0 = std::chrono::steady_clock::now();

    try {
        auto [cli, parts] = common_http_client(base_url);
        cli.set_read_timeout(timeout_s, 0);
        cli.set_connection_timeout(10, 0);

        json req = {
            {"text", prefix_text},
            {"sampling_params", {
                {"max_new_tokens", max_new_tokens},
                {"temperature",    temperature},
            }},
            {"stream", false},
        };

        auto res = cli.Post("/generate", req.dump(), "application/json");

        const auto t1 = std::chrono::steady_clock::now();
        out.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!res) {
            out.error = "no response from SGLang at " + base_url;
            return out;
        }
        if (res->status != 200) {
            out.error = "SGLang HTTP " + std::to_string(res->status) + ": " + res->body;
            return out;
        }

        json body = json::parse(res->body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded() || !body.contains("text")) {
            out.error = "unexpected SGLang response: " + res->body;
            return out;
        }

        out.text = body["text"].get<std::string>();
        out.ok   = true;
        return out;
    } catch (const std::exception & e) {
        out.error = std::string("exception: ") + e.what();
        return out;
    }
}
