#pragma once

#include <string>

// Minimal client for SGLang's native /generate endpoint.
//
// Request  (JSON): {"text": <prefix>, "sampling_params": {"max_new_tokens": N,
//                                                          "temperature": T, "stop": [...]}}
// Response (JSON): {"text": <continuation>, "meta_info": {...}}
//
// We use text-in / text-out at the protocol boundary; the caller re-tokenizes the
// returned continuation with the (shared Qwen3) vocab to feed it back into the
// local model. SGLang's RadixAttention prefix-caches the (repeated) prefix, so
// re-sending the full prefix each span is cheap.
struct sglang_result {
    bool        ok = false;
    std::string text;     // generated continuation (on success)
    std::string error;    // error message (on failure)
    double      latency_ms = 0.0;
};

struct sglang_client {
    std::string base_url;       // e.g. http://127.0.0.1:30000
    int         timeout_s = 120;

    // Generate a continuation of `prefix_text`, up to `max_new_tokens`.
    sglang_result generate(const std::string & prefix_text,
                           int   max_new_tokens,
                           float temperature) const;

    // GET /health -> true if the server answers ok
    bool health() const;
};
