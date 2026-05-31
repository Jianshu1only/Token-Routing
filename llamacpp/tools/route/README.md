# llama-route — token-level edge↔cloud routing

A llama.cpp tool implementing **token-level routing**: a small on-device model (e.g. Qwen3-0.6B)
decodes token-by-token; at each step an MLP reads the small model's **last-layer hidden state** and
decides whether to defer the next span to a large cloud model (Qwen3-32B served by **SGLang**).

This is the llama.cpp counterpart of the ONNX-runtime version in the parent
[Token-Routing](../../../README.md) repo. Moving the small model into llama.cpp adds an incremental
**KV cache** (the ONNX version recomputes the full sequence each step) and GGUF quantization.

## How it works (per generation step — see `route.cpp`)

1. `decode` the current tokens — the context is created with `embedding=true` + `pooling_type=NONE`,
   so a single `llama_decode` yields **both logits and the per-token hidden state**.
2. Read the last position's hidden state via `llama_get_embeddings_ith(ctx, -1)` (size `n_embd`).
3. `router.decide(hidden, logits)`:
   - **MLP mode** (`--router-weights`): `Linear(n_embd→H) → ReLU → Linear(H→1) → sigmoid`; route when
     `score > threshold`.
   - **Placeholder** (default): route when the small model's top-token probability is low
     (uncertainty heuristic), or fixed probability via `--route-random P`.
4. Not routed → sample one local token, decode, continue.
5. Routed → send the full prefix to SGLang, get a span (≤ `--route-span` tokens), tokenize it
   (Qwen3 shares vocab 151936 → token ids interchangeable), feed it back into the KV cache, continue.
   `--route-span 1` reproduces the original single-token routing behaviour.

Cloud tokens print in cyan; a routing summary (local/cloud counts, route rate, latency) prints at
the end. `--dump-hidden <file>` writes per-step `(pos, token, routed, score, hidden[])` records as
router training data.

## Download models

The model weights are not in the repo. Fetch the small GGUF (and optionally the large model) with:

```bash
./tools/route/download_model.sh                  # Qwen3-0.6B Q8_0 -> models/
./tools/route/download_model.sh Q4_K_M           # a different quant
./tools/route/download_model.sh Q8_0 --with-large  # also pull Qwen3-32B for SGLang
```

(The cloud Qwen3-32B is otherwise auto-downloaded by SGLang on launch.)

## Build

From the llama.cpp root (this tree), `route` is already registered in `tools/CMakeLists.txt`:

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90   # adjust arch; CPU/Metal also work
cmake --build build --config Release -j --target llama-route
```

## Run

```bash
# cloud model
python -m sglang.launch_server --model-path Qwen/Qwen3-32B --tp 2 --port 30000

# router
./build/bin/llama-route \
  -m models/Qwen3-0.6B-Q8_0.gguf -ngl 99 \
  --route-url http://127.0.0.1:30000 \
  --route-threshold 0.35 --route-span 16 \
  -p "Question: If a train travels 60 km in 1.5 hours, what is its average speed? Answer:" \
  -n 80
```

## Options

| flag | meaning | default |
|---|---|---|
| `--route-url` | SGLang base URL | `http://127.0.0.1:30000` |
| `--route-threshold` | route when router score > threshold | `0.5` |
| `--route-span` | max cloud tokens per route event (`1` = single-token, like the ONNX version) | `32` |
| `--router-weights` | trained MLP weights (`RMLP` binary); omitted → placeholder heuristic | — |
| `--route-random P` | force routing with probability `P` (debug) | off |
| `--route-temp` | cloud sampling temperature | `0.7` |
| `--dump-hidden` | write per-step hidden states (`RHID` binary) for router training | — |

Plus the usual llama.cpp flags (`-m`, `-ngl`, `-n`, `--temp`, `-c`, …).

## RMLP weight format

```
char  magic[4] = "RMLP"
int32 version  = 1
int32 n_embd
int32 n_hidden
float w1[n_hidden * n_embd]   # row-major w1[h*n_embd + e]
float b1[n_hidden]
float w2[n_hidden]
float b2[1]
```

> The original `Qwen2Confidence` MLP was trained on Qwen2.5-0.5B (`hidden=896`); this build defaults
> to Qwen3-0.6B (`n_embd=1024`). To reuse a trained router, match the small model's hidden size (or
> retrain). The router is untrained in this milestone — placeholder/heuristic by default.
