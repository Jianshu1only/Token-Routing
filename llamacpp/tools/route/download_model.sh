#!/usr/bin/env bash
#
# Download the models for llama-route (token-level routing).
#
#   small model : Qwen3-0.6B GGUF  -> downloaded into <llama.cpp>/models/ for llama.cpp
#   large model : Qwen3-32B        -> only needed for the SGLang cloud side (optional;
#                                     SGLang also auto-downloads it on launch)
#
# Usage:
#   ./tools/route/download_model.sh                 # small model, Q8_0 (default)
#   ./tools/route/download_model.sh Q4_K_M          # small model, a different quant
#   ./tools/route/download_model.sh Q8_0 --with-large   # also fetch Qwen3-32B for SGLang
#
set -euo pipefail

QUANT="${1:-Q8_0}"
WITH_LARGE=0
for arg in "$@"; do
    [ "$arg" = "--with-large" ] && WITH_LARGE=1
done

# repo root = two levels up from this script (tools/route/ -> repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

SMALL_REPO="Qwen/Qwen3-0.6B-GGUF"
SMALL_FILE="Qwen3-0.6B-${QUANT}.gguf"
SMALL_URL="https://huggingface.co/${SMALL_REPO}/resolve/main/${SMALL_FILE}?download=true"
SMALL_DST="$MODELS_DIR/$SMALL_FILE"

echo "==> small model: $SMALL_FILE"
if [ -f "$SMALL_DST" ]; then
    echo "    already present at $SMALL_DST (skipping)"
else
    echo "    downloading from $SMALL_REPO ..."
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail -o "$SMALL_DST" "$SMALL_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$SMALL_DST" "$SMALL_URL"
    else
        echo "    error: need curl or wget" >&2
        exit 1
    fi
    echo "    saved to $SMALL_DST"
fi

if [ "$WITH_LARGE" = "1" ]; then
    echo "==> large model: Qwen/Qwen3-32B (for SGLang)"
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download Qwen/Qwen3-32B \
            --include "*.safetensors" "*.json" "*.txt" "tokenizer*" "*.model"
    elif python3 -c "import huggingface_hub" >/dev/null 2>&1; then
        python3 - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-32B",
        allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","*.model"])
print("    downloaded to", p)
PY
    else
        echo "    note: huggingface_hub not found; skipping."
        echo "    SGLang will auto-download Qwen3-32B on launch instead."
    fi
fi

cat <<EOF

Done. Next steps:

  1. Build:
       cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90
       cmake --build build --config Release -j --target llama-route

  2. Start the cloud model (SGLang):
       python -m sglang.launch_server --model-path Qwen/Qwen3-32B --tp 2 --port 30000

  3. Run:
       ./build/bin/llama-route -m models/$SMALL_FILE -ngl 99 \\
         --route-url http://127.0.0.1:30000 --route-threshold 0.35 --route-span 16 \\
         -p "Question: If a train travels 60 km in 1.5 hours, what is its average speed? Answer:" -n 80
EOF
