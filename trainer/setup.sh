#!/usr/bin/env bash
# One-time setup for the trainer: vendors llama.cpp for its HF->GGUF converter.
# Usage: bash trainer/setup.sh [llama.cpp git ref]
# Pin a release tag (https://github.com/ggml-org/llama.cpp/releases) for
# reproducible demos; defaults to master.
set -euo pipefail

TRAINER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$TRAINER_DIR/vendor"
LLAMA_CPP_REF="${1:-master}"

if [ -f "$VENDOR_DIR/llama.cpp/convert_hf_to_gguf.py" ]; then
  echo "llama.cpp already vendored at $VENDOR_DIR/llama.cpp — nothing to do."
  exit 0
fi

mkdir -p "$VENDOR_DIR"
echo "Cloning llama.cpp (@$LLAMA_CPP_REF) for convert_hf_to_gguf.py ..."
git clone --depth 1 --branch "$LLAMA_CPP_REF" \
  https://github.com/ggml-org/llama.cpp "$VENDOR_DIR/llama.cpp"

echo "Done. Converter: $VENDOR_DIR/llama.cpp/convert_hf_to_gguf.py"
