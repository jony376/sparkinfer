#!/usr/bin/env bash
# Shared helpers for the sparkinfer bench / accuracy scripts.
# Sourced by bench.sh and accuracy.sh. Everything auto-detects / auto-builds so a
# contributor can run a single command on a fresh Blackwell box.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (bench/scripts -> root)
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-30B-A3B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3-30B-A3B-Q4_K_M.gguf}"
TOK_REPO="${TOK_REPO:-Qwen/Qwen3-30B-A3B}"
LLAMACPP_DIR="${LLAMACPP_DIR:-$ROOT/.llamacpp}"   # override to reuse an existing checkout

# compute capability -> CUDA arch (12.0 -> 120). RTX 5090 / PRO 6000 = 120, Spark/Thor = 121.
detect_arch() {
  local cc; cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
  echo "${ARCH:-${cc:-120}}"
}

ensure_sparkinfer() {  # $1 = arch
  [ -x "$ROOT/build/runtime/qwen3_gguf_bench" ] && [ -x "$ROOT/build/runtime/qwen3_gguf_score" ] && return
  echo ">> building sparkinfer (sm_$1) ..." >&2
  cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_CUDA_ARCHITECTURES="$1" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$ROOT/build" -j"$(nproc)" >/dev/null
}

ensure_model() {
  [ -f "$MODELS_DIR/$MODEL_FILE" ] && return
  echo ">> downloading $MODEL_REPO/$MODEL_FILE -> $MODELS_DIR (~17 GB) ..." >&2
  mkdir -p "$MODELS_DIR"
  # HF Xet can stall on some boxes; plain HTTPS is reliable. aria2c (if present) is faster.
  HF_HUB_DISABLE_XET=1 hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODELS_DIR" >&2 \
    || python3 -c "from huggingface_hub import hf_hub_download as d; d('$MODEL_REPO','$MODEL_FILE',local_dir='$MODELS_DIR')" >&2
}

ensure_tokenizer() {
  [ -f "$MODELS_DIR/tokenizer.json" ] && return
  echo ">> downloading tokenizer.json ..." >&2
  mkdir -p "$MODELS_DIR"
  HF_HUB_DISABLE_XET=1 hf download "$TOK_REPO" tokenizer.json --local-dir "$MODELS_DIR" >&2 \
    || python3 -c "from huggingface_hub import hf_hub_download as d; d('$TOK_REPO','tokenizer.json',local_dir='$MODELS_DIR')" >&2
}

ensure_llamacpp() {  # $1 = arch ; builds llama-bench + llama-server (one-time, slow)
  [ -x "$LLAMACPP_DIR/build/bin/llama-bench" ] && [ -x "$LLAMACPP_DIR/build/bin/llama-server" ] && return
  echo ">> building llama.cpp (CUDA sm_$1) — one-time, several minutes ..." >&2
  [ -d "$LLAMACPP_DIR/.git" ] || git clone --depth=1 https://github.com/ggml-org/llama.cpp "$LLAMACPP_DIR" >&2
  cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$1" \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null 2>&1
  cmake --build "$LLAMACPP_DIR/build" -j"$(nproc)" --target llama-bench llama-server >/dev/null 2>&1
}
