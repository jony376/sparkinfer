# sparkinfer bench & accuracy harness

Turnkey scripts for a fresh NVIDIA Blackwell box (`sm_120` RTX 5090 / PRO 6000,
`sm_121` RTX Spark / Jetson Thor). They auto-detect the GPU arch, build what's
missing, fetch the model, and print results — no manual path-passing.

**Prereqs:** CUDA 12.8+ (or 13), CMake ≥ 3.20, a C++17 compiler, `git`, and
`pip install huggingface_hub tokenizers` (the accuracy script also needs `curl`).

## Quickstart

```bash
# 1) Decode throughput (downloads Qwen3-30B-A3B Q4_K_M on first run)
bench/scripts/bench.sh --download

# 2) Head-to-head vs llama.cpp on the same GGUF + same GPU (builds llama.cpp once)
bench/scripts/bench.sh --download --compare

# 3) Accuracy gate vs llama.cpp (token-match / KL / perplexity)
bench/scripts/accuracy.sh --download
```

Use your own model instead of `--download`:
```bash
bench/scripts/bench.sh /path/to/model.gguf --tokens 256 --compare
```

## What you get

`bench.sh` → sparkinfer decode tok/s + VRAM (and, with `--compare`, the llama.cpp
`tg128` number on the same card).

`accuracy.sh` → the correctness gate:
```
token-match (top-1)   : 100/100 = 1.000   (bar >= 0.90)
mean KL(llama||spark) : 0.136 nats
PPL sparkinfer        : 6.13   (exact)
PPL llama.cpp         : 7.76   (top-k+floor; inflated — see accuracy results doc)
```

## Using the accuracy gate for optimization (no silent regressions)

The same `score` tool gates an optimization against the **previous** sparkinfer build,
not just llama.cpp — expect **~100% top-1 + KL ≈ 0**:
```bash
build/runtime/qwen3_gguf_score model.gguf 20 <token-ids...>   # baseline, save output
# ... apply your kernel optimization, rebuild ...
build/runtime/qwen3_gguf_score model.gguf 20 <token-ids...>   # compare argmax + logprobs
```

## Knobs (env vars)

| var | default | purpose |
|---|---|---|
| `ARCH` | auto (`compute_cap`) | CUDA arch, e.g. `121` for RTX Spark |
| `MODELS_DIR` | `./models` | where the GGUF + tokenizer live |
| `MODEL_REPO` / `MODEL_FILE` | Qwen3-30B-A3B GGUF | model to fetch |
| `LLAMACPP_DIR` | `./.llamacpp` | reuse an existing llama.cpp checkout/build |

Files: `bench.sh`, `accuracy.sh`, `accuracy_compare.py`, `eval_text.txt`, `_common.sh`.
Results from reference runs live in [`../results/`](../results).
