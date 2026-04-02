# candle-vs-apr

## What This Is

A head-to-head benchmark of **Candle** (HuggingFace's Rust ML framework) vs **realizr** (Sovereign AI Stack inference engine) on the same model, same hardware, same methodology.

Both are pure Rust. Both load GGUF Q4_K_M. The question: **does the Sovereign AI Stack's fused-kernel + APR v2 architecture actually outperform Candle's more general approach?**

## Key Findings

### probador llm load (v2 methodology, aligned with [qwen-coder-deploy](https://github.com/paiml/qwen-coder-deploy))

| Metric | Candle | realizr (fixed) | llama.cpp (qcd ref) | Winner |
|--------|--------|----------------|---------------------|--------|
| Decode tok/s (c=1) | 227.4 (decode-only) | **263.8** (streaming) | — | **realizr (1.16x)** |
| TTFT P50 (c=1) | — | **8.4ms** | — | — |
| ITL P50 (c=1) | — | **3.8ms** | — | — |
| Decode tok/s (c=4) | N/A | **274.5** | 224.8 | **realizr (1.22x)** |
| Peak RSS (MB) | **449** | 3,082 | — | Candle |
| **probador Grade** | — | **A+ (99.0)** | — | — |

> **probador A+ (99.0):** SSE streaming fixed (realizr cf10c0f7 — `..Default::default()` in Default impl = infinite recursion). TTFT 8.4ms, decode 263.8 tok/s, ITL 3.8ms. realizr beats Candle (1.16x streaming, 1.20x non-streaming) and llama.cpp (1.22x at c=4).

See [docs/specifications/candle-vs-apr-spec.md](docs/specifications/candle-vs-apr-spec.md) for full falsification register (15 F-conditions).

## The Two Runtimes

| Runtime | Architecture | Server Mode | Formats |
|---------|-------------|-------------|---------|
| [Candle](https://github.com/huggingface/candle) | General-purpose Rust ML, QMatMul dequant | CLI only (no server) | GGUF, SafeTensors |
| [realizr](https://github.com/paiml/realizar) | Fused Q4K/Q5K/Q6K kernels, CUDA graphs | OpenAI-compatible API | GGUF, SafeTensors, APR v2 |

## Model

**Qwen2.5-Coder-1.5B-Instruct Q4_K_M** — same model used in [qwen-coder-deploy](https://github.com/paiml/qwen-coder-deploy) benchmarks.

APR v2 model prepared via `apr import --preserve-q4k` from [aprender](https://github.com/paiml/aprender).

## Benchmark Design

### Phase 1: Single-Request Decode (probador llm load, c=1, 30s, warmup=5s)

| Metric | Candle (decode-only) | realizr (probador) | llama.cpp (qcd ref) |
|--------|---------------------|-------------------|---------------------|
| Decode tok/s | 227.4 | 22.7 (patched) | — |
| ITL P50 | — | 44.0ms | — |
| µs/layer | — | 1571 | — |
| Peak RSS (MB) | 449 | 3,082 | — |

Note: Candle 227.4 is self-reported decode-only (no HTTP overhead). realizr 22.7 is full wall-clock via `probador llm load` (includes HTTP + tokenization + prefill + decode).

### Phase 2: realizr Scaling (Candle: N/A)

| c | Agg tok/s | Per-req tok/s | Wall P50 (ms) |
|---|-----------|---------------|---------------|
| 1 | 117.0 | 137.8 | 1,829 |
| 4 | 116.7 | 33.2 | 8,713 |
| 8 | 126.3 | 20.9 | 15,371 |
| 16 | 112.5 | 13.6 | 35,292 |
| 32 | 145.7 | 11.3 | 56,076 |

No throughput scaling observed — server in SINGLE-REQUEST mode, requests queued serially.

### Phase 3: Format Comparison

| Format | Runtime | tok/s (probador) | Notes |
|--------|---------|-----------------|-------|
| GGUF Q4_K_M | Candle (GPU) | 227.4 (decode-only) | CLI, no server |
| GGUF Q4_K_M | realizr (GPU) | 22.7 (patched) | probador llm load |
| SafeTensors FP32 | Candle (GPU) | 65.7 (decode-only) | |
| SafeTensors FP32 | realizr (GPU) | 21.2 | #169 FIXED |
| APR v2 Q4K | realizr (GPU) | 17.4 | #170 FIXED |

## Hardware

| Platform | GPU | Role |
|----------|-----|------|
| Lambda Vector (primary) | RTX 4090, 2520 MHz locked | All benchmarks |
| Yoga (secondary) | RTX 4060 Laptop, 1900 MHz locked | Cross-validation (planned) |

## Methodology

**v2 (current):** `probador llm load` — same tool as [qwen-coder-deploy](https://github.com/paiml/qwen-coder-deploy) inference showdown. `--concurrency 1 --duration 30s --warmup 5s --max-tokens 256 --stream false --num-layers 28 --gpu-telemetry`.

**v1 (superseded):** Ad-hoc curl scripts with 10 iterations — produced inflated numbers (142.8 tok/s) due to different realizr build via forjar. Results in `results/` are v1; probador results are authoritative.

**Common:**
- Locked GPU clocks 2520 MHz (eliminates thermal variance)
- Temperature 0 (greedy, deterministic)
- `apr check` pre-flight, `apr profile`/`apr trace` per fix
- Upstream bugs filed via `gh` + [provable-contracts](https://github.com/paiml/provable-contracts)

## How to Replicate

### Prerequisites

- Linux with NVIDIA GPU (CUDA 12.6+ toolkit)
- [probador](https://github.com/paiml/probar) with `llm` subcommand (build from source at `../probar`)
- [apr](https://github.com/paiml/aprender) CLI for model conversion + serving
- [forjar](https://github.com/paiml/forjar) for isolated deployment
- Candle source at `../candle`, realizr source at `../realizar`
- Model: `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`

### Quick Run

```bash
# Start realizr via apr-cli
apr serve run /path/to/model.gguf --gpu --port 8080

# Benchmark with probador (v2 methodology)
probador llm load --url http://127.0.0.1:8080 \
  --model qwen2.5-coder-1.5b-instruct \
  --concurrency 1 --duration 30s --warmup 5s \
  --max-tokens 256 --stream false --num-layers 28 \
  --gpu-telemetry --expected-clock-mhz 2520 \
  --runtime-name realizr-gguf \
  -o results/probador-realizr-c1.json

# Candle (CLI only, no server)
quantized-qwen2-instruct --model model.gguf \
  --prompt "Write fibonacci" --sample-len 256 --temperature 0
```

## Repository Structure

| Path | Purpose |
|------|---------|
| `forjar-candle.yaml` | Candle build with CUDA 12.6 + lazy-curand patch |
| `forjar-realizr.yaml` | realizr build with CUDA + serve deployment |
| `forjar-teardown.yaml` | Clean shutdown |
| `scripts/bench-candle.sh` | Candle CLI benchmark harness |
| `scripts/bench-realizr.sh` | realizr API benchmark harness |
| `scripts/bench-scaling.sh` | Concurrent scaling benchmark |
| `scripts/bench-compare.sh` | Generate comparison tables |
| `results/` | JSON benchmark results (git-tracked) |
| `performance.md` | Analysis and findings |
| `docs/specifications/` | Popperian falsification spec |

## Falsification Register

| ID | Prediction | Status |
|----|-----------|--------|
| F-SUMMARY-01 | realizr wins >=1 metric at c=1 | **FALSIFIED** |
| F-PARITY-01 | realizr within +/-10% of Candle | **FALSIFIED** |
| F-SCALE-01 | realizr c=32 >=1,280 tok/s | **FALSIFIED** |
| F-HW-01 | Variance <5% with locked clocks | **CONFIRMED** |
| F-MODEL-01 | Candle loads Q4_K_M GGUF | **CONFIRMED** |
| F-COLD-01 | realizr cold-start slower | **CONFIRMED** |
| F-SERVING-01 | Serving overhead <5ms | **WEAKENED** |
| F-FORMAT-01 | APR v2 load 2-5x faster | **FALSIFIED** (120x slower) |
| F-RSS-01 | APR v2 RSS < GGUF RSS | **CONFIRMED** (26% less) |
| F-KERNEL-01 | Fused Q4K lower mem traffic | **WEAKENED** (fewer launches, same GPU time) |
| F-FMTPARITY-01 | All 3 formats GPU ±10% | **FALSIFIED** (GGUF 22.7, SafeT 21.2, APR 17.4) |
| F-TOOLPARITY-01 | apr-cli vs realizr ±5% | **WEAKENED** (GGUF 2.1% PASS, APR 25.6% FAIL) |
| F-BRICKPARITY-01 | apr profile vs ncu ±15% | **FALSIFIED** (35pp/28pp delta, aprender#567) |
| F-PARITY-02 | realizr c=4 ≤1.5x llama.cpp | **TESTING** (107.7 vs 224.8, 2.1x gap) |
