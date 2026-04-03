# candle-vs-apr

<p align="center">
  <img src="assets/hero.svg" alt="Candle vs realizr benchmark results" width="960"/>
</p>

## What This Is

Head-to-head benchmark: **Candle** (HuggingFace Rust ML) vs
**realizr** (Sovereign AI Stack inference engine).
Same model, same hardware, same methodology.

Both are pure Rust. Both load GGUF Q4_K_M.
**Does the fused-kernel + APR v2 architecture outperform
Candle's general-purpose approach?**

## Key Findings

### probador llm load (v3, aligned with [qwen-coder-deploy][qcd])

| Metric | Candle | realizr | llama.cpp (ref) | Winner |
|--------|--------|---------|-----------------|--------|
| Decode tok/s (c=1) | 227.4 (decode) | **263.8** (stream) | -- | **realizr 1.16x** |
| TTFT P50 (c=1) | -- | **8.4ms** | -- | -- |
| ITL P50 (c=1) | -- | **3.8ms** | -- | -- |
| Decode tok/s (c=4) | N/A | **274.5** | 224.8 | **realizr 1.22x** |
| Peak RSS (MB) | **449** | 3,082 | -- | Candle |
| **probador Grade** | -- | **A+ (99.0)** | -- | -- |

> **probador A+ (99.0):** SSE streaming fixed (realizr cf10c0f7 --
> `..Default::default()` in Default impl = infinite recursion).
> TTFT 8.4ms, decode 263.8 tok/s, ITL 3.8ms.
> realizr beats Candle 1.16x (streaming) / 1.20x (non-streaming)
> and llama.cpp 1.22x at c=4.

Full analysis: [performance.md](performance.md).
Falsification spec (15 F-conditions):
[candle-vs-apr-spec.md](docs/specifications/candle-vs-apr-spec.md).

[qcd]: https://github.com/paiml/qwen-coder-deploy

## The Two Runtimes

| Runtime | Architecture | Server | Formats |
|---------|-------------|--------|---------|
| [Candle][candle] | QMatMul dequant, general-purpose | CLI only | GGUF, SafeTensors |
| [realizr][realizr] | Fused Q4K/Q5K/Q6K, CUDA graphs | OpenAI API | GGUF, SafeT, APR v2 |

[candle]: https://github.com/huggingface/candle
[realizr]: https://github.com/paiml/realizar

## Model

**Qwen2.5-Coder-1.5B-Instruct Q4_K_M** -- same model as
[qwen-coder-deploy][qcd].
APR v2 prepared via `apr import --preserve-q4k`
([aprender](https://github.com/paiml/aprender)).

## Benchmark Results

### Phase 1: Single-Request Decode (c=1, 30s, warmup=5s)

> v1 numbers (22.7 tok/s) are **superseded**.
> CUDA graph capture poisoned the context in v1.
> v3 fix (realizr 81c912d2): default to eager, no graph capture.

| Metric | Candle (decode) | realizr (v3) | Status |
|--------|-----------------|--------------|--------|
| Decode tok/s | 227.4 | **273.8** | realizr 1.20x |
| ITL P50 | -- | **3.7ms** | probador |
| Peak RSS (MB) | **449** | 3,082 | Candle wins |

Candle 227.4 is self-reported decode-only (no HTTP).
realizr 273.8 is full wall-clock via `probador llm load`.

### Phase 2: realizr Scaling (Candle N/A -- no server)

| c | Agg tok/s | Per-req tok/s | Wall P50 (ms) |
|---|-----------|---------------|---------------|
| 1 | 117.0 | 137.8 | 1,829 |
| 4 | 116.7 | 33.2 | 8,713 |
| 8 | 126.3 | 20.9 | 15,371 |
| 16 | 112.5 | 13.6 | 35,292 |
| 32 | 145.7 | 11.3 | 56,076 |

Throughput flat -- server ran in SINGLE-REQUEST mode,
requests queued serially. Batch scheduler not tested.

### Phase 3: Format Comparison

| Format | Runtime | tok/s | Notes |
|--------|---------|-------|-------|
| GGUF Q4_K_M | Candle | 227.4 | CLI decode-only |
| GGUF Q4_K_M | realizr (v3) | **273.8** | graph fix |
| SafeT FP32 | Candle | 65.7 | decode-only |
| SafeT FP32 | realizr | 21.2 | #169 FIXED |
| APR v2 Q4K | realizr | 17.4 | #170 FIXED |

## Hardware

| Platform | GPU | Role |
|----------|-----|------|
| Lambda Vector | RTX 4090, 2520 MHz locked | All benchmarks |
| Yoga | RTX 4060 Laptop, 1900 MHz | Cross-validation (planned) |

## Methodology

**v2 (current):** `probador llm load` -- same tool as
[qwen-coder-deploy][qcd].
`--concurrency 1 --duration 30s --warmup 5s --max-tokens 256`
`--stream false --num-layers 28 --gpu-telemetry`.

**v1 (superseded):** Ad-hoc curl scripts, 10 iterations.
Inflated numbers (142.8 tok/s) from different realizr build
via forjar. Results in `results/` are v1; probador is authoritative.

**Common controls:**
- GPU clocks locked at 2520 MHz (eliminates thermal variance)
- Temperature 0 (greedy, deterministic)
- `apr check` pre-flight, `apr profile`/`apr trace` per fix
- Upstream bugs filed via `gh` +
  [provable-contracts](https://github.com/paiml/provable-contracts)

## How to Replicate

### Prerequisites

- Linux + NVIDIA GPU (CUDA 12.6+)
- [probador](https://github.com/paiml/probar) `llm` subcommand
- [apr](https://github.com/paiml/aprender) CLI
- [forjar](https://github.com/paiml/forjar) for isolated builds
- Candle at `../candle`, realizr at `../realizar`
- Model: `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`

### Quick Run

```bash
# Start realizr
apr serve run /path/to/model.gguf --gpu --port 8080

# Benchmark (v2 methodology)
probador llm load --url http://127.0.0.1:8080 \
  --model qwen2.5-coder-1.5b-instruct \
  --concurrency 1 --duration 30s --warmup 5s \
  --max-tokens 256 --stream false --num-layers 28 \
  --gpu-telemetry --expected-clock-mhz 2520 \
  --runtime-name realizr-gguf \
  -o results/probador-realizr-c1.json

# Candle (CLI only)
quantized-qwen2-instruct --model model.gguf \
  --prompt "Write fibonacci" \
  --sample-len 256 --temperature 0
```

## Repository Structure

| Path | Purpose |
|------|---------|
| `forjar-candle.yaml` | Candle build (CUDA 12.6, lazy-curand) |
| `forjar-realizr.yaml` | realizr build + serve deployment |
| `forjar-teardown.yaml` | Clean shutdown |
| `scripts/bench-candle.sh` | Candle CLI benchmark harness |
| `scripts/bench-realizr.sh` | realizr API benchmark harness |
| `scripts/bench-scaling.sh` | Concurrent scaling benchmark |
| `scripts/bench-compare.sh` | Generate comparison tables |
| `results/` | JSON results (git-tracked) |
| `performance.md` | Full analysis and findings |
| `docs/specifications/` | Popperian falsification spec |

## Falsification Register

Source of truth: [performance.md](performance.md) scorecard.

| ID | Prediction | Status |
|----|-----------|--------|
| F-SUMMARY-01 | realizr wins >=1 metric (c=1) | **REVISED** (v1: FALSIFIED; v3: realizr 1.20x) |
| F-PARITY-01 | realizr within +/-10% of Candle | **REVISED** (v1: 0.63x; v3: 1.20x realizr) |
| F-SCALE-01 | realizr c=32 >=1,280 tok/s | **FALSIFIED** (145.7, SINGLE-REQ mode) |
| F-HW-01 | Variance <5% with locked clocks | **CONFIRMED** (CV <1%) |
| F-MODEL-01 | Candle loads Q4_K_M GGUF | **CONFIRMED** |
| F-COLD-01 | realizr cold-start slower | **CONFIRMED** |
| F-SERVING-01 | Serving overhead <5ms | **WEAKENED** (HTTP 5ms, E2E 27ms) |
| F-FORMAT-01 | APR v2 load 2-5x faster | **FALSIFIED** (120x slower) |
| F-RSS-01 | APR v2 RSS < GGUF RSS | **CONFIRMED** (26% less) |
| F-KERNEL-01 | Fused Q4K lower mem traffic | **WEAKENED** |
| F-FMTPARITY-01 | All 3 formats GPU +/-10% | **FALSIFIED** (273.8 / 21.2 / 17.4) |
| F-TOOLPARITY-01 | apr-cli vs realizr +/-5% | **WEAKENED** |
| F-BRICKPARITY-01 | apr profile vs ncu +/-15% | **FALSIFIED** |
| F-PARITY-02 | realizr c=4 <=1.5x llama.cpp | **CONFIRMED** (274.5 tok/s, 1.22x faster) |

**Score: 5 CONFIRMED, 4 FALSIFIED, 3 WEAKENED, 2 REVISED, 0 BLOCKED**
