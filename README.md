# candle-vs-apr

## What This Is

A head-to-head benchmark of **Candle** (HuggingFace's Rust ML framework) vs **realizr** (Sovereign AI Stack inference engine) on the same model, same hardware, same methodology.

Both are pure Rust. Both load GGUF Q4_K_M. The question: **does the Sovereign AI Stack's fused-kernel + APR v2 architecture actually outperform Candle's more general approach?**

## Key Findings

| Metric | Candle | realizr | Winner |
|--------|--------|---------|--------|
| Decode tok/s (c=1, warm) | **227.4** | 142.8 | Candle (1.59x) |
| Decode tok/s (c=1, cold) | **223.1** | 134.4 | Candle (1.66x) |
| Peak RSS (MB) | **449** | 3,082 | Candle (6.9x less) |
| Run-to-run CV | 0.8% | 0.9% | Tie (<5% threshold) |
| Concurrent scaling (c=32) | N/A | 145.7 tok/s | realizr (Candle has no server) |

**Candle is 1.6x faster than realizr at single-request GPU decode.** The fused-kernel advantage does not materialize at c=1 on RTX 4090. realizr's serving overhead (HTTP + tokenization + scheduling) is measurable. Scaling was not demonstrated — realizr ran in SINGLE-REQUEST mode with no batch scheduling active.

See [performance.md](performance.md) for full analysis and [docs/specifications/candle-vs-apr-spec.md](docs/specifications/candle-vs-apr-spec.md) for the falsification register.

## The Two Runtimes

| Runtime | Architecture | Server Mode | Formats |
|---------|-------------|-------------|---------|
| [Candle](https://github.com/huggingface/candle) | General-purpose Rust ML, QMatMul dequant | CLI only (no server) | GGUF, SafeTensors |
| [realizr](https://github.com/paiml/realizar) | Fused Q4K/Q5K/Q6K kernels, CUDA graphs | OpenAI-compatible API | GGUF, SafeTensors, APR v2 |

## Model

**Qwen2.5-Coder-1.5B-Instruct Q4_K_M** — same model used in [qwen-coder-deploy](https://github.com/paiml/qwen-coder-deploy) benchmarks.

APR v2 model prepared via `apr import --preserve-q4k` from [aprender](https://github.com/paiml/aprender).

## Benchmark Design

### Phase 1: Single-Request Decode (Fair Comparison)

Candle has no server, so the fairest comparison is raw decode throughput at c=1:

| Metric | Candle | realizr | Ratio |
|--------|--------|---------|-------|
| Decode (tok/s, warm) | 227.4 | 142.8 | 0.63x |
| Decode (tok/s, cold) | 223.1 | 134.4 | 0.60x |
| Wall time (ms, mean) | 2,456 | 1,804 | — |
| Peak RSS (MB) | 449 | 3,082 | 6.9x |

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

| Format | Runtime | Status |
|--------|---------|--------|
| GGUF Q4_K_M | Candle (GPU) | 227.4 tok/s |
| GGUF Q4_K_M | realizr (GPU) | 142.8 tok/s |
| SafeTensors FP32 | Candle (GPU) | 65.7 tok/s |
| SafeTensors FP32 | realizr (CPU only) | 0.4 tok/s |
| APR v2 Q4K | realizr | BLOCKED (paiml/realizar#168) |

## Hardware

| Platform | GPU | Role |
|----------|-----|------|
| Lambda Vector (primary) | RTX 4090, 2520 MHz locked | All benchmarks |
| Yoga (secondary) | RTX 4060 Laptop, 1900 MHz locked | Cross-validation (planned) |

## Methodology

Same production methodology as qwen-coder-deploy (PMAT-177):
- 10 iterations, temperature 0 (greedy), drop first for cold start
- Locked GPU clocks (eliminates thermal variance)
- Isolated serial execution via [forjar](https://github.com/paiml/forjar)
- Results as JSON in `results/`
- Upstream bugs filed via `gh` and fixed with [provable-contracts](https://github.com/paiml/provable-contracts)

## How to Replicate

### Prerequisites

- Linux with NVIDIA GPU (CUDA 12.6+ toolkit)
- [forjar](https://github.com/paiml/forjar) for isolated deployment
- [apr](https://github.com/paiml/aprender) CLI for APR model conversion
- Candle source at `../candle`
- realizr source at `../realizar`
- Model: `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`

### Quick Run

```bash
# Build (via forjar)
forjar apply -f forjar-candle.yaml    # Build Candle with CUDA 12.6
forjar apply -f forjar-realizr.yaml   # Build + start realizr

# Phase 1: Single-request head-to-head
make bench-candle        # Candle CLI decode
make bench-realizr-c1    # realizr single-request decode
make compare             # Side-by-side table

# Phase 2: realizr scaling
make bench-realizr-scaling   # c=1,4,8,16,32

# Phase 3: Format comparison
make bench-formats       # GGUF vs SafeTensors vs APR v2

# Teardown
forjar apply -f forjar-teardown.yaml
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
| F-FORMAT-01 | APR v2 load 2-5x faster | BLOCKED |
| F-RSS-01 | APR v2 RSS < GGUF RSS | BLOCKED |
| F-KERNEL-01 | Fused Q4K lower mem traffic | UNTESTED |
