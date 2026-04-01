# candle-vs-apr

## What This Is

A head-to-head benchmark of **Candle** (HuggingFace's Rust ML framework) vs **realizar** (Sovereign AI Stack inference engine) on the same model, same hardware, same methodology.

Both are pure Rust. Both load GGUF Q4_K_M. The question: **does the Sovereign AI Stack's fused-kernel + APR v2 architecture actually outperform Candle's more general approach?**

## The Two Runtimes

| Runtime | Architecture | Server Mode | Formats |
|---------|-------------|-------------|---------|
| [Candle](https://github.com/huggingface/candle) | General-purpose Rust ML, QMatMul dequant | CLI only (no server) | GGUF, SafeTensors |
| [realizar](https://github.com/paiml/realizar) | Fused Q4K/Q5K/Q6K kernels, CUDA graphs | OpenAI-compatible API | GGUF, SafeTensors, APR v2 |

## Model

**Qwen2.5-Coder-1.5B-Instruct Q4_K_M** — same model used in [qwen-coder-deploy](https://github.com/paiml/qwen-coder-deploy) benchmarks.

## Benchmark Design

### Phase 1: Single-Request Decode (Fair Comparison)

Candle has no server, so the fairest comparison is raw decode throughput at c=1:

| Metric | How Measured |
|--------|-------------|
| Tokens/sec (decode) | Same prompt, same max_tokens, 10-run average |
| TTFT | Time to first token |
| Model load time | Cold start (GGUF parse + weight load) |
| Peak RSS | Memory footprint during inference |
| Model load (APR v2) | realizr only — zero-copy mmap vs GGUF parse |

### Phase 2: What Candle Can't Do

Show realizr scaling at c=1 through c=32 (Candle is inherently single-request):

| c | realizr tok/s | Candle | Note |
|---|---------------|--------|------|
| 1 | measured | measured | Head-to-head |
| 4 | measured | N/A | No server mode |
| 8 | measured | N/A | No server mode |
| 16 | measured | N/A | No server mode |
| 32 | measured | N/A | No server mode |

### Phase 3: Format Comparison

| Format | Load Time | Decode tok/s | RSS |
|--------|-----------|-------------|-----|
| GGUF (Candle) | measured | measured | measured |
| GGUF (realizr) | measured | measured | measured |
| SafeTensors (Candle) | measured | measured | measured |
| SafeTensors (realizr) | measured | measured | measured |
| APR v2 (realizr only) | measured | measured | measured |

## Hardware

| Platform | GPU | Use |
|----------|-----|-----|
| Yoga (primary) | RTX 4060 Laptop, 1900 MHz locked | Main benchmark |
| Lambda Vector | RTX 4090 | Cross-validation |

## Methodology

Same production methodology as qwen-coder-deploy (PMAT-177):
- 60-second runs, 5-second warmup
- Locked GPU clocks (eliminates thermal variance)
- Isolated serial execution via [forjar](https://github.com/paiml/forjar)
- Results as JSON in `results/`
- Scored via [probador](https://github.com/paiml/probador)

## How to Replicate

### Prerequisites

- Linux with NVIDIA GPU (CUDA 12.0+)
- [forjar](https://github.com/paiml/forjar) for isolated deployment
- [probador](https://github.com/paiml/probador) for load testing
- Candle built with CUDA: `cargo build --release --features cuda`
- realizar built with CUDA: `cargo build --release --features cuda`
- Model: `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`

### Quick Run

```bash
# Phase 1: Single-request head-to-head
make bench-candle        # Candle CLI decode
make bench-realizr-c1    # realizr single-request decode
make compare             # Side-by-side table

# Phase 2: realizr scaling
make bench-realizr-scaling   # c=1,4,8,16,32

# Phase 3: Format comparison
make bench-formats       # GGUF vs SafeTensors vs APR v2
```

## Repository Structure

| Path | Purpose |
|------|---------|
| `forjar-candle.yaml` | Candle build + benchmark deployment |
| `forjar-realizr.yaml` | realizr build + serve deployment |
| `forjar-teardown.yaml` | Clean shutdown |
| `scripts/bench-candle.sh` | Candle CLI benchmark harness |
| `scripts/bench-compare.sh` | Generate comparison tables |
| `prompts/` | Standardized test prompts |
| `results/` | JSON benchmark results (git-tracked) |
| `performance.md` | Analysis and findings |

## Why This Comparison Matters

Candle is what most Rust developers reach for when they want ML inference. This benchmark answers:

1. **Is the Sovereign AI Stack actually faster?** Or is Candle's simpler architecture good enough?
2. **Does APR v2 matter?** Is the zero-copy format worth the conversion step?
3. **What do fused kernels buy you?** Candle dequantizes then multiplies. realizr fuses them. How much does that matter on real hardware?
4. **What's the serving gap?** Candle has no server. How much throughput do you leave on the table?
