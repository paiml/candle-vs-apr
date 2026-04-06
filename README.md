# candle-vs-apr

<p align="center">
  <img src="assets/hero.svg" alt="Candle vs realizr benchmark results" width="960"/>
</p>

## What This Is

Head-to-head benchmark: **Candle** (HuggingFace Rust ML) vs
**realizr** (Sovereign AI Stack inference engine).
Same model, same hardware, same methodology.

Both are pure Rust. Both load GGUF Q4_K_M.
**Does the fused-kernel + CUDA graph architecture outperform
Candle's general-purpose approach?**

**Answer: Yes. Decisively. 1.63x faster.**

## Key Findings

### Showdown v15 (RTX 4090, 2520 MHz, probador N=3)

| Engine | Decode tok/s | vs Candle | Notes |
|--------|-------------|-----------|-------|
| llama.cpp b7746 | **443.6** | **1.95x** | `-ngl 99`, Flash Attention |
| **realizr 0.8.6** | **369.9** | **1.63x** | CUDA graph, Flash Decoding |
| Candle | 227.4 | 1.00x | CLI native, per-op dispatch |

### Why realizr Wins

| realizr advantage | Candle limitation | Impact |
|---|---|---|
| CUDA graph (647 kernels, 1 launch) | Per-op dispatch (~640 launches) | **+26%** |
| Flash Decoding (chunked KV) | Standard SDPA (sequential) | **+15%** long ctx |
| Fused DP4A GEMV (4-bit native) | Separate dequant + matmul | **~10%** fewer mem passes |
| Continuous batching (Orca-style) | CLI only, no server | **3,220 tok/s** at c=32 |
| GPU-resident KV + FP8 cache | Per-call allocation | Lower TTFT |

Candle cannot close this gap without: (a) CUDA graph support
(requires unsafe FFI redesign), (b) fused quantized GEMV kernels
(requires custom PTX), (c) a server architecture for batching.
These are fundamental design differences, not tuning parameters.

### Scaling (realizr only -- Candle has no server)

| c | Agg tok/s | Scaling | Note |
|---|-----------|---------|------|
| 1 | 367.0 | 1.0x | bootstrap N=5 CI [365, 369] |
| 4 | 634.1 | 1.73x | continuous batching |
| 8 | 954.4 | 2.60x | |
| 16 | 1,771.5 | 4.83x | |
| 32 | **3,219.9** | **8.77x** | Orca-style iteration scheduling |

### Gap to llama.cpp (16.6%)

Phase 16 five-whys decomposition:

| Component | % of gap | Root cause | Status |
|-----------|---------|------------|--------|
| Attention | **51%** | Flash Decoding occupancy (3% vs FA2) | 3 multi-warp approaches **FALSIFIED** |
| GEMV | **21%** | DP4A Q4K vs cuBLAS | trueno#239 (Marlin), trueno#175 (half-warp) |
| Other | 28% | RoPE, RmsNorm, residuals | Near parity |

**Next steps:** FlashInfer TC attention (P1), Marlin GEMV pre-packing (P2).
See [chain of thought in spec](docs/specifications/candle-vs-apr-spec.md).

Full analysis: [performance.md](performance.md).
Falsification spec (29 F-conditions, 28 tested):
[candle-vs-apr-spec.md](docs/specifications/candle-vs-apr-spec.md).

[qcd]: https://github.com/paiml/qwen-coder-deploy

## The Two Runtimes

| Runtime | Architecture | Server | Formats |
|---------|-------------|--------|---------|
| [Candle][candle] | QMatMul dequant, per-op dispatch | CLI only | GGUF, SafeTensors |
| [realizr][realizr] | Fused DP4A GEMV, CUDA graph (647 nodes) | OpenAI API + SSE | GGUF, SafeT, APR v2 |

[candle]: https://github.com/huggingface/candle
[realizr]: https://github.com/paiml/realizar

## Model

**Qwen2.5-Coder-1.5B-Instruct Q4_K_M** -- same model as
[qwen-coder-deploy][qcd].
APR v2 prepared via `apr import`
([aprender](https://github.com/paiml/aprender)).
Default produces Q4K (raw passthrough, `--preserve-q4k` deprecated).

## Perplexity

| Path | PPL | Method |
|------|-----|--------|
| realizr CPU FP32 | **12.72** | Q4K dequant + FP32 matmul |
| llama.cpp GPU | **12.97** | cuBLAS FP32 dequant |
| realizr GPU FP8 | 41.31 | Batched prefill cuBLASLt |
| realizr GPU DP4A | 42.94 | Sequential int8 accumulation |

CPU FP32 matches llama.cpp (2% delta). GPU DP4A is 3.2x worse --
the gap is entirely from int8 accumulation precision, not
dequantization. FP8 prefill narrows it 3.8%.

## Hardware

| Platform | GPU | Role |
|----------|-----|------|
| Lambda Vector | RTX 4090, 2520 MHz locked | Primary |
| Yoga | RTX 4060 Laptop, 1900 MHz | Scaling validation |

## Methodology

**v2 (current):** `probador llm load` -- same tool as
[qwen-coder-deploy][qcd].
`--concurrency 1 --duration 30s --warmup 5s --max-tokens 256`
`--stream false --num-layers 28 --gpu-telemetry`.

**CRITICAL:** llama.cpp requires `-ngl 99` (all layers GPU).
`-ngl 28` = 310 tok/s (29% penalty from CPU embedding transfer).
realizr loads all weights to GPU natively.

**Common controls:**
- GPU clocks locked at 2520 MHz (eliminates thermal variance)
- Temperature 0 (greedy, deterministic)
- `nvidia-smi` pre-flight (GPU isolation, realizr#190 lesson)
- Binary fingerprinting (PATH ordering, 0.4.11 vs 0.4.12 lesson)
- Upstream bugs filed via `gh` +
  [provable-contracts](https://github.com/paiml/provable-contracts)

## How to Replicate

```bash
# Start realizr
realizr serve --model /path/to/model.gguf --gpu --port 8081 \
  --openai-api --context-length 4096

# Benchmark
probador llm load --url http://127.0.0.1:8081 \
  --concurrency 1 --duration 30s --warmup 5s \
  --max-tokens 256 --stream false --num-layers 28 \
  --gpu-telemetry --expected-clock-mhz 2520 \
  --runtime-name realizr -o results/benchmark.json

# Candle (CLI only)
quantized-qwen2-instruct --model model.gguf \
  --prompt "Write fibonacci" \
  --sample-len 256 --temperature 0

# llama.cpp
llama-server --model model.gguf --port 8082 \
  -ngl 99 --parallel 1 --flash-attn on --ctx-size 4096
```

## Repository Structure

| Path | Purpose |
|------|---------|
| `scripts/bootstrap-ci.sh` | Bootstrap CIs + Mann-Whitney U |
| `scripts/bottleneck-gate.sh` | Pre-experiment roofline validation |
| `scripts/run-showdown.sh` | Multi-framework showdown runner |
| `configs/showdown.yaml` | Framework definitions (realizr, llama.cpp, vLLM, ollama) |
| `results/` | JSON results (git-tracked) |
| `docs/specifications/` | Popperian falsification spec |

## Falsification Register

Source of truth:
[candle-vs-apr-spec.md &sect;7](docs/specifications/candle-vs-apr-spec.md).

**Score (v15.0.0, 29 F-conditions):**
28 tested (12 confirmed, 6 revised, 4 falsified, 2 weakened, 2 fixed,
1 measured, 1 wired). 1 proposed.

**Headline results:**
- F-1.5X-01 **CONFIRMED**: realizr 369.9 tok/s = 1.63x Candle
- F-SCALE-01 **CONFIRMED**: c=32 at 3,220 tok/s on RTX 4090 (8.77x)
- F-PARITY-02 **FIXED**: c=4 scaling 1.03x &rarr; 1.73x (realizr#211)
- F-STREAM-01 **CONFIRMED**: stream=false within 1% of true (realizr#212)
- F-MULTIWARPC-01 **FALSIFIED**: 2-warp chunk kernel -1.7%/+1.9% (barrier O(n))
- F-TCATTN-01 **FALSIFIED**: multi-warp block-level -13.7% (12 blocks on 128 SMs)
- F-QUALITY-01 **FALSIFIED**: DP4A PPL 42.94 vs FP32 12.97 (int8 precision)
- F-NCU-01 **CONFIRMED**: 2.15% occupancy root cause identified via NCU

### Upstream Fixes from This Project

| Repo | Issue | Fix | Impact |
|------|-------|-----|--------|
| realizr | #198 | Graph capture missing SwiGLU recording | +26% (262&rarr;329 tok/s) |
| realizr | #211 | Non-streaming batch scheduler routing | +82% c=4 |
| realizr | #212 | stream=false bulk-send after generation | +4.3% c=1 |
| realizr | #203 | FP8 batched prefill PPL | 3.8% PPL improvement |
| trueno | #246 | chunk_size 32&rarr;16 | +7.4% short / +45% long ctx |
| trueno | #253 | 2-warp flash decode (falsified) | Correct but not faster |

See spec for full register and evidence.
