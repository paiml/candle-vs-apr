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

### Showdown v14.11 (RTX 4090, 2520 MHz, probador N=3)

| Metric | Candle | realizr 0.8.6 | llama.cpp b7746 | Winner |
|--------|--------|---------------|-----------------|--------|
| Decode tok/s (c=1) | 227.4 | **369.9** | **443.6** | llama.cpp |
| vs Candle | 1.00x | **1.63x** | **1.95x** | llama.cpp |
| Decode tok/s (c=4) | N/A | **634.1** | -- | realizr |
| Decode tok/s (c=32) | N/A | **3,219.9** (8.77x) | -- | realizr |
| WikiText-2 PPL | -- | 41.3 (FP8) | **12.97** (FP32) | llama.cpp |
| GPU util | -- | **98%** | 91% | realizr |
| Continuous batching | No | **Yes** (Orca-style) | Yes | -- |

> **Rankings at c=1:** llama.cpp (1.20x realizr) > realizr (1.63x Candle) > Candle.
> Gap analysis (Phase 16): attention occupancy 51%, GEMV efficiency 21%.
> llama.cpp requires `-ngl 99` (all layers GPU). `-ngl 28` = 310 tok/s (29% penalty).
> realizr#212 fix: stream=false bulk-send (+4.3%). trueno#253 filed for multi-warp attention.

Full analysis: [performance.md](performance.md).
Falsification spec (28 F-conditions, 27 tested):
[candle-vs-apr-spec.md](docs/specifications/candle-vs-apr-spec.md).

[qcd]: https://github.com/paiml/qwen-coder-deploy

## The Two Runtimes

| Runtime | Architecture | Server | Formats |
|---------|-------------|--------|---------|
| [Candle][candle] | QMatMul dequant, general-purpose | CLI only | GGUF, SafeTensors |
| [realizr][realizr] | Fused Q4K/Q5K/Q6K DP4A, CUDA graph dispatch | OpenAI API | GGUF, SafeT, APR v2 |

[candle]: https://github.com/huggingface/candle
[realizr]: https://github.com/paiml/realizar

## Model

**Qwen2.5-Coder-1.5B-Instruct Q4_K_M** -- same model as
[qwen-coder-deploy][qcd].
APR v2 prepared via `apr import`
([aprender](https://github.com/paiml/aprender)).
Default produces Q4K (raw passthrough, `--preserve-q4k` deprecated).

## Benchmark Results

### Phase 1: Single-Request Decode (c=1, 30s, warmup=5s)

> v1 numbers (22.7 tok/s) are **superseded** — CUDA graph capture
> poisoned the context. v14.6 uses CUDA graph dispatch (647 kernels →
> 1 launch) + chunk_size=16 attention tuning.

| Metric | Candle (decode) | realizr (v14.7 #211 fix) | Status |
|--------|-----------------|--------------------------|--------|
| Decode tok/s | 227.4 | **378.3** [372.4, 382.4] | realizr **1.66x** |
| ITL P50 | -- | **2.6ms** | probador |
| µs/layer | -- | 94.3 | 28 layers |
| GPU util | -- | 98% | `--gpu-telemetry` |
| Peak RSS (MB) | **449** | 3,082 | Candle wins |

Candle 227.4 is self-reported decode-only (no HTTP).
realizr 378.3 is full wall-clock via `probador llm load`,
bootstrap N=5 runs × 30s, CV=1.1%. Post realizr#211 batch scheduler fix.

### Phase 2: realizr Scaling (Candle N/A -- no server)

| c | v1 (4090, flat) | v5 (Yoga, batch) | Scaling |
|---|-----------------|------------------|---------|
| 1 | 117.0 | **132.6** | baseline |
| 4 | 116.7 | **302.2** | 2.3x |
| 8 | 126.3 | **519.7** | 3.9x |
| 16 | 112.5 | **980.2** | 7.4x |
| 32 | 145.7 | **1,776.5** | 13.4x |

v1 was flat (SINGLE-REQUEST mode, no batching).
v5 Yoga confirms batch scheduling: **1,776.5 tok/s at c=32**.

### Phase 3: Format Comparison

| Format | Runtime | v14 (4090) | v5 (Yoga) | Notes |
|--------|---------|------------|-----------|-------|
| GGUF Q4_K_M | Candle | 227.4 | -- | CLI decode-only |
| GGUF Q4_K_M | realizr | **378.3** | **132.5** | graph + chunk=16 + #211 |
| FP16 APR | realizr | -- | **151.6** | #180 FIXED (7.15x) |
| APR v2 Q4K | realizr | -- | **132.3** | parity with GGUF |

v5 Yoga: all 3 formats GPU, within 14.6%. Old v3 SafeT/APR
gaps were bugs (#169 F32 SGEMM, #170 dequant, #180 F16 dtype).

## Hardware

| Platform | GPU | Role |
|----------|-----|------|
| Lambda Vector | RTX 4090, 2520 MHz locked | Primary (phases 1-7) |
| Yoga | RTX 4060 Laptop, 1900 MHz | Validated (scaling, format, tool parity) |

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
| `scripts/bootstrap-ci.sh` | Bootstrap CIs + Mann-Whitney U (Phase 13) |
| `scripts/measure-vram.sh` | VRAM polling during probador runs |
| `scripts/run-showdown.sh` | 4-way framework showdown runner |
| `configs/showdown.yaml` | Showdown framework definitions |
| `results/` | JSON results (git-tracked) |
| `performance.md` | Full analysis and findings |
| `docs/specifications/` | Popperian falsification spec |

## Falsification Register

Source of truth:
[candle-vs-apr-spec.md §7](docs/specifications/candle-vs-apr-spec.md).

**Current score (v14.7.0, 27 F-conditions):**
25 tested (11 confirmed, 5 revised, 3 falsified, 2 weakened, 2 fixed,
1 measured, 1 wired). 2 proposed.

**Headline results:**
- F-1.5X-01 **CONFIRMED**: realizr 378.3 tok/s = 1.66x Candle (#211 fix)
- F-SCALE-01 **CONFIRMED**: c=32 @ 1,776 tok/s on Yoga (13.4x scaling)
- F-PARITY-02 **FIXED**: c=4 non-streaming 1.03x → 1.76x (realizr#211)
- F-PARITY-04 **REVISED**: realizr 0.88x llama.cpp at c=1 (was 0.82x)
- F-QUALITY-01 **FALSIFIED**: DP4A PPL 24.2 vs FP32 12.97 (int8 precision)
- F-RSS-02 **FALSIFIED**: min 2,930 MB RSS (server + weights irreducible)
- F-NCU-01 **CONFIRMED**: 2.15% occupancy → chunk=16 fix (trueno#246)

See spec for full register and evidence.
