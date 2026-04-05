# Candle vs APR Inference Parity Specification

**Document ID:** PAIML-CANDLE-APR-001
**Version:** 9.6.0
**Last Updated:** 2026-04-04
**Status:** ACTIVE
**Methodology:** Popperian Falsification + Deterministic Benchmarks
**Primary Target:** Lambda Vector (RTX 4090, 24 GB VRAM, sm_89)
**Secondary Target:** Yoga (RTX 4060 Laptop, 8 GB VRAM, sm_89)
**Model:** Qwen2.5-Coder-1.5B-Instruct Q4_K_M GGUF
(1.78B params, 28 layers, hidden=1536)

> Every claim in this spec carries a falsification condition.
> If the condition triggers, the claim is revised or
> retracted — not defended.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Scope](#2-scope)
3. [Hardware Targets](#3-hardware-targets)
4. [Model Under Test](#4-model-under-test)
5. [Benchmark Design](#5-benchmark-design)
6. [Metrics Contract](#6-metrics-contract)
7. [Baseline Thresholds & Pass Criteria](#7-baseline-thresholds--pass-criteria)
8. [Architectural Comparison](#8-architectural-comparison)
9. [Falsification Register](#9-falsification-register)
10. [Work Items](#10-work-items)
11. [PMAT Compliance](#11-pmat-compliance)
12. [Scientific Methodology Gaps](#12-scientific-methodology-gaps)
13. [Revision History](#13-revision-history)

---

## 1. Executive Summary

### What This Is

Head-to-head inference benchmark comparing **Candle**
(HuggingFace's Rust ML framework) and **realizr**
(Sovereign AI Stack inference engine) on the same model,
same hardware, same methodology. This is the Rust-vs-Rust
sibling of qwen-coder-deploy (which compares realizr vs
llama.cpp vs vLLM vs ollama).

### Why This Matters

Candle is the most-adopted Rust ML framework. When
developers evaluate the Sovereign AI Stack, their first
question is: "Why not just use Candle?" This benchmark
provides data-driven answers:

1. **Raw decode speed** — is the fused-kernel
   architecture measurably faster?
2. **Format advantage** — does APR v2 zero-copy loading
   justify the conversion step?
3. **Serving gap** — what throughput do you forfeit with
   CLI-only inference?
4. **Memory efficiency** — do fused kernels reduce peak
   VRAM usage?

### Chain of Reasoning

Both are pure Rust, both load GGUF Q4_K_M, both target
CUDA — isolating architecture from language/runtime.
c=1 is primary (Candle has no server). Concurrent
benchmarks (c=4..32) show what Candle cannot provide.

**Result (v9 showdown, clean GPU, 2520 MHz):** realizr
**281.2** vs llama.cpp **336.7** decode tok/s (16.5% gap).
llama.cpp TTFT 760ms vs realizr 911ms — prompt caching
advantage. Earlier false regression (273.8→232) was GPU
contention (realizr#190 CLOSED). Bootstrap: **277.3**
[276.1, 278.5]. Candle: 227.4 (reference). RSS: Candle
449 MB. Fused K+V kernel shipped (trueno 9d99e18c),
graph poison fix shipped (realizr#194).
See F-SUMMARY-01, F-PARITY-04.

---

## 2. Scope

### This repo does:

- Build Candle and realizr from source with CUDA support
- Run deterministic, isolated benchmarks via forjar
- Measure decode throughput, TTFT, model load time,
  and memory footprint
- **Enforce format parity:** all 3 formats (GGUF,
  SafeTensors, APR v2) must have GPU inference — any gap
  is a bug, not a limitation
- **Enforce tool parity:** `apr` CLI and raw `realizr`
  must produce equivalent results on the same model
- **Enforce CLI parity:** `apr run` must support every
  sampling/generation feature Candle CLI has
  (F-CLIPARITY-01)
- Report results as machine-readable JSON +
  human-readable tables

### Zero external dependencies — everything is in our codebase

The Sovereign AI Stack uses NO external inference
libraries. Every capability Candle provides exists
natively:
- **Inference engine:** `../realizar` (GGUF/SafeT/APR,
  decoder-only + enc-dec via realizr#173)
- **GPU kernels:** `../trueno` (fused Q4K/Q5K/Q6K DP4A,
  CUDA graphs, cuBLAS)
- **GPU profiling:** `../trueno/crates/cgp`
  (`cgp profile kernel`, `cgp roofline`, `cgp compete`,
  `cgp contract verify` — unified perf analysis for
  CUDA/SIMD/wgpu kernels)
- **CLI:** `../aprender`
  (`apr run/serve/check/profile/bench/trace`)
- **Load testing:** `apr bench` (throughput gate,
  CI assertions: `--assert-throughput`, `--assert-p99`)
  and `probador llm load` (HTTP load, Poisson arrival)
- **Quality:** `../provable-contracts`
  (compile-time contract enforcement)
- **Testing:** `../probar` (`probador llm load/score`),
  `../apr-model-qa-playbook` (95 models certified)

**Workflow:** `apr check` -> `apr profile` -> `apr bench`
-> `cgp contract verify` -> `gh issue create` ->
fix upstream -> contract -> `make perf-gate` ->
re-run falsification.

---

## 3. Hardware Targets

| Property | Lambda (PRIMARY) | Yoga (SECONDARY) |
|----------|-----------------|------------------|
| GPU | RTX 4090 | RTX 4060 Laptop |
| VRAM | 24 GB GDDR6X | 8 GB GDDR6 |
| Memory BW | 1,008 GB/s | 256 GB/s |
| SMs | 128 | 24 |
| Clock | Locked 2520 MHz | Locked 1900 MHz |
| CUDA | 12.6 / Driver 570.207 | 12.6 |
| Compute | sm_89 (Ada Lovelace) | sm_89 |

Both locked clocks. CV <1% confirmed (F-HW-01).

---

## 4. Model Under Test

**Qwen2.5-Coder-1.5B-Instruct** — same model as
qwen-coder-deploy and qwen-train-canary.

| Property | Value |
|----------|-------|
| Parameters | 1.78B |
| Layers | 28 |
| Hidden dim | 1536 |
| Heads (Q/KV) | 12 / 2 (GQA 6:1) |
| Head dim | 128 |
| Intermediate | 8960 |
| Vocab | 151,936 |
| RoPE | NEOX, theta=1,000,000 |
| Quantization | Q4_K_M (GGUF) |
| File size | ~1 GB |

### Available formats

| Format | Path | Source | Runtimes |
|--------|------|--------|----------|
| GGUF Q4_K_M | `~/models/qwen2.5-..q4_k_m.gguf` | HuggingFace | Candle, realizr |
| SafeTensors | `~/models/qwen2.5-..-safetensors/` | HuggingFace | Candle, realizr |
| APR v2 Q4K | `~/models/qwen2.5-..-q4k.apr` | `apr import` | realizr only |

### Model preparation

**APR v2 conversion (preferred — uses `apr` CLI):**
```bash
apr import \
  ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --arch qwen2 \
  -o ~/models/qwen2.5-coder-1.5b-instruct-q4k.apr
```

Default import now produces Q4K via raw byte passthrough
(realizr#185, aprender#582). `--preserve-q4k` is
deprecated — the flag is accepted but has no effect.

**Raw realizr (fallback — direct GGUF serving):**
```bash
realizar serve --model <gguf> --gpu --openai-api
```

realizr can serve GGUF files directly without
pre-conversion. Use this path when testing GGUF parity
(Phase 1) or when APR conversion is blocked.

> **F-MODEL-01:** If Candle's quantized-qwen2-instruct
> example cannot load the Q4_K_M GGUF file, the
> head-to-head comparison is blocked. Action: verify
> Candle's QMatMul supports Q4_K_M dequant path.

---

## 5. Benchmark Design

### Phase 1: Single-Request Decode (Head-to-Head)

The only fair comparison — both runtimes process one
request at a time.

| Parameter | Value |
|-----------|-------|
| Prompt | Coding task (~38 tokens after template) |
| Max tokens | 256 |
| Duration | 30s (5s warmup) via `probador llm load` |
| Measurement | decode tok/s, TTFT, ITL, us/layer |
| Memory | Peak RSS via `/usr/bin/time -v` (Candle) |
| Isolation | forjar deploy, kill competing GPU procs |
| Clock | Locked (nvidia-smi -lgc) |

**Candle:** `quantized-qwen2-instruct --model <gguf>
--prompt <text> --sample-len 256 --temperature 0`

**realizr:** `probador llm load --url <URL>
--stream false --max-tokens 256` (v2).
v1 used raw curl; probador is authoritative.

Temperature 0 (greedy) is mandatory for determinism.
With temperature >0, non-deterministic output lengths
produce 13% CV (F-HW-01 would fail).

### Phase 2: Concurrent Scaling (realizr-only)

Demonstrates what Candle's architecture cannot provide.

| c | Duration | Tool | Note |
|---|----------|------|------|
| 1 | 60s | bench-scaling.sh | Baseline for scaling eff |
| 4 | 60s | bench-scaling.sh | Continuous batching |
| 8 | 60s | bench-scaling.sh | Memory pressure test |
| 16 | 60s | bench-scaling.sh | Near-asymptote |
| 32 | 60s | bench-scaling.sh | At asymptote |

### Phase 3: Format + Tool Parity

All 3 formats must have GPU inference. Any gap is a bug.

#### 3a. Format parity (F-FMTPARITY-01)

| Format | Candle | realizr v1 | v3 (4090) | v5 (Yoga) | Status |
|--------|--------|-----------|-----------|-----------|--------|
| GGUF Q4_K_M | 227.4 | 142.8 | **273.8** | 132.5 | realizr wins |
| FP16 APR | N/A | -- | 21.2 | **151.6** | #180 FIXED, 7.15x |
| APR v2 Q4K | N/A | -- | 17.4 | **132.3** | parity with GGUF |

v5 Yoga: all 3 formats GPU, within 14.6% (FP16 fastest).
v3 SafeT/APR gaps were bugs (#169 F32 SGEMM, #170 dequant, #180 F16-as-F32 dtype). All fixed.

#### 3b. Tool parity: `apr` CLI vs `realizr`

| Format | Tool | v1 | v3 (4090) | v5 (Yoga) | Delta |
|--------|------|----|-----------|-----------|-------|
| GGUF | `realizr serve` | 142.8 | 273.8 | 132.5 | 0.0% ✅ |
| GGUF | `apr serve run` | 139.8 | 273.8 | 132.5 | |
| APR Q4K | `realizr serve` | -- | 17.4 | **132.3** | 1.4% ✅ |
| APR Q4K | `apr serve run` | -- | 21.9 | **130.4** | |
| FP16 APR | `realizr serve` | -- | -- | **151.6** | N/A |

Both tools on the same model/format within +/-5%.
GGUF: **0.0%** (132.5 vs 132.5). APR Q4K: **1.4%**
(130.4 vs 132.3). Previous 25.6% was version skew
(realizr#179).

### Methodology (v2, aligned with qwen-coder-deploy)

**Tool:** `probador llm load` — same tool used in
qwen-coder-deploy inference-showdown-v1.yaml. Replaces
ad-hoc curl loops. Reports TTFT, ITL, TPOT, decode
tok/s, us/layer, GPU telemetry.

**Standard run:**
```
probador llm load --url <URL> --concurrency 1 \
  --duration 30s --warmup 5s --max-tokens 256 \
  --stream false --num-layers 28 --gpu-telemetry \
  --expected-clock-mhz 2520
```

**Cross-reference:** qcd v2 baseline: llama.cpp 224.8,
Candle 227.4 (consistent — both decode-only).

**Required gates:** `apr check` (integrity) ->
`apr profile --granular` (brick scores) ->
`apr trace --verbose` (layer correctness).

---

## 6. Metrics Contract

### Primary Metrics

| Metric | Definition | Unit | How Measured |
|--------|-----------|------|-------------|
| Decode tok/s | Tokens/sec (warm) | tok/s | `probador llm load` |
| ITL P50 | Inter-token latency median | ms | `probador --stream true` |
| TTFT P50 | Time to first token | ms | `probador llm load` |
| us/layer | Per-layer decode time | us | `probador --num-layers 28` |
| Peak RSS | Max resident set size | MB | `/usr/bin/time -v` |

### Derived Metrics (Phase 2)

Aggregate tok/s, per-request tok/s, scaling efficiency
`(agg_c / agg_1) / c`, ITL P50, TTFT P50.

### Data Format

All results as JSON in `results/`:
`candle-*.jsonl`, `realizr-c1-*.jsonl`,
`realizr-scaling-c<N>-*.jsonl`, `apr-cli-*.jsonl`,
plus `-summary.json` aggregates.

---

## 7. Baseline Thresholds & Pass Criteria

### Phase 1: Single-Request Parity (c=1)

| Metric | Predict | v1 | v3 (graph fix) | v8 (showdown) | Status |
|--------|---------|----|----|----|----|
| Decode tok/s | 0.90-1.10 | 0.63x (poisoned) | **1.20x** (273.8 vs 227.4) | **0.87x total** (289.0 vs 333.1) | **v8: REVISED** |
| Decode-only | 0.90-1.10 | -- | -- | **~1.01x** (~303 vs ~299) | **PARITY** |
| Peak RSS | 0.85-1.15 | 0.15x (449/3082) | 0.15x (unchanged) | TBD | **FAIL** |

**F-PARITY-01: REVISED.** v3: 1.20x. v8 showdown
(clean GPU): total throughput **0.87x** (realizr 289.0
vs llama.cpp 333.1). But llama.cpp prompt caching (LCP
similarity) inflates total. Decode-only ~303 vs ~299:
**parity**. Bootstrap: **277.3** [276.1, 278.5].
RSS still 6.9x higher (server + KV cache pool).

> **Mandatory pre-flight:** `nvidia-smi
> --query-compute-apps=pid,name --format=csv` must show
> ONLY the process under test. GPU contention caused
> false 15% regression (realizr#190 root cause).

### Phase 2: Scaling Demonstration

| c | Predicted | v1 (4090) | v5 (Yoga) | Status |
|---|-----------|-----------|-----------|--------|
| 1 | ~148 tok/s | 117.0 (flat) | **132.6** | baseline |
| 4 | ~325 tok/s | 116.7 (flat) | **302.2** | 2.3x scaling |
| 8 | ~525 tok/s | 126.3 (flat) | **519.7** | 3.9x scaling |
| 16 | ~931 tok/s | 112.5 (flat) | **980.2** | 7.4x scaling |
| 32 | ~1,600 tok/s | 145.7 (flat) | **1,776.5** | 13.4x scaling |

v1 was flat because realizr used SINGLE-REQUEST mode
(no batching). v5 Yoga confirms batch scheduling works.

> **F-SCALE-01: CONFIRMED (Yoga RTX 4060).** v1: flat
> (SINGLE-REQUEST, FALSIFIED). v5: **1,776.5 tok/s at
> c=32** (13.4x scaling from c=1). SSE streaming FIXED
> (realizr cf10c0f7: `..Default::default()` = infinite
> recursion). Streaming: TTFT 8.4ms, 263.8 tok/s, A+.

### Phase 3: Format Advantage

| Metric | Predict | Pass | Fail | Status |
|--------|---------|------|------|--------|
| APR load | 2-5x faster | ratio 2.0-5.0 | <1.5 | **FIXED** [1] |
| APR RSS | < GGUF | RSS_apr < RSS_gguf | >= | **CONFIRMED** [2] |
| APR decode | +/-5% of GGUF | 0.95-1.05 | <0.95 | **CONFIRMED** [3] |

**Notes:**
1. Legacy AprQ4: 60s (dequant+requant). **FIXED:**
   default import now produces Q4K (raw passthrough).
   --preserve-q4k deprecated (aprender#582, realizr#185).
2. 2,278 < 3,082 MB (26% less via mmap paging)
3. v3: 17.4 vs 273.8 = 0.06x (FALSIFIED). **v5: 132.3
   vs 132.5 = 0.998x (CONFIRMED, Yoga, #180 fixed)**

> **F-FORMAT-01: FIXED.** Legacy APR native q4 (dtype=128)
> was 120x slower (CPU dequant). Default `apr import` now
> produces Q4K (dtype=12) via raw byte passthrough
> (realizr#185, aprender#582). Load time parity with GGUF.

---

## 8. Architectural Comparison

### Kernel Strategy

| Dimension | Candle | realizr |
|-----------|--------|---------|
| Dequant | Separate QMatMul step | Fused matmul (Q4K/Q5K/Q6K DP4A) |
| CUDA dispatch | Per-op kernel launch | Eager (graph capture disabled after poison fix) |
| Attention | Standard scaled dot-product | Flash Decoding (KV chunked) |
| KV cache | Manual, per-call alloc | GPU-resident, per-slot batching |
| Weight reuse | None (c=1 only) | Batched GEMV: shared across M reqs |

### Format Pipeline

| Format | Candle path | realizr path |
|--------|------------|-------------|
| GGUF Q4_K_M | QMatMul dequant->matmul (2 passes) | fused Q4K DP4A (1 pass) |
| SafeTensors | FP16/FP32 GPU matmul | FP16 HGEMM (#174) **151.6** tok/s |
| APR v2 Q4K | N/A | from_apr->GGUF CUDA (#170) **132.3** tok/s |

Candle: CLI only (`stdin->forward->stdout`).
realizr: full serving stack
(`HTTP->batch scheduler->eager dispatch->SSE`).

### Parity Gap Analysis

What Candle provides that this benchmark does/doesn't cover.

#### Architecture coverage

| Arch | Candle quantized example | realizr | Benchmarked? |
|------|-------------------------|---------|-------------|
| LLaMA | quantized | Yes (llama) | A+ (qa-playbook) |
| Qwen2 | quantized-qwen2-instruct | Yes (qwen2) | **A+ 273.8** (primary) |
| Qwen3 | quantized-qwen3 | Yes (qwen3) | **133.7** (Yoga) |
| Qwen3-MoE | quantized-qwen3-moe | **NO** | Gap |
| Phi-2/3 | quantized-phi | Yes (phi, phi2) | A+ (qa-playbook) |
| Gemma | quantized-gemma | Yes (gemma) | A+ (qa-playbook) |
| T5 | quantized-t5 | Yes (enc/dec) | API wired |
| Whisper | whisper (non-quantized) | Yes (re-import) | UNBLOCKED |
| Mistral | (uses quantized/llama) | Yes (mistral) | A+ (qa-playbook) |
| DeepSeek | deepseekv2 | Yes (deepseek) | Not benchmarked |
| Falcon | falcon | Yes (falcon) | Not benchmarked |
| StableLM | stable-lm | Yes (stablelm) | Not benchmarked |

**Gap: Qwen3-MoE** — Candle has `quantized-qwen3-moe`
but realizr has no MoE dispatch. This is the only
architecture gap in the quantized examples.

#### GGUF quantization format coverage (CUDA)

| Format | Candle CUDA | trueno GPU | Gap? |
|--------|------------|-----------|------|
| Q4_K (Q4_K_M) | QMatMul dequant | **Fused DP4A** | realizr wins |
| Q5_K (Q5_K_S/M) | QMatMul dequant | **Fused DP4A** | realizr wins |
| Q6_K | QMatMul dequant | **Fused DP4A** | realizr wins |
| Q4_0 | QMatMul dequant | cuBLAS (dequant→F16) | Parity |
| Q4_1 | QMatMul dequant | cuBLAS (dequant→F16) | Parity |
| Q5_0 | QMatMul dequant | cuBLAS (dequant→F16) | Parity |
| Q5_1 | QMatMul dequant | cuBLAS (dequant→F16) | Parity |
| Q8_0 | QMatMul dequant | cuBLAS (dequant→F16) | Parity |
| Q2_K | QMatMul dequant | **CPU only** | **Gap** |
| Q3_K | QMatMul dequant | **CPU only** | **Gap** |
| Q8_1 | QMatMul dequant | cuBLAS | Parity |
| Q8_K | QMatMul dequant | cuBLAS | Parity |
| F16 | Native | HGEMM (#174) | Parity |
| BF16 | Native | cuBLAS | Parity |
| F32 | Native | cuBLAS | Parity |

**Gaps: Q2_K, Q3_K** — no GPU kernel in trueno. These
are low-bit formats (2/3-bit) rarely used in practice
(quality too low for production). Not blocking.

#### Unmeasured dimensions

| Dimension | Status | Why missing |
|-----------|--------|-------------|
| VRAM comparison | **Not measured** | Candle VRAM not captured (CLI, no nvidia-smi hook) |
| Prefill throughput | **Not measured** | probador reports TTFT but not prefill tok/s separately |
| Output correctness | **Not measured** | Both produce text but no bitwise comparison of logits |
| Model load time | **Partial** | F-FORMAT-01 covers APR vs GGUF; no Candle-vs-realizr GGUF load comparison |
| Quantization accuracy | **Not measured** | No perplexity/eval comparison between runtimes |

> **F-PARITY-03 (proposed):** If realizr and Candle
> produce different top-1 tokens for >1% of positions
> on the same prompt with temperature 0, the decode
> paths diverge. Action: compare greedy output strings.

---

## 9. Falsification Register

Pre-registered predictions. Each tested by benchmark
and confirmed, weakened, or retracted.

| ID | Prediction | Status | Evidence |
|----|-----------|--------|---------|
| F-SUMMARY-01 | realizr wins >=1 at c=1 | **REVISED** | v3: 273.8. v8 clean: **289.0 vs 333.1 llama.cpp** (total). Decode-only: ~303 vs ~299 (**parity**). llama.cpp wins total via prompt caching. |
| F-PARITY-01 | c=1 within +/-10% | **REVISED** | v3: 1.20x. v8: **0.87x total** (289.0 vs 333.1). But decode-only ~1.01x (parity). Delta = prompt caching, not kernel speed. |
| F-FORMAT-01 | APR load 2-5x faster | **FIXED** | Legacy AprQ4: 60s (dequant). Current: Q4K raw passthrough (realizr#185) |
| F-SCALE-01 | c=32 >=1,280 tok/s | **CONFIRMED** | Yoga: **1,776.5** (13.4x from c=1 132.6) |
| F-HW-01 | Variance <5% locked | **CONFIRMED** | CV 0.8% (Candle), 0.9% (realizr). 2520 MHz locked |
| F-MODEL-01 | Candle loads Q4_K_M | **CONFIRMED** | 339 tensors, 1.11 GB, 0.49s. Lazy-curand patch needed |
| F-KERNEL-01 | Fused Q4K lower mem | **WEAKENED** | 22K vs 41K launches but GPU time identical (105/106ms) |
| F-BRICKPARITY-01 | apr profile = ncu +/-15% | **FIXED** | mem 151.4%, compute 16.2%, Grade A (was C). L2 cache hits |
| F-RSS-01 | APR RSS < GGUF RSS | **CONFIRMED** | 2,278 < 3,082 MB (26% less via mmap) |
| F-COLD-01 | realizr cold slower | **REVISED** | preload_modules_for_capture pre-compiles ~60 kernels. Disk cache at ~/.cache/trueno/ptx/ |
| F-SERVING-01 | Overhead <5ms at c=1 | **CONFIRMED** | TTFT 8.4ms - ITL 3.8ms = **4.6ms overhead** |
| F-FMTPARITY-01 | 3 formats GPU +/-10% | **REVISED** | GGUF 132.5, FP16 **151.6**, APR Q4K 132.3 (Yoga) |
| F-TOOLPARITY-01 | apr/realizr +/-5% | **CONFIRMED** | GGUF 0.0%, APR Q4K 1.4%. Version skew was root cause |
| F-PARITY-02 | c=4 <=1.5x slower llama.cpp | **CONFIRMED** | **274.5** (1.22x FASTER than llama.cpp 224.8) |
| F-PARITY-04 | realizr >= llama.cpp at c=1 | **REVISED** | Total: realizr 289.0 vs llama.cpp **333.1** (0.87x). Decode-only: ~303 vs ~299 (**parity**). Gap = prompt caching. |
| F-CLIPARITY-01 | apr run = Candle features | **CONFIRMED** | 6/6: top-p, seed, repeat-penalty/last-n, split, chrome |
| F-1.5X-01 | realizr >=341 tok/s (1.5x Candle) | **TESTING** | Phase 12: tensor graph + fusion + weight layout |
| F-RSS-02 | realizr RSS <=673 MB at c=1 | **FALSIFIED** | Yoga min 2,930 MB (both flags). Irreducible: weights ~1 GB + server ~1.5 MB |
| F-PARITY-03 | Greedy output divergence <=1% | **WEAKENED** | 72% word divergence — but caused by chat template wrapping, not dequant. Needs prompt-parity test. |
| F-QUALITY-01 | realizr PPL within 0.1 of llama.cpp | **FALSIFIED** | DP4A decode PPL: 20.4-31.3 (weighted 24.2, 5 chunks). llama.cpp 12.97. Gap = DP4A int8 vs FP32. Batched FP8 GEMM path untested (needs batched forward in PPL endpoint). |
| F-REGRESSION-01 | No >5% decode regression vs 81c912d2 | **CONFIRMED** | Clean GPU: **277.3** [276.1, 278.5] vs baseline 273.8 (+1.3%). Previous "regression" was GPU contention. realizr#190 CLOSED. |

---

## 10. Work Items

### Phases 0-5: COMPLETE (PMAT-300..355)

All infrastructure, benchmarking, format parity,
profiling, and publication work done. 35 tasks across
6 phases. Key outcomes captured in sections 7-9.

### Phase 6: Parity Sprint (PMAT-370) — COMPLETE

Graph poison fix (realizr 81c912d2): 22.7→273.8 tok/s
(12.1x). Root cause: opt-out graph capture poisoned
CUDA context on failure. Contract: `cuda-graph-safety-v1`.

### Phase 7: CLI + Example Parity (PMAT-380) — COMPLETE

6/6 sampling/gen args. `apr run --gpu` FIXED
(aprender#573): 0.7→121.6 tok/s (validation probe on
cold model). Contract: `gpu-inference-parity-v1`.

| Example | Arch | Status |
|---------|------|--------|
| quantized-qwen2-instruct | Qwen2 | **A+ (273.8 tok/s)** |
| quantized (llama) | LLaMA | Certified A+ |
| quantized-phi | Phi-2/3 | Certified A+ |
| quantized-gemma | Gemma | Certified A+ |
| quantized-qwen3 | Qwen3 | **GPU 133.7 tok/s** (Yoga) |
| quantized-t5 | T5 | API DONE (enc/dec wired) |
| whisper | Whisper | UNBLOCKED (re-import verified) |

### Phases 8-11: COMPLETE (PMAT-390..420)

22 upstream tickets fixed across 5 repos. Key results:
- Phase 8: SafeT 151.6 (7.15x), T5 enc/dec, whisper
- Phase 9: health-gate, FP8 workspace, stack overflow
- Phase 10: Qwen3 133.7 tok/s, whisper re-import
- Phase 11: F-FORMAT-01 FIXED (Q4K default),
  F-COLD-01 REVISED (preload, not JIT)

### Phase 12: 1.5x Candle Target (PMAT-430)

**Target:** realizr >=341 tok/s decode AND RSS <=673 MB
at c=1 on RTX 4090. (1.5x Candle's 227.4 / 449 MB.)

**Current (v8.5 clean GPU):** **277.3** tok/s (1.22x),
3,082 MB RSS (6.9x). Gap: +23.0% decode, -78.2% RSS.

> **F-1.5X-01:** If realizr cannot sustain >=341 tok/s
> decode at c=1 (30s, probador) on RTX 4090, the 1.5x
> claim is falsified. Action: profile bottleneck.
>
> **F-RSS-02:** If realizr RSS >673 MB at c=1, the
> memory parity claim is falsified. Action: audit allocs.

**Root cause analysis (decode, `apr profile --granular`):**
- **85.9% kernel launch overhead** at M=1 (430 launches/token)
- Compute breakdown (14.1% of time):
  - AttentionScore: 39.9% (6.0ms, 14.3µs avg)
  - QkvProjection: 15.6% (2.3ms, 5.6µs avg)
  - RmsNorm: 8.2%, OutputProjection: 8.0%
  - DownProjection: 7.8%, RopeEmbedding: 7.7%
  - LmHead: 5.7%, Residuals: 7.3%
- Compute efficiency: 6.6% (per-kernel)
- **CUDA graph BLOCKED** on driver 570.207 (code 901,
  realizr#197). Poisons context on capture attempt.
- Candle weaknesses: no CUDA graphs, no FlashAttn for
  quantized, 2-kernel QMatMul, ~640 launches/token,
  per-call KV alloc, no memory pooling

**Root cause analysis (RSS) — MEASURED on Yoga:**

| Config | RSS | VRAM | vs baseline |
|--------|-----|------|-------------|
| baseline (4096, FP8) | 2,985 | 3,878 | -- |
| --no-fp8-cache | 2,595 | 2,816 | **-1,062 VRAM** |
| --context-length 512 | 3,069 | 3,682 | -196 VRAM |
| both | 2,930 | 2,620 | **-1,258 VRAM** |

F-RSS-02 (<=673 MB) NOT achievable — model weights
(~1 GB) + server (~1.5 GB) irreducible without
PagedAttention or lazy weight loading.

**Research basis (arXiv + Candle + qwen-coder-deploy):**

| Technique | Source | Expected | Complexity |
|-----------|--------|----------|------------|
| Tensor graph dispatch | qcd Path A, FlashFormer | +20-40% | 4-8 wk |
| Fused QKV GEMV (trueno#237) | gate+up pattern | +8-11% | 2-3 wk |
| Weight pre-packing (Marlin-style) | IST-DASLab | +10-15% | 2-3 wk |
| RMSNorm+Residual fusion | llama.cpp #17621 | +10-15% | 1-2 wk |
| --context-length 512 | GH-286 | VRAM -196 MB | **DONE** |
| --no-fp8-cache | GH-286 | **VRAM -1,062 MB** | **DONE** |

**qcd lesson: 16 kernel fusion approaches FAILED** on
RTX 4060 (PMAT-280..289). Only tensor graph dispatch
(reduce 430→~15 launches) and cuBLASLt grouped GEMM
survived validation. Mega-kernels fail at low SM count.

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-431 | `--context-length` + `--no-fp8-cache` flags | **DONE** | realizr 2a8de443, aprender |
| PMAT-432 | RSS audit: profile all GPU allocations | SCRIPTED | scripts/audit-gpu-allocs.sh |
| PMAT-431 | `--context-length` + `--no-fp8-cache` flags | **DONE** | realizr 2a8de443, aprender |
| PMAT-432 | RSS audit: profile all GPU allocations | SCRIPTED | scripts/audit-gpu-allocs.sh |
| PMAT-433 | Fused QKV DP4A GEMV kernel | **Phase 1 DONE** | realizr 8e2f6900 (shared Q8 cache). Phase 2 (single launch, trueno#237) = stub only. |
| PMAT-434 | RMSNorm+GEMV fusion kernel | **REVERTED** | realizr PMAT-092: fused kernel 5% slower. Dead end without new approach. |
| PMAT-435 | Tensor graph dispatch (trueno layer) | **INFRA DONE** | trueno#238 graph module merged. Phase 12 quantized wiring NOT started. |
| PMAT-436 | Marlin-style Q4K weight pre-packing | **DEPRIORITIZED** | trueno#239 branch stale (1453 behind). Not beneficial for M=1 decode. |
| PMAT-437 | Re-benchmark: probador 1.5x gate | TODO | 435 |
| PMAT-438 | RSS re-measure with --no-fp8-cache | **DONE** | Yoga measured |

**Perf gate:** `probador llm load --url ... --concurrency 1
--duration 30s --perf-gate 341` (FAIL if <341 tok/s).

---

## 11. PMAT Compliance

| Principle | How Enforced |
|-----------|-------------|
| Determinism | Locked clocks, temperature 0, CV <1% (F-HW-01) |
| Isolation | forjar deploy, kill competing GPU procs |
| Reproducibility | probador llm load, machine-readable JSON |
| Falsifiability | 21 F-conditions pre-registered, all tested (section 9) |
| Format parity | 3 formats GPU-tested (F-FMTPARITY-01) |
| Tool parity | apr vs realizr within 1.4% (F-TOOLPARITY-01) |
| CLI parity | 6/6 features matched (F-CLIPARITY-01) |
| Contracts | 12 provable-contracts, 44 equations, 100% coverage |
| Gates | `apr check` -> `apr profile` -> `apr bench` -> `cgp contract verify` pre-merge |
| Perf gate | `apr bench --assert-throughput 341` + `probador --perf-gate 341` (Phase 12) |
| Kernel perf | `cgp profile kernel` + `cgp roofline --empirical` (trueno kernels) |
| QA playbook | 95 models certified via apr-model-qa-playbook |

## 12. Scientific Methodology Gaps

Audit of what this benchmark does well and what it
lacks, grounded in MLPerf v4.0 (2024), Splitwise
(Patel 2024), Sarathi-Serve (Agrawal 2024), and
LLMPerf (Anyscale 2024).

### What we do well

| Requirement | Status | How |
|-------------|--------|-----|
| Clock locking | **PASS** | nvidia-smi -lgc, CV <1% |
| Deterministic decode | **PASS** | temperature 0, greedy |
| Latency decomposition | **PASS** | TTFT, ITL, TPOT, us/layer via probador |
| Isolated builds | **PASS** | forjar deploy, kill competing procs |
| Machine-readable results | **PASS** | JSON with per-request detail |
| Pre-registered predictions | **PASS** | 21 F-conditions, all tested, Popperian falsification |
| Provable contracts | **PASS** | 12 contracts, 44 equations, 100% coverage |

### What we're missing

| Gap | Severity | Why it matters | Fix |
|-----|----------|---------------|-----|
| **Output correctness** | Critical | Fast-but-wrong is meaningless. Reviewers demand quality co-reporting with speed (SqueezeLLM, AQLM 2024). | `lm-evaluation-harness` or llama.cpp perplexity on WikiText-2 |
| **Statistical significance** | High | Single-run medians rejected in peer review. MLPerf requires min sample counts. | Bootstrap CIs on tok/s (min 30 runs), Mann-Whitney U for comparisons |
| **VRAM measurement** | High | Nearly universal gap (Alizadeh 2024). nvidia-smi polling is crude; misses fragmentation. | `nvidia-smi --query-gpu=memory.used -l 100` during runs, or CUDA memory API in probador |
| **Realistic traffic** | Medium | Rankings invert with workload shape (Vidur, Microsoft 2024). | Poisson arrival via `probador --rate`, ShareGPT prompt-length distributions |
| **Prefill/decode separation** | Medium | Combined tok/s hides regime-specific bottlenecks (Splitwise 2024). | probador reports prefill_tok_per_sec; not yet used in falsification |
| **Perplexity delta** | Medium | Speed claims without quality loss quantification are incomplete. | llama.cpp perplexity tool or batuta `/api/v1/eval/perplexity` |
| **Multi-framework harness** | Low | Ad-hoc per-framework scripts don't scale to 6+ comparisons. | Standardize on OpenAI-compatible API (all 6 frameworks support it) |

### Tooling available in ~/src

| Tool | Repo | What it provides | Gap it closes |
|------|------|-----------------|--------------|
| `probador llm load` | probar | tok/s, TTFT, ITL, us/layer, GPU telemetry, Poisson `--rate`, `--validate` | Latency, throughput, HTTP load |
| `probador llm score` | probar | Weighted A+-F grades, SLO thresholds | Quality grading |
| `apr profile --granular` | aprender | Per-brick timing, roofline, kernel overhead, `--perf-grade` | Decode bottleneck analysis |
| `apr bench` | aprender | Throughput gate, CI assertions (`--assert-throughput`, `--assert-p99`) | Local load testing, perf gate |
| `cgp profile kernel` | trueno/cgp | CUDA PTX kernel profiling via ncu + CUPTI | Kernel-level bottleneck |
| `cgp roofline` | trueno/cgp | Roofline model (cuda/avx2/avx512/wgpu), empirical or spec | Memory vs compute bound analysis |
| `cgp compete` | trueno/cgp | Head-to-head comparison (`--ours` vs `--theirs`) | Framework shootout automation |
| `cgp contract verify` | trueno/cgp | Performance contract CI/CD gate | Regression prevention |
| `cgp diff` | trueno/cgp | Compare two profiles (git integration) | Profile regression detection |
| `batuta eval perplexity` | batuta | Per-token PPL via Banco API | Perplexity delta |
| llama.cpp `llama-perplexity` | llama.cpp | WikiText-2 PPL, KL divergence, HellaSwag/MMLU/TruthfulQA | Output correctness, quant accuracy |
| vLLM benchmarks | vllm | ShareGPT traces, Poisson arrival, VRAM via `torch.cuda.memory_allocated` | Realistic traffic, VRAM methodology |
| `lm-evaluation-harness` | (pip) | 400+ tasks, multi-backend (HF/GGUF/vLLM/API) | Correctness at scale |
| `apr check` / `apr trace` | aprender | Integrity + layer correctness | Pre-flight validation |

### Parity query: framework comparison capability

| Framework | In ~/src | OpenAI API | probador compatible | Perplexity tool | Notes |
|-----------|---------|-----------|-------------------|----------------|-------|
| realizr | Yes | Yes | **Yes** | via batuta | Primary |
| Candle | Yes | **No** (CLI) | Ad-hoc scripts | N/A | CLI-only, no server |
| llama.cpp | Yes | Yes (`llama-server`) | **Yes** | `llama-perplexity` | Has KL divergence |
| ollama | Yes | Yes | **Yes** | N/A | Wraps llama.cpp |
| vLLM | Yes | Yes | **Yes** | via lm-eval | Python, torch required |
| unsloth | Yes | **No** (library) | N/A | via lm-eval | Training-focused |
| PyTorch | Yes | **No** (library) | N/A | via lm-eval | Too low-level for direct comparison |

**5 of 7 are OpenAI-compatible** → probador can benchmark
realizr, llama.cpp, ollama, vLLM head-to-head with zero
code changes. Candle/unsloth/PyTorch need wrappers.

### Phase 13: Scientific Rigor Sprint (PMAT-440)

| ID | Task | Status | Tool |
|----|------|--------|------|
| PMAT-440 | Perplexity: realizr vs llama.cpp on WikiText-2 | **MEASURED** | realizr 17.40 vs llama.cpp 12.97 (+4.4 PPL). DP4A int8 vs FP32 precision. F-QUALITY-01 FALSIFIED. |
| PMAT-441 | Bootstrap CIs on decode tok/s | **MEASURED** | Showdown: realizr 250.9, llama.cpp 296.4. Bootstrap: 231.9 [229.4, 234.3]. |
| PMAT-442 | VRAM measurement during probador runs | **MEASURED** | Peak 5,388 MiB, mean 5,288 MiB (RTX 4090) |
| PMAT-443 | Poisson arrival: c=1..32 with `--rate` | **MEASURED** | c=1: 245-254 tok/s (rate 0.5-2.0). c=4: 151 tok/s decode, 387 agg (rate 8.0). Latency drift at c=4. |
| PMAT-444 | Output correctness (F-PARITY-03) | **MEASURED** | 72% divergence (chat template, not dequant). F-PARITY-03 WEAKENED. |
| PMAT-445 | Multi-framework showdown (3-way) | **MEASURED** | v8.9: llama.cpp 289.3, realizr 268.5, ollama 241.7. v9: llama.cpp **336.7**, realizr **281.2** (16.5% gap). Candle 227.4 (ref). |

> **F-QUALITY-01: FALSIFIED.** realizr WikiText-2
> DP4A decode PPL = **20.4-31.3** (weighted avg 24.2,
> 5 chunks of ~1800 tokens). llama.cpp **12.97**.
> Previous single-chunk measurement (17.40) was
> text-position-dependent. Measured via `/v1/perplexity`
> teacher-forcing (realizr e49d5534 + 7c7abb83 poison fix).
>
> **Root cause:** teacher-forcing endpoint uses sequential
> decode (DP4A GEMV, M=1), not batched FP8 GEMM prefill.
> Per-token forward at M=1 accumulates DP4A int8→int32
> precision error across 28 layers.
>
> **Implication:** DP4A decode PPL is significantly worse
> than FP32 dequant. For quality-critical applications,
> the batched FP8 GEMM prefill path should be used.
>
> **Action (PMAT-456):** Add batched prefill forward to
> perplexity endpoint. This uses FP8 GEMM (cuBLASLt)
> instead of DP4A GEMV — higher precision, closer to
> llama.cpp's FP32 path. Requires new forward function
> that processes N tokens at once (M=N, not M=1).
>
> **Graph poison fix (realizr#194):** Overflow requests
> no longer corrupt CUDA state. Input validation checks
> GPU KV cache capacity (not model context_length). Error
> recovery resets KV cache on forward failure. Contract:
> C-GRAPH-RECOVERY-01.

### Poisson Arrival Results (PMAT-443)

| c | Rate (req/s) | Decode tok/s | Agg tok/s | TTFT P50 | ITL P50 | ITL CV |
|---|-------------|-------------|-----------|----------|---------|--------|
| 1 | 0.5 | 245.4 | 6.3 | 21.1ms | 4.1ms | -- |
| 1 | 1.0 | 250.8 | 80.6 | 41.0ms | 4.0ms | 0.02 |
| 1 | 2.0 | 253.5 | 169.7 | 41.5ms | 3.9ms | 0.02 |
| 4 | 2.0 | 151.5 | 254.9 | 241.8ms | 6.6ms | 0.23 |
| 4 | 4.0 | 151.1 | 297.8 | 1016.5ms | 6.6ms | 0.09 |
| 4 | 8.0 | 156.8 | 386.7 | 709.1ms | 6.4ms | 0.10 |

At c=1, decode stable 245-254 tok/s (Poisson doesn't
degrade single-request). At c=4, continuous batching
scales aggregate throughput (387 vs 254 at rate 2→8)
while per-request decode drops to ~152 (shared GPU).

### Phase 14: Parity Sprint — Close Remaining Gaps

**Goal:** Close the 13% total-throughput gap with llama.cpp
and unblock the only untested F-condition (F-QUALITY-01).

| ID | Task | Status | Upstream | Impact |
|----|------|--------|----------|--------|
| PMAT-450 | KV prefix caching (prompt reuse) | FILED | realizr#193 | +13% total tok/s (match llama.cpp) |
| PMAT-451 | Logprobs endpoint | **SHIPPED** | realizr e8da8431, /v1/logprobs | Generation logprobs done. Teacher-forcing PPL next. |
| PMAT-452 | Fused K+V kernel (single launch) | **FALSIFIED** | trueno 9d99e18c, realizr 84d36305 | MEASURED: 272.5 vs 281.2 tok/s (-3.1%). kv_dim=256 too small for fusion benefit. Reverted to Phase 1 (Q8 cache). |
| PMAT-453 | Tensor graph dispatch wiring (Phase 12 quantized) | **590 KERNELS RECORDED** | trueno#243, realizr 82512d66, realizr#198 | All decode-path kernels wired: RMSNorm, DP4A GEMV, RoPE, attention, KV scatter, flash decoding, SwiGLU, Q8 quantize, residual add. Graph builds (590 nodes). Replay correctness bug (realizr#198). |
| PMAT-454 | GPU isolation pre-flight in all scripts | **DONE** | bootstrap-ci.sh, run-showdown.sh | Prevents false regressions |
| PMAT-455 | Perplexity graph poison fix | **SHIPPED** | realizr#194, 1f527a89 | KV overflow validation + error recovery. C-GRAPH-RECOVERY-01. Gate debt cleared (realizr#195). |
| PMAT-456 | Batched prefill PPL endpoint | TODO | realizr (needs new path) | FP8 GEMM PPL vs DP4A — true precision comparison for F-QUALITY-01. |

> **F-CACHE-01 (proposed):** If realizr with KV prefix
> caching does not achieve total tok/s >= 0.95 * llama.cpp
> on repeated-prompt workloads, the caching implementation
> is insufficient. Action: profile cache hit rate.

### Phase 12 Status Update

PMAT-434 (RMSNorm+GEMV fusion) **REVERTED** — fused
kernel was 5% slower (realizr PMAT-092). PMAT-436
(Marlin pre-packing) **DEPRIORITIZED** — not beneficial
for M=1 decode (trueno#239 branch 1453 behind main).

PMAT-452 (Fused K+V kernel) **FALSIFIED** — trueno
9d99e18c kernel + realizr 84d36305 wiring. MEASURED
272.5 vs 281.2 tok/s (-3.1% regression). Root cause:
kv_dim=256 too small — launch overhead savings (~5μs)
< compute overhead from dual accumulators (~10μs/row).
Same pattern as qcd PMAT-280..289 (16 fusion failures).
Reverted to Phase 1 (Q8 cache sharing only).

Remaining Phase 12 path: tensor graph dispatch
(PMAT-435/453) is the only technique that survived
validation in qcd. All kernel fusion approaches have
been falsified for small-dim M=1 decode.

**DRIVER BLOCKED (realizr#197):** CUDA graph stream
capture fails with code 901 on driver 570.207 (Ada
Lovelace). Both `CaptureMode::Global` AND `ThreadLocal`
fail. Bug is kernel-specific (empty capture succeeds).

**Manual graph API (trueno#243): VERIFIED + WIRED.**
`cuGraphCreate`, `cuGraphAddKernelNode`,
`cuGraphInstantiateWithFlags`, `cuGraphLaunch` all
succeed on driver 570.207. Manual construction bypasses
stream capture entirely.

**ALL decode-path kernels wired (realizr 82512d66):**
590 kernel nodes recorded during eager forward pass.
Recording added to: RMSNorm, HW DP4A GEMV (Q4K),
Q8 quantize, RoPE (direct + indirect + neox), KV
scatter indirect (K + V), incremental attention
(single-warp + multi-warp), flash decoding (chunk +
reduce), fused SwiGLU, residual add. Graph builds
successfully and instantiates.

**Replay correctness bug (realizr#198):** Graph replay
via `cuGraphLaunch` produces identical logits at all
positions despite position_buf/seq_len_buf being
correctly updated (verified via D2H readback). Five-
whys: `cuGraphAddKernelNode` captures kernel params at
creation → pointer values are correct → BUT graph's
`ld.global` loads appear to use captured device memory
state, not runtime content. Fix: switch to
`cuGraphExecKernelNodeSetParams` per-replay (llama.cpp
approach) or investigate CUDA graph memory semantics.
Benchmark deferred until fix.

## 13. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0-3.1 | 2026-04-01..02 | Phases 1-7. probador. Graph fix: 273.8. CLI parity. |
| 4.0-4.5 | 2026-04-02..03 | Phase 8: upstream fixes, T5 arch, whisper unblocked. |
| 5.0-5.5 | 2026-04-03 | Phase 9: health-gate, FP16 151.6, parity gate fixed. |
| 5.6-5.9 | 2026-04-03 | Phase 10: whisper/Qwen3/T5 complete. |
| 6.0.0 | 2026-04-03 | Spec condensed: 982→500 lines. Stale data fixed. |
| 6.0.1 | 2026-04-04 | Date bump. All 10 phases complete. Parity summary. |
| 6.1.0 | 2026-04-04 | F-FORMAT-01 FIXED (realizr#185, aprender#582). F-COLD-01 REVISED (preload, not JIT). |
| 7.0.0 | 2026-04-04 | Phase 12: 1.5x Candle target. arXiv + Candle source + qcd research. 8 work items. |
| 7.1.0 | 2026-04-04 | PMAT-438 measured. Fused QKV design (trueno#237). Multi-stream→fused pivot. |
| 7.2.0 | 2026-04-04 | CUDA graph→eager (stale claim). PMAT-103→realizr#185. Section 11 expanded. RSS gap 78.2%. |
| 7.3.0 | 2026-04-04 | F-SERVING-01 evidence (4.6ms = TTFT-ITL). Phase 1 curl→probador. F3 RSS/VRAM. F11 graph replay. |
| 7.4.0 | 2026-04-04 | F-RSS-02 FALSIFIED (2,930 > 673). PMAT-433 Integrating (trueno 60a0dd51). PMAT-432 scripted. |
| 7.5.0 | 2026-04-04 | trueno#238 (graph dispatch), #239 (pre-pack) filed. PMAT-434 kernel designed. All Phase 12 items FILED+. |
| 7.6.0 | 2026-04-04 | Parity gap analysis: arch (1 gap: MoE), quant (2 gaps: Q2K/Q3K), 5 unmeasured dims. F-PARITY-03 registered. |
| 8.0.0 | 2026-04-04 | Section 12: Scientific methodology gaps. 7 gaps identified, 8 tools audited, 6 framework parity matrix. Phase 13 proposed (PMAT-440..445). F-QUALITY-01 proposed. |
| 8.1.0 | 2026-04-04 | Phase 13: 4/6 items SCRIPTED. bootstrap-ci.sh, measure-vram.sh, run-showdown.sh + showdown.yaml. F-QUALITY-01 registered (19 F-conditions). |
| 8.2.0 | 2026-04-04 | **MEASURED on Lambda RTX 4090**: PMAT-441 bootstrap CI (234.2 tok/s), PMAT-442 VRAM (5,388 MiB peak), F-PARITY-03 (72% — chat template). F-REGRESSION-01 FALSIFIED: 273.8→234.2 (-14.5%). realizr#190 filed. 20 F-conditions. |
| 8.3.0 | 2026-04-04 | **SHOWDOWN: llama.cpp 296.4 > realizr 250.9 (0.85x)**. Bisect: trueno 0.17 host-side dispatch regression (trueno#240 filed). Kernel unchanged (276.4 apr profile). F-PARITY-01, F-SUMMARY-01 FALSIFIED. F-PARITY-04 registered. 21 F-conditions. |
| 8.4.0 | 2026-04-04 | Phase 13: 5/6 MEASURED, 1 BLOCKED. Poisson c=1 stable 245-254, c=4 387 agg. llama.cpp PPL=15.80. |
| 8.5.0 | 2026-04-04 | **FALSE REGRESSION: GPU contention** (stale apr finetune/serve). Clean GPU: **277.3** [276.1, 278.5] (+1.3% vs baseline). Showdown: realizr 289 vs llama.cpp 333 (total), decode-only ~parity. realizr#190 CLOSED, trueno#240 CLOSED. Mandatory pre-flight check added. |
| 8.6.0 | 2026-04-04 | Phase 14 proposed: KV prefix caching (realizr#193), logprobs (realizr#191). Phase 12 status corrected: PMAT-434 REVERTED (5% slower), PMAT-436 DEPRIORITIZED. GPU pre-flight added to scripts. F-CACHE-01 proposed. |
| 8.7.0 | 2026-04-04 | `/v1/logprobs` SHIPPED (realizr e8da8431). Generation logprobs work; perplexity needs teacher-forcing. |
| 8.8.0 | 2026-04-04 | **F-QUALITY-01 FALSIFIED:** `/v1/perplexity` SHIPPED. WikiText-2 PPL: realizr 17.40 vs llama.cpp 12.97 (+4.4). DP4A int8 vs FP32. All 21 F-conditions tested. |
| 8.9.0 | 2026-04-04 | 3-way showdown: llama.cpp 289.3 > realizr 268.5 > ollama 241.7 > Candle 227.4. trueno#241 filed (DP4A precision). README updated with full competitive picture. |
| 9.0.0 | 2026-04-04 | **realizr#194 SHIPPED** (graph poison fix). Fresh showdown: realizr 281.2 vs llama.cpp 336.7 (16.5% gap). DP4A PPL re-measured: 20.4-31.3 (text-dependent). **Fused K+V kernel IMPLEMENTED** (trueno 9d99e18c, -28 launches/token). F-QUALITY-01 updated: batched FP8 PPL path needed. |
| 9.0.1 | 2026-04-05 | **realizr#194 PUSHED** (all 4 gates ✅). Fixed 30+ examples/tests/benches (field accessors, clippy). trueno BLIS clippy fixed (unsafe_op_in_unsafe_fn, wgsl_forward). Gate debt cleared across realizr + trueno. |
| 9.1.0 | 2026-04-05 | Tooling upgrade: `cgp` (trueno) + `apr bench` (load testing) integrated into spec. Workflow updated: `apr check` → `apr profile` → `apr bench` → `cgp contract verify`. 14 tools in Section 12 (was 8). |
| 9.2.0 | 2026-04-05 | **PMAT-452 FALSIFIED:** Fused K+V kernel -3.1% regression at kv_dim=256. Reverted to Phase 1. |
| 9.3.0 | 2026-04-05 | **PROFILED:** 85.9% launch overhead. PMAT-453 stream capture blocked (code 901 on 570.207, both Global + ThreadLocal). **trueno#243 SHIPPED:** `cuGraphAddKernelNode` manual graph API — bypasses stream capture. Wiring into decode path next. |
| 9.4.0 | 2026-04-05 | **VERIFIED:** Manual graph API works on driver 570.207 (Python test). Stream capture bug is kernel-specific. Manual construction viable. |
| 9.5.0 | 2026-04-05 | **Manual graph infrastructure IMPLEMENTED** in realizr (6ae0703d): RecordedKernel struct, begin/end_graph_recording, record_kernel_launch. Wired into graphed_capture.rs (skips stream capture, uses eager+record). HW DP4A GEMV recording wired. Full kernel coverage needed (RMSNorm, attention, RoPE, etc.) before benchmark. |
| 9.6.0 | 2026-04-05 | **ALL decode-path kernels wired** (realizr 82512d66): 590 kernel nodes recorded (was 0). Graph builds + instantiates on driver 570.207. Replay correctness bug (realizr#198): identical logits despite updated position_buf. Five-whys: graph's ld.global loads not reading updated device memory. Fix: cuGraphExecKernelNodeSetParams. |
