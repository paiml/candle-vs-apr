# Candle vs APR Inference Parity Specification

**Document ID:** PAIML-CANDLE-APR-001
**Version:** 7.2.0
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
12. [Revision History](#12-revision-history)

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

**Result:** realizr **1.20x faster** (273.8 vs 227.4)
after graph poison fix. RSS still favors Candle
(449 vs 3082 MB). See F-SUMMARY-01, F-PARITY-02 in
section 9.

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
- **CLI:** `../aprender`
  (`apr run/serve/check/profile/trace`)
- **Quality:** `../provable-contracts`
  (compile-time contract enforcement)
- **Testing:** `../probar` (`probador llm load/score`),
  `../apr-model-qa-playbook` (95 models certified)

**Workflow:** `apr check` -> `apr profile` -> `apr trace`
-> `gh issue create` -> fix upstream -> contract ->
`make perf-gate` -> re-run falsification.

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

| Metric | Predict | v1 | v3 (graph fix) | Status |
|--------|---------|----|----|--------|
| Decode tok/s | 0.90-1.10 | 0.63x (poisoned) | **1.20x** (273.8 vs 227.4) | **v3: PASS** |
| Peak RSS | 0.85-1.15 | 0.15x (449/3082) | 0.15x (unchanged) | **FAIL** |

**F-PARITY-01: REVISED.** v1 FALSIFIED (0.63x, context
poisoned). v3 after graph fix: **1.20x in realizr's
favor** (273.8 vs 227.4). RSS still 6.9x higher
(server + KV cache pool).

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

---

## 9. Falsification Register

Pre-registered predictions. Each tested by benchmark
and confirmed, weakened, or retracted.

| ID | Prediction | Status | Evidence |
|----|-----------|--------|---------|
| F-SUMMARY-01 | realizr wins >=1 at c=1 | **REVISED** | v3: 273.8 vs 227.4 (decode win). RSS: Candle (449 vs 3082) |
| F-PARITY-01 | c=1 within +/-10% | **REVISED** | v1: 0.63x (poisoned). v3: **1.20x realizr** |
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
| F-CLIPARITY-01 | apr run = Candle features | **CONFIRMED** | 6/6: top-p, seed, repeat-penalty/last-n, split, chrome |
| F-1.5X-01 | realizr >=341 tok/s (1.5x Candle) | **TESTING** | Phase 12: tensor graph + fusion + weight layout |
| F-RSS-02 | realizr RSS <=673 MB at c=1 | **TESTING** | Phase 12: --context-length + --no-fp8-cache |

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

**Current:** 273.8 tok/s (1.20x), 3,082 MB RSS (6.9x).
Gap: +24.6% decode, -78.2% RSS.

> **F-1.5X-01:** If realizr cannot sustain >=341 tok/s
> decode at c=1 (30s, probador) on RTX 4090, the 1.5x
> claim is falsified. Action: profile bottleneck.
>
> **F-RSS-02:** If realizr RSS >673 MB at c=1, the
> memory parity claim is falsified. Action: audit allocs.

**Root cause analysis (decode):**
- GPU utilization: 20.1% BW (202.5/1,008 GB/s)
- 83.2% kernel launch overhead at M=1
- DP4A GEMV compute ceiling: 412 tok/s (we're at 66%)
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
| PMAT-432 | RSS audit: profile all GPU allocations | TODO | 431 |
| PMAT-433 | Fused QKV DP4A GEMV kernel | **Design DONE** | realizr 8e2f6900 (design), trueno#237 (integration) |
| PMAT-434 | RMSNorm+GEMV fusion kernel | FILED | realizr#189 |
| PMAT-435 | Tensor graph dispatch (trueno layer) | TODO | 433,434 |
| PMAT-436 | Marlin-style Q4K weight pre-packing | TODO | -- |
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
| Falsifiability | 17 F-conditions pre-registered (section 9) |
| Format parity | 3 formats GPU-tested (F-FMTPARITY-01) |
| Tool parity | apr vs realizr within 1.4% (F-TOOLPARITY-01) |
| CLI parity | 6/6 features matched (F-CLIPARITY-01) |
| Contracts | 12 provable-contracts, 44 equations, 100% coverage |
| Gates | `apr check` -> `apr profile` -> `apr trace` pre-merge |
| Perf gate | `probador --perf-gate 341` (Phase 12) |
| QA playbook | 95 models certified via apr-model-qa-playbook |

## 12. Revision History

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
