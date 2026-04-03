# Candle vs APR Inference Parity Specification

**Document ID:** PAIML-CANDLE-APR-001
**Version:** 5.7.0
**Last Updated:** 2026-04-03
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

**Step 1: Why Candle specifically?** Both are pure Rust,
both load GGUF Q4_K_M, both target CUDA. This isolates
architectural decisions (fused kernels, format design,
serving layer) from language/runtime differences. Comparing
realizr to llama.cpp/vLLM mixes Rust-vs-C++ and
Rust-vs-Python performance characteristics.

**Step 2: Why c=1 as the primary comparison?** Candle has
no server mode — it's CLI-only. The only fair head-to-head
is single-request decode. Concurrent benchmarks (c=4..32)
demonstrate what Candle architecturally cannot provide.

**Step 3: What constitutes a win?** realizr must
demonstrate measurable advantage in at least one of:
decode throughput, model load time, or memory footprint
at c=1. **v1 result: Candle won (F-SUMMARY-01
FALSIFIED).** **v3 result: realizr wins decode (273.8 vs
227.4, F-PARITY-02 CONFIRMED)** after graph poison fix.
RSS still favors Candle (449 vs 3082 MB).

> **F-SUMMARY-01: REVISED.** v1 (FALSIFIED): Candle
> 1.59x faster with poisoned context. v3 (graph fix):
> **realizr 1.20x faster** (273.8 vs 227.4 tok/s).
> RSS still favors Candle. The "fused-kernel advantage"
> was masked by a CUDA driver bug, not absent.

> **F-PARITY-02: CONFIRMED.** After fixing CUDA graph
> poison bug (realizr 81c912d2), probador llm load
> measures **273.8 tok/s decode at c=1** (was 22.7,
> 12.1x improvement) and **274.5 tok/s at c=4** (1.22x
> FASTER than llama.cpp 224.8). realizr now **beats both
> Candle (227.4) and llama.cpp (224.8)** at decode
> throughput.

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

**Before building ANY new feature:** `pv coverage`
(provable-contracts) -> `pmat query` (find existing
code) -> `batuta oracle` (architecture guidance) -> THEN
implement with contract-first design.

### Measure-and-Fix Policy

**Measure:** `apr check` -> `apr profile --granular` ->
`apr trace --verbose` -> `apr cbtop --headless`. NVIDIA
`nsys`/`ncu` as parity validation.

**Fix:** `gh issue create` -> fix upstream ->
`provable-contracts` binding -> `apr trace`/`apr profile`
verify -> `make perf-gate` (probador >=200 tok/s) ->
rebuild -> re-run falsification.

### Sister repos

qwen-coder-deploy (llama.cpp/vLLM/ollama) .
qwen-train-canary (training) . aprender (apr CLI) .
realizar (engine) . trueno (kernels) .
provable-contracts (contracts) .
**apr-model-qa-playbook** (95 models certified,
18 tests/model)

---

## 3. Hardware Targets

### Lambda Vector (PRIMARY — RTX 4090)

| Property | Value |
|----------|-------|
| GPU | NVIDIA RTX 4090 |
| Compute | sm_89 (Ada Lovelace) |
| VRAM | 24 GB GDDR6X |
| Memory BW | 1,008 GB/s |
| SMs | 128 |
| Clock | Locked 2520 MHz (nvidia-smi -lgc 2520) |
| CUDA | 12.6 Toolkit / 12.8 Driver (570.207) |
| CPU | Intel Xeon |
| Transport | Local |

### Yoga (SECONDARY — RTX 4060 Laptop)

| Property | Value |
|----------|-------|
| GPU | NVIDIA RTX 4060 Laptop GPU |
| Compute | sm_89 (Ada Lovelace) |
| VRAM | 8 GB GDDR6 |
| Memory BW | 256 GB/s |
| SMs | 24 |
| Clock | Locked 1900 MHz |
| CUDA | 12.6 Toolkit / TBD Driver |
| Transport | SSH (192.168.50.38) |

> **F-HW-01:** If run-to-run variance exceeds 5% with
> locked clocks, the determinism claim is falsified.
> Action: investigate thermal throttle or background GPU
> processes.

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
  --preserve-q4k --arch qwen2 \
  -o ~/models/qwen2.5-coder-1.5b-instruct-q4k.apr
```

`--preserve-q4k` keeps Q4_K superblock layout intact for
fused DP4A kernels; without it, weights are dequantized
to F32 and requantized.

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
| Iterations | 10 (drop first for cold-start) |
| Measurement | Wall time, tok/s (runtime output) |
| Memory | Peak RSS via `/usr/bin/time -v` (Candle) |
| Isolation | forjar deploy, kill competing GPU procs |
| Clock | Locked (nvidia-smi -lgc) |

**Candle:** `quantized-qwen2-instruct --model <gguf>
--prompt <text> --sample-len 256 --temperature 0`

**realizr:** `curl /v1/chat/completions` with
`stream: false`, `temperature: 0`,
extract `usage.completion_tokens`

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

**Invariant:** All 3 formats must have GPU inference.
Both `apr` CLI and raw `realizr` must produce equivalent
results. Any format without a GPU path is a bug, not a
limitation.

#### 3a. Format parity (F-FMTPARITY-01)

| Format | Candle | realizr v1 | v3 (4090) | v5 (Yoga) | Status |
|--------|--------|-----------|-----------|-----------|--------|
| GGUF Q4_K_M | 227.4 | 142.8 | **273.8** | 132.5 | realizr wins |
| FP16 APR | N/A | -- | 21.2 | **151.6** | #180 FIXED, 7.15x |
| APR v2 Q4K | N/A | -- | 17.4 | **132.3** | parity with GGUF |

v5 Yoga: all 3 formats GPU, within 14.4% (FP16 fastest).
v3 SafeT/APR gaps were bugs (#169 F32 SGEMM, #170 dequant, #180 F16-as-F32 dtype). All fixed.

#### 3b. Tool parity: `apr` CLI vs `realizr`

| Format | Tool | v1 | v3 (4090) | v5 (Yoga) | Delta |
|--------|------|----|-----------|-----------|-------|
| GGUF | `realizr serve` | 142.8 | 273.8 | 132.5 | 0.0% ✅ |
| GGUF | `apr serve run` | 139.8 | 273.8 | 132.5 | |
| APR Q4K | `realizr serve` | -- | 17.4 | **132.3** | 1.6% ✅ |
| APR Q4K | `apr serve run` | -- | 21.9 | **130.4** | |
| FP16 APR | `realizr serve` | -- | -- | **151.6** | N/A |

Both tools on the same model/format within +/-5%.
GGUF: **0.0%** (132.5 vs 132.5). APR Q4K: **1.6%**
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

**Cross-reference:** qwen-coder-deploy v2 baseline
(c=4, 60s, 3 runs, 95% CI): llama.cpp 224.8 tok/s,
apr 107.7 tok/s (2.1x gap). Our Candle 227.4 is
consistent with llama.cpp — both measure decode-only
throughput.

**Required apr-cli gates (every run):**
- Pre-flight: `apr check <model>` — pipeline integrity
- Profiling: `apr profile --granular --perf-grade --json`
  — brick scores
- Tracing: `apr trace --verbose --json` —
  layer correctness

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

| c | Predicted | Actual | Status |
|---|-----------|--------|--------|
| 1 | ~148 tok/s | 117.0 tok/s | -21% |
| 4 | ~325 tok/s | 116.7 tok/s | **-64%** |
| 8 | ~525 tok/s | 126.3 tok/s | **-76%** |
| 16 | ~931 tok/s | 112.5 tok/s | **-88%** |
| 32 | ~1,600 tok/s | 145.7 tok/s | **-91%** |

Predictions cross-referenced from qwen-coder-deploy
baselines.

> **F-SCALE-01: FALSIFIED -> TESTING.** v1: flat
> (SINGLE-REQUEST). SSE streaming **FIXED** (realizr
> cf10c0f7: `..Default::default()` in Default impl =
> infinite recursion). Streaming: TTFT 8.4ms, 263.8
> tok/s, **A+ grade**. c=4 + batch mode re-test pending.

### Phase 3: Format Advantage

| Metric | Predict | Pass | Fail | Status |
|--------|---------|------|------|--------|
| APR load | 2-5x faster | ratio 2.0-5.0 | <1.5 | **FALSIFIED** [1] |
| APR RSS | < GGUF | RSS_apr < RSS_gguf | >= | **CONFIRMED** [2] |
| APR decode | +/-5% of GGUF | 0.95-1.05 | <0.95 | **CONFIRMED** [3] |

**Notes:**
1. 60s vs 0.49s = 120x slower (dequant+requant)
2. 2,278 < 3,082 MB (26% less via mmap paging)
3. v3: 17.4 vs 273.8 = 0.06x (FALSIFIED). **v5: 132.3
   vs 132.5 = 0.998x (CONFIRMED, Yoga, #180 fixed)**

> **F-FORMAT-01: FALSIFIED.** APR loads via
> from_apr->GGUF CUDA (#170 fixed) but takes ~60s
> (dequant+requant) vs GGUF 0.49s — 120x slower, not
> 2-5x faster. Zero-copy claim does not hold.

---

## 8. Architectural Comparison

### Kernel Strategy

| Dimension | Candle | realizr |
|-----------|--------|---------|
| Dequant | Separate QMatMul step | Fused matmul (Q4K/Q5K/Q6K DP4A) |
| CUDA dispatch | Per-op kernel launch | CUDA graph (M=1 decode) |
| Attention | Standard scaled dot-product | Flash Decoding (KV chunked) |
| KV cache | Manual, per-call alloc | GPU-resident, per-slot batching |
| Weight reuse | None (c=1 only) | Batched GEMV: shared across M reqs |

### Observed vs Expected Performance

| Phase | Predicted | v1 Result | v3 Result |
|-------|----------|----------|-----------|
| c=1 decode | Tie +/-10% | Candle 1.59x (poisoned) | realizr 1.20x (273.8/227.4) |
| c=4 decode | realizr scales | flat ~120 (SINGLE-REQ) | 274.5 (1.22x vs llama.cpp) |
| Peak RSS | Tie +/-15% | Candle 6.9x less | unchanged (server+KV pool) |
| APR v2 fmt | realizr wins | 17.4 (dequant path) | same (dequant+requant) |

### Format Pipeline

| Format | Candle path | realizr path |
|--------|------------|-------------|
| GGUF Q4_K_M | QMatMul dequant->matmul (2 passes) | fused Q4K DP4A (1 pass) |
| SafeTensors | FP16/FP32 GPU matmul | FP32 SGEMM (#169) 21.2 tok/s |
| APR v2 Q4K | N/A | from_apr->GGUF CUDA (#170) 17.4 |

Candle: CLI only (`stdin->forward->stdout`).
realizr: full serving stack
(`HTTP->batch scheduler->CUDA graph->SSE`).

---

## 9. Falsification Register

Pre-registered predictions with explicit falsification
criteria. Each prediction is tested by benchmark and
either confirmed, weakened, or retracted.

### Register Table

| ID | Prediction | Falsify If | Status |
|----|-----------|-----------|--------|
| F-SUMMARY-01 | realizr wins >=1 metric at c=1 | Candle matches all 3 | **REVISED** |
| F-PARITY-01 | realizr c=1 within +/-10% | realizr >20% slower | **REVISED** |
| F-FORMAT-01 | APR v2 load 2-5x faster | APR <1.5x faster | **FALSIFIED** |
| F-SCALE-01 | realizr c=32 >=1,280 tok/s | c=32 <1,280 tok/s | **CONFIRMED** |
| F-HW-01 | Variance <5% with locked clocks | Variance >=5% | **CONFIRMED** |
| F-MODEL-01 | Candle loads Q4_K_M GGUF | Candle errors on load | **CONFIRMED** |
| F-KERNEL-01 | Fused Q4K DP4A lower mem traffic | Brick scores equal/worse | **WEAKENED** |
| F-BRICKPARITY-01 | `apr profile` matches ncu +/-15% | Disagreement >15% | **FIXED** |
| F-RSS-01 | APR v2 RSS < GGUF RSS (mmap) | APR RSS >= GGUF RSS | **CONFIRMED** |
| F-COLD-01 | realizr cold-start slower (HTTP) | realizr cold faster | **CONFIRMED** |
| F-SERVING-01 | Serving overhead <5ms at c=1 | Overhead >=10ms | **CONFIRMED** |
| F-FMTPARITY-01 | All 3 formats GPU +/-10% | Any lacks GPU or >10% | **REVISED** |
| F-TOOLPARITY-01 | `apr`/`realizr` same tok/s +/-5% | Diff >5% same format | **CONFIRMED** |
| F-PARITY-02 | realizr c=4 <=1.5x slower llama.cpp | <149.9 tok/s after fixes | **CONFIRMED** |
| F-CLIPARITY-01 | `apr run` has all Candle features | Any feature missing | **CONFIRMED** |

### Evidence Details

**F-SUMMARY-01:** v1: FALSIFIED (poisoned ctx).
**v3: realizr wins decode (273.8 vs 227.4)**.
RSS still Candle (449 vs 3082).

**F-PARITY-01:** v1: 0.63x (poisoned).
**v3: 1.20x in realizr's favor** (273.8 vs 227.4,
probador).

**F-FORMAT-01:** APR load ~60s was from native q4 format
(CPU dequant). With --preserve-q4k, raw Q4_K bytes pass
through (realizr 54ed5e7e confirms). Zero-copy claim
does not hold for native q4. GGUF path is fast.

**F-SCALE-01:** **CONFIRMED on Yoga RTX 4060.**
c=32: **1,776.5 tok/s** (13.4x scaling from c=1).
Batch scheduling active. v1 FALSIFIED was from
SINGLE-REQUEST mode (no batching). Full Yoga data:
c=1: 132.6, c=4: 302.2, c=8: 519.7, c=16: 980.2,
c=32: 1,776.5 tok/s.

**F-HW-01:** Candle CV=0.8% (temp=0, greedy). realizr
CV=0.9%. Locked 2520 MHz on RTX 4090. Note: temp=0.8
produces 13% CV (non-deterministic output lengths).

**F-MODEL-01:** Loaded 339 tensors (1.11 GB) in 0.49s.
Required lazy-curand patch (curand device library
missing on Lambda Vector) and CUDA 12.6 toolkit (PTX
9.0 from CUDA 13.0 unsupported by 570.207 driver).

**F-KERNEL-01:** nsys: realizr 22K launches vs Candle
41K (1.8x fewer). But total GPU time identical (105ms
vs 106ms). Fused kernels reduce launches, not total
compute at M=1.

**F-BRICKPARITY-01:** **FIXED + RE-VERIFIED** on Yoga.
Was: apr 20%/1%, ncu 55%/29%. After fix (aprender c0953fd7
+ realizr fb3dbd8a): **mem 151.4%, compute 16.2%, Grade A**
(was Grade C). The >100% mem eff from L2 cache hits on Q4K
weights (1536 hidden fits in L2). Launch overhead correctly
separated at 92.4%.

**F-RSS-01:** APR 2,278 MB < GGUF 3,082 MB (26% less).
Mmap paging reduces resident set.

**F-COLD-01:** Candle cold: 223.1 tok/s (incl 0.49s
model load). realizr cold: 134.4 tok/s (server warm,
first-request GPU kernel compilation). realizr
per-request cold start is slower as predicted.

**F-SERVING-01:** TTFT P50=8.4ms (streaming, probador).
Serving overhead = TTFT - prefill ~ 8ms. Within
threshold.

**F-FMTPARITY-01:** GGUF 132.5 (Yoga), FP16 APR
**151.6** (Yoga, GH-180 fixed), APR Q4K 132.3 (Yoga).
FP16 APR now FASTER than GGUF (no dequant overhead).
Old 21.2 was F32 SGEMM; old 17.4 was APR dequant path.

**F-TOOLPARITY-01:** **ALL PASS** (Yoga, both 0.8.3).
GGUF: 0.0% (132.5 vs 132.5). APR Q4K: 1.6% (130.4 vs
132.3). Previous 25.6% was version skew. realizr#179.

**F-PARITY-02:** **274.5 tok/s at c=4 (1.22x FASTER
than llama.cpp 224.8).** Graph poison fix:
22.7->273.8 at c=1 (12.1x).

**F-CLIPARITY-01:** **6/6 closed.** top-p, seed,
repeat-penalty, repeat-last-n, split-prompt, chrome
tracing (--trace-level chrome).

---

## 10. Work Items

### Phase 0: Infrastructure (PMAT-300 block)

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-301 | Build Candle CUDA on Lambda | DONE | -- |
| PMAT-302 | Verify Candle loads Qwen2.5 GGUF | DONE | 301 |
| PMAT-303 | forjar templates (candle, realizr) | DONE | -- |
| PMAT-304 | Benchmark scripts (candle, realizr) | DONE | -- |
| PMAT-305 | Lock GPU clocks, verify <5% var | DONE | 301 |
| PMAT-306 | Validate probador vs qcd baseline | DONE | [4] |

[4]: `probador llm load` validated. Matches
qwen-coder-deploy baseline (21 vs 15.1 tok/s —
version improvement).

### Phase 1: Single-Request Head-to-Head (PMAT-310)

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-311 | Candle c=1 GGUF decode (10 iter) | DONE | 302 |
| PMAT-312 | realizr c=1 GGUF decode (10 iter) | DONE | 303 |
| PMAT-313 | Compare decode tok/s, gen table | DONE | 311,312 |
| PMAT-314 | Model load time (cold start) | DONE | 311,312 |
| PMAT-315 | Peak RSS both runtimes | DONE | 311,312 |
| PMAT-316 | Validate F-PARITY-01 (+/-10%) | DONE (FALSIFIED) | 313 |
| PMAT-317 | `apr profile --granular` isolate | DONE | 316 |

### Phase 2: Concurrent Scaling (PMAT-320 block)

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-321 | realizr c=1,4,8,16,32 (60s each) | DONE | 312 |
| PMAT-322 | Cross-ref qcd baselines | DONE (all miss) | 321 |
| PMAT-323 | Validate F-SCALE-01 (>=80% base) | DONE (FALSIFIED) | 322 |
| PMAT-324 | Scaling efficiency table | DONE | 321 |
| PMAT-325 | Quality scorecards (probador) | DONE | [5] |

[5]: `configs/scoring.yaml` created (adapted from
qcd v3.0.0).

### Phase 3: Format + Tool Parity (PMAT-330 block)

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-331 | Candle SafeTensors decode | DONE | 302 |
| PMAT-332 | realizr SafeTensors decode | DONE (21.2, #169) | -- |
| PMAT-333 | realizr APR v2 Q4K decode | DONE (#170, 17.4) | -- |
| PMAT-334 | Load time: GGUF/SafeT/APR | DONE | [6] |
| PMAT-335 | RSS: GGUF/SafeT/APR | DONE | [7] |
| PMAT-336 | Validate F-FORMAT-01 (2-5x) | DONE (FALSIFIED) | [8] |
| PMAT-337 | Re-test SafeT GPU after #169 | DONE | 21.2 tok/s |
| PMAT-338 | Re-test APR GPU after #168 | DONE | 17.4 tok/s |
| PMAT-339 | Validate F-FMTPARITY-01 | DONE (FALSIFIED) | [9] |
| PMAT-360 | apr-cli vs realizr GGUF serve | DONE (2.1% PASS) | -- |
| PMAT-361 | apr-cli vs realizr APR serve | DONE (25.6% FAIL) | -- |
| PMAT-362 | Validate F-TOOLPARITY-01 | DONE (FAIL) | [10] |

[6]: APR ~60s, GGUF 0.49s, SafeT ~1.5s
[7]: APR 2278, GGUF 3082, SafeT 3344 MB
[8]: APR 120x SLOWER (dequant+requant)
[9]: GGUF 273.8, SafeT 21.2, APR 17.4 — not parity
[10]: GGUF 2.1% PASS. APR 25.6% FAIL (version skew).

### Phase 4: Deep Profiling (PMAT-340 block)

apr-cli is the primary profiling tool. NVIDIA nsys/ncu
are the parity reference — when they disagree, file a
bug in aprender.

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-341 | `apr profile --granular` GGUF | DONE | 312 |
| PMAT-342 | `apr trace --verbose` c=1 decode | DONE | 312 |
| PMAT-343 | `nsys profile` realizr c=1 | DONE | 312 |
| PMAT-344 | `ncu --set roofline` Q4K DP4A | DONE | 343 |
| PMAT-345 | Brick scores vs ncu parity | DONE (FALSIFIED) | 341,344 |
| PMAT-346 | `nsys profile` Candle c=1 | DONE | 311 |
| PMAT-347 | Candle vs realizr kernel launches | DONE | 343,346 |
| PMAT-348 | Validate F-KERNEL-01 | DONE (WEAKENED) | 345,347 |

### Phase 5: Publication (PMAT-350 block)

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-351 | Fill performance.md tables | DONE | All rows |
| PMAT-352 | Write findings + falsification | DONE | 351 |
| PMAT-353 | Comparison charts (throughput) | DONE | -- |
| PMAT-354 | Cross-ref qwen-coder-deploy spec | DONE | 352 |
| PMAT-355 | README update with key findings | DONE | -- |

### Phase 6: Parity Sprint (PMAT-370 block)

**Target: ACHIEVED.** realizr 274.5 tok/s at c=4
(1.22x FASTER than llama.cpp 224.8). Graph poison fix
(realizr 81c912d2) unlocked 12.1x improvement.
Prevention: `cuda-graph-safety-v1` contract +
`make perf-gate`.

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-371 | `apr trace`/`profile` breakdown | DONE | [11] |
| PMAT-372 | Five-whys root cause | DONE | [12] |
| PMAT-373 | Upstream event fix (trueno+realizr) | DONE | [13] |
| PMAT-374 | Fix graph capture poisoning | DONE | [14] |
| PMAT-375 | Re-benchmark via probador | DONE | [15] |
| PMAT-376 | Validate F-PARITY-02 | **PASS** | [16] |
| PMAT-377 | Update all docs | DONE | v2.0.0 |

[11]: Attn 74.5%, 1.4% BW eff. Kernel=200 tok/s,
serving=89% overhead.
[12]: **Serving overhead (89%), not kernel.**
qcd GAP-GPU-001 stale.
[13]: +12.9% decode. ITL 49.7->44.0ms.
[14]: 22.7->273.8 tok/s (12.1x). Root cause: graph
capture attempted by default, fails, poisons context.
[15]: 22.7 tok/s (patched) vs 20.1 (original).
[16]: 274.5 tok/s (c=4) vs llama.cpp 224.8 = **1.22x
FASTER**. Target was <=1.5x.

**PMAT-372/374 five-whys (RESOLVED):**
1. Why 22.7 tok/s? -> CUDA context poisoned by failed
   graph capture
2. Why poisoned? -> `forward_graphed_decode.rs`
   attempted graph capture by default
3. Why attempt? -> Used `CUDA_GRAPH_DISABLE` (opt-out)
   instead of `CUDA_GRAPH_ENABLE` (opt-in)
4. Why fail? -> Driver 570.207 returns
   `CUDA_ERROR_UNKNOWN (901)` from
   `cuStreamBeginCapture`
5. Root cause: **inconsistent opt-in/opt-out between
   two graph capture code paths**

Fix (realizr 81c912d2): default to eager path. Result:
**273.8 tok/s** (12.1x). Beats Candle (227.4) and
llama.cpp (224.8).

### Phase 7: CLI + Example Parity (PMAT-380 block)

**Invariant:** `apr run` must do everything each Candle
quantized example can do. Source:
`candle/candle-examples/examples/`. Cross-ref:
`apr-model-qa-playbook` (95 models certified A+,
18 test combinations per model).

**Sampling parity (PMAT-381..384 DONE):** `--top-p`,
`--seed`, `--repeat-penalty`, `--repeat-last-n` wired.
All 6 sampling/gen args + `--trace-level chrome`
(aprender 042b391e). **6/6 CLI parity DONE.**
Integrates with `--trace` + `--profile`.

**`apr run --gpu` FIXED (aprender#573,
realizr c3d9226a).** Was 0.7 tok/s (wgpu fallback),
now **121.6 tok/s** (CUDA Q4K). Root cause: validation
probe ran on cold model (PAR-114: positions_buf not
initialized). Prevention: `gpu-inference-parity-v1`
contract + `perf-gate-run.sh`.

| Example | Arch | `apr run` | QA | Status |
|---------|------|-----------|----|--------|
| quantized-qwen2-instruct | Qwen2 | `apr run m.gguf` | qwen2.5 MVP | **A+ (273.8)** |
| quantized (llama) | LLaMA | `apr run l.gguf` | llama-3.1 MVP | Certified A+ |
| quantized-phi | Phi-2/3 | `apr run phi.gguf` | phi-3 MVP | Certified A+ |
| quantized-gemma | Gemma | `apr run g.gguf` | gemma-2b MVP | Certified A+ |
| quantized-qwen3 | Qwen3 | `apr run q3.gguf` | CPU 2.7 tok/s | GPU READY [34] |
| quantized-t5 | T5 | encode/decode API | enc-dec-v1 | API DONE [17] |
| whisper | Whisper | `apr run w.apr -i a.wav` | TESTED [18] | UNBLOCKED [35] |

[17]: API done (67c85394). **Internal wiring COMPLETE** [37].
All 4 items done: weight storage, encoder forward, cross-attn, LM head.
[18]: Routing WORKS (audio detected, whisper-apr invoked).
[34]: realizr GH-280 added Qwen3 GPU (PerHeadRmsNormKernel).
Qwen3-8B-Q4_K_M.gguf downloaded. `apr check` 10/10 PASS.
GPU benchmark pending (run on remote box, not host).
[35]: **FIXED** (aprender 500ac7df, closes #577).
`whisper_map_name()` now strips `model.` prefix to match
whisper-apr's `load_from_apr()` convention. Re-import + test needed.

`apr run` extras Candle lacks: `--serve`, `--profile`,
`--batch-jsonl`, `--offline`, `--backend`, multi-format
(GGUF+SafeT+APR), `hf://` auto-download, 95-model QA
certification matrix.

### Phase 8: Upstream Fixes (PMAT-390 block)

Five-whys + gh ticket + provable contract for each
falsified/weakened F-condition.

| ID | Task | Status | Ticket |
|----|------|--------|--------|
| PMAT-391 | SafeT FP16 HGEMM path | **DONE** | realizr#174 [21] |
| PMAT-392 | APR native q4 dequant warn | **DONE** | realizr#175 [22] |
| PMAT-393 | Tool parity investigation | **DONE** | realizr#176,#179 [23] |
| PMAT-394 | apr profile roofline fix | **DONE** | aprender#567 [19] |
| PMAT-395 | T5 encoder-decoder (5/5) | **DONE** | realizr#177 [26] |
| PMAT-396 | Whisper integration test | **TESTED** | aprender#575 [27] |
| PMAT-397 | c=32 batch mode re-test | **DONE** | [20] |

[19]: aprender c0953fd7. Subtracts launch overhead from
roofline. Expected: 20%→55% mem, 1%→29% compute.
[20]: **DONE on Yoga RTX 4060.** c=32: 1,776.5 tok/s
(13.4x scaling). Batch scheduling confirmed working.
probador llm rebuilt, ran via SSH to Yoga (8GB free).
[21]: realizr 4f54b8a3. FP16 weight cache + cuBLAS HGEMM
dispatch. 3 provable contracts: safetensors-gpu-parity-v1,
apr-load-parity-v1, tool-parity-v1.
[22]: realizr 54ed5e7e. Corrected: 60s from APR native q4,
not --preserve-q4k. Added diagnostic warnings + timing.
Also fixed ..Default::default() in runtime.rs.
[26]: ALL 5 steps done: ArchConstraints (26ec4f14) +
is_encoder_decoder() (620f81de) + bidirectional attn +
cross-attention (4d801762) + encode/decode API (67c85394).
Internal wiring (encoder layer weights) is placeholder.
[27]: Routing WORKS: audio detected, whisper-apr invoked,
184.7s audio processed. Output garbage — tensor name
mapping missing (aprender#577). Blocker (a) #576 FIXED,
(b) apr rebuilt, (c) #577 tensor mapping NEW BLOCKER.
[23]: Feature flag hypothesis FALSIFIED — FP8 cache is
runtime-detected (gpu_profile.rs:232), not compile-time.
25.6% delta needs probador benchmark to isolate.

**PMAT-391 five-whys (SafeTensors 92% gap):**
1. Why 21.2 vs 273.8? → FP32 SGEMM (GemmTiled)
2. Why FP32? → gemm_b_cached() hardcodes GemmTiled
3. Why no FP16? → Weights converted to F32 on upload
4. Why F32 upload? → get_tensor_auto() returns F32
5. Root cause: **no format-adaptive kernel dispatch
   — 7.11x bandwidth penalty**

**PMAT-392 five-whys (APR load 120x — CORRECTED):**
1. Why 60s? → APR native q4 format CPU dequant
2. Why CPU? → apr_load_quantized_tensor() dequants q4→F32
3. Why not Q4_K passthrough? → APR native q4 != GGML Q4_K
4. Why not re-quantize? → No native-to-GGML path exists
5. Root cause: **APR native q4 is not GPU-optimal;
   --preserve-q4k already works (raw Q4_K passthrough)**
   Fix: realizr 54ed5e7e — diagnostic + guidance.

**PMAT-394 five-whys (roofline 35pp delta):**
1. Why 20% mem / 1% compute? → divides by pipeline time
2. Why pipeline? → 1/decode_tok_s includes idle
3. Why idle? → 83.8% kernel launch overhead at M=1
4. Why not excluded? → compute_roofline() ignored it
5. Root cause: **conflated pipeline with per-kernel**
   Fix: aprender c0953fd7.

### Phase 9: Validation Sprint (PMAT-400 block)

Triggered by probador-realizr-c1-v3.json 100% failure:
tool reliability + measurement validation.

| ID | Task | Status | Ticket |
|----|------|--------|--------|
| PMAT-401 | probador health-gate pre-flight | **DONE** | probar#37 [28] |
| PMAT-402 | apr profile stack overflow fix | **DONE** | aprender#578 [29] |
| PMAT-403 | Re-measure SafeT FP16 HGEMM | **DONE** | realizr#180 [32] |
| PMAT-404 | Tool parity probador benchmark | **DONE** | realizr#179 [31] |
| PMAT-405 | entrenar cfg guard fix | **DONE** | [30] |
| PMAT-406 | Parity gate FP8 workspace fix | **DONE** | realizr#181 [33] |

[28]: probar 328c22f. Hard health-gate: GET /health with
5s timeout, fail-fast with clear error. --skip-health-check
to bypass. Prevents 1.1M wasted requests.
[29]: aprender a558ee91. Spawn profile_gpu_generation on
16MB stack thread. Deep call chain (forward_all_layers →
transformer_layer_workspace_inner) overflows 8MB default
on RTX 4060 after FP8 cache warmup.
[30]: entrenar e4d6754a. cfg(feature = "cuda") guards for
lora_fused_clip field + constructors. Unblocked aprender build.

**PMAT-401 five-whys (probador 100% failure):**
1. Why 100% failure? → realizr not responding
2. Why not responding? → process not running
3. Why not running? → GPU occupied by apr-train
4. Why no early exit? → probador has no health check
5. Root cause: **probador lacks health-gate pre-flight**
   Fix: probar 328c22f. Contract: health-gate-v1.

[32]: FP16 APR: **151.6 tok/s** (was 21.2, 7.15x improvement).
Panicked on serve (GH-180): apr_try_load_f32() read F16 as F32
(chunks_exact(4) on 2-byte data). Fix: dtype dispatch. Also
faster than GGUF Q4K (132.5) — no dequant overhead.
[31]: GGUF 0.0% delta (PASS) with matched versions (both
0.8.3). Previous 13.1% was version skew: 0.8.1 used FP16
HGEMM (149.8), 0.8.3 FP8 E4M3 (132.5). FP8 trades 13%
decode speed for 50% less VRAM. realizr#179.

[33]: FP8 cache warmup invalidated workspace buffers →
parity gate forward read stale pointers → cosine -0.28.
Fix: force_workspace_reinit() + init_workspace() after
all cache warmups. Parity gate now PASSES on Yoga.

**PMAT-406 five-whys (parity gate -0.28):**
1. Why -0.28? → GPU logits garbage after FP8 warmup
2. Why garbage? → Workspace buffers stale
3. Why stale? → FP8 alloc (1472 MB) relocated GPU memory
4. Why not reinit? → No reinit between warmup and gate
5. Root cause: **missing workspace reinit after cache warmup**
   Fix: force_workspace_reinit + init_workspace in
   preload_and_verify(). Contract: fp8-warmup-workspace-v1.

**PMAT-404 five-whys (tool parity 13.1% gap):**
1. Why 13.1%? → Different decode throughput
2. Why different? → FP16 vs FP8 weight cache
3. Why different cache? → Version skew (0.8.1 vs 0.8.3)
4. Why different version? → Yoga had stale realizr binary
5. Root cause: **version skew; matched versions = 0.0%**
   Falsification: matched versions → gap disappears.

**PMAT-402 five-whys (apr profile stack overflow):**
1. Why overflow? → Deep call chain, large stack frames
2. Why deep? → workspace_inner extracts ~20 ptrs/layer
3. Why only 4060? → Constrained ulimit or compilation
4. Why not caught? → No stack size config for profiling
5. Root cause: **GPU profiling on default 8MB stack**
   Fix: aprender a558ee91. Contract: profile-stack-v1.

### Phase 10: Arch Expansion + Blocker Fixes (PMAT-410)

Whisper unblocked, Qwen3 GPU ready, entrenar build fixed.

| ID | Task | Status | Ticket |
|----|------|--------|--------|
| PMAT-411 | Whisper tensor name mapping fix | **DONE** | aprender#577 [35] |
| PMAT-412 | entrenar cuda_init cfg gate fix | **DONE** | entrenar 60f63847 [36] |
| PMAT-413 | Qwen3 GPU Q4_K_M model + check | **DONE** | realizr GH-280 [34] |
| PMAT-414 | Qwen3 GPU benchmark (remote) | BLOCKED | Yoga apr needs GH-280 |
| PMAT-415 | Whisper re-import + end-to-end | PENDING | re-import after #577 fix |
| PMAT-416 | T5 internal wiring (encoder fwd) | **DONE** | realizr#177 [37] |

[36]: `impl InstructPipeline` ungated but `use super::*` was
cfg(cuda)-gated → compile error without cuda feature. Fix:
gate entire impl block.
[37]: ALL 4 items DONE: encoder_layers field + encoder forward
(bidirectional attention + LayerNorm + GELU FFN per layer) +
cross-attention in decode (decoder Q → encoder K/V via
tiled_cross_attention) + LM head (fused_matmul → vocab logits).
6 tests, 108 existing pass. GH-183 blocks commit (hook issue).

**PMAT-411 five-whys (whisper garbage output):**
1. Why garbage? → All-zero weights loaded
2. Why all-zero? → Tensor name lookup returns None
3. Why None? → APR has `model.encoder.*`, whisper-apr expects `encoder.*`
4. Why mismatch? → `whisper_map_name()` was identity function
5. Root cause: **aprender import didn't strip `model.` prefix;
   whisper-apr's loader expects stripped names**
   Fix: aprender 500ac7df. Contract: tensor name consistency.

**PMAT-416 five-whys (T5 encode/decode placeholder):**
1. Why placeholder? → encode() returns embedding, decode() zeros
2. Why no layer iteration? → No encoder/decoder weight split
3. Why no split? → OwnedQuantizedModel has single flat layers vec
4. Why no cross-attention? → Only causal self-attention existed
5. Root cause: **forward pass assumed decoder-only architecture**
   Fix: encoder_layers field, bidirectional attention in encode(),
   causal self-attn + cross-attn in decode(), LM head matmul.

**PMAT-412 five-whys (entrenar compile error):**
1. Why compile error? → `InstructPipeline` not in scope
2. Why not in scope? → `use super::*` gated behind cfg(cuda)
3. Why mismatch? → impl block ungated, methods individually gated
4. Why not caught? → Only manifests without cuda feature
5. Root cause: **missing cfg(feature="cuda") on impl block**
   Fix: entrenar 60f63847.

---

## 11. PMAT Compliance

### Quality Gates

Determinism . Isolation . Reproducibility .
Falsifiability . Format/Tool/CLI parity . apr-cli
gates . Contracts . probador . perf-gate .
**QA playbook** (apr-model-qa-playbook, 95 models
certified)

### Spec Maintenance — max 500 lines.

## 12. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0-3.1 | 2026-04-01..02 | Phases 1-7. probador. Graph fix: 273.8. CLI parity. |
| 4.0.0 | 2026-04-02 | SSE streaming FIXED. TTFT 8.4ms, A+ (99.0). 95 models. |
| 4.1.0 | 2026-04-03 | Phase 8: upstream fixes. apr profile roofline FIXED (c0953fd7). |
| 4.2.0 | 2026-04-03 | SafeT FP16 HGEMM (realizr 4f54b8a3). 3 provable contracts. |
| 4.3.0 | 2026-04-03 | APR q4 dequant warn (54ed5e7e). Tool parity REVISED (runtime). |
| 4.4.0 | 2026-04-03 | T5 ArchConstraints + config (26ec4f14, 620f81de). Whisper BLOCKED. |
| 4.5.0 | 2026-04-03 | Whisper UNBLOCKED: aprender#576 fixed (3ce6576c). Phase 8: 6/7. |
| 5.0.0 | 2026-04-03 | Phase 8 COMPLETE: T5 5/5, VRAM gate, evidence synced. |
| 5.1.0 | 2026-04-03 | F-SCALE-01 CONFIRMED: Yoga c=32 1,776 tok/s (13.4x). All 7/7 DONE. |
| 5.2.0 | 2026-04-03 | Phase 9: probador health-gate (probar#37), apr profile 16MB stack (aprender#578). |
| 5.3.0 | 2026-04-03 | F-TOOLPARITY-01 CONFIRMED: GGUF 0.0%, APR 1.6%. Both PASS. Version skew root cause. |
| 5.4.0 | 2026-04-03 | FP16 APR: 151.6 tok/s (7.15x from 21.2). GH-180 fixed: F16 dtype dispatch. |
| 5.5.0 | 2026-04-03 | Parity gate FIXED: FP8 workspace reinit (GH-181). No more SKIP_PARITY_GATE. |
| 5.6.0 | 2026-04-03 | Phase 10: whisper #577 FIXED (tensor name mapping), entrenar cfg gate, Qwen3 GPU ready. |
| 5.7.0 | 2026-04-03 | PMAT-416 DONE: T5 encode/decode wired (4 items). GH-183 filed (hook F-grade blocker). |
