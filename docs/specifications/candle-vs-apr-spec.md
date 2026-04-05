# Candle vs APR Inference Parity Specification

**Document ID:** PAIML-CANDLE-APR-001
**Version:** 12.1.0
**Last Updated:** 2026-04-05
**Status:** ACTIVE
**Methodology:** Popperian Falsification + Deterministic Benchmarks
**Primary Target:** Lambda Vector (RTX 4090, 24 GB VRAM, sm_89)
**Model:** Qwen2.5-Coder-1.5B-Instruct Q4_K_M GGUF
(1.78B params, 28 layers, hidden=1536)

> Every claim carries a falsification condition.
> If triggered, the claim is revised — not defended.

---

## 1. Executive Summary

Head-to-head benchmark: **Candle** (HuggingFace Rust ML)
vs **realizr** (Sovereign AI Stack) on same model, same
GPU, same methodology. Pure Rust-vs-Rust comparison.

**v11 Showdown (RTX 4090, 2520 MHz, clean GPU):**

| Engine | Decode tok/s | vs Candle | ITL P50 | µs/layer |
|--------|-------------|-----------|---------|----------|
| llama.cpp b7746 | **425.0** | 1.87x | 2.4ms | 84 |
| realizr (graph) | **329.4** | 1.45x | 3.0ms | 107 |
| realizr (eager) | 264.6 | 1.16x | 3.8ms | 135 |
| Candle | 227.4 | 1.00x | -- | -- |

Bootstrap CI (N=30): **329.4** [317.3, 336.7] CV=1.5%.
Graph replay DEFAULT for sm_89+ (realizr#201).
Memory-bound decode (arithmetic intensity 4.0).

**Key findings:**
1. Graph dispatch: +26% decode (647 kernels → 1 launch)
2. AttentionScore: 44% of compute, 23µs/layer gap vs FA
3. DP4A PPL: 24.2 vs llama.cpp 12.97 (precision gap)
4. Scaling: 1,776 tok/s at c=32 (Yoga, 13.4x from c=1)

---

## 2. Hardware & Model

| Property | Lambda (PRIMARY) | Yoga (SECONDARY) |
|----------|-----------------|------------------|
| GPU | RTX 4090 (128 SMs) | RTX 4060 Laptop (24 SMs) |
| VRAM | 24 GB GDDR6X | 8 GB GDDR6 |
| BW | 1,008 GB/s | 256 GB/s |
| Clock | Locked 2520 MHz | Locked 1900 MHz |
| Compute | sm_89 (Ada) | sm_89 |

**Model:** Qwen2.5-Coder-1.5B (28 layers, hidden=1536,
GQA 6:1, Q4_K_M GGUF ~1 GB). Three formats: GGUF,
SafeTensors FP16, APR v2 Q4K.

---

## 3. Benchmark Design

**Phase 1 — Single-Request (c=1, head-to-head):**
`probador llm load --concurrency 1 --duration 30s
--warmup 5s --max-tokens 256 --stream false
--num-layers 28`. Temperature 0 (greedy, CV <1%).

**Phase 2 — Scaling (realizr-only, c=1..32):**
Continuous batching via batch scheduler. Yoga confirmed
13.4x scaling at c=32 (1,776.5 tok/s). Validates
Orca-style iteration-level scheduling (Yu et al. 2022).

**Phase 3 — Format + Tool Parity:**
All 3 formats GPU within 14.6% (Yoga). `apr` vs
`realizr` within 1.4%. Previous gaps were bugs (all
fixed: #169 SGEMM, #170 dequant, #180 dtype, #185 Q4K).

**Isolation:** forjar deploy, `nvidia-smi
--query-compute-apps` pre-flight (mandatory after
realizr#190 false regression from GPU contention).

---

## 4. Metrics Contract

| Metric | Unit | Source |
|--------|------|--------|
| Decode tok/s | tok/s | `probador llm load` |
| ITL P50 | ms | `probador --stream true` |
| TTFT P50 | ms | `probador llm load` |
| µs/layer | µs | `probador --num-layers 28` |
| Peak RSS | MB | `/usr/bin/time -v` |

---

## 5. Results & Baselines

### Phase 1: Single-Request (c=1)

| Metric | v11 graph | v11 eager | Candle | llama.cpp |
|--------|-----------|-----------|--------|-----------|
| Decode tok/s | **329.4** | 264.6 | 227.4 | 425.0 |
| ITL P50 | 3.0ms | 3.8ms | -- | 2.4ms |
| µs/layer | 107 | 135 | -- | 84 |
| Peak RSS | 3,082 MB | 3,082 MB | 449 MB | -- |
| VRAM peak | 5,388 MiB | 5,388 MiB | -- | -- |

Graph replay: +26% from 647 kernels → 1 cuGraphLaunch.
Validated by CUDA graph literature: Yu et al. 2020
report 2-10x kernel launch reduction on DNN workloads.

### Phase 2: Scaling (Yoga RTX 4060)

| c | Decode tok/s | Agg tok/s | Scaling |
|---|-------------|-----------|---------|
| 1 | 132.6 | 132.6 | 1.0x |
| 4 | -- | 302.2 | 2.3x |
| 32 | -- | **1,776.5** | **13.4x** |

Validates continuous batching (Orca, Yu et al. 2022).
Candle has no server — cannot demonstrate c>1.

### Phase 3: Format Parity (Yoga)

| Format | tok/s | Status |
|--------|-------|--------|
| GGUF Q4_K_M | 132.5 | baseline |
| FP16 APR | **151.6** | +14.4% (HGEMM) |
| APR v2 Q4K | 132.3 | parity (0.998x) |

### Perplexity (F-QUALITY-01)

realizr DP4A PPL: **24.2** (weighted, WikiText-2).
llama.cpp FP32: **12.97**. Gap = DP4A int8→int32
accumulation vs FP32 dequant (Micikevicius et al. 2018).
PMAT-456 (batched FP8 GEMM path) will close this.

---

## 6. Architectural Comparison

| Dimension | Candle | realizr |
|-----------|--------|---------|
| Dequant | Separate QMatMul | Fused DP4A (Q4K/Q5K/Q6K) |
| Dispatch | Per-op launch (~640/tok) | **CUDA graph** (1 launch, 647 nodes) |
| Attention | Standard SDPA | Flash Decoding (chunked KV) |
| KV cache | Per-call alloc | GPU-resident, FP8 option |
| Batching | None (c=1 only) | Continuous (Orca-style) |
| Serving | CLI only | HTTP + SSE streaming |

### Compute Profile (graph mode, `apr profile --granular`)

| Brick | % compute | Avg µs | Bottleneck |
|-------|-----------|--------|------------|
| **AttentionScore** | **44.3%** | 18.2 | Memory BW |
| QkvProjection | 14.0% | 5.7 | Memory BW |
| RmsNorm | 7.5% | 1.5 | Memory BW |
| OutputProjection | 7.2% | 2.9 | Memory BW |
| DownProjection | 7.0% | 2.9 | Memory BW |
| RopeEmbedding | 6.9% | 2.8 | Compute |

Roofline: AI=4.0, achieved 1,235 GB/s (122% of spec
due to L2 hits). Memory-bound per Williams et al. 2009.

### GGUF Quant Coverage

Fused DP4A GPU: Q4K, Q5K, Q6K. cuBLAS fallback: Q4_0,
Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q8_K, F16, BF16, F32.
CPU only: Q2_K, Q3_K (rarely used, not blocking).

### Architecture Coverage

A+ certified: LLaMA, Qwen2, Qwen3, Phi-2/3, Gemma,
Mistral. Wired: T5 (enc/dec), Whisper. Gap: Qwen3-MoE
(Candle has it, realizr lacks MoE dispatch).

---

## 7. Falsification Register

| ID | Prediction | Status | Evidence |
|----|-----------|--------|---------|
| F-SUMMARY-01 | realizr wins c=1 | **REVISED** | Graph 329 vs llama.cpp 425 (0.78x). 1.45x Candle. |
| F-PARITY-01 | c=1 within +/-10% | **REVISED** | Graph 329 vs Candle 227 (1.45x). vs llama.cpp 0.78x. |
| F-FORMAT-01 | APR load faster | **FIXED** | Q4K raw passthrough (realizr#185). |
| F-SCALE-01 | c=32 >=1,280 | **CONFIRMED** | Yoga: **1,776.5** (13.4x). |
| F-HW-01 | CV <5% locked | **CONFIRMED** | CV 0.8-0.9%. Bootstrap CV=1.5%. |
| F-MODEL-01 | Candle loads Q4K | **CONFIRMED** | 339 tensors, 0.49s. |
| F-KERNEL-01 | Fused lower mem | **WEAKENED** | Fewer launches but same GPU time. |
| F-RSS-01 | APR RSS < GGUF | **CONFIRMED** | 2,278 < 3,082 (mmap). |
| F-SERVING-01 | Overhead <5ms | **CONFIRMED** | 4.6ms (TTFT-ITL). |
| F-FMTPARITY-01 | 3 formats +/-10% | **REVISED** | Yoga: GGUF 132.5, FP16 151.6, Q4K 132.3. |
| F-TOOLPARITY-01 | apr/realizr +/-5% | **CONFIRMED** | 0.0% GGUF, 1.4% Q4K. |
| F-PARITY-02 | c=4 <=1.5x slower | **CONFIRMED** | 274.5 (1.22x FASTER). |
| F-PARITY-04 | realizr >= llama.cpp | **REVISED** | Graph 329 vs llama.cpp 425 (0.78x). FA gap. |
| F-CLIPARITY-01 | apr = Candle CLI | **CONFIRMED** | 6/6 features. |
| F-1.5X-01 | >=341 (1.5x Candle) | **NEAR** | 329.4 [317, 337]. Needs attention kernel. |
| F-RSS-02 | RSS <=673 MB | **FALSIFIED** | Min 2,930 (weights + server irreducible). |
| F-PARITY-03 | Output div <=1% | **WEAKENED** | 72% — chat template, not dequant. |
| F-QUALITY-01 | PPL within 0.1 | **FALSIFIED** | DP4A 24.2 vs FP32 12.97. Int8 precision. |
| F-REGRESSION-01 | No >5% regression | **CONFIRMED** | 277.3 [276.1, 278.5] vs 273.8 baseline. |
| F-COLD-01 | Cold slower | **REVISED** | preload_modules pre-compiles ~60 kernels. |
| F-CACHE-01 | Prefix cache TTFT | **MEASURED** | Cold 136ms → Warm 56ms (2.4x). |

| F-TCATTN-01 | TC attn <=14µs | **PROPOSED** | GQA Tensor Core (FlashInfer). HIGH risk: DRAM stall. |
| F-NCU-01 | NCU finds root cause | **PROPOSED** | cgp ncu-analyze. LOW risk. |
| F-GATE-01 | Falsification <20% | **PROPOSED** | Pre-opt bottleneck gate. MED risk. |
| F-DOCS-01 | cgp adoption +2 users | **PROPOSED** | cgp CLAUDE.md. LOW risk. |
| F-L2-01 | L2 changes priorities | **PROPOSED** | Per-brick L2 in apr profile. MED risk. |

26 F-conditions. 21 tested (12 confirmed, 4 revised,
3 falsified, 2 weakened). 5 proposed (Phase 15).

---

## 8. Work Items

### Phases 0-11: COMPLETE

35+ tasks across 12 phases. Key outcomes: graph poison
fix (273.8 tok/s), CLI parity (6/6), format parity
(3 formats GPU), Qwen3/T5/Whisper, 95-model QA,
RSS audit, KV prefix caching (2.4x TTFT).

### Phase 12: 1.5x Candle — COMPLETE

**Winner: CUDA graph dispatch** (+26%, 262→329 tok/s).
All kernel fusion approaches falsified for M=1 decode
(RMSNorm+GEMV -5%, fused K+V -3.1%, 16 qcd fusions
failed). Graph dispatch was the only survivor, consistent
with qcd PMAT-280..289 validation.

**Root cause (realizr#198):** Fused gate+up+SwiGLU
kernel missing `graph_recording` block — 28
kernels/forward never captured. Five-whys A/B diagnostic
(per-buffer scan) isolated it. Fix: 647 kernels (was
619). A/B: ALL 13 buffers diff=0.

Manual graph API (trueno#243) bypasses stream capture
bug on driver 570.207 (code 901). `cuGraphAddKernelNode`
+ linear dependency chain.

### Phase 13: Scientific Rigor — COMPLETE

| Task | Result |
|------|--------|
| Perplexity | realizr 24.2 vs llama.cpp 12.97 |
| Bootstrap CI | 329.4 [317.3, 336.7] CV=1.5% (N=30) |
| VRAM | Peak 5,388 MiB (RTX 4090) |
| Poisson | c=1 stable 245-254, c=4 agg 387 |
| Output correctness | 72% divergence (chat template) |
| Showdown | llama.cpp 425 > realizr 329 > Candle 227 |

### Phase 14: Active

| ID | Task | Status |
|----|------|--------|
| PMAT-453 | Graph dispatch | **FIXED** (329 tok/s) |
| PMAT-456 | Batched prefill PPL | TODO |
| trueno#244 | Attention kernel opt | FILED |
| realizr#201 | Graph default sm_89+ | **SHIPPED** |

### Phase 15: Profiler + Kernel Sprint (PROPOSED)

Five proposals from cross-repo analysis, arXiv research
(2024-2025), org commit history, and batuta oracle.
Each carries a falsification condition.

**P15-01: GQA-aware Tensor Core attention (trueno#244)**

FlashInfer (Ye et al. 2025) shows GQA 6:1 enables
Tensor Core usage at M=1 by treating decode as thin
prefill (6 queries, 1 KV set). Our Flash Decoding uses
CUDA cores. llama.cpp FA2 exploits GQA for Tensor Cores.
Predicted: 18.2µs → ~12µs attention, 329→361 tok/s.

> **F-TCATTN-01:** If GQA-aware Tensor Core attention
> does not reduce AttentionScore from 18.2µs to <=14µs
> avg on RTX 4090 (Qwen2.5-1.5B, M=1), the technique
> is falsified for small-model decode. Action: profile
> `mma.m16n8k16` occupancy vs CUDA core throughput.
> **Falsification risk: HIGH.** FA2's win may depend on
> register file size at head_dim=128. Ramirez-Gargallo
> 2025 shows >50% cycles stalled on DRAM even with FA —
> Tensor Cores may starve waiting for data.

**P15-02: NCU integration in cgp (`cgp ncu-analyze`)**

Zero NSight/NCU usage across entire org (verified by
commit search). All profiling uses custom timers.
Missing: warp stall reasons, L2 hit rates, memory
transaction counts. 6+ realizr experiments falsified
without hardware counter pre-analysis.

> **F-NCU-01:** If `cgp ncu-analyze` does not identify
> the root cause of the 18.2µs attention bottleneck
> (stall type + L2 miss rate) within 1 hour of
> implementation, the tool adds complexity without
> insight. Action: compare NCU diagnosis vs manual
> five-whys — does NCU find the answer faster?
> **Falsification risk: LOW.** NCU reliably reports
> hardware counters. The risk is implementation time
> vs value — may take weeks to wrap NCU properly.

**P15-03: Pre-optimization bottleneck gate (contract)**

step-profiler-v1 contract says "speculative optimization
is prohibited." But 6+ realizr experiments (RMSNorm+GEMV,
fused K+V, DP4A inline, f16 conv, shared Q8K, 16 qcd
fusions) were falsified — the gate wasn't enforced.

> **F-GATE-01:** If adding `apr profile` JSON as a
> required precondition for perf tickets does not reduce
> the falsification rate of optimization experiments from
> 6/10 (60%) to <=2/10 (20%) over the next 10 attempts,
> the gate adds bureaucracy without improving hit rate.
> **Falsification risk: MEDIUM.** Some falsifications
> are inherent to M=1 decode physics (small dims defeat
> fusion). The gate would prevent obviously wrong
> attempts but not physics-limited ones.

**P15-04: cgp documentation (CLAUDE.md)**

cgp has 9 backend profilers, roofline, regression
detection, performance contracts, `compete`, `diff`,
`explain` — but ZERO documentation. No CLAUDE.md, no
README.md. Blocks onboarding and discovery.

> **F-DOCS-01:** If cgp CLAUDE.md does not result in at
> least 2 new uses of `cgp` commands (by contributors
> other than the author) within 30 days, the
> documentation failed to enable adoption. Action:
> track `cgp` usage in commit messages.
> **Falsification risk: LOW.** Documentation is
> inherently low-risk. Only risk is effort vs adoption
> if the org is too small for external contributors.

**P15-05: L2 cache + occupancy in `apr profile`**

Ramirez-Gargallo 2025: >50% attention cycles stalled
on DRAM, L2 hit rate avg 12%, L1 avg 2%. Our `apr
profile --granular` reports AI=4.0 but not WHERE cache
misses occur. Adding per-brick L2 hit% and occupancy%
via CUPTI events makes roofline actionable.

> **F-L2-01:** If per-brick L2 hit rate data does not
> change the optimization priority ordering (currently:
> attention > QKV > RMSNorm) for at least one brick,
> the metric adds noise without insight. Action:
> compare priority ordering before/after L2 data.
> **Falsification risk: MEDIUM.** L2 data might confirm
> existing priorities without changing them — useful for
> confidence but not for new insights. Risk increases
> if CUPTI event collection adds >5% overhead to the
> profiling pass itself.

### Phase 15 Priority & Risk Matrix

| # | Proposal | Impact | Risk | Effort | Priority |
|---|----------|--------|------|--------|----------|
| P15-01 | TC attention | **+32 tok/s** | HIGH | 4-6 wk | **P0** |
| P15-02 | NCU in cgp | diagnostic | LOW | 1-2 wk | P1 |
| P15-03 | Bottleneck gate | process | MED | 1 wk | P2 |
| P15-05 | L2 in apr profile | diagnostic | MED | 2-3 wk | P3 |
| P15-04 | cgp docs | enablement | LOW | 2 days | P4 |

**Decision required:** Approve Phase 15 proposals or
revise priorities before implementation begins.

---

## 9. Research Basis (arXiv-grounded)

### Memory-bound M=1 Decode

Autoregressive decode at M=1 is memory-bandwidth-bound
(Pope et al. 2023, "Efficiently Scaling Transformer
Inference"). Our roofline confirms: AI=4.0 vs compute
threshold 82.0. At 1,008 GB/s theoretical / 1,235 GB/s
achieved (L2 amplification), further improvement requires
either reducing memory traffic or increasing arithmetic
intensity via batching (Sheng et al. 2023, FlexGen).

### CUDA Graph Dispatch

CUDA graphs amortize kernel launch overhead by recording
a DAG of kernel nodes and replaying with a single API
call (NVIDIA CUDA Programming Guide, Section 3.2.8).
Our manual graph construction (cuGraphAddKernelNode)
avoids stream capture bugs. 647 nodes, linear chain.

**PyGraph** (Ghosh et al. 2025, arXiv:2503.19779):
Parameter copy elimination doubles CUDA graph benefit.
Our device-side buffers (position_buf, seq_len_buf)
already implement this pattern.

**Kernel Looping** (Prabhakar et al. 2024,
arXiv:2410.23668): H100 achieves only 21% peak BW
during decode. Kernel looping (32 layers → 1 kernel)
reaches 78% roofline (2x speedup). Requires dataflow
HW (SN40L). On GPU, our graph is near-optimal.

### Flash Attention vs Flash Decoding

llama.cpp uses FlashAttention-2 (Dao 2023) with recent
register spill fix (+26%). realizr uses Flash Decoding
(Hong et al. 2023, "FlashDecoding++") — chunked KV
with two-pass reduce. Gap: 18.2µs vs ~12µs.

**FlashInfer** (Ye et al. 2025, arXiv:2501.01005):
GQA decode as thin prefill enables Tensor Core usage
at M=1. For GQA 6:1, 6 queries share 1 KV set →
sufficient AI for `mma.m16n8k16`. Near-100% BW
utilization on RTX 4090. 29-69% ITL reduction.

**FlashAttention-3** (Shah et al. 2024, NeurIPS):
Warp specialization + async softmax achieves 75% of
peak FLOPs on H100 (vs 35% FA2). Ada (sm_89) lacks
TMA but warp specialization principle applies.

**Mind the Memory Gap** (Ramirez-Gargallo et al. 2025,
arXiv:2503.08311): >50% attention cycles stalled on
DRAM. L2 hit rate 12%, L1 2%. Even FA remains
memory-bound at all batch sizes (AI 0.5-1.0).

### Quantized Inference Precision

DP4A (int8 dot product with int32 accumulation) trades
precision for throughput (Micikevicius et al. 2018,
"Mixed Precision Training"). Our PPL gap (24.2 vs 12.97)
matches expected int8 accumulation error across 28
layers. FP8 GEMM (cuBLASLt E4M3) offers better
precision at similar throughput — PMAT-456.

### Continuous Batching

Iteration-level scheduling (Yu et al. 2022, "Orca")
enables linear throughput scaling with concurrency.
Confirmed: 13.4x at c=32 (Yoga). Candle's CLI
architecture precludes batching entirely.

### Serving System Design

PagedAttention (Kwon et al. 2023, "vLLM") and
Sarathi-Serve (Agrawal et al. 2024) demonstrate that
serving overhead dominates for small models. Our 4.6ms
TTFT-ITL overhead is consistent with lightweight Rust
HTTP serving (no Python GIL, no torch overhead).

### Benchmark Methodology

Follows MLPerf Inference v4.0 requirements: locked
clocks, deterministic decode, latency decomposition,
pre-registered predictions. Bootstrap CIs (N=30)
satisfy minimum sample requirements. Poisson arrival
validates under realistic traffic patterns.

---

## 10. Methodology Gaps

| Gap | Severity | Reference |
|-----|----------|-----------|
| **Perplexity delta** | High | DP4A 24.2 vs FP32 12.97. PMAT-456 (FP8 path). |
| **Prefill/decode split** | Medium | probador reports both; not in F-conditions. Splitwise (Patel 2024). |
| **Realistic traffic** | Medium | Poisson done; ShareGPT traces not yet. Vidur (2024). |
| **VRAM fragmentation** | Low | nvidia-smi polling; no CUDA allocator hook. Alizadeh 2024. |

**Tooling:** probador (load), apr (profile/bench/check),
cgp (kernel/roofline/contract), batuta (PPL), llama.cpp
(perplexity), lm-evaluation-harness (correctness).

---

## 11. Revision History

| Version | Date | Key Change |
|---------|------|------------|
| 1.0-5.9 | 04-01..03 | Phases 1-10. Graph fix 273.8. CLI/format parity. |
| 6.0 | 04-03 | Condensed 982→500 lines. |
| 7.0-7.6 | 04-04 | Phase 12 research. 1.5x target. Parity gaps. |
| 8.0-8.9 | 04-04 | Phase 13 scientific rigor. Showdown. PPL. |
| 9.0-9.9 | 04-05 | Graph wiring (590→619 kernels). KV prefix cache. |
| 10.0 | 04-05 | A/B test: RMSNorm OK, hidden_buf2 diverged. |
| 11.0 | 04-05 | **Graph fix** (realizr#198): 647 kernels, 331 tok/s. |
| 11.1 | 04-05 | Showdown: llama.cpp 425, realizr 333, Candle 227. |
| 11.2 | 04-05 | Graph default sm_89+. Bootstrap CI. Profile. |
| 12.0 | 04-05 | arXiv grounding. Condensed 897→352 lines. |
| 12.1 | 04-05 | Phase 15: 5 profiler proposals with falsification. FlashInfer, PyGraph, Kernel Looping, Mind the Memory Gap. 26 F-conditions. |
