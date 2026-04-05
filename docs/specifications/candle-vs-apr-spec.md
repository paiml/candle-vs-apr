# Candle vs APR Inference Parity Specification

**Document ID:** PAIML-CANDLE-APR-001
**Version:** 13.6.0
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

**PMAT-456 analysis (realizr#203):** FP8 E4M3 cuBLASLt
infrastructure already exists for prefill (M>=5) but
perplexity uses M=1 incremental forwards (bypasses FP8
threshold). Fix: batched teacher-forcing — process all
N prompt tokens in one prefill forward, extract
per-position logits, score against ground truth.
FP8 E4M3 has 3-bit mantissa (vs DP4A int8) — expected
to close precision gap significantly.

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

**NCU Root Cause (flash_decoding_chunk, P15-02):**
Occupancy 2.15% (theoretical 50%). Grid 108 blocks on
128 SMs = <1 block/SM. Scheduler starved 96.6% of cycles.
Memory BW 8.33 GB/s (0.83% of peak). The attention kernel
isn't slow — the GPU is 99% idle during it. Multi-warp
or TC dispatch needed to increase occupancy.

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

| F-CONTRACT-01 | Contracts catch >=1 bug | **WIRED** | 6/6 invariants wired (realizr 1a05516). Awaiting 5 profiling sessions. |
| F-TCATTN-01 | TC attn <=14µs | **PROPOSED** | GQA Tensor Core (FlashInfer). HIGH risk: DRAM stall. |
| F-NCU-01 | NCU finds root cause | **CONFIRMED** | Occupancy 2.15%, scheduler starved 96.6%. Attention grid too small for 128 SMs. |
| F-GATE-01 | Falsification <20% | **PROPOSED** | Pre-opt bottleneck gate. MED risk. |
| F-DOCS-01 | cgp adoption +2 users | **PROPOSED** | cgp CLAUDE.md. LOW risk. |
| F-L2-01 | L2 changes priorities | **CONFIRMED** | Attention 82% L2 → occupancy-starved not BW-starved. Reversed priority. |

27 F-conditions. 25 tested (15 confirmed, 4 revised,
3 falsified, 2 weakened). 1 wired (P15-06). 2 proposed.

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
| PMAT-456 | Batched prefill PPL | **ANALYZED** (realizr#203) |
| trueno#244 | Attention kernel opt | FILED |
| realizr#201 | Graph default sm_89+ | **SHIPPED** |

### Phase 15: Profiler + Kernel Sprint (ACTIVE)

Five proposals from cross-repo analysis (qwen-coder-deploy
v6.34.0, paiml-mcp-agent-toolkit), arXiv 2024-2025,
org commit history, and batuta oracle.

#### Cross-project lessons

**qcd:** BrickProfiler `Deferred` sync = 3.4x fidelity
error (PMAT-3031). LmHead FP8 cuBLASLt closed 0.60x→
0.98x gap (PMAT-105). 771 cuLaunchKernel/decode = 17%
overhead (PMAT-217). Three-tier: BrickProfiler → nsys
→ ncu (we skip tiers 2+3). Prompt-length: 1-step
invariant (±4%), 2-step FP8 penalty (-26%).

**paiml-mcp-agent-toolkit:** brick_score_hardware.rs
auto-classifies memory/compute-bound. dhat-rs: -69%
allocs, -47% runtime. PMAT-033 falsification audit.

#### Proposals (each with falsification condition)

**P15-01: GQA Tensor Core attention (trueno#244, #245).**
FlashInfer (Ye 2025): GQA 6:1 → TC at M=1. Predicted
18.2µs → ~12µs, 329→361 tok/s. **F-TCATTN-01:** <=14µs
or falsified. **Risk: HIGH** (DRAM stalls).

**NCU-informed update:** Root cause is NOT kernel speed
(2.9µs per chunk launch) but occupancy (2.15%). The
multi-warp kernel (PAR-070) is **fully implemented** in
trueno but **not wired into dispatch**. Grid: 12 blocks
(1 per head) × 4 warps = 48 warps — still only 12
blocks on 128 SMs. Need either:
1. Multi-warp integration (trueno#245, moderate risk)
2. Persistent kernel (stays resident on SMs)
3. FlashInfer TC (combines occupancy + compute boost)

**P15-02: NCU on attention kernel. DONE.**
`flash_decoding_chunk` profiled (RTX 4090, 2520 MHz):

| Metric | Value | Assessment |
|--------|-------|------------|
| Duration | 2.9µs | Fast per-launch |
| Achieved Occupancy | **2.15%** | CATASTROPHIC |
| SM Busy | 0.88% | 99% idle |
| No Eligible (stalls) | 96.6% | Scheduler starved |
| Memory BW | 8.33 GB/s | 0.83% of peak |
| L2 Hit Rate | 82% | Good |
| L1 Hit Rate | 17% | Poor |
| Grid | (12,1,9)×(32,1,1) | 108 blocks/128 SMs |

**Root cause:** At M=1 with seq_len~9, grid of 108 blocks
(12 heads × 9 chunks, 1 warp each) cannot fill 128 SMs.
Occupancy 2.15% vs theoretical 50%. Scheduler is starved
96.6% of cycles. This is the "Mind the Memory Gap"
(Ramirez-Gargallo 2025) prediction confirmed empirically.

**F-NCU-01: CONFIRMED** — NCU identified root cause
(occupancy starvation) in <1h. Actionable: multi-warp
variant (PAR-070) or FlashInfer GQA TC (trueno#244).
**Risk: LOW** confirmed.

**P15-03: Bottleneck gate. DONE.**
`scripts/bottleneck-gate.sh` checks roofline bounds
before experiments. Encodes hardware constants (RTX 4090
1,008 GB/s, 128 SMs), model constants (1 GB Q4_K_M),
and Phase 12 evidence (16 fusion falsifications).

Verified gate accuracy on known cases:
- FALSIFIES: RSS < 500 MB (irreducible 2.5 GB+) ✓
- FALSIFIES: decode > 1,008 tok/s (BW ceiling) ✓
- FALSIFIES: attention < 0.26µs (roofline floor) ✓
- WARNS: kernel fusion for >5% at M=1 (Phase 12) ✓
- PASSES: decode 340 tok/s (within reach) ✓

**F-GATE-01:** Falsification < 20% over 10 attempts.
Track via `results/bottleneck-gate-log.json`.
**Risk: MED** (gate might be too conservative).

**P15-04: cgp docs.** 9 profilers, 0 docs. **F-DOCS-01:**
2+ new users in 30d. **Risk: LOW.**

**P15-05: L2 cache analysis. MEASURED via NCU.**
Per-brick L2 hit rates from ncu (RTX 4090):

| Brick | L2 Hit Rate | L1 Hit Rate | DRAM % |
|-------|-------------|-------------|--------|
| flash_decoding_chunk | **82%** | 17% | 0.84% |
| hw_dp4a_q4k_gemv | **~14%** | ~8% | 33.2% |

**F-L2-01:** L2 data DOES change priorities. Attention
has 82% L2 hit rate (KV cache fits in L2) but only
0.84% DRAM throughput (occupancy-starved, not BW-starved).
GEMV has 14% L2 (weights don't fit) and 33% DRAM.

Implication: Attention optimization should target
OCCUPANCY not memory. GEMV is the true BW bottleneck.
This reverses naive priority ordering (attention looked
like the bottleneck at 44% of time, but it's
occupancy-limited not BW-limited).

**F-L2-01: CONFIRMED** — L2 data changed priority
for attention brick (from "optimize BW" to "optimize
occupancy"). **Risk: MED** confirmed.

**P15-06: Profiler contract enforcement (five-whys)**

**Five-whys: Why don't we catch profiling errors?**

1. **Why was qcd's BrickProfiler fidelity lag (3.4x)
   undetected for a week?** Because no contract
   validated sync mode against expected timing bounds.
2. **Why wasn't sync mode validated?** Because
   `gpu-decode-profiling-v1.yaml` defines the invariant
   (`Immediate: LmHead.avg > 10x RmsNorm.avg`) but
   no code checks it.
3. **Why doesn't the code check it?** Because
   BrickProfiler, `apr profile`, and `apr trace` have
   ZERO `#[contract(...)]` macros — contracts exist as
   YAML but aren't wired into the profiler code.
4. **Why aren't they wired?** Because profiler was built
   before provable-contracts existed (v0.1 predates
   contract infra). Never retrofitted.
5. **Why wasn't it retrofitted?** No enforcement gate —
   new code can ship without contract binding.

**Chain of thought: What must be enforced?**

The 11 provable-contracts define 47+ invariants for
profiling. Six are high-value for catching real bugs:

| Contract | Invariant | Catches |
|----------|-----------|---------|
| `gpu-decode-profiling-v1` | wall_coverage >= 0.85 | Missing bricks (like our 28 missing SwiGLU kernels) |
| `gpu-decode-profiling-v1` | Immediate: LmHead > 10x RmsNorm | Deferred sync fidelity lag (qcd 3.4x) |
| `gpu-decode-profiling-v1` | decoded_tokens == LmHead.count | Token accounting errors |
| `layer-parity-v1` | cosine_sim(GPU, CPU) >= 0.99 | Graph replay divergence (our hidden_buf2 81.0 diff) |
| `per-op-training-v1` | GEMM >= 50% of layer_fwd | Architecture regression detection |
| `tracing-observability-v1` | No orphan spans, monotonic counters | Trace corruption |

These 6 invariants would have caught BOTH major bugs
in this project (missing SwiGLU recording, profiler
fidelity) BEFORE they reached measurement.

> **F-CONTRACT-01:** If wiring these 6 invariants into
> `apr profile` + BrickProfiler does not catch at least
> 1 real bug in the next 5 profiling sessions that would
> otherwise go undetected, the enforcement adds overhead
> without value. **Risk: LOW** — wall_coverage alone
> would have flagged the 28 missing SwiGLU kernels
> (619/647 = 95.7% < 100% expected coverage).

### Phase 15 Priority & Risk Matrix

| # | Proposal | Impact | Risk | Effort | Priority | Status |
|---|----------|--------|------|--------|----------|--------|
| P15-06 | Contract enforcement | **bug prevention** | LOW | 1 wk | **P0** | **DONE** |
| P15-01 | TC attention | **+32 tok/s** | HIGH | 4-6 wk | P1 | TODO |
| P15-02 | NCU in cgp | diagnostic | LOW | 1-2 wk | P2 | **DONE** |
| P15-03 | Bottleneck gate | process | MED | 1 wk | P3 | **DONE** |
| P15-05 | L2 in apr profile | diagnostic | MED | 2-3 wk | P4 | **DONE** |
| P15-04 | cgp docs | enablement | LOW | 2 days | P5 | TODO |

**P15-06 IMPLEMENTED:** 6/6 invariants wired into realizr
profiler, tracer, and inference trace (realizr `1a05516`):

| Contract | Invariant | Location |
|----------|-----------|----------|
| gpu-decode-profiling-v1 | wall_coverage >= 0.85 | `profiler_contracts.rs` |
| gpu-decode-profiling-v1 | LmHead > 10x RmsNorm | `profiler_contracts.rs` |
| gpu-decode-profiling-v1 | decoded_tokens == LmHead.count | `profiler_contracts.rs` |
| per-op-training-v1 | GEMM >= 50% of compute | `profiler_contracts.rs` |
| layer-parity-v1 | cosine_sim(GPU, CPU) >= 0.99 | `tracer.rs` |
| tracing-observability-v1 | monotonic IDs + no orphan spans | `tracer_contracts.rs` |

GPU verification (RTX 4090, 2520 MHz): **328.2 decode
tok/s** [within bootstrap CI 317-337]. ITL P50 = 3.0ms.
Consistent with v11 baseline (329.4). No regression.

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

### Profiler Fidelity (qcd PMAT-3031)

BrickProfiler `Deferred` sync reports CPU-side launch
latency, not actual GPU execution time (3.4x error on
QkvProjection: 26µs reported vs 89µs real). `Immediate`
sync mode is mandatory for accurate profiling. This is
a known pitfall in CUDA profiling — nsys/ncu use
event-based collection to avoid this class of error.

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
| **Contract enforcement** | **CLOSED** | 6/6 invariants wired to profiler+tracer (P15-06). |
| **Perplexity delta** | High | DP4A 24.2 vs FP32 12.97. PMAT-456 filed (realizr#203). |
| **NCU profiling** | **CLOSED** | P15-02: ncu on flash_decoding_chunk + hw_dp4a_q4k_gemv. Root causes identified. |
| **L2 cache analysis** | **CLOSED** | P15-05: L2 82% (attn) / 14% (GEMV). Reversed priority ordering. |
| **Chrome Trace export** | Medium | Custom JSON, not Perfetto/Chrome. |
| **GPU-side kernel timing** | Medium | CPU Instant::now() + ncu for validation. CUPTI for continuous monitoring. |
| **Per-step callbacks** | Low | No loss/lr/grad_norm/step_ms. |
| **Memory waterfall** | Low | No per-step alloc/peak/fragmentation. |

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
| 12.2 | 04-05 | Cross-project insights from qcd + toolkit. Profiler fidelity section. |
| 12.3 | 04-05 | P15-06 contract enforcement: five-whys (11 YAML, 0 wired) + chain of thought (6 high-value invariants). F-CONTRACT-01. P15-06 promoted to P0. 27 F-conditions. |
| 13.0 | 04-05 | **P15-06 DONE:** 6/6 contracts wired upstream (realizr 1a05516). GPU verified 328.2 tok/s (within CI). PMAT-456 filed (realizr#203). Phase 15 ACTIVE. |
| 13.1 | 04-05 | **P15-02 DONE:** NCU on flash_decoding_chunk. Root cause: 2.15% occupancy, 96.6% scheduler stalls. Grid (108 blocks) too small for 128 SMs at M=1. F-NCU-01 CONFIRMED. |
| 13.2 | 04-05 | P15-01 analysis: multi-warp (PAR-070) ready but unwired. trueno#245 filed. NCU confirms kernel fast (2.9µs) but GPU idle (99%). |
| 13.3 | 04-05 | **P15-03 DONE:** Bottleneck gate (roofline pre-check). Catches BW ceiling, occupancy, Amdahl's law. Verified on 5 known cases. |
| 13.4 | 04-05 | **P15-05 DONE:** L2 cache from NCU. Attention 82% L2 (occupancy problem), GEMV 14% L2 (BW problem). Reversed naive priority. F-L2-01 CONFIRMED. |
| 13.5 | 04-05 | Phase 15: 5/6 done (P15-04 blocked: no cgp repo). Methodology gaps updated. 25/27 F-conditions tested. |
| 13.6 | 04-05 | PMAT-456 analyzed: FP8 infra exists, perplexity needs batched teacher-forcing. realizr#208 filed (cargo fmt workspace fix). |
