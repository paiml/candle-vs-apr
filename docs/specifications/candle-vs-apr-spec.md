# Candle vs APR Inference Parity Specification

**Document ID:** PAIML-CANDLE-APR-001
**Version:** 1.2.0
**Last Updated:** 2026-04-01
**Status:** ACTIVE
**Methodology:** Popperian Falsification + Deterministic Benchmarks
**Primary Target:** Lambda Vector (RTX 4090, 24 GB VRAM, sm_89)
**Secondary Target:** Yoga (RTX 4060 Laptop, 8 GB VRAM, sm_89)
**Model:** Qwen2.5-Coder-1.5B-Instruct Q4_K_M GGUF (1.78B params, 28 layers, hidden=1536)

> Every claim in this spec carries a falsification condition.
> If the condition triggers, the claim is revised or retracted — not defended.

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

Head-to-head inference benchmark comparing **Candle** (HuggingFace's Rust ML framework) and **realizr** (Sovereign AI Stack inference engine) on the same model, same hardware, same methodology. This is the Rust-vs-Rust sibling of qwen-coder-deploy (which compares realizr vs llama.cpp vs vLLM vs ollama).

### Why This Matters

Candle is the most-adopted Rust ML framework. When developers evaluate the Sovereign AI Stack, their first question is: "Why not just use Candle?" This benchmark provides data-driven answers:

1. **Raw decode speed** — is the fused-kernel architecture measurably faster?
2. **Format advantage** — does APR v2 zero-copy loading justify the conversion step?
3. **Serving gap** — what throughput do you forfeit with CLI-only inference?
4. **Memory efficiency** — do fused kernels reduce peak VRAM usage?

### Chain of Reasoning

**Step 1: Why Candle specifically?** Both are pure Rust, both load GGUF Q4_K_M, both target CUDA. This isolates architectural decisions (fused kernels, format design, serving layer) from language/runtime differences. Comparing realizr to llama.cpp/vLLM mixes Rust-vs-C++ and Rust-vs-Python performance characteristics.

**Step 2: Why c=1 as the primary comparison?** Candle has no server mode — it's CLI-only. The only fair head-to-head is single-request decode. Concurrent benchmarks (c=4..32) demonstrate what Candle architecturally cannot provide.

**Step 3: What constitutes a win?** realizr must demonstrate measurable advantage in at least one of: decode throughput, model load time, or memory footprint at c=1. Concurrent scaling (c>1) and APR v2 format are realizr-exclusive capabilities — Candle cannot participate.

> **F-SUMMARY-01:** If Candle matches or beats realizr at c=1 on all three metrics (decode tok/s, load time, peak RSS), the fused-kernel advantage claim for single-request inference is falsified.

---

## 2. Scope

### This repo does:

- Build Candle and realizr from source with CUDA support
- Run deterministic, isolated benchmarks via forjar
- Measure decode throughput, TTFT, model load time, and memory footprint
- Compare GGUF, SafeTensors, and APR v2 format performance
- Report results as machine-readable JSON + human-readable tables

### This repo does NOT:

- Contain inference engine code (that's `../realizar`)
- Contain GPU kernels (that's `../trueno`)
- Contain the Candle framework (that's `../candle`)
- Contain the APR CLI (that's `../aprender`, binary: `apr`)
- Benchmark training (that's `../qwen-train-canary`)

### Upstream Bug Policy

When a benchmark reveals a bug in a dependency (realizr, trueno, aprender):

1. **File a `gh` issue** — `gh issue create --repo paiml/<repo>` with reproduction steps
2. **Fix in the upstream repo** — never work around bugs locally in this benchmark repo
3. **Add a provable-contract** — if the broken invariant can be expressed as a contract, add it to `binding.yaml` in the upstream repo via `provable-contracts`. This turns the runtime bug into a compile-time guarantee. Example: tensor name resolution must succeed for all supported naming conventions → contract on the adapter's `upload_weights` function.
4. **Rebuild via forjar** — update the forjar template if the fix requires new build flags or dependencies
5. **Re-run the falsification** — re-test the blocked F-condition and update the register

**Example (paiml/realizar#167):** APR Q4K GPU scheduler hardcoded HF tensor names, failing on GGUF-converted APR files. Fix: name normalization in `upload_apr_q4k_weights`. Contract candidate: `TENSOR_NAME_RESOLUTION_V1` — all weight lookups must resolve for GGUF, SafeTensors, and HF naming conventions.

### Relationship to sister repos

| Repo | Role | Focus |
|------|------|-------|
| **qwen-coder-deploy** | Benchmark | realizr vs llama.cpp vs vLLM vs ollama |
| **qwen-train-canary** | Benchmark | apr vs unsloth vs pytorch vs cublas |
| **candle-vs-apr** (this) | Benchmark | Candle vs realizr (Rust-vs-Rust) |
| **aprender** | Tooling | APR format, `apr` CLI (model import, conversion, profiling) |
| **realizar** | Engine | Inference engine under test |
| **trueno** | Kernel lib | SIMD/GPU kernel library (trueno-gpu for CUDA) |
| **provable-contracts** | Quality | Compile-time contract enforcement for upstream fixes |

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
| Clock | Locked (nvidia-smi -lgc) |
| CUDA | 12.6 Runtime / 13.1 Driver |
| CPU | AMD Threadripper / Intel Xeon |
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
| CUDA | 12.6 Runtime / 13.1 Driver |
| Transport | SSH (192.168.50.38) |

> **F-HW-01:** If run-to-run variance exceeds 5% with locked clocks, the determinism claim is falsified. Action: investigate thermal throttle or background GPU processes.

---

## 4. Model Under Test

**Qwen2.5-Coder-1.5B-Instruct** — same model as qwen-coder-deploy and qwen-train-canary.

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

| Format | Path | Created by | Runtimes |
|--------|------|-----------|----------|
| GGUF Q4_K_M | `/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | upstream (HuggingFace) | Candle, realizr |
| SafeTensors | `/home/noah/models/qwen2.5-coder-1.5b-instruct-safetensors/` | upstream (HuggingFace) | Candle, realizr |
| APR v2 Q4K | `/home/noah/models/qwen2.5-coder-1.5b-instruct-q4k.apr` | `apr import --preserve-q4k` | realizr only |

### Model preparation

**APR v2 conversion (preferred path — uses `apr` CLI from aprender):**
```bash
apr import /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --preserve-q4k --arch qwen2 \
  -o /home/noah/models/qwen2.5-coder-1.5b-instruct-q4k.apr
```

The `apr` binary is the canonical tool for format conversion across all sister repos (qwen-coder-deploy, qwen-train-canary). `--preserve-q4k` keeps Q4_K superblock layout intact for fused DP4A kernels; without it, weights are dequantized to F32 and requantized.

**Raw realizr (fallback — direct GGUF serving):**
```bash
realizar serve --model <gguf> --gpu --openai-api
```

realizr can serve GGUF files directly without pre-conversion. Use this path when testing GGUF parity (Phase 1) or when APR conversion is blocked.

> **F-MODEL-01:** If Candle's quantized-qwen2-instruct example cannot load the Q4_K_M GGUF file, the head-to-head comparison is blocked. Action: verify Candle's QMatMul supports Q4_K_M dequant path.

---

## 5. Benchmark Design

### Phase 1: Single-Request Decode (Head-to-Head)

The only fair comparison — both runtimes process one request at a time.

| Parameter | Value |
|-----------|-------|
| Prompt | Medium (~102 tokens), coding task |
| Max tokens | 256 |
| Iterations | 10 (drop first for cold-start) |
| Measurement | Wall time, tok/s (from runtime output) |
| Memory | Peak RSS via `/usr/bin/time -v` (Candle) |
| Isolation | forjar deploy, kill competing GPU processes |
| Clock | Locked (nvidia-smi -lgc) |

**Candle:** `quantized-qwen2-instruct --model <gguf> --prompt <text> --sample-len 256`
**realizr:** `curl /v1/chat/completions` with `stream: false`, extract `usage.completion_tokens`

### Phase 2: Concurrent Scaling (realizr-only)

Demonstrates what Candle's architecture cannot provide.

| c | Duration | Tool | Note |
|---|----------|------|------|
| 1 | 60s | bench-scaling.sh | Baseline for scaling efficiency |
| 4 | 60s | bench-scaling.sh | Continuous batching benefit |
| 8 | 60s | bench-scaling.sh | Memory pressure test |
| 16 | 60s | bench-scaling.sh | Near-asymptote |
| 32 | 60s | bench-scaling.sh | At asymptote |

### Phase 3: Format Comparison

APR v2 model prepared via `apr import --preserve-q4k` (preferred). Raw realizr GGUF serving as baseline.

| Format | Candle | realizr | Prepared by | Metrics |
|--------|--------|---------|-------------|---------|
| GGUF Q4_K_M | Yes | Yes (raw) | upstream HF | Load time, decode tok/s, RSS |
| SafeTensors FP16 | Yes | Yes (raw) | upstream HF | Load time, decode tok/s, RSS |
| APR v2 Q4K | No | Yes | `apr import --preserve-q4k` | Load time, decode tok/s, RSS |

### Methodology (inherited from PMAT-177)

- **60-second runs** with 5-second warmup — steady-state, not burst
- **Locked GPU clocks** — eliminates thermal throttle variance
- **Isolated serial** — one runtime at a time, clean GPU state
- **forjar deploy/teardown** — reproducible environment setup
- **probador scoring** — standardized quality scorecards (Phase 2)

---

## 6. Metrics Contract

### Primary Metrics

| Metric | Definition | Unit | How Measured |
|--------|-----------|------|-------------|
| Decode tok/s | Tokens generated per second (warm) | tok/s | Runtime output (9-run mean, drop cold start) |
| Cold-start tok/s | First-run decode speed | tok/s | First iteration only |
| Model load time | Time from process start to first token ready | ms | Wall time delta |
| Peak RSS | Maximum resident set size during inference | MB | `/usr/bin/time -v` |
| TTFT | Time to first token | ms | Timestamp delta |

### Derived Metrics (Phase 2)

| Metric | Definition | Unit |
|--------|-----------|------|
| Aggregate tok/s | Total tokens/sec across all concurrent requests | tok/s |
| Per-request decode | Tokens/sec experienced by individual request | tok/s |
| Scaling efficiency | (agg_c / agg_1) / c | ratio (1.0 = perfect) |
| ITL P50 | Inter-token latency, median | ms |
| TTFT P50 | Time to first token, median | ms |

### Data Format

All results saved as JSON in `results/`:
- `candle-<timestamp>.jsonl` — per-iteration Candle results
- `candle-summary-<timestamp>.json` — aggregated Candle metrics
- `realizr-c<N>-<timestamp>.json` — realizr results at concurrency N
- `realizr-c<N>-summary-<timestamp>.json` — aggregated realizr metrics

---

## 7. Baseline Thresholds & Pass Criteria

### Phase 1: Single-Request Parity (c=1)

| Metric | Prediction | Pass | Fail |
|--------|-----------|------|------|
| Decode tok/s | realizr within ±10% of Candle | ratio 0.90-1.10 | ratio < 0.90 |
| Cold-start tok/s | realizr within ±20% of Candle | ratio 0.80-1.20 | ratio < 0.80 |
| Model load (GGUF) | Within ±20% | ratio 0.80-1.20 | ratio < 0.80 |
| Peak RSS | realizr within ±15% of Candle | ratio 0.85-1.15 | ratio < 0.85 |

**Rationale:** At c=1, the GPU is underutilized. Fused kernels save one memory pass but the bottleneck is compute, not bandwidth. Serving overhead (HTTP stack, tokenizer init) may penalize realizr slightly.

> **F-PARITY-01:** If realizr decode is >20% slower than Candle at c=1, the serving overhead hypothesis is confirmed. Action: benchmark realizr CLI mode (no HTTP) to isolate.

### Phase 2: Format Advantage

| Metric | Prediction | Pass | Fail |
|--------|-----------|------|------|
| APR v2 load time | 2-5x faster than GGUF | ratio 2.0-5.0 | ratio < 1.5 |
| APR v2 RSS | Lower than GGUF (mmap) | RSS_apr < RSS_gguf | RSS_apr >= RSS_gguf |
| APR v2 decode | Within ±5% of GGUF decode | ratio 0.95-1.05 | ratio < 0.95 |

> **F-FORMAT-01:** If APR v2 load is <1.5x faster than GGUF, the zero-copy claim needs qualification — metadata parsing overhead is not the bottleneck.

### Phase 3: Scaling Demonstration

| c | Prediction (realizr agg tok/s) | Cross-ref |
|---|-------------------------------|-----------|
| 1 | ~148 tok/s | qwen-coder-deploy c=1 |
| 4 | ~325 tok/s | qwen-coder-deploy c=4 |
| 8 | ~525 tok/s | qwen-coder-deploy c=8 |
| 16 | ~931 tok/s | qwen-coder-deploy c=16 |
| 32 | ~1,600 tok/s | qwen-coder-deploy c=32 |

> **F-SCALE-01:** If realizr c=32 throughput is >20% below qwen-coder-deploy numbers on same hardware, there is a regression. Action: bisect realizr commits between deploy baseline and current.

---

## 8. Architectural Comparison

### Kernel Strategy

| Dimension | Candle | realizr |
|-----------|--------|---------|
| Dequantization | Separate QMatMul step | Fused with matmul (Q4K/Q5K/Q6K DP4A) |
| CUDA dispatch | Per-op kernel launch | CUDA graph (M=1 decode) |
| Attention | Standard scaled dot-product | Flash Decoding (KV chunked across CTAs) |
| KV cache | Manual, per-call allocation | GPU-resident, per-slot for batching |
| Weight reuse | None (c=1 only) | Batched GEMV: weights shared across M requests |
| Graph capture | Not implemented | Full forward pass at M=1, eager at M>1 |

### Expected Performance Profile

| Phase | Candle Advantage | realizr Advantage |
|-------|-----------------|-------------------|
| c=1 decode | Simpler dispatch, no HTTP overhead | Fused dequant+matmul saves 1 memory pass |
| Model load | — | APR v2 zero-copy mmap (skip parsing) |
| c>1 | N/A (no server) | Batch-and-step scheduler, weight sharing |
| Long sequences | — | Flash Decoding, GPU-resident KV |
| Format flexibility | — | GGUF + SafeTensors + APR v2 |

### Format Pipeline

```
                    Candle                          realizr (raw GGUF)
                    ──────                          ──────────────────
GGUF Q4_K_M ──► QMatMul dequant ──► matmul    GGUF Q4_K_M ──► fused Q4K DP4A ──► output
                (2 memory passes)              (1 memory pass, INT8 activations)

SafeTensors ──► FP16/FP32 matmul               SafeTensors ──► FP16 HGEMM (tensor cores)

                                                realizr (APR v2, via apr-cli)
                                                ────────────────────────────
                    N/A                         apr import ──► APR v2 Q4K ──► mmap ──► fused Q4K DP4A
                                               (zero-copy, LZ4/ZSTD, 64-byte aligned)
```

### Serving Architecture

```
Candle (CLI only)                    realizr (full serving stack)
─────────────────                    ──────────────────────────
stdin ──► tokenize ──► forward       HTTP ──► /v1/chat/completions
      ──► sample ──► stdout                ──► tokenize ──► batch scheduler
                                           ──► forward (CUDA graph / eager)
                                           ──► sample ──► SSE stream
                                           ──► circuit breaker / failover
                                           ──► privacy tier enforcement
```

---

## 9. Falsification Register

Pre-registered predictions with explicit falsification criteria. Each prediction is tested by benchmark and either confirmed, weakened, or retracted.

| ID | Prediction | Falsification Condition | Status | Evidence |
|----|-----------|------------------------|--------|----------|
| F-SUMMARY-01 | realizr wins on ≥1 of: decode, load, RSS at c=1 | Candle matches/beats all three | **FALSIFIED** | Candle: 227 tok/s, 449 MB RSS. realizr: 143 tok/s, 3082 MB RSS. Candle wins decode AND RSS. |
| F-PARITY-01 | realizr c=1 decode within ±10% of Candle | realizr >20% slower | **FALSIFIED** | Ratio 0.63x — realizr 37% slower. Candle 227.4 tok/s (decode-only) vs realizr 142.8 tok/s (wall-clock incl. HTTP+prefill). |
| F-FORMAT-01 | APR v2 load 2-5x faster than GGUF | APR v2 load <1.5x faster | **BLOCKED** | APR v2 model errors: "Tensor not found: model.embed_tokens.weight". Format conversion incomplete. |
| F-SCALE-01 | realizr c=32 ≥1,280 tok/s (80% of deploy baseline) | realizr c=32 <1,280 tok/s | **FALSIFIED** | c=32 agg: 145.7 tok/s (89% below target). Server started in SINGLE-REQUEST mode; no batch scheduling active. Throughput flat across c=1..32. |
| F-HW-01 | Run-to-run variance <5% with locked clocks | Variance ≥5% | **CONFIRMED** | Candle CV=0.8% (temp=0, greedy). realizr CV=0.9%. Locked at 2520 MHz on RTX 4090. Note: temp=0.8 produces 13% CV (non-deterministic output lengths). |
| F-MODEL-01 | Candle loads Q4_K_M GGUF successfully | Candle errors on load | **CONFIRMED** | Loaded 339 tensors (1.11 GB) in 0.49s. Required lazy-curand patch (curand device library missing on Lambda Vector) and CUDA 12.6 toolkit (PTX 9.0 from CUDA 13.0 unsupported by 570.207 driver). |
| F-KERNEL-01 | Fused Q4K DP4A has lower memory traffic than QMatMul | nsys shows equal or higher BW | UNTESTED | Requires nsys profiling (Phase 4). |
| F-RSS-01 | APR v2 RSS < GGUF RSS (mmap paging) | APR v2 RSS ≥ GGUF RSS | **BLOCKED** | APR v2 model fails to load (missing embedding tensor). Cannot compare. |
| F-COLD-01 | realizr cold-start slower (HTTP + server init) | realizr cold-start faster | **CONFIRMED** | Candle cold: 223.1 tok/s (includes 0.49s model load). realizr cold: 134.4 tok/s (server warm, first-request GPU kernel compilation). realizr per-request cold start is slower as predicted. |
| F-SERVING-01 | Serving overhead <5ms per request at c=1 | Overhead ≥10ms | **WEAKENED** | HTTP health: ~5ms (at threshold). Full 1-token request: 35ms. Pure HTTP overhead meets 5ms target, but end-to-end overhead (tokenization + scheduling) is ~27ms. |

---

## 10. Work Items

### Phase 0: Infrastructure (PMAT-300 block)

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-301 | Build Candle with CUDA on Lambda Vector | DONE | — |
| PMAT-302 | Verify Candle loads Qwen2.5 Q4_K_M GGUF | DONE | PMAT-301 |
| PMAT-303 | Create forjar templates (candle, realizr, teardown) | DONE | — |
| PMAT-304 | Create benchmark scripts (candle, realizr, compare) | DONE | — |
| PMAT-305 | Lock GPU clocks, verify <5% variance | DONE | PMAT-301 |
| PMAT-306 | Validate probador scoring against qwen-coder-deploy | BLOCKED | probador has no `llm` subcommand |

### Phase 1: Single-Request Head-to-Head (PMAT-310 block)

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-311 | Candle c=1 GGUF decode (10 iterations) | DONE | PMAT-302 |
| PMAT-312 | realizr c=1 GGUF decode (10 iterations) | DONE | PMAT-303 |
| PMAT-313 | Compare decode tok/s, generate table | DONE | PMAT-311, 312 |
| PMAT-314 | Measure model load time (cold start) | DONE | PMAT-311, 312 |
| PMAT-315 | Measure peak RSS both runtimes | DONE | PMAT-311, 312 |
| PMAT-316 | Validate F-PARITY-01 (±10% decode) | DONE (FALSIFIED) | PMAT-313 |
| PMAT-317 | If F-PARITY-01 fails: profile with nsys | TODO | PMAT-316 |

### Phase 2: Concurrent Scaling (PMAT-320 block)

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-321 | realizr c=1,4,8,16,32 (60s each) | DONE | PMAT-312 |
| PMAT-322 | Cross-reference against qwen-coder-deploy baselines | DONE (all miss) | PMAT-321 |
| PMAT-323 | Validate F-SCALE-01 (≥80% of deploy baseline) | DONE (FALSIFIED) | PMAT-322 |
| PMAT-324 | Generate scaling efficiency table | DONE | PMAT-321 |
| PMAT-325 | Quality scorecards (probador llm score) | BLOCKED | probador has no `llm` command |

### Phase 3: Format Comparison (PMAT-330 block)

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-331 | Candle SafeTensors decode (non-quantized) | TODO | PMAT-302 |
| PMAT-332 | realizr SafeTensors decode | TODO | — |
| PMAT-333 | realizr APR v2 Q4K decode | BLOCKED | APR model missing embed_tokens |
| PMAT-334 | Measure load time: GGUF vs SafeTensors vs APR v2 | BLOCKED | PMAT-333 |
| PMAT-335 | Measure RSS: GGUF vs SafeTensors vs APR v2 | BLOCKED | PMAT-333 |
| PMAT-336 | Validate F-FORMAT-01 (APR v2 load 2-5x faster) | BLOCKED | PMAT-334 |

### Phase 4: Deep Profiling (PMAT-340 block)

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-341 | nsys timeline: Candle c=1 decode step | TODO | PMAT-311 |
| PMAT-342 | nsys timeline: realizr c=1 decode step | TODO | PMAT-312 |
| PMAT-343 | ncu roofline: Candle QMatMul kernel | TODO | PMAT-341 |
| PMAT-344 | ncu roofline: realizr fused Q4K DP4A kernel | TODO | PMAT-342 |
| PMAT-345 | Kernel launch count comparison | TODO | PMAT-341, 342 |
| PMAT-346 | Memory bandwidth utilization comparison | TODO | PMAT-343, 344 |
| PMAT-347 | Validate F-KERNEL-01 (fused kernel lower BW) | TODO | PMAT-346 |

### Phase 5: Publication (PMAT-350 block)

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-351 | Fill performance.md results tables | TODO | Phase 1-3 |
| PMAT-352 | Write findings section with falsification outcomes | TODO | PMAT-351 |
| PMAT-353 | Generate comparison charts (throughput, scaling) | TODO | PMAT-351 |
| PMAT-354 | Cross-reference with qwen-coder-deploy spec | TODO | PMAT-352 |
| PMAT-355 | README update with key findings table | TODO | PMAT-352 |

---

## 11. PMAT Compliance

### Quality Gates

| Gate | Requirement | Enforcement |
|------|-------------|-------------|
| Determinism | <5% run-to-run variance | F-HW-01, locked clocks |
| Isolation | One runtime at a time | forjar serial deploy/teardown |
| Reproducibility | All results as JSON | `results/` directory, git-tracked |
| Falsifiability | Every claim has F-condition | Section 9 register |
| Cross-validation | Results match sister repos | F-SCALE-01 vs qwen-coder-deploy |

### Spec Maintenance

- Maximum 500 lines (this document)
- Component specs in `docs/specifications/components/` for deep dives
- Version bump on every structural change
- Work items tracked in PMAT-300 block

---

## 12. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-04-01 | Initial spec: 3-phase benchmark design, 10 falsification conditions, 30 work items |
| 1.1.0 | 2026-04-01 | Phase 1+2 results: 3 FALSIFIED, 3 CONFIRMED, 1 WEAKENED, 3 BLOCKED/UNTESTED. Candle 1.6x faster at c=1. No scaling (SINGLE-REQUEST mode). APR v2 format broken. |
| 1.2.0 | 2026-04-01 | Prefer apr-cli for model prep. Upstream bug policy: gh tickets + provable-contracts. Fixed paiml/realizar#167 (tensor name normalization). Reconverted APR v2 via `apr import --preserve-q4k`. |
