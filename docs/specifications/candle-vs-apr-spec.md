# Candle vs APR Inference Parity Specification

**Document ID:** PAIML-CANDLE-APR-001
**Version:** 1.4.0
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

**Step 3: What constitutes a win?** realizr must demonstrate measurable advantage in at least one of: decode throughput, model load time, or memory footprint at c=1. **Result: Candle won all three at c=1 (F-SUMMARY-01 FALSIFIED).** Concurrent scaling (c>1) was not demonstrated (SINGLE-REQUEST mode). APR v2 format comparison is blocked (paiml/realizar#168).

> **F-SUMMARY-01: FALSIFIED.** Candle beats realizr on decode (1.59x) and RSS (6.9x less) at c=1. The fused-kernel advantage does not materialize for single-request inference on RTX 4090. realizr's serving overhead (HTTP + prefill) is the dominant factor.

---

## 2. Scope

### This repo does:

- Build Candle and realizr from source with CUDA support
- Run deterministic, isolated benchmarks via forjar
- Measure decode throughput, TTFT, model load time, and memory footprint
- **Enforce format parity:** all 3 formats (GGUF, SafeTensors, APR v2) must have GPU inference — any gap is a bug, not a limitation
- **Enforce tool parity:** `apr` CLI and raw `realizr` must produce equivalent results on the same model
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
| Clock | Locked 2520 MHz (nvidia-smi -lgc 2520) |
| CUDA | 12.6 Toolkit (forced) / 12.8 Driver (570.207) |
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
| Prompt | Coding task (~38 tokens after chat template) |
| Max tokens | 256 |
| Iterations | 10 (drop first for cold-start) |
| Measurement | Wall time, tok/s (from runtime output) |
| Memory | Peak RSS via `/usr/bin/time -v` (Candle) |
| Isolation | forjar deploy, kill competing GPU processes |
| Clock | Locked (nvidia-smi -lgc) |

**Candle:** `quantized-qwen2-instruct --model <gguf> --prompt <text> --sample-len 256 --temperature 0`
**realizr:** `curl /v1/chat/completions` with `stream: false`, `temperature: 0`, extract `usage.completion_tokens`

Temperature 0 (greedy) is mandatory for determinism. With temperature >0, non-deterministic output lengths produce 13% CV (F-HW-01 would fail).

### Phase 2: Concurrent Scaling (realizr-only)

Demonstrates what Candle's architecture cannot provide.

| c | Duration | Tool | Note |
|---|----------|------|------|
| 1 | 60s | bench-scaling.sh | Baseline for scaling efficiency |
| 4 | 60s | bench-scaling.sh | Continuous batching benefit |
| 8 | 60s | bench-scaling.sh | Memory pressure test |
| 16 | 60s | bench-scaling.sh | Near-asymptote |
| 32 | 60s | bench-scaling.sh | At asymptote |

### Phase 3: Format + Tool Parity

**Invariant:** All 3 formats must have GPU inference. Both `apr` CLI and raw `realizr` must produce equivalent results. Any format without a GPU path is a bug, not a limitation.

#### 3a. Format parity (F-FMTPARITY-01)

| Format | Candle (GPU) | realizr (GPU) | Status | Blocker |
|--------|-------------|---------------|--------|---------|
| GGUF Q4_K_M | 227.4 tok/s | 142.8 tok/s | **Measured** | — |
| SafeTensors FP32 | 65.7 tok/s | 0.4 tok/s (CPU!) | **BUG** | paiml/realizar#169 |
| APR v2 Q4K | N/A | garbage output | **BUG** | paiml/realizar#168 |

#### 3b. Tool parity: `apr` CLI vs `realizr` (F-TOOLPARITY-01)

| Format | Tool | Command | Status |
|--------|------|---------|--------|
| GGUF Q4_K_M | `realizr serve --gpu` | direct GGUF serving | Measured (142.8 tok/s) |
| GGUF Q4_K_M | `apr serve run --gpu` | apr-cli GGUF serving | UNTESTED |
| APR v2 Q4K | `realizr serve --gpu` | raw APR serving | BLOCKED (#168) |
| APR v2 Q4K | `apr serve run --gpu` | apr-cli APR serving | UNTESTED |

Both tools loading the same model in the same format must produce tok/s within ±5%.

### Methodology (inherited from PMAT-177)

- **Phase 1:** 10 iterations, greedy (temp=0), drop first for cold-start, 9-run mean
- **Phase 2:** 60-second runs with 5-second warmup — steady-state, not burst
- **Locked GPU clocks** — eliminates thermal throttle variance (<1% CV measured)
- **Isolated serial** — one runtime at a time, clean GPU state
- **forjar deploy/teardown** — reproducible environment setup
- **bench-scaling.sh** — concurrent load testing (Phase 2). Note: `probador` on this system is a WASM test tool, not an LLM load tester.

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
- `realizr-c1-<timestamp>.jsonl` — realizr c=1 per-iteration results
- `realizr-c1-summary-<timestamp>.json` — aggregated realizr c=1 metrics
- `realizr-scaling-c<N>-<timestamp>.jsonl` — scaling per-request results
- `realizr-scaling-c<N>-summary.json` — aggregated scaling metrics

---

## 7. Baseline Thresholds & Pass Criteria

### Phase 1: Single-Request Parity (c=1)

| Metric | Prediction | Actual | Status |
|--------|-----------|--------|--------|
| Decode tok/s | ratio 0.90-1.10 | **0.63x** (227.4 vs 142.8) | **FAIL** |
| Cold-start tok/s | ratio 0.80-1.20 | **0.60x** (223.1 vs 134.4) | **FAIL** |
| Model load (GGUF) | ratio 0.80-1.20 | 0.49s (Candle) vs amortized (realizr) | N/A (different model) |
| Peak RSS | ratio 0.85-1.15 | **0.15x** (449 vs 3,082 MB) | **FAIL** |

**Original rationale (pre-test):** At c=1, the GPU is underutilized. Fused kernels save one memory pass but the bottleneck is compute, not bandwidth. Serving overhead (HTTP stack, tokenizer init) may penalize realizr slightly.

**Post-test finding:** The penalty is not "slight" — it's 37%. The metric asymmetry (Candle self-reported decode-only vs realizr wall-clock including HTTP+prefill) accounts for part of the gap. A realizr CLI-mode benchmark (PMAT-317) is needed to isolate serving overhead from kernel performance.

> **F-PARITY-01: FALSIFIED.** realizr 37% slower. Action: PMAT-317 (nsys profile) + realizr CLI-mode benchmark to isolate HTTP overhead.

### Phase 2: Scaling Demonstration

| c | Predicted | Actual | Status |
|---|-----------|--------|--------|
| 1 | ~148 tok/s | 117.0 tok/s | -21% |
| 4 | ~325 tok/s | 116.7 tok/s | **-64%** |
| 8 | ~525 tok/s | 126.3 tok/s | **-76%** |
| 16 | ~931 tok/s | 112.5 tok/s | **-88%** |
| 32 | ~1,600 tok/s | 145.7 tok/s | **-91%** |

Predictions cross-referenced from qwen-coder-deploy baselines.

> **F-SCALE-01: FALSIFIED.** Throughput flat at ~120-146 tok/s across all concurrency levels. Root cause: realizr started in SINGLE-REQUEST mode (no batch scheduler). The `--openai-api` flag does not activate continuous batching.

### Phase 3: Format Advantage

| Metric | Prediction | Pass | Fail | Status |
|--------|-----------|------|------|--------|
| APR v2 load time | 2-5x faster than GGUF | ratio 2.0-5.0 | ratio < 1.5 | BLOCKED |
| APR v2 RSS | Lower than GGUF (mmap) | RSS_apr < RSS_gguf | RSS_apr >= RSS_gguf | BLOCKED |
| APR v2 decode | Within ±5% of GGUF decode | ratio 0.95-1.05 | ratio < 0.95 | BLOCKED |

> **F-FORMAT-01: BLOCKED.** APR model loads (paiml/realizar#167 fixed) but inference produces garbage output (paiml/realizar#168). GPU adapter weight name mapping incomplete.

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

### Observed vs Expected Performance

| Phase | Predicted Winner | Actual Winner | Notes |
|-------|-----------------|---------------|-------|
| c=1 decode | Tie (±10%) | **Candle (1.59x)** | Candle 227 tok/s vs realizr 143 tok/s. Serving overhead + prefill dominates. |
| Peak RSS | Tie (±15%) | **Candle (6.9x less)** | 449 MB vs 3,082 MB. realizr includes server + KV cache pool for batch_size=32. |
| Cold start | realizr slower | **Confirmed** | 223 vs 134 tok/s. realizr JIT-compiles PTX on first request. |
| c>1 scaling | realizr scales | **Not demonstrated** | SINGLE-REQUEST mode: throughput flat at ~120-146 tok/s c=1..32. |
| APR v2 format | realizr wins | **BLOCKED** | Inference garbage output (paiml/realizar#168). |

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

Candle: `stdin → tokenize → forward → sample → stdout` (CLI only, no server).
realizr: `HTTP → /v1/chat/completions → tokenize → batch scheduler → forward (CUDA graph) → SSE stream`.

---

## 9. Falsification Register

Pre-registered predictions with explicit falsification criteria. Each prediction is tested by benchmark and either confirmed, weakened, or retracted.

| ID | Prediction | Falsification Condition | Status | Evidence |
|----|-----------|------------------------|--------|----------|
| F-SUMMARY-01 | realizr wins on ≥1 of: decode, load, RSS at c=1 | Candle matches/beats all three | **FALSIFIED** | Candle: 227 tok/s, 449 MB RSS. realizr: 143 tok/s, 3082 MB RSS. Candle wins decode AND RSS. |
| F-PARITY-01 | realizr c=1 decode within ±10% of Candle | realizr >20% slower | **FALSIFIED** | Ratio 0.63x — realizr 37% slower. Candle 227.4 tok/s (decode-only) vs realizr 142.8 tok/s (wall-clock incl. HTTP+prefill). |
| F-FORMAT-01 | APR v2 load 2-5x faster than GGUF | APR v2 load <1.5x faster | **BLOCKED** | APR reconverted via `apr import --preserve-q4k`. Loading fixed (paiml/realizar#167). Norm aliasing fixed (paiml/realizar#168). Inference produces garbage — further GPU adapter investigation needed in realizr. |
| F-SCALE-01 | realizr c=32 ≥1,280 tok/s (80% of deploy baseline) | realizr c=32 <1,280 tok/s | **FALSIFIED** | c=32 agg: 145.7 tok/s (89% below target). Server started in SINGLE-REQUEST mode; no batch scheduling active. Throughput flat across c=1..32. |
| F-HW-01 | Run-to-run variance <5% with locked clocks | Variance ≥5% | **CONFIRMED** | Candle CV=0.8% (temp=0, greedy). realizr CV=0.9%. Locked at 2520 MHz on RTX 4090. Note: temp=0.8 produces 13% CV (non-deterministic output lengths). |
| F-MODEL-01 | Candle loads Q4_K_M GGUF successfully | Candle errors on load | **CONFIRMED** | Loaded 339 tensors (1.11 GB) in 0.49s. Required lazy-curand patch (curand device library missing on Lambda Vector) and CUDA 12.6 toolkit (PTX 9.0 from CUDA 13.0 unsupported by 570.207 driver). |
| F-KERNEL-01 | Fused Q4K DP4A has lower memory traffic than QMatMul | nsys shows equal or higher BW | UNTESTED | Requires nsys profiling (Phase 4). |
| F-RSS-01 | APR v2 RSS < GGUF RSS (mmap paging) | APR v2 RSS ≥ GGUF RSS | **BLOCKED** | APR loads successfully but inference output is garbage. Blocked on paiml/realizar#168 resolution. |
| F-COLD-01 | realizr cold-start slower (HTTP + server init) | realizr cold-start faster | **CONFIRMED** | Candle cold: 223.1 tok/s (includes 0.49s model load). realizr cold: 134.4 tok/s (server warm, first-request GPU kernel compilation). realizr per-request cold start is slower as predicted. |
| F-SERVING-01 | Serving overhead <5ms per request at c=1 | Overhead ≥10ms | **WEAKENED** | HTTP health: ~5ms (at threshold). Full 1-token request: 35ms. Pure HTTP overhead meets 5ms target, but end-to-end overhead (tokenization + scheduling) is ~27ms. |
| F-FMTPARITY-01 | All 3 formats produce equivalent GPU tok/s (±10%) | Any format lacks GPU path or differs >10% | **FALSIFIED** | SafeTensors has no GPU path (0.4 tok/s CPU vs 143 GGUF GPU = 357x gap, paiml/realizar#169). APR v2 inference garbage (paiml/realizar#168). Only GGUF has working GPU inference. |
| F-TOOLPARITY-01 | `apr serve` and `realizr serve` produce same tok/s on same model (±5%) | Difference >5% on same format | UNTESTED | Requires apr-cli serve path + realizr serve path on same GGUF and same APR. |

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

### Phase 3: Format + Tool Parity (PMAT-330 block)

| ID | Task | Status | Depends |
|----|------|--------|---------|
| PMAT-331 | Candle SafeTensors decode (non-quantized) | DONE | PMAT-302 |
| PMAT-332 | realizr SafeTensors decode | DONE (BUG: CPU only, #169) | — |
| PMAT-333 | realizr APR v2 Q4K decode | BLOCKED | #167 fixed, #168 open — inference garbage |
| PMAT-334 | Measure load time: GGUF vs SafeTensors vs APR v2 | BLOCKED | PMAT-333 |
| PMAT-335 | Measure RSS: GGUF vs SafeTensors vs APR v2 | BLOCKED | PMAT-333 |
| PMAT-336 | Validate F-FORMAT-01 (APR v2 load 2-5x faster) | BLOCKED | PMAT-334 |
| PMAT-337 | Re-test SafeTensors GPU after #169 fix | TODO | paiml/realizar#169 |
| PMAT-338 | Re-test APR v2 GPU after #168 fix | TODO | paiml/realizar#168 |
| PMAT-339 | Validate F-FMTPARITY-01 (all 3 formats GPU ±10%) | TODO | PMAT-337, 338 |
| PMAT-360 | apr-cli serve GGUF vs realizr serve GGUF | TODO | — |
| PMAT-361 | apr-cli serve APR vs realizr serve APR | TODO | PMAT-338 |
| PMAT-362 | Validate F-TOOLPARITY-01 (apr vs realizr ±5%) | TODO | PMAT-360, 361 |

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
| PMAT-351 | Fill performance.md results tables | PARTIAL | Phase 1-2 + SafeTensors filled; APR v2 BLOCKED |
| PMAT-352 | Write findings section with falsification outcomes | DONE | PMAT-351 |
| PMAT-353 | Generate comparison charts (throughput, scaling) | TODO | PMAT-351 |
| PMAT-354 | Cross-reference with qwen-coder-deploy spec | TODO | PMAT-352 |
| PMAT-355 | README update with key findings table | DONE | — |

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
| **Format parity** | All 3 formats (GGUF, SafeTensors, APR v2) tested on GPU | F-FMTPARITY-01 |
| **Tool parity** | `apr` CLI and raw `realizr` produce equivalent inference | F-TOOLPARITY-01 |

### Spec Maintenance

Maximum 500 lines. Version bump on structural changes. Work items in PMAT-300 block.

## 12. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-04-01 | Initial spec: 3-phase benchmark design, 10 falsification conditions, 30 work items |
| 1.1.0 | 2026-04-01 | Phase 1+2 results: 3 FALSIFIED, 3 CONFIRMED, 1 WEAKENED, 3 BLOCKED/UNTESTED. Candle 1.6x faster at c=1. No scaling (SINGLE-REQUEST mode). APR v2 format broken. |
| 1.2.0 | 2026-04-01 | Prefer apr-cli for model prep. Upstream bug policy: gh tickets + provable-contracts. Fixed paiml/realizar#167 (tensor name normalization). Reconverted APR v2 via `apr import --preserve-q4k`. |
| 1.3.0 | 2026-04-01 | Reconcile all predictions with actuals. Section 7 Phase numbering fixed (scaling=2, format=3). Section 8 observed vs expected. Section 1 summary updated with F-SUMMARY-01 FALSIFIED. Temperature=0 documented as mandatory. |
| 1.4.0 | 2026-04-02 | Format + tool parity as hard requirements. F-FMTPARITY-01 (all 3 formats GPU), F-TOOLPARITY-01 (apr-cli vs realizr). SafeTensors CPU-only is a bug (#169), not a trade-off. 6 new PMAT items (337-339, 360-362). |
