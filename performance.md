# Candle vs realizr — Inference Performance

## Methodology

Same production methodology as qwen-coder-deploy (PMAT-177):
- Model: Qwen2.5-Coder-1.5B-Instruct Q4_K_M GGUF (1.11 GB)
- Hardware: RTX 4090 (Lambda Vector), 2520 MHz locked, sm_89
- Prompt: Fixed coding task (~38 tokens), max_tokens=256, temperature=0 (greedy)
- Iterations: 10 per runtime, drop first for cold start
- Isolation: forjar serial deployment (one runtime at a time)
- Candle: CUDA 12.6 PTX, lazy-curand patch (device curand missing on Lambda)
- realizr: CUDA build, GGUF SINGLE-REQUEST mode

## Predictions (Pre-Registration)

Before running benchmarks, we register falsifiable predictions per Popperian methodology:

### P1: Single-Request Decode (c=1)

**Prediction:** realizr within ±10% of Candle at c=1.

**Rationale:** At c=1, both are single-threaded decode on the same GPU. The fused Q4K kernel saves one memory pass vs Candle's separate dequant+matmul, but at c=1 the GPU is underutilized so memory bandwidth isn't the bottleneck.

**Falsification criterion:** If realizr is >20% slower at c=1, the serving overhead (HTTP, tokenizer) is a measurable penalty for single-request workloads.

### P2: Model Load Time

**Prediction:** GGUF load times within ±20%. APR v2 load 2-5x faster than GGUF.

**Rationale:** GGUF parsing is similar in both. APR v2 skips parsing entirely (mmap + binary index).

**Falsification criterion:** If APR v2 load is <1.5x faster, the zero-copy claim needs qualification.

### P3: Memory Footprint

**Prediction:** Similar RSS for GGUF. APR v2 RSS lower (mmap pages in only on access).

**Falsification criterion:** If APR v2 RSS is higher, alignment padding or metadata overhead dominates.

### P4: Scaling

**Prediction:** realizr reaches 1,500+ tok/s at c=32 (matching qwen-coder-deploy results). Candle: N/A (no server).

**Falsification criterion:** If realizr c=32 throughput is >20% below qwen-coder-deploy numbers on same hardware, there's a regression.

## Results

### Phase 1: Single-Request Decode (c=1)

| Metric | Candle | realizr | Ratio | Status |
|--------|--------|---------|-------|--------|
| Decode (tok/s, warm) | 227.4 | 142.8 | 0.63x | **F-PARITY-01: FALSIFIED** |
| Decode (tok/s, cold) | 223.1 | 134.4 | 0.60x | |
| Decode CV (%) | 0.8% | 0.9% | — | **F-HW-01: CONFIRMED** |
| Wall time (ms, mean) | 2,456 | 1,804 | 0.73x | realizr wins wall-clock (no model load) |
| Model load (ms) | 490 | amortized | — | |
| Peak RSS (MB) | 449 | 3,082 | 6.9x | realizr 6.9x higher (server + KV cache) |

**Note on decode metric asymmetry:** Candle 227.4 tok/s is self-reported decode-only (excludes prompt processing). realizr 142.8 tok/s is wall-clock end-to-end (includes HTTP round-trip + tokenization + prefill + decode). A fairer realizr decode-only estimate: ~148 tok/s (subtracting ~60ms prefill).

**F-SUMMARY-01: FALSIFIED** — Candle beats realizr on both decode throughput (1.6x) and RSS (6.9x less). realizr only wins on amortized wall-clock (no model reload per request).

### Phase 2: realizr Scaling (Candle: N/A)

| c | Agg tok/s | Per-req tok/s | Wall P50 (ms) | Predicted |
|---|-----------|---------------|---------------|-----------|
| 1 | 117.0 | 137.8 | 1,829 | ~148 |
| 4 | 116.7 | 33.2 | 8,713 | ~325 |
| 8 | 126.3 | 20.9 | 15,371 | ~525 |
| 16 | 112.5 | 13.6 | 35,292 | ~931 |
| 32 | 145.7 | 11.3 | 56,076 | ~1,600 |

**F-SCALE-01: FALSIFIED** — Aggregate throughput flat at ~120-146 tok/s (no scaling). Server in SINGLE-REQUEST mode; continuous batching NOT active. c=32 at 145.7 tok/s is 91% below predicted 1,600 tok/s.

**Root cause:** realizr started in `Mode: SINGLE-REQUEST` with `--openai-api`. Batch scheduling requires `--batch` mode, which was not tested. Requests are queued serially.

### Phase 3: Format Comparison

| Format | Runtime | Load (ms) | Decode (tok/s) | RSS (MB) | Status |
|--------|---------|-----------|----------------|----------|--------|
| GGUF Q4_K_M | Candle | 490 | 227.4 | 449 | Measured |
| GGUF Q4_K_M | realizr | amortized | 142.8 | 3,082 | Measured |
| SafeTensors FP32 | Candle (GPU) | ~1,500 | 65.7 | 3,344 | Measured |
| SafeTensors FP32 | realizr (GPU) | ~11,000 | 21.2 | — | **#169 FIXED** (was 0.4 CPU) |
| APR v2 Q4K | realizr (GPU) | ~60,000 | 17.4 | 2,278 | **#170 FIXED** (via from_apr→GGUF CUDA) |

**F-FORMAT-01: FALSIFIED** — APR load is ~60s (vs GGUF 0.49s) due to dequant+requant roundtrip — 120x slower, not 2-5x faster. Zero-copy claim does not hold for the `from_apr` path.
**F-RSS-01: CONFIRMED** — APR RSS 2,278 MB < GGUF RSS 3,082 MB (26% less, mmap paging).

## Comparison Charts

### Decode Throughput (tok/s, c=1, RTX 4090)

```
  Candle GGUF Q4K (GPU)        ████████████████████████████████████████ 227.4
  realizr GGUF Q4K (GPU)       █████████████████████████ 142.8
  apr-cli GGUF Q4K (GPU)       ████████████████████████ 139.8
  Candle SafeT FP32 (GPU)      ███████████ 65.7
  realizr SafeT FP32 (GPU)     ████ 21.2
  apr-cli APR Q4K (GPU)        ████ 21.9
  realizr APR Q4K (GPU)        ███ 17.4
```

### realizr Scaling (SINGLE-REQUEST mode)

```
  c=1   ███████████████████████████████ 117.0
  c=4   ███████████████████████████████ 116.7
  c=8   █████████████████████████████████ 126.3
  c=16  ██████████████████████████████ 112.5
  c=32  ██████████████████████████████████████ 145.7
  (flat — no batch scheduling active)
```

### Kernel Launches (32 tokens, nsys)

```
  Candle   ████████████████████████████████████████ 40,513
  realizr  ██████████████████████ 22,360
  (1.8x fewer launches, same total GPU time: 106ms vs 105ms)
```

## Architectural Comparison

### Kernel Strategy

| Aspect | Candle | realizr |
|--------|--------|---------|
| Dequantization | Separate step (QMatMul) | Fused with matmul (Q4K/Q5K/Q6K) |
| CUDA dispatch | Per-op kernel launch | CUDA graph (M=1) |
| Attention | Standard | FlashAttention-style tiled |
| KV cache | Manual management | Integrated with serving layer |

### Format Support

| Format | Candle | realizr |
|--------|--------|---------|
| GGUF | Direct load | Direct load |
| SafeTensors | Direct load | Direct load |
| APR v2 | Not supported | Zero-copy mmap, LZ4/ZSTD |

### Serving Capabilities

| Feature | Candle | realizr |
|---------|--------|---------|
| HTTP API | None | OpenAI-compatible |
| Concurrent requests | Not supported | Batch-and-step / iteration scheduler |
| Streaming | stdout only | SSE streaming |
| Failover | None | Circuit breakers |
| Privacy tiers | None | Sovereign/Private/Standard |

## Findings

### Finding 1: Candle wins c=1 decode by 1.6x (F-SUMMARY-01, F-PARITY-01 FALSIFIED)

**What:** Candle decodes at 227.4 tok/s vs realizr at 142.8 tok/s on the same GGUF Q4_K_M model, same RTX 4090.

**Why (five-whys):**
1. Why is realizr slower? → The 142.8 tok/s includes HTTP round-trip + tokenization + prefill, while Candle's 227.4 is decode-only.
2. Why does the metric asymmetry matter? → Subtracting ~60ms prefill gives ~148 tok/s — still 35% slower.
3. Why is realizr's raw decode slower? → realizr runs Q4K GEMV through CUDA with JIT-compiled PTX. Candle uses pre-compiled QMatMul kernels without a JIT step.
4. Why doesn't the fused kernel help? → At c=1, the RTX 4090 is bandwidth-limited only for large matrices. For 1536-dim hidden state, compute is the bottleneck, and fused dequant saves ~0.1ms per layer — negligible vs the 7ms/token total.
5. Why not? → The fused kernel architecture is designed for throughput at c>1 (batched GEMV shares weight loads across M requests). At c=1, M=1, there's no sharing benefit.

**Implication:** The "fused-kernel advantage" is a batching advantage, not a single-request advantage. Marketing realizr on c=1 performance against Candle would be misleading.

### Finding 2: No scaling demonstrated (F-SCALE-01 FALSIFIED)

**What:** realizr throughput was flat at ~120-146 tok/s from c=1 to c=32. Predicted: 148 → 1,600 tok/s.

**Why:** realizr started in `Mode: SINGLE-REQUEST` with `--openai-api`. The `--openai-api` flag enables the API format but does NOT activate the batch scheduler. Requests are queued and processed serially. The batch-and-step scheduler requires explicit configuration (environment variable or `--batch` flag).

**Implication:** The scaling numbers from qwen-coder-deploy used a different server configuration. This benchmark did not test the batch scheduler path. Re-testing with batch mode enabled would address F-SCALE-01 properly.

### Finding 3: realizr RSS 6.9x higher at c=1 (not a fair comparison)

**What:** Candle: 449 MB. realizr: 3,082 MB.

**Why:** realizr pre-allocates KV cache for `max_batch=32` slots at startup (`[PMAT-399] Auto-sized CUDA_MAX_BATCH=32`). Each slot uses ~0.2 GB for KV storage. 32 slots × 0.2 GB = 6.4 GB. The server also includes the tokio runtime, axum HTTP stack, and tokenizer. Candle is a CLI tool that exits after each run — no server overhead, no KV cache pool.

**Implication:** RSS comparison is only meaningful at matched concurrency. At c=1, realizr over-provisions by 32x.

### Finding 4: SafeTensors GPU path fixed (paiml/realizar#169 FIXED)

**What:** Before fix: Candle 65.7 tok/s (GPU FP32), realizr 0.4 tok/s (CPU FP32) — 164x gap. After fix: realizr 21.2 tok/s (GPU FP32) — 3.1x gap remains.

**Why (five-whys):**
1. Why was realizr CPU-only? → GPU path only supported quantized formats (Q4K, Q6K via DP4A).
2. Why 21.2 vs 65.7 after fix? → realizr uses FP32 SGEMM; Candle uses optimized QMatMul with FP16 tensor cores.
3. Why not use tensor cores? → SafeTensors FP32 weights need FP16 downcast for HGEMM, not yet implemented.
4. Why not auto-quantize on load? → Would add latency and change numerical behavior.
5. Root cause: **FP32 SGEMM is compute-limited vs Candle's FP16 tensor core path.**

**Implication:** #169 fixed the missing GPU path. The 3.1x gap is now a performance optimization issue, not a correctness bug. FP16 HGEMM or on-load quantization would close it.

### Finding 5: Infrastructure blockers on Lambda Vector

Two infrastructure issues required workarounds to run benchmarks:

1. **curand device library missing** — Candle eagerly initializes curand at GPU device creation, but the Lambda Vector CUDA 13.0 install lacks `libcurand_device.a`. Fix: lazy-curand patch in Candle (defer init until first `rand_*` call).

2. **CUDA 13.0 PTX incompatible with driver** — nvcc 13.0 generates PTX 9.0, but driver 570.207 only supports PTX 8.7. Fix: force CUDA 12.6 toolkit in forjar (temporarily disable nvcc 13.0 during build).

Both fixes are encoded in `forjar-candle.yaml` for reproducibility.

### Finding 6: Tool parity confirmed for GGUF (F-TOOLPARITY-01 PARTIAL)

**What:** apr-cli `apr serve run --gpu` produces 139.8 tok/s. realizr `realizr serve --gpu` produces 142.8 tok/s. Delta: 2.1% — within ±5% threshold.

**Why:** Both tools embed the same realizr inference engine. apr-cli adds a thin wrapper for model import/profiling but uses the same GPU kernels and serving stack. The 2.1% delta is within measurement noise.

**Implication:** GGUF tool parity confirmed. APR v2 tool parity **FAIL** — apr-cli 21.9 vs realizr 17.4 tok/s (25.6% delta). Root cause: version skew — apr-cli embeds realizr with FP8 weight cache (1472 MB), which the earlier realizr standalone build lacked.

### Finding 7: 83.8% kernel launch overhead (apr profile, PMAT-341)

**What:** `apr profile --granular --perf-grade` reports 83.8% of decode time is kernel launch overhead. Grade: C. Memory bound (arithmetic intensity 4.0 vs roofline threshold 82.0). Achieved 808 GFLOPS / 202 GB/s vs RTX 4090 peak of 82,580 GFLOPS / 1,008 GB/s.

**Why (five-whys):**
1. Why 83.8% overhead? → Each decode step launches many small kernels (RMSNorm, Q4K GEMV ×3, attention, FFN ×3, output norm).
2. Why not fused? → The Q4K GEMV kernels are fused (dequant+matmul), but RMSNorm, attention, and sampling are separate launches.
3. Why does launch overhead dominate? → At M=1 (single request), each kernel does very little work — the GPU is idle between launches.
4. Why memory bound at M=1? → With 1536-dim hidden state, each GEMV reads ~1 MB of weights for ~3 MFLOP of compute. Arithmetic intensity = 3, well below the roofline crossover at 82.
5. Why does Candle not have this problem? → Candle also launches separate kernels, but its QMatMul is a single fused call per projection, and it avoids the HTTP/tokenization/scheduling overhead.

**Implication:** The fused Q4K kernel saves one memory pass but the launch overhead between kernels is the dominant cost. CUDA graph capture (realizr has this at M=1) should help — investigate whether CUDA graphs are actually active in the benchmarked configuration.

### Finding 8: Cross-reference with qwen-coder-deploy (PMAT-354)

| c | qwen-coder-deploy | candle-vs-apr | Delta | Notes |
|---|-------------------|---------------|-------|-------|
| 1 | 148.6 tok/s | 142.8 tok/s | -3.9% | Same engine, same model, same GPU |
| 4 | 325.2 tok/s | 116.7 tok/s | -64% | No batching (SINGLE-REQUEST mode) |
| 32 | ~1,500 tok/s | 145.7 tok/s | -90% | qwen-coder-deploy used BATCH=32 |

c=1 match (3.9% delta) validates methodology. Scaling gap = server mode, not a regression.

### Finding 9: Fused kernels reduce launches, not total GPU time (F-KERNEL-01 WEAKENED)

**What:** nsys profiles (32 tokens, RTX 4090):
- Candle: 40,513 kernel launches, 106.0ms total GPU time
- realizr: 22,360 kernel launches, 105.1ms total GPU time
- realizr launches 1.8x fewer kernels but total GPU time is identical.

**Why:** Candle's QMatMul does dequant + matmul in two separate kernel launches per projection. realizr fuses them into one Q4K GEMV launch. This halves the launch count. But at M=1, each kernel does so little work that the compute saved by fusion is negligible — the memory traffic (reading 1 MB of weights per projection) dominates regardless of whether it's 1 or 2 launches.

**Implication:** F-KERNEL-01 is **weakened**, not falsified. The fused kernels DO reduce launches (1.8x) as predicted, but the total GPU time benefit is <1% at M=1. The advantage would be more meaningful at higher concurrency where launch overhead becomes a larger fraction of total time, but we couldn't test this (SINGLE-REQUEST mode).

### Finding 10: apr profile disagrees with ncu roofline (F-BRICKPARITY-01 FALSIFIED)

**What:** `apr profile` reports 20% memory efficiency, 1% compute efficiency. `ncu --set roofline` on the dominant Q4K kernel: 55% memory throughput, 29% compute throughput. Delta: 35pp memory, 28pp compute.

**Why (five-whys):**
1. Why disagree? → apr profile measures end-to-end pipeline, ncu measures individual kernels
2. Why does end-to-end differ from per-kernel? → 83.8% of decode time is kernel launch overhead (idle GPU)
3. Why include idle in "efficiency"? → apr profile divides achieved FLOPS by peak, counting gaps
4. Why label it "roofline"? → The UI says "Roofline Analysis" but computes pipeline throughput
5. Root cause: **apr profile conflates pipeline efficiency with per-kernel roofline**

**Implication:** The "Grade C, 20% efficiency" report is misleading. Individual Q4K kernels achieve 55% memory BW — respectable for a memory-bound workload. The actual bottleneck is launch overhead between kernels, not kernel efficiency. Filed paiml/aprender#567.

### Upstream bugs discovered

| Issue | Repo | Status | Contract |
|-------|------|--------|----------|
| paiml/realizar#167 | GPU scheduler hardcodes HF tensor names | Fixed | `tensor-name-resolution-v1` (FALSIFY-TNR-004) |
| paiml/realizar#168 | RMSNorm cache aliasing mismatch | **Fixed** (#170) | `tensor-name-resolution-v1` (FALSIFY-TNR-001) |
| paiml/realizar#169 | SafeTensors GPU inference missing | **Fixed** | `tensor-name-resolution-v1` (format_parity eq) |
| paiml/realizar#170 | 0 contracts on tensor name resolution | **CONTRACT ADDED** | `tensor-name-resolution-v1.yaml` — 3 eq, 4 ob, 4 ft, 2 kani |
| paiml/aprender#567 | apr profile conflates pipeline/kernel roofline | Filed | Needs `PROFILING_ACCURACY_V1` contract |

`pv coverage` (realizr): 12 contracts, 44 equations, 100% obligation coverage.

## Falsification Scorecard

| ID | Prediction | Outcome |
|----|-----------|---------|
| F-SUMMARY-01 | realizr wins >=1 metric at c=1 | **FALSIFIED** |
| F-PARITY-01 | realizr within +/-10% of Candle | **FALSIFIED** (0.63x) |
| F-SCALE-01 | realizr c=32 >=1,280 tok/s | **FALSIFIED** (145.7) |
| F-HW-01 | Variance <5% with locked clocks | **CONFIRMED** (CV <1%) |
| F-MODEL-01 | Candle loads Q4_K_M GGUF | **CONFIRMED** |
| F-COLD-01 | realizr cold-start slower | **CONFIRMED** |
| F-SERVING-01 | Serving overhead <5ms | **WEAKENED** (HTTP 5ms, E2E 27ms) |
| F-FORMAT-01 | APR v2 load 2-5x faster | **FALSIFIED** (60s vs 0.49s — 120x slower) |
| F-RSS-01 | APR v2 RSS < GGUF RSS | **CONFIRMED** (2,278 < 3,082 MB, 26% less) |
| F-KERNEL-01 | Fused Q4K lower mem traffic | **WEAKENED** (1.8x fewer launches, same GPU time) |
| F-FMTPARITY-01 | All 3 formats GPU ±10% | **FALSIFIED** (all GPU now: GGUF 142.8, SafeT 21.2, APR 17.4 — not at parity) |
| F-TOOLPARITY-01 | apr-cli vs realizr ±5% | **WEAKENED** (GGUF 2.1% PASS, APR 25.6% FAIL — version skew) |
| F-BRICKPARITY-01 | apr profile vs ncu ±15% | **FALSIFIED** (20% vs 55% mem, aprender#567) |

**Score: 6 FALSIFIED, 4 CONFIRMED, 3 WEAKENED, 0 PARTIAL, 0 BLOCKED, 0 UNTESTED**
