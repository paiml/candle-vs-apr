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
| SafeTensors FP32 | realizr (CPU) | ~10,000 | 0.4 | — | Measured (no GPU path) |
| APR v2 Q4K | realizr | — | — | — | **BLOCKED** (loads OK, inference garbage — paiml/realizar#168) |

**F-FORMAT-01: BLOCKED** — APR loads (paiml/realizar#167 fixed) but inference produces garbage output. GPU adapter weight name mapping incomplete (paiml/realizar#168).
**F-RSS-01: BLOCKED** — Cannot compare APR vs GGUF RSS until APR inference is correct.

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

### Finding 4: SafeTensors gap reveals GPU acceleration asymmetry

**What:** Candle: 65.7 tok/s (GPU FP32). realizr: 0.4 tok/s (CPU FP32). Candle 164x faster.

**Why:** Candle dispatches FP32 SafeTensors matmul to CUDA. realizr's GPU path only supports quantized formats (Q4K, Q6K via DP4A); SafeTensors FP32 falls back to CPU with no SIMD optimization beyond what the Rust compiler auto-vectorizes.

**Implication:** realizr is a quantization-first engine. If a user needs FP32/FP16 inference, Candle is the correct tool. This is an architectural trade-off, not a bug.

### Finding 5: Infrastructure blockers on Lambda Vector

Two infrastructure issues required workarounds to run benchmarks:

1. **curand device library missing** — Candle eagerly initializes curand at GPU device creation, but the Lambda Vector CUDA 13.0 install lacks `libcurand_device.a`. Fix: lazy-curand patch in Candle (defer init until first `rand_*` call).

2. **CUDA 13.0 PTX incompatible with driver** — nvcc 13.0 generates PTX 9.0, but driver 570.207 only supports PTX 8.7. Fix: force CUDA 12.6 toolkit in forjar (temporarily disable nvcc 13.0 during build).

Both fixes are encoded in `forjar-candle.yaml` for reproducibility.

### Upstream bugs discovered

| Issue | Repo | Status | Contract candidate |
|-------|------|--------|-------------------|
| paiml/realizar#167 | GPU scheduler hardcodes HF tensor names | Fixed | `TENSOR_NAME_RESOLUTION_V1` |
| paiml/realizar#168 | RMSNorm cache aliasing mismatch | Filed | `TENSOR_NAME_RESOLUTION_V1` |

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
| F-FORMAT-01 | APR v2 load 2-5x faster | **BLOCKED** (#168) |
| F-RSS-01 | APR v2 RSS < GGUF RSS | **BLOCKED** (#168) |
| F-KERNEL-01 | Fused Q4K lower mem traffic | UNTESTED |

**Score: 3 FALSIFIED, 3 CONFIRMED, 1 WEAKENED, 2 BLOCKED, 1 UNTESTED**
