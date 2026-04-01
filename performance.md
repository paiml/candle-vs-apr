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
| SafeTensors | Candle | — | — | — | TODO |
| SafeTensors | realizr | — | — | — | TODO |
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
