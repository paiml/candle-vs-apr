# Candle vs realizr — Inference Performance

## Methodology

Same production methodology as qwen-coder-deploy (PMAT-177):
- Model: Qwen2.5-Coder-1.5B-Instruct Q4_K_M GGUF
- Hardware: RTX 4060 Laptop GPU, 1900 MHz locked
- Duration: 60s runs, 5s warmup
- Isolation: forjar serial deployment (one runtime at a time)
- Output: uniform(16, 256) tokens, streaming

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

*To be filled after benchmarks run.*

### Phase 1: Single-Request Decode (c=1)

| Metric | Candle | realizr | Ratio | P1 Status |
|--------|--------|---------|-------|-----------|
| Decode (tok/s, warm) | — | — | — | — |
| Decode (tok/s, cold) | — | — | — | — |
| TTFT (ms) | — | — | — | — |
| Wall time (ms, mean) | — | — | — | — |
| Peak RSS (MB) | — | — | — | — |

### Phase 2: realizr Scaling (Candle: N/A)

| c | realizr agg tok/s | realizr dec tok/s | vs Candle c=1 |
|---|-------------------|-------------------|---------------|
| 1 | — | — | — |
| 4 | — | — | — |
| 8 | — | — | — |
| 16 | — | — | — |
| 32 | — | — | — |

### Phase 3: Format Comparison

| Format | Runtime | Load (ms) | Decode (tok/s) | RSS (MB) |
|--------|---------|-----------|----------------|----------|
| GGUF Q4_K_M | Candle | — | — | — |
| GGUF Q4_K_M | realizr | — | — | — |
| SafeTensors | Candle | — | — | — |
| SafeTensors | realizr | — | — | — |
| APR v2 Q4K | realizr | — | — | — |

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
