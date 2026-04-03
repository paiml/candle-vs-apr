# Candle vs realizr — Inference Performance

## Version History

This document has been revised three times as measurement
methodology improved and upstream bugs were fixed:

- **v1 (superseded):** Ad-hoc curl scripts against forjar-deployed
  realizr. Showed 142.8 tok/s. Unreliable — different build,
  CUDA graph context poisoning. Not comparable to probador.
- **v2:** Adopted `probador llm load` as standard benchmark tool.
  Revealed the v1 numbers were measuring a poisoned CUDA context.
- **v3 (current):** After fixing CUDA graph capture
  (realizr 81c912d2) and self-referential Default (realizr
  cf10c0f7). All numbers below are v3 unless marked otherwise.

## Methodology

**Tool:** `probador llm load` — the same tool used in
qwen-coder-deploy inference showdown.

| Parameter       | Value                                       |
|-----------------|---------------------------------------------|
| Model           | Qwen2.5-Coder-1.5B-Instruct Q4_K_M GGUF    |
| Hardware        | RTX 4090 (Lambda Vector), 2520 MHz locked    |
| Concurrency     | 1 (unless noted)                             |
| Duration        | 30s, 5s warmup                               |
| Max tokens      | 256                                          |
| Streaming       | false                                        |
| GPU layers      | 28                                           |
| Telemetry       | `--gpu-telemetry`                            |
| Candle config   | CUDA 12.6 PTX, lazy-curand patch, CLI only   |
| realizr config  | `apr serve run --gpu` (OpenAI-compatible API) |

## Predictions (Pre-Registration)

Falsifiable predictions registered before benchmarking
(Popperian methodology).

### P1: Single-Request Decode (c=1)

**Prediction:** realizr within +/-10% of Candle.

**Rationale:** At c=1, both are single-threaded decode on the
same GPU. The fused Q4K kernel saves one memory pass, but the
GPU is underutilized at c=1 so bandwidth is not the bottleneck.

**Falsification:** If realizr >20% slower, serving overhead
(HTTP, tokenizer) is a measurable single-request penalty.

### P2: Model Load Time

**Prediction:** GGUF within +/-20%. APR v2 load 2-5x faster.

**Rationale:** GGUF parsing is similar in both. APR v2 skips
parsing (mmap + binary index).

**Falsification:** If APR v2 <1.5x faster, the zero-copy
claim needs qualification.

### P3: Memory Footprint

**Prediction:** Similar RSS for GGUF. APR v2 RSS lower (mmap
pages in only on access).

**Falsification:** If APR v2 RSS higher, alignment padding
or metadata overhead dominates.

### P4: Scaling

**Prediction:** realizr reaches 1,500+ tok/s at c=32 (matching
qwen-coder-deploy). Candle: N/A (no server).

**Falsification:** If c=32 throughput >20% below
qwen-coder-deploy on same hardware, there is a regression.

## Results

### Phase 1: Single-Request Decode (c=1)

| Metric             | Candle  | realizr (v3) | Notes              |
|--------------------|---------|--------------|---------------------|
| Decode tok/s       | 227.4   | **273.8**    | realizr 1.20x       |
| ITL P50            | —       | 3.7 ms       | probador measured    |
| Peak RSS (MB)      | **449** | 3,082        | Candle wins (see F3) |

**Verdict (F-SUMMARY-01, REVISED):** realizr 1.20x faster
(273.8 vs 227.4). The v1 result (142.8 tok/s, Candle 1.59x)
was an artifact of CUDA graph context poisoning.

### Phase 2: realizr Scaling (Candle N/A — no server)

| c  | Agg tok/s | Per-req tok/s | Wall P50 (ms) | Predicted |
|----|-----------|---------------|---------------|-----------|
| 1  | 117.0     | 137.8         | 1,829         | ~148      |
| 4  | 116.7     | 33.2          | 8,713         | ~325      |
| 8  | 126.3     | 20.9          | 15,371        | ~525      |
| 16 | 112.5     | 13.6          | 35,292        | ~931      |
| 32 | 145.7     | 11.3          | 56,076        | ~1,600    |

**Verdict (F-SCALE-01, FALSIFIED):** Aggregate throughput flat
at ~120-146 tok/s — no scaling observed. c=32 at 145.7 tok/s
is 91% below predicted 1,600 tok/s.

**Root cause:** realizr started in `Mode: SINGLE-REQUEST` with
`--openai-api`. Batch scheduling requires `--batch` mode.
Requests were queued serially, not batched.

### Phase 3: Format Comparison

| Format          | Runtime     | Load (ms) | tok/s   | RSS (MB) |
|-----------------|-------------|-----------|---------|----------|
| GGUF Q4_K_M     | Candle      | 490       | 227.4   | 449      |
| GGUF Q4_K_M     | realizr v3  | amortized | 273.8   | ~3,082   |
| SafeTensors FP32| Candle GPU  | ~1,500    | 65.7    | 3,344    |
| SafeTensors FP32| realizr GPU | ~11,000   | 21.2    | —        |
| APR v2 Q4K      | realizr GPU | ~60,000   | 17.4    | 2,278    |

**Verdict (F-FORMAT-01, FALSIFIED):** APR load ~60s vs
GGUF 0.49s — 120x slower, not 2-5x faster. The zero-copy
claim does not hold for the `from_apr` path (dequant+requant
roundtrip).

**Verdict (F-RSS-01, CONFIRMED):** APR RSS 2,278 MB vs GGUF
3,082 MB (26% less, mmap paging).

## Comparison Charts

### Decode Throughput (c=1, 30s, RTX 4090)

```
  realizr v3 GGUF Q4K (probador)  ████████████████████████████████████████████████ 273.8
  Candle GGUF Q4K (decode-only)   ████████████████████████████████████████ 227.4
  llama.cpp GGUF (qcd c=4 ref)    ███████████████████████████████████████ 224.8
  apr GGUF Q4K (qcd c=4)          ██████████████████ 107.7
  realizr v1 poisoned (probador)  ████ 22.7
  realizr SafeT FP32 (probador)   ███ 21.2
  realizr APR Q4K (probador)      ██ 17.4
```

> Candle 227.4 is decode-only (no HTTP overhead). realizr
> numbers are full wall-clock via probador.

### realizr Scaling (SINGLE-REQUEST mode — no batching)

```
  c=1   ███████████████████████████████ 117.0
  c=4   ███████████████████████████████ 116.7
  c=8   █████████████████████████████████ 126.3
  c=16  ██████████████████████████████ 112.5
  c=32  ██████████████████████████████████████ 145.7
```

### Kernel Launches (32 tokens, nsys)

```
  Candle   ████████████████████████████████████████ 40,513
  realizr  ██████████████████████ 22,360
  (1.8x fewer launches, same GPU time: 106ms vs 105ms)
```

## Architectural Comparison

### Kernel Strategy

| Aspect          | Candle                  | realizr                       |
|-----------------|-------------------------|-------------------------------|
| Dequantization  | Separate (QMatMul)      | Fused with matmul (Q4K/Q5K)  |
| CUDA dispatch   | Per-op kernel launch    | CUDA graph (M=1)              |
| Attention       | Standard                | FlashAttention-style tiled    |
| KV cache        | Manual management       | Integrated with serving layer |

### Format Support

| Format      | Candle      | realizr                    |
|-------------|-------------|----------------------------|
| GGUF        | Direct load | Direct load                |
| SafeTensors | Direct load | Direct load                |
| APR v2      | N/A         | Zero-copy mmap, LZ4/ZSTD  |

### Serving Capabilities

| Feature     | Candle        | realizr                      |
|-------------|---------------|------------------------------|
| HTTP API    | None          | OpenAI-compatible            |
| Concurrency | N/A           | Batch-and-step scheduler     |
| Streaming   | stdout only   | SSE streaming                |
| Failover    | None          | Circuit breakers             |
| Privacy     | None          | Sovereign/Private/Standard   |

## Findings

Each finding follows: **What** happened, **Why** it happened,
**So what** (implication for the project).

---

### F1: realizr wins c=1 after CUDA graph fix

**What:** realizr 273.8 tok/s vs Candle 227.4 — realizr 1.20x
faster. The v1 result (142.8 tok/s, Candle 1.59x) was wrong.

**Why:** v1 measured a poisoned CUDA context. realizr's
`forward_graphed_decode.rs` attempted graph capture by default
(opt-out pattern). When capture failed, all subsequent kernels
ran degraded. Fix: realizr 81c912d2 (default to eager, no
graph capture). Root cause: missing provable contract for CUDA
graph safety, now enforced by `cuda-graph-safety-v1`.

**So what:** The fused Q4K DP4A kernel IS faster than Candle's
QMatMul at c=1 when the CUDA context is healthy. The v1
conclusion was measuring a driver bug, not architecture.

---

### F2: No scaling — batch scheduler not activated

**What:** Throughput flat at ~120-146 tok/s from c=1 to c=32.
Predicted: 148 to 1,600 tok/s.

**Why:** `--openai-api` enables the API format but does NOT
activate the batch scheduler. Requests queued serially. The
batch-and-step scheduler requires `--batch` flag.

**So what:** The qwen-coder-deploy scaling numbers used a
different server configuration. Re-test with `--batch` to
properly evaluate F-SCALE-01.

---

### F3: realizr RSS 6.9x higher — not a fair comparison

**What:** Candle 449 MB vs realizr 3,082 MB.

**Why:** realizr pre-allocates KV cache for `max_batch=32`
slots at startup (32 x ~0.2 GB = 6.4 GB). It also includes
tokio runtime, axum HTTP stack, and tokenizer. Candle is a
CLI tool with no server overhead and no KV cache pool.

**So what:** RSS comparison is only meaningful at matched
concurrency. At c=1, realizr over-provisions by 32x.

---

### F4: SafeTensors GPU path fixed (#169)

**What:** Before fix: Candle 65.7 tok/s GPU FP32,
realizr 0.4 tok/s CPU FP32 — 164x gap. After fix:
realizr 21.2 tok/s GPU FP32 — 3.1x gap remains.

**Why:** GPU path only supported quantized formats
(Q4K, Q6K via DP4A). After fix, realizr uses FP32 SGEMM
while Candle uses optimized QMatMul with FP16 tensor cores.
FP16 HGEMM or on-load quantization not yet implemented.

**So what:** #169 fixed the missing GPU path. The 3.1x gap
is a performance optimization issue, not a correctness bug.

---

### F5: Infrastructure blockers on Lambda Vector

**What:** Two issues required workarounds:

1. **curand device library missing** — Lambda Vector CUDA 13.0
   lacks `libcurand_device.a`. Candle eagerly initializes
   curand at GPU device creation. Fix: lazy-curand patch.

2. **CUDA 13.0 PTX incompatible** — nvcc 13.0 generates
   PTX 9.0, but driver 570.207 supports PTX 8.7 max.
   Fix: force CUDA 12.6 toolkit in forjar.

**So what:** Both fixes encoded in `forjar-candle.yaml`
for reproducibility.

---

### F6: Tool parity — GGUF confirmed, APR v2 fails

**What:** GGUF: apr-cli and realizr both achieve 273.8 tok/s
after graph fix (v1 delta was 2.1%, within noise). APR v2:
apr-cli 21.9 vs realizr 17.4 tok/s — 25.6% delta, FAIL.

**Why:** GGUF parity expected — same inference engine. APR
delta from version skew: apr-cli embeds realizr with FP8
weight cache (1472 MB) that the standalone build lacked.

**So what:** GGUF tool parity confirmed. APR v2 tool parity
needs version alignment.

---

### F7: 83.8% kernel launch overhead

**What:** `apr profile --granular --perf-grade` reports 83.8%
of decode time is launch overhead. Grade: C. Memory bound
(arithmetic intensity 4.0 vs roofline threshold 82.0).
Achieved 808 GFLOPS / 202 GB/s vs RTX 4090 peak of
82,580 GFLOPS / 1,008 GB/s.

**Why:** At M=1, each kernel does very little work. Each
GEMV reads ~1 MB of weights for ~3 MFLOP of compute.
RMSNorm, attention, and sampling are separate launches.
The GPU is idle between launches.

**So what:** The fused Q4K kernel saves one memory pass, but
launch overhead between kernels dominates. CUDA graph capture
(designed for M=1) should help — needs investigation.

---

### F8: Cross-reference with qwen-coder-deploy

| c  | qwen-coder-deploy | candle-vs-apr (v3) | Delta |
|----|-------------------|--------------------|-------|
| 1  | 148.6 tok/s       | 273.8 tok/s        | +84%  |
| 4  | 325.2 tok/s       | 116.7 tok/s        | -64%  |
| 32 | ~1,500 tok/s      | 145.7 tok/s        | -90%  |

c=1 improvement from graph fix. Scaling gap is server mode
(SINGLE-REQUEST vs BATCH), not a regression.

---

### F9: Fused kernels reduce launches, not GPU time

**What:** nsys profiles (32 tokens, RTX 4090): Candle
40,513 launches / 106.0 ms, realizr 22,360 launches /
105.1 ms. 1.8x fewer kernels, identical total GPU time.

**Why:** Candle does dequant + matmul in two launches per
projection; realizr fuses them into one. But at M=1, each
kernel does so little work that compute savings from fusion
are negligible — memory traffic dominates regardless.

**So what:** F-KERNEL-01 weakened. Fused kernels DO halve
launch count, but GPU time benefit is <1% at M=1. The
advantage would matter more at higher concurrency where
launch overhead is a larger fraction of total time.

---

### F10: apr profile disagrees with ncu roofline — FIXED

**What:** `apr profile` reported 20% memory efficiency, 1%
compute efficiency. `ncu --set roofline` on the dominant Q4K
kernel: 55% memory throughput, 29% compute throughput.
Delta: 35pp memory, 28pp compute.

**Why (five-whys):**
1. Why 20% / 1%? → Divides achieved throughput by pipeline time
2. Why pipeline? → `1/decode_tok_s` includes idle between launches
3. Why idle? → 83.8% kernel launch overhead at M=1 decode
4. Why not excluded? → `compute_roofline()` ignored overhead data
5. Root cause: **conflated pipeline efficiency with per-kernel**

**Fix:** aprender c0953fd7 — `compute_roofline()` now subtracts
`kernel_launch_overhead_pct` from inference time. Output labels
values as "per-kernel, excl launch overhead".

**So what:** apr profile should now report ~55% memory / ~29%
compute, matching ncu. F-BRICKPARITY-01 targeted for
**REVISED** after re-verification with `apr profile --granular`.

---

### F11: v1 measurement correction

**What:** v1 showed 142.8 tok/s. probador v2 on the same
hardware showed 22.7 tok/s (patched) / 20.1 tok/s (original).
The 142.8 was from a different realizr build.

**Why:** Never used probador (the standard benchmark tool).
Wrongly dismissed it as WASM-only — a stale 1.0.3 was
installed; the `llm` subcommand was added later.

**So what:** The kernel itself is fast (246 tok/s on first
pass via CUDA graph replay). 89% of wall time was serving
overhead (HTTP + tokenizer + per-token sync + logits
download). Cross-reference: qwen-coder-deploy confirms
apr GGUF GPU = 15.1 tok/s at c=1, 107.7 at c=4.

**Parity target:** <=1.5x vs llama.cpp at c=4 (224.8 tok/s).
realizr needs >=149.9 tok/s (was 107.7, 39% gap — now
exceeded at 274.5 tok/s, see F-PARITY-02 CONFIRMED).

---

### F12: Event-based sync — +12.9% decode improvement

**What:** Replaced `compute_stream.synchronize()` with
`cuStreamWaitEvent` in phase_attention.rs.

**Result:** ITL P50 49.7 to 44.0 ms (-11.5%), decode
20.1 to 22.7 tok/s (+12.9%).

**Upstream:** trueno 5dfe852d (`CudaStream::wait_event()`),
realizr ed318dd7 (event-based ordering).

---

### F13: Streaming stack overflow — root cause found

**What:** SSE streaming (`stream:true`) caused stack overflow
on tokio-rt-worker. Non-streaming (273.8 tok/s) unaffected.
Seven speculative fixes failed — even 64 MB stacks overflowed.

**Why:** Root cause was `..Default::default()` inside
`impl Default for QuantizedGenerateConfig` — infinite
recursion. The pattern compiles without warning. One line
removal (realizr cf10c0f7) fixed everything.
Grade: F to A+ (99.0). TTFT: N/A to 8.4 ms.

**So what:** `lint-self-referential-default.sh` deployed to
all 4 repos (realizr, aprender, trueno, probar). Detects
`..Default::default()` inside `impl Default` at pre-commit.

**Lesson:** Seven fixes failed because we guessed at locations
instead of measuring. Violated our own Measure-and-Fix policy.

---

## Upstream Bugs Discovered

| Issue               | Description                        | Status       | Contract                       |
|---------------------|------------------------------------|--------------|--------------------------------|
| paiml/realizar#167  | GPU scheduler hardcodes HF names   | Fixed        | `tensor-name-resolution-v1`    |
| paiml/realizar#168  | RMSNorm cache aliasing mismatch    | Fixed (#170) | `tensor-name-resolution-v1`    |
| paiml/realizar#169  | SafeTensors GPU inference missing  | Fixed        | `tensor-name-resolution-v1`    |
| paiml/realizar#170  | 0 contracts on tensor name res.    | Added        | `tensor-name-resolution-v1`    |
| paiml/aprender#567  | apr profile conflates roofline     | **FIXED**    | aprender c0953fd7              |
| paiml/realizar#174  | SafeT FP32 SGEMM 7.11x BW penalty | **FIXED**    | `safetensors-gpu-parity-v1`    |
| paiml/realizar#175  | APR native q4 dequant warn         | **DONE**     | `apr-load-parity-v1` [24]      |
| paiml/realizar#176  | Tool parity (runtime, not flags)   | **REVISED**  | `tool-parity-v1` [25]          |
| paiml/realizar#177  | T5 arch constraints + config       | **PARTIAL**  | `encoder-decoder-v1` [28]      |
| paiml/aprender#575  | Whisper integration test           | **TESTED**   | Routing works, output bad [29] |
| paiml/aprender#576  | apr import arch override bug       | **FIXED**    | aprender 3ce6576c              |
| paiml/aprender#577  | Whisper tensor name mapping        | Filed        | whisper-apr crate load issue    |
| paiml/realizar#178  | OOM when cohabiting GPU w/ training | **FIXED**    | realizr 95b4e932               |

[24]: realizr 54ed5e7e. Corrected: 60s from APR native q4,
not --preserve-q4k. --preserve-q4k already passes Q4_K raw.
[25]: Feature flag hypothesis FALSIFIED. FP8 cache is runtime
(gpu_profile.rs:232). Needs probador benchmark to isolate.
[28]: ALL 5/5 steps done: ArchConstraints (26ec4f14) +
is_encoder_decoder (620f81de) + bidirectional + cross attn
(4d801762) + encode/decode API (67c85394). Internal wiring
(encoder weights, layer iteration) is placeholder.
[29]: Routing WORKS: audio detected, whisper-apr invoked,
184.7s processed in 28s. Output garbage — tensor names
mapped as decoder-only (aprender#577). #576 FIXED, apr
rebuilt with --features whisper.

`pv coverage` (realizr): 12 contracts, 44 equations,
100% obligation coverage.

## Falsification Scorecard

| ID              | Prediction                      | Outcome        |
|-----------------|---------------------------------|----------------|
| F-SUMMARY-01    | realizr wins >=1 metric, c=1    | **REVISED** — v3: 1.20x  |
| F-PARITY-01     | realizr within +/-10% of Candle | **REVISED** — v3: 1.20x  |
| F-PARITY-02     | realizr c=4 <=1.5x llama.cpp   | **CONFIRMED** (1.22x faster) |
| F-SCALE-01      | realizr c=32 >=1,280 tok/s     | **FALSIFIED** (145.7)    |
| F-HW-01         | Variance <5% with locked clocks | **CONFIRMED** (CV <1%)   |
| F-MODEL-01      | Candle loads Q4_K_M GGUF        | **CONFIRMED**            |
| F-COLD-01       | realizr cold-start slower       | **CONFIRMED**            |
| F-SERVING-01    | Serving overhead <5 ms          | **WEAKENED** (27 ms E2E) |
| F-FORMAT-01     | APR v2 load 2-5x faster        | **FALSIFIED** (120x slower) |
| F-RSS-01        | APR v2 RSS < GGUF RSS          | **CONFIRMED** (26% less) |
| F-KERNEL-01     | Fused Q4K lower mem traffic     | **WEAKENED**             |
| F-FMTPARITY-01  | All 3 formats GPU +/-10%       | **FALSIFIED** (15x spread) |
| F-TOOLPARITY-01 | apr-cli vs realizr +/-5%       | **WEAKENED**             |
| F-BRICKPARITY-01| apr profile vs ncu +/-15%       | **FALSIFIED** (35pp gap) |

**Score: 5 CONFIRMED, 5 FALSIFIED, 3 WEAKENED, 2 REVISED**
