# ============================================================================
# candle-vs-apr — Candle vs realizr inference benchmark
# ============================================================================
# Phase 1 (c=1):  make bench-candle && make bench-realizr-c1 && make compare
# Phase 2 (scale): make bench-realizr-scaling
# Phase 3 (fmt):   make bench-formats
# Full run:        make bench-all
# ============================================================================

.NOTPARALLEL:

DATE := $(shell date +%Y%m%d)

GGUF_MODEL := /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
APR_MODEL  := /home/noah/models/qwen2.5-coder-1.5b-instruct-q4k.apr
ST_MODEL   := /home/noah/models/qwen2.5-coder-1.5b-instruct-safetensors

CANDLE_DIR := /home/noah/src/candle
CANDLE_BIN := $(CANDLE_DIR)/target/release/examples/quantized-qwen2-instruct
REALIZR_BIN := /mnt/nvme-raid0/targets/realizar/release/realizar
REALIZR_URL := http://127.0.0.1:8081

ITERATIONS := 10
MAX_TOKENS := 256
DURATION   := 60

# --- Build ---

.PHONY: build-candle build-realizr build

build-candle:
	forjar apply -f forjar-candle.yaml

build-realizr:
	forjar apply -f forjar-realizr.yaml

build: build-candle

# --- Phase 1: Single-Request Head-to-Head ---

.PHONY: bench-candle bench-realizr-c1 compare

bench-candle: build-candle
	@echo "=== Phase 1: Candle c=1 decode ==="
	ITERATIONS=$(ITERATIONS) MODEL=$(GGUF_MODEL) CANDLE_BIN=$(CANDLE_BIN) \
		MAX_TOKENS=$(MAX_TOKENS) bash scripts/bench-candle.sh

bench-realizr-c1:
	@echo "=== Phase 1: realizr c=1 decode ==="
	@# Ensure realizr is running
	@curl -sf $(REALIZR_URL)/health > /dev/null 2>&1 || \
		(echo "Start realizr first: make deploy-realizr" && exit 1)
	ITERATIONS=$(ITERATIONS) CONCURRENCY=1 REALIZR_URL=$(REALIZR_URL) \
		MAX_TOKENS=$(MAX_TOKENS) bash scripts/bench-realizr.sh

compare:
	@bash scripts/bench-compare.sh

# --- Phase 2: realizr Scaling ---

.PHONY: bench-realizr-scaling

bench-realizr-scaling:
	@echo "=== Phase 2: realizr scaling (Candle: N/A) ==="
	@for c in 1 4 8 16 32; do \
		echo "--- c=$$c ---"; \
		CONCURRENCY=$$c REALIZR_URL=$(REALIZR_URL) DURATION=$(DURATION) \
			MAX_TOKENS=$(MAX_TOKENS) bash scripts/bench-realizr.sh; \
	done

# --- Phase 3: Format Comparison ---

.PHONY: bench-formats

bench-formats:
	@echo "=== Phase 3: Format comparison ==="
	@echo "GGUF (Candle):"
	ITERATIONS=5 MODEL=$(GGUF_MODEL) CANDLE_BIN=$(CANDLE_BIN) \
		MAX_TOKENS=$(MAX_TOKENS) bash scripts/bench-candle.sh
	@echo ""
	@echo "GGUF (realizr): use probador llm load at c=1"
	@echo "SafeTensors (Candle): TODO — requires non-quantized candle-qwen example"
	@echo "APR v2 (realizr): TODO — restart realizr with --model $(APR_MODEL)"

# --- Deploy / Teardown ---

.PHONY: deploy-realizr teardown

deploy-realizr:
	forjar apply -f forjar-realizr.yaml

teardown:
	forjar apply -f forjar-teardown.yaml

# --- Full Benchmark ---

.PHONY: bench-all

bench-all: build-candle deploy-realizr
	@echo "=== Full benchmark: Candle vs realizr ==="
	$(MAKE) bench-candle
	$(MAKE) bench-realizr-c1
	$(MAKE) compare
	$(MAKE) bench-realizr-scaling
	@echo "=== Done. Results in results/ ==="

# --- Utilities ---

.PHONY: clean results

clean:
	rm -rf results/

results:
	@ls -la results/ 2>/dev/null || echo "No results yet. Run: make bench-all"
