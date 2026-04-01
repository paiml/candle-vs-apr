#!/usr/bin/env bash
# bench-candle.sh — Benchmark Candle quantized-qwen2-instruct decode speed
#
# Runs N iterations of the same prompt through Candle CLI, captures:
#   - tokens/sec (from Candle's own output)
#   - wall time
#   - peak RSS (via /usr/bin/time)
#
# Usage: ./scripts/bench-candle.sh [--iterations 10] [--model /path/to/gguf]

set -euo pipefail

ITERATIONS="${ITERATIONS:-10}"
MODEL="${MODEL:-/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
CANDLE_BIN="${CANDLE_BIN:-/home/noah/src/candle/target/release/examples/quantized-qwen2-instruct}"
MAX_TOKENS="${MAX_TOKENS:-256}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PROMPT="Write a Rust function that computes the nth Fibonacci number using matrix exponentiation for O(log n) time complexity. Include proper error handling for overflow cases."

mkdir -p "$RESULTS_DIR"

echo "=== Candle Benchmark ==="
echo "Model:      $MODEL"
echo "Binary:     $CANDLE_BIN"
echo "Iterations: $ITERATIONS"
echo "Max tokens: $MAX_TOKENS"
echo ""

if [ ! -x "$CANDLE_BIN" ]; then
    echo "ERROR: Candle binary not found at $CANDLE_BIN"
    echo "Build with: cd ~/src/candle && cargo build --release --features cuda --example quantized-qwen2-instruct"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    exit 1
fi

RESULTS_FILE="$RESULTS_DIR/candle-$(date +%Y%m%d-%H%M%S).jsonl"
SUMMARY_FILE="$RESULTS_DIR/candle-summary-$(date +%Y%m%d-%H%M%S).json"

declare -a TOK_SEC_VALUES
declare -a WALL_TIMES
declare -a RSS_VALUES

for i in $(seq 1 "$ITERATIONS"); do
    echo "--- Iteration $i/$ITERATIONS ---"

    # Use /usr/bin/time for RSS, capture stderr for Candle's tok/s output
    START=$(date +%s%N)

    OUTPUT=$( /usr/bin/time -v "$CANDLE_BIN" \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --sample-len "$MAX_TOKENS" \
        2>&1 ) || true

    END=$(date +%s%N)
    WALL_MS=$(( (END - START) / 1000000 ))

    # Extract tokens/sec from Candle output (format: "N tokens generated (X.XX token/s)")
    TOK_SEC=$(echo "$OUTPUT" | grep -oP '[\d.]+\s+token/s' | grep -oP '[\d.]+' | head -1 || echo "0")

    # Extract peak RSS from /usr/bin/time output (in KB)
    RSS_KB=$(echo "$OUTPUT" | grep -oP 'Maximum resident set size.*?:\s*\K\d+' || echo "0")

    echo "  tok/s: $TOK_SEC | wall: ${WALL_MS}ms | RSS: ${RSS_KB}KB"

    # Write JSONL record
    echo "{\"iteration\": $i, \"tok_sec\": $TOK_SEC, \"wall_ms\": $WALL_MS, \"rss_kb\": $RSS_KB, \"max_tokens\": $MAX_TOKENS}" >> "$RESULTS_FILE"

    TOK_SEC_VALUES+=("$TOK_SEC")
    WALL_TIMES+=("$WALL_MS")
    RSS_VALUES+=("$RSS_KB")
done

# Compute summary stats
python3 -c "
import json, sys

tok_sec = [${TOK_SEC_VALUES[*]}]
wall_ms = [${WALL_TIMES[*]}]
rss_kb = [${RSS_VALUES[*]}]

# Drop first iteration (cold start) for tok/s stats
warm_tok = tok_sec[1:] if len(tok_sec) > 1 else tok_sec

summary = {
    'runtime': 'candle',
    'model': '$MODEL',
    'max_tokens': $MAX_TOKENS,
    'iterations': $ITERATIONS,
    'tok_sec': {
        'mean': sum(warm_tok) / len(warm_tok),
        'min': min(warm_tok),
        'max': max(warm_tok),
        'cold_start': tok_sec[0] if tok_sec else 0,
    },
    'wall_ms': {
        'mean': sum(wall_ms) / len(wall_ms),
        'min': min(wall_ms),
        'max': max(wall_ms),
    },
    'rss_kb': {
        'mean': sum(rss_kb) / len(rss_kb),
        'min': min(rss_kb),
        'max': max(rss_kb),
    },
}
print(json.dumps(summary, indent=2))
with open('$SUMMARY_FILE', 'w') as f:
    json.dump(summary, f, indent=2)
"

echo ""
echo "Results: $RESULTS_FILE"
echo "Summary: $SUMMARY_FILE"
