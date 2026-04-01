#!/usr/bin/env bash
# bench-realizr.sh — Benchmark realizr decode speed via OpenAI-compatible API
#
# Runs N iterations of the same prompt through realizr's /v1/chat/completions endpoint.
# For single-request (c=1), measures raw decode speed comparable to Candle CLI.
# For concurrent (c>1), uses probador for proper load testing.
#
# Usage: ./scripts/bench-realizr.sh [--iterations 10] [--concurrency 1]

set -euo pipefail

ITERATIONS="${ITERATIONS:-10}"
CONCURRENCY="${CONCURRENCY:-1}"
REALIZR_URL="${REALIZR_URL:-http://127.0.0.1:8081}"
MAX_TOKENS="${MAX_TOKENS:-256}"
RESULTS_DIR="${RESULTS_DIR:-results}"
DURATION="${DURATION:-60}"
PROMPT="Write a Rust function that computes the nth Fibonacci number using matrix exponentiation for O(log n) time complexity. Include proper error handling for overflow cases."

mkdir -p "$RESULTS_DIR"

echo "=== realizr Benchmark ==="
echo "URL:         $REALIZR_URL"
echo "Concurrency: $CONCURRENCY"
echo "Max tokens:  $MAX_TOKENS"
echo ""

# Health check
if ! curl -sf "$REALIZR_URL/health" > /dev/null 2>&1; then
    echo "ERROR: realizr not responding at $REALIZR_URL/health"
    echo "Start with: forjar apply -f forjar-realizr.yaml"
    exit 1
fi

if [ "$CONCURRENCY" -eq 1 ]; then
    # Single-request mode: iterate like Candle for fair comparison
    RESULTS_FILE="$RESULTS_DIR/realizr-c1-$(date +%Y%m%d-%H%M%S).jsonl"
    SUMMARY_FILE="$RESULTS_DIR/realizr-c1-summary-$(date +%Y%m%d-%H%M%S).json"

    declare -a TOK_SEC_VALUES
    declare -a WALL_TIMES
    declare -a TTFT_VALUES

    for i in $(seq 1 "$ITERATIONS"); do
        echo "--- Iteration $i/$ITERATIONS ---"

        START=$(date +%s%N)

        RESPONSE=$(curl -sf "$REALIZR_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"qwen2.5-coder-1.5b-instruct\",
                \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful coding assistant.\"}, {\"role\": \"user\", \"content\": \"$PROMPT\"}],
                \"max_tokens\": $MAX_TOKENS,
                \"stream\": false
            }" 2>&1)

        END=$(date +%s%N)
        WALL_MS=$(( (END - START) / 1000000 ))

        # Extract usage from OpenAI response
        COMPLETION_TOKENS=$(echo "$RESPONSE" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")
        TOK_SEC=$(python3 -c "print(round($COMPLETION_TOKENS / ($WALL_MS / 1000), 1) if $WALL_MS > 0 else 0)")

        echo "  tok/s: $TOK_SEC | tokens: $COMPLETION_TOKENS | wall: ${WALL_MS}ms"

        echo "{\"iteration\": $i, \"tok_sec\": $TOK_SEC, \"wall_ms\": $WALL_MS, \"completion_tokens\": $COMPLETION_TOKENS, \"max_tokens\": $MAX_TOKENS}" >> "$RESULTS_FILE"

        TOK_SEC_VALUES+=("$TOK_SEC")
        WALL_TIMES+=("$WALL_MS")
    done

    # Summary
    python3 -c "
import json

tok_sec = [${TOK_SEC_VALUES[*]}]
wall_ms = [${WALL_TIMES[*]}]
warm_tok = tok_sec[1:] if len(tok_sec) > 1 else tok_sec

summary = {
    'runtime': 'realizr',
    'concurrency': 1,
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
}
print(json.dumps(summary, indent=2))
with open('$SUMMARY_FILE', 'w') as f:
    json.dump(summary, f, indent=2)
"

    echo ""
    echo "Results: $RESULTS_FILE"
    echo "Summary: $SUMMARY_FILE"

else
    # Concurrent mode: use probador
    echo "Using probador for c=$CONCURRENCY (duration=${DURATION}s)"
    RESULTS_FILE="$RESULTS_DIR/realizr-c${CONCURRENCY}-$(date +%Y%m%d-%H%M%S).json"

    probador llm load \
        --url "$REALIZR_URL" \
        --concurrency "$CONCURRENCY" \
        --duration "$DURATION" \
        --warmup 5 \
        --stream true \
        --max-tokens "$MAX_TOKENS" \
        --output "$RESULTS_FILE" \
        --prompt-profile medium

    echo "Results: $RESULTS_FILE"
fi
