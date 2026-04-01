#!/usr/bin/env bash
# bench-scaling.sh — Concurrent scaling benchmark for realizr
# Runs N concurrent requests for DURATION seconds, measures aggregate tok/s
#
# Usage: CONCURRENCY=4 DURATION=60 bash scripts/bench-scaling.sh

set -euo pipefail

CONCURRENCY="${CONCURRENCY:-1}"
DURATION="${DURATION:-60}"
REALIZR_URL="${REALIZR_URL:-http://127.0.0.1:8081}"
MAX_TOKENS="${MAX_TOKENS:-256}"
RESULTS_DIR="${RESULTS_DIR:-results}"
WARMUP="${WARMUP:-5}"
PROMPT="Write a Rust function that computes the nth Fibonacci number using matrix exponentiation for O(log n) time complexity. Include proper error handling for overflow cases."

mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/realizr-scaling-c${CONCURRENCY}-$(date +%Y%m%d-%H%M%S).jsonl"

echo "=== realizr Scaling: c=$CONCURRENCY, duration=${DURATION}s ==="

# Health check
if ! curl -sf "$REALIZR_URL/health" > /dev/null 2>&1; then
    echo "ERROR: realizr not responding at $REALIZR_URL/health"
    exit 1
fi

# Worker function: runs requests in a loop until STOP_FILE exists
worker() {
    local id=$1
    local stop_file=$2
    local out_file=$3
    local count=0
    local total_tokens=0

    while [ ! -f "$stop_file" ]; do
        local start=$(date +%s%N)
        local response
        response=$(curl -sf "$REALIZR_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"qwen2.5-coder-1.5b-instruct\",
                \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful coding assistant.\"}, {\"role\": \"user\", \"content\": \"$PROMPT\"}],
                \"max_tokens\": $MAX_TOKENS,
                \"stream\": false
            }" 2>/dev/null) || continue

        local end=$(date +%s%N)
        local wall_ms=$(( (end - start) / 1000000 ))
        local tokens
        tokens=$(echo "$response" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('usage',{}).get('completion_tokens',0))" 2>/dev/null) || tokens=0

        if [ "$tokens" -gt 0 ]; then
            count=$((count + 1))
            total_tokens=$((total_tokens + tokens))
            echo "{\"worker\": $id, \"request\": $count, \"tokens\": $tokens, \"wall_ms\": $wall_ms}" >> "$out_file"
        fi
    done
}

STOP_FILE=$(mktemp)
rm -f "$STOP_FILE"

echo "Warming up (${WARMUP}s)..."
# Single warmup request
curl -sf "$REALIZR_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"qwen2.5-coder-1.5b-instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":1,\"stream\":false}" > /dev/null 2>&1
sleep "$WARMUP"

echo "Running $CONCURRENCY workers for ${DURATION}s..."
START_TIME=$(date +%s)

# Launch workers
for i in $(seq 1 "$CONCURRENCY"); do
    worker "$i" "$STOP_FILE" "$RESULTS_FILE" &
done

# Wait for duration
sleep "$DURATION"
touch "$STOP_FILE"

# Wait for workers to finish current request
sleep 5
wait 2>/dev/null || true
rm -f "$STOP_FILE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Compute summary
python3 -c "
import json, sys

results = []
with open('$RESULTS_FILE') as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

if not results:
    print('ERROR: No results collected')
    sys.exit(1)

total_tokens = sum(r['tokens'] for r in results)
total_requests = len(results)
wall_times = [r['wall_ms'] for r in results]

agg_tok_sec = total_tokens / $ELAPSED
per_req_tok_sec = sum(r['tokens'] / (r['wall_ms'] / 1000) for r in results) / total_requests
p50_wall = sorted(wall_times)[len(wall_times) // 2]

summary = {
    'concurrency': $CONCURRENCY,
    'duration_s': $ELAPSED,
    'total_requests': total_requests,
    'total_tokens': total_tokens,
    'agg_tok_sec': round(agg_tok_sec, 1),
    'per_req_tok_sec': round(per_req_tok_sec, 1),
    'wall_p50_ms': p50_wall,
    'wall_min_ms': min(wall_times),
    'wall_max_ms': max(wall_times),
}

print(json.dumps(summary, indent=2))
summary_file = '$RESULTS_DIR/realizr-scaling-c${CONCURRENCY}-summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary: {summary_file}')
"
