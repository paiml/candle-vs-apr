#!/bin/bash
# PMAT-442: VRAM measurement during probador benchmark runs
#
# Polls nvidia-smi memory.used at 100ms intervals during a
# probador run. Captures peak, mean, and timeline.
#
# Usage:
#   bash scripts/measure-vram.sh --url http://127.0.0.1:8081 --name realizr
#
# Output:
#   results/vram-<name>-<date>.json
#   results/vram-<name>-<date>.csv (timeline)
#
# Addresses "nearly universal gap" in LLM inference benchmarks
# (Alizadeh et al. 2024). nvidia-smi polling is crude but
# portable — CUDA memory API would require probador changes.
set -euo pipefail

URL="${URL:-http://127.0.0.1:8081}"
NAME="${NAME:-realizr}"
DURATION=30
WARMUP=5
MAX_TOKENS=256
CLOCK_MHZ=2520
POLL_MS=100

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url) URL="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --clock) CLOCK_MHZ="$2"; shift 2 ;;
        --poll) POLL_MS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"
DATE=$(date +%Y%m%d-%H%M%S)
VRAM_CSV="$RESULTS_DIR/vram-${NAME}-${DATE}.csv"
VRAM_JSON="$RESULTS_DIR/vram-${NAME}-${DATE}.json"
PROBADOR_OUT="$RESULTS_DIR/vram-probador-${NAME}-${DATE}.json"

echo "PMAT-442: VRAM Measurement"
echo "Name: $NAME"
echo "URL: $URL"
echo "Poll interval: ${POLL_MS}ms"
echo ""

# Pre-benchmark VRAM snapshot
PRE_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
echo "Pre-benchmark VRAM: ${PRE_VRAM} MiB"

# Start nvidia-smi polling in background
echo "timestamp_ms,vram_mib" > "$VRAM_CSV"
POLL_INTERVAL=$(python3 -c "print($POLL_MS / 1000)")
START_TS=$(date +%s%3N)

(
    while true; do
        VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        NOW=$(date +%s%3N)
        ELAPSED=$((NOW - START_TS))
        echo "${ELAPSED},${VRAM}" >> "$VRAM_CSV"
        sleep "$POLL_INTERVAL"
    done
) &
POLL_PID=$!

# Run probador benchmark
echo "Running probador (${DURATION}s + ${WARMUP}s warmup)..."
probador llm load \
    --url "$URL" \
    --concurrency 1 \
    --duration "${DURATION}s" \
    --warmup "${WARMUP}s" \
    --max-tokens "$MAX_TOKENS" \
    --stream false \
    --num-layers 28 \
    --gpu-telemetry \
    --expected-clock-mhz "$CLOCK_MHZ" \
    --runtime-name "$NAME" \
    -o "$PROBADOR_OUT" 2>/dev/null

# Stop polling
kill "$POLL_PID" 2>/dev/null || true
wait "$POLL_PID" 2>/dev/null || true

# Post-benchmark VRAM
POST_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')

# Analyze
python3 -c "
import json, csv

# Read VRAM timeline
samples = []
with open('$VRAM_CSV') as f:
    reader = csv.DictReader(f)
    for row in reader:
        samples.append(int(row['vram_mib']))

# Read probador results
probador = json.load(open('$PROBADOR_OUT'))
tok_s = probador.get('decode_tok_per_sec', probador.get('tokens_per_sec', 0))

peak = max(samples) if samples else 0
mean = sum(samples) / len(samples) if samples else 0
min_v = min(samples) if samples else 0

result = {
    'name': '$NAME',
    'url': '$URL',
    'pre_vram_mib': $PRE_VRAM,
    'post_vram_mib': $POST_VRAM,
    'peak_vram_mib': peak,
    'mean_vram_mib': round(mean, 1),
    'min_vram_mib': min_v,
    'samples': len(samples),
    'poll_ms': $POLL_MS,
    'decode_tok_s': round(tok_s, 1),
    'vram_timeline_csv': '$VRAM_CSV',
}
json.dump(result, open('$VRAM_JSON', 'w'), indent=2)

print(f'  Decode: {tok_s:.1f} tok/s')
print(f'  VRAM peak:  {peak} MiB')
print(f'  VRAM mean:  {mean:.0f} MiB')
print(f'  VRAM range: [{min_v}, {peak}] MiB')
print(f'  Pre/Post:   {int(\"$PRE_VRAM\")} / {int(\"$POST_VRAM\")} MiB')
print(f'  Samples:    {len(samples)} ({$POLL_MS}ms interval)')
print(f'  Saved:      $VRAM_JSON')
"
