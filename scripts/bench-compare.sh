#!/usr/bin/env bash
# bench-compare.sh — Generate side-by-side comparison from latest results
#
# Usage: ./scripts/bench-compare.sh

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results}"

CANDLE_SUMMARY=$(ls -t "$RESULTS_DIR"/candle-summary-*.json 2>/dev/null | head -1)
REALIZR_SUMMARY=$(ls -t "$RESULTS_DIR"/realizr-c1-summary-*.json 2>/dev/null | head -1)

if [ -z "$CANDLE_SUMMARY" ] || [ -z "$REALIZR_SUMMARY" ]; then
    echo "ERROR: Need both candle and realizr c=1 results."
    echo "  Run: make bench-candle && make bench-realizr-c1"
    exit 1
fi

python3 -c "
import json

with open('$CANDLE_SUMMARY') as f:
    candle = json.load(f)
with open('$REALIZR_SUMMARY') as f:
    realizr = json.load(f)

c_tok = candle['tok_sec']['mean']
r_tok = realizr['tok_sec']['mean']
ratio = r_tok / c_tok if c_tok > 0 else 0

print()
print('=' * 65)
print('  Candle vs realizr — Single-Request Decode (c=1)')
print('=' * 65)
print()
print(f\"{'Metric':<25} {'Candle':>15} {'realizr':>15} {'Ratio':>8}\")
print('-' * 65)
print(f\"{'Decode (tok/s, warm)':<25} {c_tok:>15.1f} {r_tok:>15.1f} {ratio:>7.2f}x\")
print(f\"{'Decode (tok/s, cold)':<25} {candle['tok_sec']['cold_start']:>15.1f} {realizr['tok_sec']['cold_start']:>15.1f}\")
print(f\"{'Wall time (ms, mean)':<25} {candle['wall_ms']['mean']:>15.0f} {realizr['wall_ms']['mean']:>15.0f}\")

if 'rss_kb' in candle:
    c_rss = candle['rss_kb']['mean'] / 1024
    print(f\"{'Peak RSS (MB)':<25} {c_rss:>15.0f} {'N/A':>15}\")

print()
if ratio > 1.0:
    print(f'  realizr is {ratio:.2f}x faster than Candle at c=1')
elif ratio < 1.0:
    print(f'  Candle is {1/ratio:.2f}x faster than realizr at c=1')
else:
    print('  Dead heat at c=1')

# Check for scaling results
import glob
scaling = sorted(glob.glob('$RESULTS_DIR/realizr-c*-summary-*.json'))
scaling = [s for s in scaling if 'c1' not in s]

if scaling:
    print()
    print('=' * 65)
    print('  realizr Scaling (Candle: N/A — no server mode)')
    print('=' * 65)
    print()
    print(f\"{'c':<6} {'Agg tok/s':>12} {'Per-req tok/s':>15} {'vs Candle c=1':>15}\")
    print('-' * 50)
    print(f\"{'1':<6} {r_tok:>12.1f} {r_tok:>15.1f} {ratio:>14.2f}x\")

print()
"
