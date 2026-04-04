#!/bin/bash
# PMAT-441: Bootstrap confidence intervals for decode tok/s
#
# Runs probador N times, collects decode_tok_per_sec from each run,
# computes 95% bootstrap CI. Optionally compares two frameworks
# with Mann-Whitney U test.
#
# Usage:
#   # Single framework CI:
#   bash scripts/bootstrap-ci.sh --url http://127.0.0.1:8081 --runs 30 --name realizr
#
#   # Two-framework comparison:
#   bash scripts/bootstrap-ci.sh \
#     --url-a http://127.0.0.1:8081 --name-a realizr \
#     --url-b http://127.0.0.1:8082 --name-b llama-cpp \
#     --runs 30
#
# Output:
#   results/bootstrap-<name>-<date>.json (per-run tok/s array + CI)
#
# Scientific basis: MLPerf v4.0 requires min sample counts.
# LLMPerf (Anyscale 2024) and Splitwise (Patel 2024) demand CIs.
set -euo pipefail

# PMAT-445 pre-flight: GPU isolation check
# False regression (realizr#190) was caused by stale GPU processes.
# Abort if unexpected compute processes are detected.
gpu_preflight() {
    local procs
    procs=$(nvidia-smi --query-compute-apps=pid,name,used_memory \
        --format=csv,noheader 2>/dev/null || true)
    if [ -n "$procs" ]; then
        local count
        count=$(echo "$procs" | wc -l)
        echo "WARNING: $count GPU compute process(es) detected:"
        echo "$procs" | sed 's/^/  /'
        echo ""
        echo "GPU contention causes false regressions (realizr#190)."
        echo "Kill competing processes or set SKIP_GPU_PREFLIGHT=1 to override."
        if [ "${SKIP_GPU_PREFLIGHT:-}" != "1" ]; then
            exit 1
        fi
        echo "SKIP_GPU_PREFLIGHT=1 — continuing anyway."
    fi
}
gpu_preflight

# Defaults
RUNS=30
DURATION=30
WARMUP=5
MAX_TOKENS=256
URL_A=""
URL_B=""
NAME_A="framework-a"
NAME_B="framework-b"
CLOCK_MHZ=2520

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --url|--url-a) URL_A="$2"; shift 2 ;;
        --url-b) URL_B="$2"; shift 2 ;;
        --name|--name-a) NAME_A="$2"; shift 2 ;;
        --name-b) NAME_B="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --clock) CLOCK_MHZ="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$URL_A" ]; then
    echo "Usage: $0 --url <URL> --runs <N> --name <NAME>"
    exit 1
fi

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"
DATE=$(date +%Y%m%d-%H%M%S)

run_benchmark() {
    local url="$1"
    local name="$2"
    local outfile="$RESULTS_DIR/bootstrap-${name}-${DATE}.json"
    local tmpdir=$(mktemp -d)

    echo "=== $name: $RUNS runs × ${DURATION}s ==="
    echo "URL: $url"

    # Collect tok/s from each run
    local tok_array="["
    for i in $(seq 1 "$RUNS"); do
        local run_out="$tmpdir/run-$i.json"
        probador llm load \
            --url "$url" \
            --concurrency 1 \
            --duration "${DURATION}s" \
            --warmup "${WARMUP}s" \
            --max-tokens "$MAX_TOKENS" \
            --stream false \
            --num-layers 28 \
            --gpu-telemetry \
            --expected-clock-mhz "$CLOCK_MHZ" \
            --runtime-name "$name" \
            -o "$run_out" 2>/dev/null

        local tok_s=$(python3 -c "
import json
d = json.load(open('$run_out'))
print(d.get('decode_tok_per_sec', d.get('tokens_per_sec', 0)))
")
        printf "  Run %2d/%d: %.1f tok/s\n" "$i" "$RUNS" "$tok_s"

        if [ "$i" -gt 1 ]; then tok_array="$tok_array,"; fi
        tok_array="$tok_array$tok_s"
    done
    tok_array="$tok_array]"

    # Compute bootstrap CI
    python3 -c "
import json, random, sys

samples = $tok_array
n = len(samples)
mean = sum(samples) / n
std = (sum((x - mean)**2 for x in samples) / (n - 1)) ** 0.5
cv = std / mean * 100 if mean > 0 else 0

# Bootstrap: 10,000 resamples
random.seed(42)
B = 10000
boot_means = []
for _ in range(B):
    resample = [random.choice(samples) for _ in range(n)]
    boot_means.append(sum(resample) / n)
boot_means.sort()

ci_lo = boot_means[int(B * 0.025)]
ci_hi = boot_means[int(B * 0.975)]

result = {
    'name': '$name',
    'url': '$url',
    'runs': n,
    'duration_per_run': $DURATION,
    'samples': samples,
    'mean': round(mean, 2),
    'std': round(std, 2),
    'cv_pct': round(cv, 2),
    'ci_95_lo': round(ci_lo, 2),
    'ci_95_hi': round(ci_hi, 2),
    'median': round(sorted(samples)[n // 2], 2),
    'min': round(min(samples), 2),
    'max': round(max(samples), 2),
}
json.dump(result, open('$outfile', 'w'), indent=2)

print(f'  Mean: {mean:.1f} tok/s')
print(f'  Std:  {std:.1f} tok/s (CV {cv:.1f}%)')
print(f'  95% CI: [{ci_lo:.1f}, {ci_hi:.1f}]')
print(f'  Range: [{min(samples):.1f}, {max(samples):.1f}]')
print(f'  Saved: $outfile')
"
    rm -rf "$tmpdir"
    echo ""
}

compare_frameworks() {
    local file_a="$RESULTS_DIR/bootstrap-${NAME_A}-${DATE}.json"
    local file_b="$RESULTS_DIR/bootstrap-${NAME_B}-${DATE}.json"

    echo "=== Mann-Whitney U Test: $NAME_A vs $NAME_B ==="
    python3 -c "
import json

a = json.load(open('$file_a'))
b = json.load(open('$file_b'))
sa, sb = a['samples'], b['samples']

# Mann-Whitney U (no scipy dependency)
na, nb = len(sa), len(sb)
combined = [(v, 'a') for v in sa] + [(v, 'b') for v in sb]
combined.sort(key=lambda x: x[0])

rank_sum_a = 0
for i, (val, group) in enumerate(combined, 1):
    if group == 'a':
        rank_sum_a += i

U_a = rank_sum_a - na * (na + 1) / 2
U_b = na * nb - U_a
U = min(U_a, U_b)

# Normal approximation for p-value
mu = na * nb / 2
sigma = (na * nb * (na + nb + 1) / 12) ** 0.5
z = (U - mu) / sigma if sigma > 0 else 0

# Two-tailed p-value approximation
import math
p_approx = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / 2**0.5)))

ratio = a['mean'] / b['mean'] if b['mean'] > 0 else float('inf')
sig = 'YES (p < 0.05)' if p_approx < 0.05 else 'NO (p >= 0.05)'

print(f'  {a[\"name\"]}: {a[\"mean\"]:.1f} [{a[\"ci_95_lo\"]:.1f}, {a[\"ci_95_hi\"]:.1f}]')
print(f'  {b[\"name\"]}: {b[\"mean\"]:.1f} [{b[\"ci_95_lo\"]:.1f}, {b[\"ci_95_hi\"]:.1f}]')
print(f'  Ratio: {ratio:.3f}x')
print(f'  U = {U:.0f}, z = {z:.2f}, p ≈ {p_approx:.4f}')
print(f'  Significant: {sig}')

# CIs overlap?
overlap = a['ci_95_lo'] <= b['ci_95_hi'] and b['ci_95_lo'] <= a['ci_95_hi']
print(f'  CIs overlap: {\"YES\" if overlap else \"NO\"} (non-overlap = strong evidence)')
"
}

echo "PMAT-441: Bootstrap Confidence Intervals"
echo "Date: $(date -Iseconds)"
echo ""

run_benchmark "$URL_A" "$NAME_A"

if [ -n "$URL_B" ]; then
    run_benchmark "$URL_B" "$NAME_B"
    compare_frameworks
fi
