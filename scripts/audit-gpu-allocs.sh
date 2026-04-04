#!/bin/bash
# PMAT-432: Audit GPU allocations for realizr serving
# Profiles VRAM allocation categories to identify reduction targets.
#
# Usage: bash scripts/audit-gpu-allocs.sh [--port 8090] [--model /path/to/model.gguf]
#
# Outputs:
#   - Host RSS breakdown (smaps categories)
#   - GPU VRAM per-process
#   - CUDA memory allocation summary
#   - Delta with/without --no-fp8-cache
#
# Run on machine with GPU (Yoga or Lambda).
set -euo pipefail

PORT="${1:-8090}"
MODEL="${2:-$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
URL="http://127.0.0.1:$PORT"

cleanup() {
    pkill -f "realizr serve.*$PORT" 2>/dev/null || true
    pkill -f "apr serve.*$PORT" 2>/dev/null || true
}
trap cleanup EXIT

audit_process() {
    local label="$1"
    local pid="$2"

    echo "=== $label (PID $pid) ==="

    # Host RSS from /proc
    local rss_kb=$(awk '/^VmRSS:/ {print $2}' /proc/$pid/status 2>/dev/null || echo "?")
    local rss_mb=$((rss_kb / 1024))
    echo "Host RSS: ${rss_mb} MB"

    # smaps breakdown: Rss by mapping type
    echo "--- smaps breakdown ---"
    if [ -r /proc/$pid/smaps ]; then
        awk '
        /^[0-9a-f]/ { mapping = $0 }
        /^Rss:/ {
            rss = $2;
            if (mapping ~ /\.so/) so += rss;
            else if (mapping ~ /heap/) heap += rss;
            else if (mapping ~ /stack/) stack += rss;
            else if (mapping ~ /nvidia|cuda/) gpu += rss;
            else if (mapping ~ /\.gguf|\.apr|model/) model += rss;
            else anon += rss;
            total += rss;
        }
        END {
            printf "  .so libs:    %6d MB\n", so/1024;
            printf "  heap:        %6d MB\n", heap/1024;
            printf "  nvidia/cuda: %6d MB\n", gpu/1024;
            printf "  model files: %6d MB\n", model/1024;
            printf "  anon/other:  %6d MB\n", anon/1024;
            printf "  stack:       %6d MB\n", stack/1024;
            printf "  TOTAL:       %6d MB\n", total/1024;
        }' /proc/$pid/smaps
    else
        echo "  (smaps not readable — try with sudo)"
    fi

    # GPU VRAM via nvidia-smi
    echo "--- GPU VRAM ---"
    nvidia-smi --query-compute-apps=pid,used_memory \
        --format=csv,noheader 2>/dev/null | grep "^$pid" || echo "  (no GPU allocation found)"

    # CUDA memory info
    echo "--- CUDA memory summary ---"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
        --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

    echo ""
}

start_server() {
    local label="$1"
    shift
    local flags="$@"

    cleanup
    sleep 1

    echo "Starting server: apr serve run $MODEL --gpu --port $PORT $flags"
    apr serve run "$MODEL" --gpu --port "$PORT" $flags &
    local pid=$!

    # Wait for health
    for i in $(seq 1 30); do
        if curl -s "$URL/health" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ! curl -s "$URL/health" >/dev/null 2>&1; then
        echo "ERROR: Server failed to start for $label"
        return 1
    fi

    # Warm up with one request
    curl -s "$URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen","messages":[{"role":"user","content":"hello"}],"max_tokens":16,"temperature":0}' \
        >/dev/null 2>&1

    sleep 2
    audit_process "$label" "$pid"
    cleanup
    sleep 2
}

echo "PMAT-432: GPU Allocation Audit"
echo "Model: $MODEL"
echo "Date: $(date -Iseconds)"
echo "Host: $(hostname)"
echo ""

# Lock GPU clocks if possible
sudo nvidia-smi -lgc 1900 2>/dev/null || sudo nvidia-smi -lgc 2520 2>/dev/null || true

# Baseline
start_server "baseline (default)" ""

# No FP8 cache
start_server "no-fp8-cache" "--no-fp8-cache"

# Short context + no FP8
start_server "minimal (ctx512 + no-fp8)" "--context-length 512 --no-fp8-cache"

echo "=== AUDIT COMPLETE ==="
echo "Next: compare output to identify largest reducible allocation category."
