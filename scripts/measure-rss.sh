#!/bin/bash
# PMAT-438: Measure RSS with --context-length and --no-fp8-cache
# Run on Yoga RTX 4060 with Qwen2.5-Coder-1.5B Q4_K_M
set -euo pipefail

MODEL="$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
PORT=8090
URL="http://127.0.0.1:$PORT"

measure_rss() {
    local label="$1"
    shift
    local flags="$@"

    echo "=== $label ==="
    echo "Flags: $flags"

    # Kill any existing realizr/apr processes
    pkill -f "realizr serve" 2>/dev/null || true
    pkill -f "apr serve" 2>/dev/null || true
    sleep 2

    # Lock GPU clocks
    sudo nvidia-smi -lgc 1900 2>/dev/null || true

    # Start server in background, capture PID
    apr serve run "$MODEL" --gpu --port "$PORT" $flags &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"

    # Wait for server to be ready
    for i in $(seq 1 30); do
        if curl -s "$URL/health" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    # Measure RSS before any requests
    RSS_COLD=$(ps -o rss= -p $SERVER_PID 2>/dev/null | tr -d ' ')
    echo "RSS cold (KB): $RSS_COLD"
    echo "RSS cold (MB): $((RSS_COLD / 1024))"

    # GPU VRAM
    VRAM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep $SERVER_PID | awk -F',' '{print $2}' | tr -d ' ')
    echo "GPU VRAM: ${VRAM:-unknown}"

    # Run 1 warmup request
    curl -s "$URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen","messages":[{"role":"user","content":"hello"}],"max_tokens":16,"temperature":0}' \
        >/dev/null 2>&1

    # Measure RSS after warmup
    RSS_WARM=$(ps -o rss= -p $SERVER_PID 2>/dev/null | tr -d ' ')
    echo "RSS warm (KB): $RSS_WARM"
    echo "RSS warm (MB): $((RSS_WARM / 1024))"

    # Run probador benchmark (c=1, 30s)
    echo "Running probador..."
    probador llm load --url "$URL" \
        --concurrency 1 --duration 30s --warmup 5s \
        --max-tokens 256 --stream false --num-layers 28 \
        --gpu-telemetry --expected-clock-mhz 1900 \
        --runtime-name "apr-$label" \
        -o "/tmp/pmat-438-$label.json" 2>&1 | grep -E "decode|tok/s|TTFT|ITL|grade"

    # Final RSS
    RSS_FINAL=$(ps -o rss= -p $SERVER_PID 2>/dev/null | tr -d ' ')
    echo "RSS final (MB): $((RSS_FINAL / 1024))"

    # GPU VRAM final
    VRAM_FINAL=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep $SERVER_PID | awk -F',' '{print $2}' | tr -d ' ')
    echo "GPU VRAM final: ${VRAM_FINAL:-unknown}"

    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo ""
}

echo "PMAT-438: RSS Measurement Suite"
echo "Model: $MODEL"
echo "Hardware: Yoga RTX 4060 Laptop (8 GB VRAM)"
echo "Date: $(date -Iseconds)"
echo ""

# Baseline: current defaults (max_seq_len=4096, FP8 enabled)
measure_rss "baseline" ""

# No FP8 cache (saves ~1.5 GB)
measure_rss "no-fp8" "--no-fp8-cache"

# Short context (saves KV)
measure_rss "ctx512" "--context-length 512"

# Both optimizations
measure_rss "minimal" "--context-length 512 --no-fp8-cache"

echo "=== SUMMARY ==="
echo "Results in /tmp/pmat-438-*.json"
