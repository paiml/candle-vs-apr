#!/bin/bash
# PMAT-445: Multi-Framework Inference Showdown
#
# Runs probador against each framework in configs/showdown.yaml.
# All use identical benchmark params for apples-to-apples comparison.
#
# Usage:
#   bash scripts/run-showdown.sh                    # All 4 frameworks
#   bash scripts/run-showdown.sh realizr llama-cpp  # Specific frameworks
#
# Prerequisites:
#   - Frameworks built and models available
#   - GPU clocks locked: sudo nvidia-smi -lgc <MHz>
#   - No competing GPU processes
set -euo pipefail

# PMAT-445 pre-flight: GPU isolation check (realizr#190 lesson)
gpu_preflight() {
    local procs
    procs=$(nvidia-smi --query-compute-apps=pid,name,used_memory \
        --format=csv,noheader 2>/dev/null || true)
    if [ -n "$procs" ]; then
        echo "ERROR: GPU compute processes detected before showdown:"
        echo "$procs" | sed 's/^/  /'
        echo "Kill all GPU processes first. GPU contention = false data."
        if [ "${SKIP_GPU_PREFLIGHT:-}" != "1" ]; then
            exit 1
        fi
    fi
}
gpu_preflight

MODEL="/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
RESULTS_DIR="results/showdown"
DATE=$(date +%Y%m%d-%H%M%S)
DURATION=30
WARMUP=5
MAX_TOKENS=256

# Detect clock speed
CLOCK=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "2520")

mkdir -p "$RESULTS_DIR"

# Framework definitions (port, start cmd, stop cmd)
declare -A PORTS=( [realizr]=8081 [llama-cpp]=8082 [vllm]=8083 [ollama]=8084 )

start_framework() {
    local name="$1"
    local port="${PORTS[$name]}"

    echo "Starting $name on port $port..."
    case "$name" in
        realizr)
            apr serve run "$MODEL" --gpu --port "$port" &
            ;;
        llama-cpp)
            /home/noah/src/llama.cpp/build/bin/llama-server \
                --model "$MODEL" --port "$port" \
                --n-gpu-layers 28 --ctx-size 4096 --threads 4 &
            ;;
        vllm)
            python3 -m vllm.entrypoints.openai.api_server \
                --model /home/noah/models/qwen2.5-coder-1.5b-instruct-safetensors/ \
                --port "$port" --gpu-memory-utilization 0.9 \
                --dtype float16 --max-model-len 4096 &
            ;;
        ollama)
            OLLAMA_HOST="127.0.0.1:$port" ollama serve &
            sleep 5
            ;;
        *)
            echo "Unknown framework: $name"
            return 1
            ;;
    esac

    # Wait for health
    local url="http://127.0.0.1:$port"
    for i in $(seq 1 60); do
        if curl -sf "$url/health" >/dev/null 2>&1 || \
           curl -sf "$url/v1/models" >/dev/null 2>&1 || \
           curl -sf "$url/api/tags" >/dev/null 2>&1; then
            echo "  $name healthy on :$port"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: $name failed to start on :$port"
    return 1
}

stop_framework() {
    local name="$1"
    local port="${PORTS[$name]}"
    case "$name" in
        realizr) pkill -f "realizr serve.*$port" 2>/dev/null || true ;;
        llama-cpp) pkill -f "llama-server.*$port" 2>/dev/null || true ;;
        vllm) pkill -f "vllm.entrypoints.*$port" 2>/dev/null || true ;;
        ollama) pkill -f "ollama serve" 2>/dev/null || true ;;
    esac
    sleep 2
}

benchmark_framework() {
    local name="$1"
    local port="${PORTS[$name]}"
    local url="http://127.0.0.1:$port"
    local outfile="$RESULTS_DIR/${name}-c1-${DATE}.json"

    echo "=== Benchmarking $name (c=1, ${DURATION}s) ==="
    probador llm load \
        --url "$url" \
        --concurrency 1 \
        --duration "${DURATION}s" \
        --warmup "${WARMUP}s" \
        --max-tokens "$MAX_TOKENS" \
        --stream false \
        --num-layers 28 \
        --gpu-telemetry \
        --expected-clock-mhz "$CLOCK" \
        --runtime-name "$name" \
        -o "$outfile" 2>/dev/null

    # Extract key metrics
    python3 -c "
import json
d = json.load(open('$outfile'))
tok = d.get('decode_tok_per_sec', d.get('tokens_per_sec', 0))
ttft = d.get('ttft_p50_ms', 0)
itl = d.get('itl_p50_ms', 0)
err = d.get('error_rate', 0)
print(f'  Decode: {tok:.1f} tok/s | TTFT P50: {ttft:.1f}ms | ITL P50: {itl:.2f}ms | Errors: {err:.1%}')
"
    echo ""
}

# Determine which frameworks to run
if [ $# -gt 0 ]; then
    FRAMEWORKS=("$@")
else
    FRAMEWORKS=(realizr llama-cpp vllm ollama)
fi

echo "PMAT-445: Multi-Framework Inference Showdown"
echo "Date: $(date -Iseconds)"
echo "Model: $(basename "$MODEL")"
echo "Clock: ${CLOCK} MHz"
echo "Frameworks: ${FRAMEWORKS[*]}"
echo ""

# Kill everything first
for fw in "${FRAMEWORKS[@]}"; do
    stop_framework "$fw"
done

# Run each framework sequentially (isolation)
for fw in "${FRAMEWORKS[@]}"; do
    if start_framework "$fw"; then
        benchmark_framework "$fw"
        stop_framework "$fw"
    else
        echo "SKIP: $fw (failed to start)"
    fi
done

# Summary table
echo "=== SHOWDOWN RESULTS ==="
printf "%-12s %10s %10s %10s\n" "Framework" "Decode" "TTFT P50" "ITL P50"
printf "%-12s %10s %10s %10s\n" "---------" "------" "--------" "-------"
for fw in "${FRAMEWORKS[@]}"; do
    local_file="$RESULTS_DIR/${fw}-c1-${DATE}.json"
    if [ -f "$local_file" ]; then
        python3 -c "
import json
d = json.load(open('$local_file'))
tok = d.get('decode_tok_per_sec', d.get('tokens_per_sec', 0))
ttft = d.get('ttft_p50_ms', 0)
itl = d.get('itl_p50_ms', 0)
print(f'$fw'.ljust(12) + f'{tok:>10.1f}' + f'{ttft:>10.1f}' + f'{itl:>10.2f}')
" 2>/dev/null || printf "%-12s %10s %10s %10s\n" "$fw" "ERR" "ERR" "ERR"
    fi
done

echo ""
echo "Results in $RESULTS_DIR/"
