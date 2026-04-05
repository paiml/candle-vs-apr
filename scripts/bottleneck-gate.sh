#!/usr/bin/env bash
# P15-03: Bottleneck Gate — Pre-experiment falsification predictor
#
# Before running an optimization experiment, this gate checks whether
# the proposed change CAN succeed given roofline constraints.
#
# Usage:
#   ./scripts/bottleneck-gate.sh <experiment-type> <target-metric> <target-value>
#
# Examples:
#   ./scripts/bottleneck-gate.sh kernel-fusion attention 12  # Can attention reach 12µs?
#   ./scripts/bottleneck-gate.sh dispatch-opt decode 400     # Can decode reach 400 tok/s?
#   ./scripts/bottleneck-gate.sh memory-opt rss 500          # Can RSS reach 500 MB?
#
# F-GATE-01: If this gate falsifies < 20% of experiments over 10 attempts,
# the bottleneck gate adds overhead without value.
#
# Reference: candle-vs-apr spec v13.x, Phase 15 P15-03

set -euo pipefail

# Hardware constants (RTX 4090)
PEAK_BW_GBS=1008       # Memory bandwidth (GB/s)
PEAK_FLOPS_TFLOPS=83   # FP32 TFLOPS
NUM_SMS=128
WARP_SIZE=32
MAX_WARPS_PER_SM=48
L2_SIZE_MB=72
CLOCK_MHZ=2520

# Model constants (Qwen2.5-Coder-1.5B Q4_K_M)
NUM_LAYERS=28
HIDDEN_DIM=1536
NUM_HEADS=12
NUM_KV_HEADS=2
HEAD_DIM=128
VOCAB_SIZE=151936
MODEL_SIZE_MB=1024    # ~1 GB Q4_K_M

# Current baselines (from spec v14.7 measurements, post-#211 fix)
BASELINE_DECODE_TOKS=378.3       # bootstrap CI [372.4, 382.4], CV 1.1%
BASELINE_DECODE_TOKS_LONG=338.9  # at ~420 ctx (chunk=16)
BASELINE_ATTN_US=18.2            # Per-layer average (chunk=16: 3.09% occupancy)
BASELINE_GEMV_US=4.2             # hw_dp4a_q4k_gemv average
BASELINE_ATTN_OCCUPANCY=3.09     # chunk=16 (was 2.15 with chunk=32)
BASELINE_GEMV_BW_PCT=33.2

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

experiment_type="${1:-}"
target_metric="${2:-}"
target_value="${3:-}"

if [[ -z "$experiment_type" || -z "$target_metric" || -z "$target_value" ]]; then
    echo "Usage: $0 <experiment-type> <target-metric> <target-value>"
    echo ""
    echo "Experiment types: kernel-fusion, dispatch-opt, memory-opt, batching, quantization"
    echo "Target metrics: attention (µs), decode (tok/s), rss (MB), occupancy (%)"
    exit 1
fi

gate_result="PASS"
reasons=()

check_roofline_bound() {
    local metric="$1"
    local target="$2"

    case "$metric" in
        attention)
            # Attention is memory-bound. Minimum time = data_volume / peak_BW
            # Q: num_heads * head_dim * 4B = 12 * 128 * 4 = 6144 B
            # K/V: num_kv_heads * seq_len * head_dim * 4B (variable)
            # At seq_len=256: 2 * 256 * 128 * 4 = 262,144 B per K/V read
            # Minimum: 262144 / (1008e9) ≈ 0.26µs (roofline floor)
            local roofline_floor="0.26"
            if (( $(echo "$target < $roofline_floor" | bc -l) )); then
                gate_result="FALSIFIED"
                reasons+=("Target ${target}µs below roofline floor ${roofline_floor}µs (memory BW limit)")
            fi

            # Occupancy check: 12 heads on 128 SMs = max ~9.4% occupancy with 1 warp/head
            # Even multi-warp (4 warps/head): 12 blocks * 4 warps / (128 * 48) ≈ 0.8%
            if (( $(echo "$target < 1.0" | bc -l) )); then
                gate_result="FALSIFIED"
                reasons+=("Cannot achieve <1µs attention with 12 heads on 128 SMs (occupancy limit)")
            fi

            # Current: 2.9µs per chunk * N chunks. Multi-warp helps within-block only.
            if (( $(echo "$target < 2.0" | bc -l) )); then
                reasons+=("WARNING: Below 2µs requires fundamental algorithm change (persistent kernel or TC)")
            fi
            ;;

        decode)
            # Decode throughput bound by weight loading (memory BW)
            # Model weights: ~1 GB Q4_K_M. Must load all for each token.
            # Max tok/s = peak_BW / model_size = 1008 GB/s / 1 GB ≈ 1008 tok/s
            local bw_ceiling="1008"
            if (( $(echo "$target > $bw_ceiling" | bc -l) )); then
                gate_result="FALSIFIED"
                reasons+=("Target ${target} tok/s exceeds BW ceiling ${bw_ceiling} tok/s (must load 1GB weights/token)")
            fi

            # Practical ceiling with overhead: ~70% of BW ceiling
            local practical_ceiling=$(echo "$bw_ceiling * 0.70" | bc -l | cut -d. -f1)
            if (( $(echo "$target > $practical_ceiling" | bc -l) )); then
                reasons+=("WARNING: ${target} tok/s above practical ceiling ~${practical_ceiling} tok/s (70% BW utilization)")
            fi

            # Current decode limited by attention occupancy (44% of time)
            # If attention goes to 0, max improvement = 1/(1-0.44) = 1.79x
            local attn_speedup_ceiling=$(echo "$BASELINE_DECODE_TOKS * 1.79" | bc -l | cut -d. -f1)
            if (( $(echo "$target > $attn_speedup_ceiling" | bc -l) )); then
                reasons+=("WARNING: Even with zero attention overhead, max ${attn_speedup_ceiling} tok/s (Amdahl's law)")
            fi
            ;;

        rss)
            # RSS minimum = model weights + KV cache + overhead
            # Weights: ~1 GB (Q4_K_M)
            # Server overhead: ~1.5 GB (runtime, allocator, CUDA context)
            local min_rss=2500
            if (( $(echo "$target < $min_rss" | bc -l) )); then
                gate_result="FALSIFIED"
                reasons+=("Target ${target} MB below minimum ${min_rss} MB (weights + server + CUDA context)")
            fi
            ;;

        occupancy)
            # Occupancy limited by grid size and block configuration
            local max_warps=$((NUM_HEADS * 4))  # 4 warps/head
            local total_capacity=$((NUM_SMS * MAX_WARPS_PER_SM))
            local max_occupancy=$(echo "scale=1; $max_warps * 100 / $total_capacity" | bc -l)
            if (( $(echo "$target > $max_occupancy" | bc -l) )); then
                reasons+=("WARNING: Max occupancy ${max_occupancy}% with ${NUM_HEADS} heads × 4 warps on ${NUM_SMS} SMs")
            fi
            ;;
    esac
}

check_experiment_feasibility() {
    local exp_type="$1"

    case "$exp_type" in
        kernel-fusion)
            # Kernel fusion at M=1 decode is memory-bound → fusion doesn't help
            if [[ "$target_metric" == "decode" ]]; then
                local improvement_pct=$(echo "scale=1; ($target_value - $BASELINE_DECODE_TOKS) / $BASELINE_DECODE_TOKS * 100" | bc -l)
                if (( $(echo "$improvement_pct > 5" | bc -l) )); then
                    reasons+=("WARNING: Kernel fusion at M=1 unlikely to improve >5% (memory-bound, AI=4.0)")
                    reasons+=("Evidence: Phase 12 — 16 qcd fusions all falsified for M=1 decode")
                fi
            fi
            ;;

        dispatch-opt)
            # Dispatch optimization already at minimum (CUDA graph = 1 launch)
            if [[ "$target_metric" == "decode" ]]; then
                local current_overhead_pct=5  # Graph replay overhead ~5%
                local max_gain=$(echo "scale=1; $BASELINE_DECODE_TOKS * $current_overhead_pct / 100" | bc -l)
                reasons+=("INFO: Graph dispatch already minimal (647 kernels → 1 launch). Max gain: ~${max_gain} tok/s")
            fi
            ;;

        quantization)
            # Quantization changes precision, not memory BW
            if [[ "$target_metric" == "decode" ]]; then
                reasons+=("INFO: Quantization affects precision (PPL) not throughput at M=1 (memory-bound)")
            fi
            ;;
    esac
}

echo "╔══════════════════════════════════════════════╗"
echo "║  P15-03: BOTTLENECK GATE (Pre-Experiment)   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Experiment: $experiment_type"
echo "Target: $target_metric = $target_value"
echo "Hardware: RTX 4090 (${NUM_SMS} SMs, ${PEAK_BW_GBS} GB/s, ${CLOCK_MHZ} MHz)"
echo ""

check_roofline_bound "$target_metric" "$target_value"
check_experiment_feasibility "$experiment_type"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$gate_result" == "FALSIFIED" ]]; then
    echo -e "${RED}GATE: FALSIFIED${NC} — experiment cannot succeed"
    for reason in "${reasons[@]}"; do
        echo -e "  ${RED}✗${NC} $reason"
    done
    echo ""
    echo "Recommendation: Do NOT run this experiment. Revise target or approach."
    exit 1
elif [[ ${#reasons[@]} -gt 0 ]]; then
    echo -e "${YELLOW}GATE: PASS (with warnings)${NC}"
    for reason in "${reasons[@]}"; do
        if [[ "$reason" == WARNING* ]]; then
            echo -e "  ${YELLOW}⚠${NC} $reason"
        else
            echo -e "  ${GREEN}ℹ${NC} $reason"
        fi
    done
    echo ""
    echo "Experiment may proceed, but review warnings."
    exit 0
else
    echo -e "${GREEN}GATE: PASS${NC} — no roofline violations detected"
    echo ""
    echo "Experiment may proceed."
    exit 0
fi
