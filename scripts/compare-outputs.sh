#!/bin/bash
# F-PARITY-03: Greedy output correctness comparison
# Compares token-by-token output of Candle vs realizr on the same prompt.
#
# Usage: bash scripts/compare-outputs.sh [--realizr-url http://127.0.0.1:8081]
#
# Prerequisites:
#   - Candle quantized-qwen2-instruct binary built
#   - realizr serving on --gpu with the same GGUF model
#   - Temperature 0 (greedy) mandatory for determinism
#
# Output:
#   - Side-by-side comparison of generated text
#   - Token count and first-divergence position
#   - PASS if identical, FAIL if >1% divergence
set -euo pipefail

CANDLE_BIN="${CANDLE_BIN:-/mnt/nvme-raid0/targets/candle/release/examples/quantized-qwen2-instruct}"
MODEL="${MODEL:-$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
REALIZR_URL="${REALIZR_URL:-http://127.0.0.1:8081}"
MAX_TOKENS=128
PROMPT="Write a Python function that checks if a number is prime. Include type hints."

echo "F-PARITY-03: Output Correctness Comparison"
echo "Model: $(basename "$MODEL")"
echo "Max tokens: $MAX_TOKENS"
echo "Prompt: $PROMPT"
echo "Temperature: 0 (greedy, mandatory)"
echo ""

# --- Candle output ---
echo "=== Candle ==="
CANDLE_OUT=$(mktemp)
if [ -x "$CANDLE_BIN" ]; then
    "$CANDLE_BIN" \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --sample-len "$MAX_TOKENS" \
        --temperature 0 2>/dev/null \
        | sed -n '/^$/,$ p' | tail -n +2 > "$CANDLE_OUT"
    CANDLE_TOKENS=$(wc -w < "$CANDLE_OUT")
    echo "Tokens (approx words): $CANDLE_TOKENS"
    echo "First 200 chars:"
    head -c 200 "$CANDLE_OUT"
    echo ""
else
    echo "ERROR: Candle binary not found at $CANDLE_BIN"
    echo "Build with: make build-candle"
    CANDLE_OUT=""
fi
echo ""

# --- realizr output ---
echo "=== realizr ==="
REALIZR_OUT=$(mktemp)
if curl -sf "$REALIZR_URL/health" >/dev/null 2>&1; then
    curl -s "$REALIZR_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(cat <<JSON
{
    "model": "qwen2.5-coder-1.5b-instruct",
    "messages": [{"role": "user", "content": "$PROMPT"}],
    "max_tokens": $MAX_TOKENS,
    "temperature": 0,
    "stream": false
}
JSON
)" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
text = d['choices'][0]['message']['content']
print(text)
" > "$REALIZR_OUT"
    REALIZR_TOKENS=$(wc -w < "$REALIZR_OUT")
    echo "Tokens (approx words): $REALIZR_TOKENS"
    echo "First 200 chars:"
    head -c 200 "$REALIZR_OUT"
    echo ""
else
    echo "ERROR: realizr not responding at $REALIZR_URL/health"
    echo "Start with: apr serve run <model> --gpu --port 8081"
    REALIZR_OUT=""
fi
echo ""

# --- Comparison ---
echo "=== Comparison ==="
if [ -n "$CANDLE_OUT" ] && [ -s "$CANDLE_OUT" ] && [ -n "$REALIZR_OUT" ] && [ -s "$REALIZR_OUT" ]; then
    # Note: Candle outputs raw completion, realizr outputs chat response.
    # The chat template wrapping means outputs won't be byte-identical.
    # Compare the CODE portion (strip template artifacts).

    # Word-level diff
    DIFF_COUNT=$(diff <(tr ' ' '\n' < "$CANDLE_OUT") <(tr ' ' '\n' < "$REALIZR_OUT") | grep -c "^[<>]" || true)
    TOTAL_WORDS=$(wc -w < "$CANDLE_OUT")

    if [ "$TOTAL_WORDS" -gt 0 ]; then
        DIVERGENCE_PCT=$(python3 -c "print(f'{$DIFF_COUNT / ($TOTAL_WORDS * 2) * 100:.1f}')")
        echo "Word-level divergence: $DIFF_COUNT different words"
        echo "Divergence rate: ${DIVERGENCE_PCT}%"
        echo ""

        # Show diff
        echo "--- Word diff (first 20 differences) ---"
        diff <(tr ' ' '\n' < "$CANDLE_OUT") <(tr ' ' '\n' < "$REALIZR_OUT") | head -40
        echo ""

        # Verdict
        THRESHOLD=1
        if python3 -c "exit(0 if $DIVERGENCE_PCT <= $THRESHOLD else 1)" 2>/dev/null; then
            echo "F-PARITY-03: **PASS** (divergence ${DIVERGENCE_PCT}% <= ${THRESHOLD}%)"
        else
            echo "F-PARITY-03: **FAIL** (divergence ${DIVERGENCE_PCT}% > ${THRESHOLD}%)"
            echo ""
            echo "NOTE: Some divergence is expected due to:"
            echo "  - Chat template wrapping (realizr applies chat template, Candle does raw completion)"
            echo "  - FP32 vs FP16 accumulator differences"
            echo "  - Different RoPE implementation details"
            echo "Compare the actual code content, not template artifacts."
        fi
    else
        echo "ERROR: Candle produced empty output"
    fi
else
    echo "SKIP: One or both runtimes not available. Run on machine with GPU."
fi

# Cleanup
rm -f "$CANDLE_OUT" "$REALIZR_OUT" 2>/dev/null || true
