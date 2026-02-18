#!/bin/bash
# ============================================================================
# Start vLLM server for ADE Pipeline v2
# ============================================================================
# Usage: bash deploy/start_vllm.sh [--model MODEL] [--port PORT] [--background]
#
# Reads defaults from config.yaml, overridable via CLI args.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_DIR/config.yaml"

# ---------------------------------------------------------------------------
# Parse config.yaml with Python (no external deps needed)
# ---------------------------------------------------------------------------
read_config() {
    python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
# Navigate nested keys like 'vllm.port'
keys = sys.argv[1].split('.')
val = cfg
for k in keys:
    val = val.get(k, '')
print(val)
" "$1" 2>/dev/null || echo "$2"
}

# Defaults from config.yaml
MODEL="${MODEL:-$(read_config 'model.name' 'Qwen/Qwen2.5-VL-72B-Instruct-AWQ')}"
PORT="${PORT:-$(read_config 'vllm.port' '8000')}"
DTYPE="${DTYPE:-$(read_config 'vllm.dtype' 'float16')}"
QUANT="${QUANT:-$(read_config 'vllm.quantization' 'awq_marlin')}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(read_config 'vllm.max_model_len' '8192')}"
GPU_MEM="${GPU_MEM:-$(read_config 'vllm.gpu_memory_utilization' '0.90')}"
MAX_SEQS="${MAX_SEQS:-$(read_config 'vllm.max_num_seqs' '10')}"
TP_SIZE="${TP_SIZE:-$(read_config 'vllm.tensor_parallel_size' '1')}"
MM_LIMIT="${MM_LIMIT:-$(read_config 'vllm.limit_mm_per_prompt' '{"image": 1}')}"
MM_PROC="${MM_PROC:-$(read_config 'vllm.mm_processor_kwargs' '{"min_pixels": 784, "max_pixels": 1003520}')}"

BACKGROUND=false

# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     MODEL="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --dtype)     DTYPE="$2"; shift 2 ;;
        --quant)     QUANT="$2"; shift 2 ;;
        --background) BACKGROUND=true; shift ;;
        -h|--help)
            echo "Usage: bash start_vllm.sh [--model M] [--port P] [--dtype D] [--quant Q] [--background]"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Build vLLM command
# ---------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CMD="python3 -m vllm.entrypoints.openai.api_server"
CMD+=" --model $MODEL"
CMD+=" --port $PORT"
CMD+=" --dtype $DTYPE"
CMD+=" --max-model-len $MAX_MODEL_LEN"
CMD+=" --gpu-memory-utilization $GPU_MEM"
CMD+=" --max-num-seqs $MAX_SEQS"
CMD+=" --tensor-parallel-size $TP_SIZE"
CMD+=" --trust-remote-code"
CMD+=" --limit-mm-per-prompt '$MM_LIMIT'"
CMD+=" --mm-processor-kwargs '$MM_PROC'"

# Add quantization if set
if [ "$QUANT" != "null" ] && [ "$QUANT" != "None" ] && [ -n "$QUANT" ]; then
    CMD+=" --quantization $QUANT"
fi

echo "============================================================"
echo "  Starting vLLM Server"
echo "============================================================"
echo "  Model:    $MODEL"
echo "  Port:     $PORT"
echo "  Dtype:    $DTYPE"
echo "  Quant:    $QUANT"
echo "  Max Seqs: $MAX_SEQS"
echo "  GPU Mem:  $GPU_MEM"
echo "  TP Size:  $TP_SIZE"
echo "============================================================"
echo ""

if [ "$BACKGROUND" = true ]; then
    LOG_FILE="$PROJECT_DIR/vllm_server.log"
    echo "  Running in background. Log: $LOG_FILE"
    echo "  PID file: $PROJECT_DIR/vllm_server.pid"
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    echo $! > "$PROJECT_DIR/vllm_server.pid"
    echo "  PID: $(cat "$PROJECT_DIR/vllm_server.pid")"
    echo ""
    echo "  Waiting for server to start..."
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
            echo "  Server is ready!"
            exit 0
        fi
        sleep 5
        echo "    ... waiting ($((i*5))s)"
    done
    echo "  WARNING: Server not ready after 5 minutes. Check $LOG_FILE"
else
    echo "  Running in foreground (Ctrl+C to stop)..."
    echo ""
    eval "$CMD"
fi
