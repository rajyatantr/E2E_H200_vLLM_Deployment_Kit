#!/usr/bin/env bash
# =============================================================================
# serve.sh - Launch vLLM OpenAI-compatible API server on H200
# =============================================================================
# Usage:
#   ./serve.sh                       # Serve base model
#   ./serve.sh --lora /path/to/adapter  # Serve with LoRA adapter
#   ./serve.sh --background          # Run in background (logs to file)
#   ./serve.sh --stop                # Stop background server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# -- Helpers -------------------------------------------------------------------
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

PIDFILE="${SCRIPT_DIR}/.vllm.pid"
LORA_ADAPTER=""
BACKGROUND=false

# -- Parse arguments -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lora)
            LORA_ADAPTER="$2"
            shift 2
            ;;
        --background|-bg)
            BACKGROUND=true
            shift
            ;;
        --stop)
            if [ -f "${PIDFILE}" ]; then
                PID=$(cat "${PIDFILE}")
                if kill -0 "${PID}" 2>/dev/null; then
                    kill "${PID}"
                    rm -f "${PIDFILE}"
                    ok "vLLM server (PID ${PID}) stopped"
                else
                    rm -f "${PIDFILE}"
                    info "Server was not running (stale PID file cleaned)"
                fi
            else
                info "No PID file found. Checking for running vLLM processes..."
                pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && ok "vLLM processes killed" || info "No vLLM processes found"
            fi
            exit 0
            ;;
        --help|-h)
            echo "Usage: ./serve.sh [--lora /path/to/adapter] [--background] [--stop]"
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# -- Pre-flight checks --------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi not found. Are you on the H200 instance?"
    exit 1
fi

if ! python -c "import vllm" 2>/dev/null; then
    err "vLLM not installed. Run ./setup.sh first."
    exit 1
fi

# Check if port is already in use
if ss -tlnp 2>/dev/null | grep -q ":${SERVE_PORT} " || \
   lsof -i ":${SERVE_PORT}" &>/dev/null; then
    err "Port ${SERVE_PORT} is already in use. Run ./serve.sh --stop first."
    exit 1
fi

# -- Resolve model path --------------------------------------------------------
# Try to find the model in the local cache first, then fall back to HF ID
RESOLVED_MODEL="${MODEL_ID}"
CACHED_PATH=$(python -c "
from huggingface_hub import try_to_load_from_cache, scan_cache_dir
import os
cache_dir = '${MODEL_CACHE_DIR}'
if os.path.isdir(cache_dir):
    cache = scan_cache_dir(cache_dir)
    for repo in cache.repos:
        if repo.repo_id == '${MODEL_ID}':
            for rev in repo.revisions:
                print(rev.snapshot_path)
                break
            break
" 2>/dev/null || true)

if [ -n "${CACHED_PATH}" ] && [ -d "${CACHED_PATH}" ]; then
    RESOLVED_MODEL="${CACHED_PATH}"
    info "Using cached model: ${RESOLVED_MODEL}"
else
    info "Using model ID (will download if needed): ${MODEL_ID}"
fi

# -- Build vLLM arguments -----------------------------------------------------
VLLM_ARGS=(
    --model "${RESOLVED_MODEL}"
    --host "${SERVE_HOST}"
    --port "${SERVE_PORT}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --dtype "${DTYPE}"
    --trust-remote-code
    --served-model-name "${MODEL_ID}"
    --download-dir "${MODEL_CACHE_DIR}"
)

if [ -n "${QUANTIZATION}" ]; then
    VLLM_ARGS+=(--quantization "${QUANTIZATION}")
fi

if [ "${ENABLE_CHUNKED_PREFILL}" = true ]; then
    VLLM_ARGS+=(--enable-chunked-prefill)
fi

# LoRA support
if [ -n "${LORA_ADAPTER}" ]; then
    if [ ! -d "${LORA_ADAPTER}" ]; then
        err "LoRA adapter path does not exist: ${LORA_ADAPTER}"
        exit 1
    fi
    VLLM_ARGS+=(--enable-lora --max-loras "${MAX_LORAS}" --lora-modules "finetuned=${LORA_ADAPTER}")
    info "LoRA adapter loaded: ${LORA_ADAPTER}"
fi

# -- Determine log file --------------------------------------------------------
LOG_DIR="${VLLM_LOGS_DIR}"
if [ ! -d "${LOG_DIR}" ]; then
    LOG_DIR="${SCRIPT_DIR}/logs"
    mkdir -p "${LOG_DIR}"
fi
LOG_FILE="${LOG_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log"

# -- Launch server -------------------------------------------------------------
info "Starting vLLM server..."
info "Model: ${MODEL_ID}"
info "Endpoint: http://${SERVE_HOST}:${SERVE_PORT}"
info "OpenAI base URL: http://localhost:${SERVE_PORT}/v1"
echo ""

if [ "${BACKGROUND}" = true ]; then
    nohup python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" \
        > "${LOG_FILE}" 2>&1 &
    SERVER_PID=$!
    echo "${SERVER_PID}" > "${PIDFILE}"
    ok "vLLM running in background (PID: ${SERVER_PID})"
    info "Log file: ${LOG_FILE}"
    info "Stop with: ./serve.sh --stop"

    # Wait briefly and check it didn't crash immediately
    sleep 3
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        err "Server exited immediately. Check log: ${LOG_FILE}"
        tail -20 "${LOG_FILE}" 2>/dev/null
        rm -f "${PIDFILE}"
        exit 1
    fi
    ok "Server is starting up. It may take 1-2 minutes to load the model."
    info "Test with: curl http://localhost:${SERVE_PORT}/v1/models"
else
    info "Running in foreground (Ctrl+C to stop)..."
    info "Log file: ${LOG_FILE}"
    python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
fi
