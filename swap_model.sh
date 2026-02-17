#!/usr/bin/env bash
# =============================================================================
# swap_model.sh - Hot-swap between base model and fine-tuned adapters
# =============================================================================
# Usage:
#   ./swap_model.sh --finetuned             # Use latest adapter
#   ./swap_model.sh --adapter /path/to/adapter  # Use specific adapter
#   ./swap_model.sh --base                  # Revert to base model
#   ./swap_model.sh --list                  # List available adapters
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# -- Helpers -------------------------------------------------------------------
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

MODE=""
ADAPTER_PATH=""
BACKGROUND=false

# -- Parse arguments -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --finetuned)
            MODE="finetuned"
            shift
            ;;
        --base)
            MODE="base"
            shift
            ;;
        --adapter)
            MODE="adapter"
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --list)
            MODE="list"
            shift
            ;;
        --background|-bg)
            BACKGROUND=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./swap_model.sh [--finetuned|--base|--adapter PATH|--list] [--background]"
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "${MODE}" ]; then
    err "Specify --finetuned, --base, --adapter <path>, or --list"
    exit 1
fi

# -- List adapters -------------------------------------------------------------
if [ "${MODE}" = "list" ]; then
    echo "Available adapters in ${ADAPTER_DIR}:"
    echo ""
    if [ -d "${ADAPTER_DIR}" ] && [ "$(ls -A "${ADAPTER_DIR}" 2>/dev/null)" ]; then
        for dir in "${ADAPTER_DIR}"/*/; do
            if [ -f "${dir}/adapter_config.json" ]; then
                ADAPTER_NAME=$(basename "${dir}")
                SIZE=$(du -sh "${dir}" 2>/dev/null | cut -f1)
                MODIFIED=$(stat -c '%y' "${dir}/adapter_config.json" 2>/dev/null | cut -d'.' -f1 || \
                           stat -f '%Sm' "${dir}/adapter_config.json" 2>/dev/null || echo "unknown")
                echo "  ${ADAPTER_NAME}  (${SIZE}, modified: ${MODIFIED})"
            fi
        done
    else
        echo "  (no adapters found)"
        echo ""
        echo "  Run ./finetune.sh <data.jsonl> to create one."
    fi
    exit 0
fi

# -- Find latest adapter (for --finetuned) ------------------------------------
if [ "${MODE}" = "finetuned" ]; then
    if [ ! -d "${ADAPTER_DIR}" ]; then
        err "No adapter directory found: ${ADAPTER_DIR}"
        err "Run ./finetune.sh <data.jsonl> first."
        exit 1
    fi

    # Find the most recently modified adapter
    LATEST=$(find "${ADAPTER_DIR}" -name "adapter_config.json" -type f -printf '%T@ %h\n' 2>/dev/null | \
             sort -rn | head -1 | cut -d' ' -f2- || true)

    # Fallback for macOS (no -printf)
    if [ -z "${LATEST}" ]; then
        LATEST=$(find "${ADAPTER_DIR}" -name "adapter_config.json" -type f -exec stat -f '%m %N' {} \; 2>/dev/null | \
                 sort -rn | head -1 | sed 's|/adapter_config.json||' | awk '{print $2}' || true)
    fi

    if [ -z "${LATEST}" ]; then
        err "No adapters found in ${ADAPTER_DIR}"
        err "Run ./finetune.sh <data.jsonl> first."
        exit 1
    fi

    ADAPTER_PATH="${LATEST}"
    info "Using latest adapter: ${ADAPTER_PATH}"
fi

# -- Validate adapter path (for --finetuned and --adapter) ---------------------
if [ "${MODE}" != "base" ]; then
    if [ ! -f "${ADAPTER_PATH}/adapter_config.json" ]; then
        err "Invalid adapter: ${ADAPTER_PATH}"
        err "Expected adapter_config.json in the directory."
        exit 1
    fi
    ok "Adapter validated: $(basename "${ADAPTER_PATH}")"
fi

# -- Stop existing server ------------------------------------------------------
info "Stopping existing vLLM server..."
"${SCRIPT_DIR}/serve.sh" --stop 2>/dev/null || true
sleep 2

# -- Restart with new configuration -------------------------------------------
SERVE_ARGS=()

if [ "${BACKGROUND}" = true ]; then
    SERVE_ARGS+=(--background)
fi

if [ "${MODE}" = "base" ]; then
    info "Restarting with base model: ${MODEL_ID}"
    "${SCRIPT_DIR}/serve.sh" "${SERVE_ARGS[@]}"
else
    info "Restarting with LoRA adapter: $(basename "${ADAPTER_PATH}")"
    "${SCRIPT_DIR}/serve.sh" --lora "${ADAPTER_PATH}" "${SERVE_ARGS[@]}"
fi
