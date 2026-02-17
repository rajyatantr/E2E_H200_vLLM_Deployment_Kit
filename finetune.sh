#!/usr/bin/env bash
# =============================================================================
# finetune.sh - Fine-tune a model with LoRA/QLoRA on H200
# =============================================================================
# Usage:
#   ./finetune.sh my_data.jsonl
#   ./finetune.sh my_data.jsonl --method qlora --epochs 5 --lr 1e-4
#   ./finetune.sh my_data.jsonl --adapter-name my_project
#
# Data format (JSONL, one per line):
#   {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# -- Helpers -------------------------------------------------------------------
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

# -- Parse arguments -----------------------------------------------------------
DATA_FILE=""
ADAPTER_NAME=""
METHOD="${FINETUNE_METHOD}"
EPOCHS="${NUM_EPOCHS}"
LR="${LEARNING_RATE}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)       METHOD="$2";       shift 2 ;;
        --epochs)       EPOCHS="$2";       shift 2 ;;
        --lr)           LR="$2";           shift 2 ;;
        --adapter-name) ADAPTER_NAME="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: ./finetune.sh <data.jsonl> [options]"
            echo ""
            echo "Options:"
            echo "  --method lora|qlora     Fine-tuning method (default: ${FINETUNE_METHOD})"
            echo "  --epochs N              Number of training epochs (default: ${NUM_EPOCHS})"
            echo "  --lr RATE               Learning rate (default: ${LEARNING_RATE})"
            echo "  --adapter-name NAME     Name for the output adapter"
            exit 0
            ;;
        -*)
            err "Unknown option: $1"
            exit 1
            ;;
        *)
            DATA_FILE="$1"
            shift
            ;;
    esac
done

if [ -z "${DATA_FILE}" ]; then
    err "No data file specified."
    echo "Usage: ./finetune.sh <data.jsonl>"
    exit 1
fi

if [ ! -f "${DATA_FILE}" ]; then
    err "Data file not found: ${DATA_FILE}"
    exit 1
fi

# -- Validate data file --------------------------------------------------------
info "Validating data file: ${DATA_FILE}"
LINE_COUNT=$(wc -l < "${DATA_FILE}" | tr -d ' ')
if [ "${LINE_COUNT}" -eq 0 ]; then
    err "Data file is empty"
    exit 1
fi

python -c "
import json, sys
errors = 0
with open('${DATA_FILE}') as f:
    for i, line in enumerate(f, 1):
        try:
            obj = json.loads(line.strip())
            if 'messages' not in obj:
                print(f'Line {i}: missing \"messages\" key')
                errors += 1
        except json.JSONDecodeError as e:
            print(f'Line {i}: invalid JSON - {e}')
            errors += 1
        if errors >= 5:
            print('... (too many errors, stopping)')
            break
if errors:
    sys.exit(1)
print(f'Valid: {i} examples')
"
ok "Data validated: ${LINE_COUNT} examples"

# -- Resolve adapter output path -----------------------------------------------
if [ -z "${ADAPTER_NAME}" ]; then
    MODEL_SHORT=$(basename "${MODEL_ID}")
    ADAPTER_NAME="${MODEL_SHORT}_lora_$(date +%Y%m%d_%H%M%S)"
fi
OUTPUT_DIR="${ADAPTER_DIR}/${ADAPTER_NAME}"
mkdir -p "${OUTPUT_DIR}"
info "Adapter will be saved to: ${OUTPUT_DIR}"

# -- Resolve base model --------------------------------------------------------
RESOLVED_MODEL="${MODEL_ID}"
CACHED_PATH=$(python -c "
from huggingface_hub import scan_cache_dir
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
fi

# -- Build and run training script --------------------------------------------
info "Starting ${METHOD^^} fine-tuning..."
info "Model: ${MODEL_ID}"
info "Method: ${METHOD} | Epochs: ${EPOCHS} | LR: ${LR} | Batch: ${BATCH_SIZE}"
echo ""

python "${SCRIPT_DIR}/train.py" \
    --model_name_or_path "${RESOLVED_MODEL}" \
    --data_path "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --method "${METHOD}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --learning_rate "${LR}" \
    --num_epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}"

ok "Fine-tuning complete!"
echo ""
echo "========================================="
echo "  Adapter saved: ${OUTPUT_DIR}"
echo ""
echo "  To serve with this adapter:"
echo "    ./swap_model.sh --adapter ${OUTPUT_DIR}"
echo ""
echo "  Or directly:"
echo "    ./serve.sh --lora ${OUTPUT_DIR}"
echo "========================================="
