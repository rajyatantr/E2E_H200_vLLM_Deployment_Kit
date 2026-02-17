#!/usr/bin/env bash
# =============================================================================
# setup.sh - Bootstrap the E2E H200 instance for vLLM deployment
# =============================================================================
# Usage: ./setup.sh [--skip-model-download]
#
# This script:
#   1. Creates directory structure on persistent & ephemeral storage
#   2. Installs/upgrades vLLM and fine-tuning dependencies
#   3. Downloads the configured model from HuggingFace
#   4. Validates GPU access
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# -- Helpers -------------------------------------------------------------------
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

SKIP_MODEL_DOWNLOAD=false
for arg in "$@"; do
    case "$arg" in
        --skip-model-download) SKIP_MODEL_DOWNLOAD=true ;;
        --help|-h)
            echo "Usage: ./setup.sh [--skip-model-download]"
            exit 0
            ;;
    esac
done

# -- Step 1: Validate GPU -----------------------------------------------------
info "Checking GPU availability..."
if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi not found. Are you on an H200 instance with NVIDIA drivers?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
ok "GPU detected: ${GPU_NAME} (${GPU_MEM} MiB)"

if [[ "${GPU_NAME}" == *"H200"* ]]; then
    ok "H200 GPU confirmed"
else
    warn "Expected H200, found: ${GPU_NAME}. Scripts may still work."
fi

# -- Step 2: Create directories ------------------------------------------------
info "Setting up directory structure..."
mkdir -p "${MODEL_CACHE_DIR}" "${ADAPTER_DIR}" "${TRAINING_OUTPUT_DIR}"
ok "Persistent dirs ready: ${MODEL_CACHE_DIR}, ${ADAPTER_DIR}"

if [ -d "$(dirname "${EPHEMERAL_DIR}")" ]; then
    mkdir -p "${VLLM_LOGS_DIR}" 2>/dev/null || warn "Could not create ephemeral log dir (may need sudo)"
    ok "Ephemeral log dir: ${VLLM_LOGS_DIR}"
else
    warn "Ephemeral storage ${EPHEMERAL_DIR} not available. Logs will go to ./logs/"
    VLLM_LOGS_DIR="${SCRIPT_DIR}/logs"
    mkdir -p "${VLLM_LOGS_DIR}"
fi

# -- Step 3: Install Python dependencies --------------------------------------
info "Installing/upgrading Python packages..."

pip install --upgrade pip setuptools wheel 2>&1 | tail -1

info "Installing vLLM..."
pip install --upgrade vllm 2>&1 | tail -1
ok "vLLM installed: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'version check failed')"

info "Installing fine-tuning stack (transformers, peft, trl, bitsandbytes)..."
pip install --upgrade \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    bitsandbytes \
    scipy \
    sentencepiece \
    protobuf \
    huggingface_hub[cli] \
    2>&1 | tail -1
ok "Fine-tuning dependencies installed"

# -- Step 4: HuggingFace login ------------------------------------------------
if [ -n "${HF_TOKEN}" ]; then
    info "Logging into HuggingFace..."
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null
    ok "HuggingFace login successful"
else
    if ! huggingface-cli whoami &>/dev/null; then
        warn "HF_TOKEN not set and not logged in. Set it in config.env or run: huggingface-cli login"
        warn "Gated models (Llama, etc.) will fail to download without auth."
    else
        ok "Already logged into HuggingFace"
    fi
fi

# -- Step 5: Download model ----------------------------------------------------
if [ "${SKIP_MODEL_DOWNLOAD}" = true ]; then
    info "Skipping model download (--skip-model-download)"
else
    info "Downloading model: ${MODEL_ID} → ${MODEL_CACHE_DIR}"
    info "This may take a while for large models..."

    python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${MODEL_ID}',
    cache_dir='${MODEL_CACHE_DIR}',
    resume_download=True,
)
print('Download complete.')
"
    ok "Model cached at ${MODEL_CACHE_DIR}"
fi

# -- Step 6: Verify installation -----------------------------------------------
info "Running verification checks..."

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_mem / 1024**3:.1f} GB)')
"

ok "Setup complete!"
echo ""
echo "========================================="
echo "  Next steps:"
echo "  1. ./serve.sh              - Start vLLM server"
echo "  2. ./finetune.sh data.jsonl - Fine-tune with your data"
echo "  3. ./swap_model.sh --finetuned - Serve fine-tuned model"
echo "========================================="
