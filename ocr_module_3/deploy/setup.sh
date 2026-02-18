#!/bin/bash
# ============================================================================
# ADE Pipeline v2 — One-Command H200 Setup
# ============================================================================
# Usage: bash deploy/setup.sh
#
# What this does:
#   1. Installs vLLM + PyMuPDF (skipping NGC torch constraint conflicts)
#   2. Fixes known vLLM registry bug on NGC containers
#   3. Downloads the model (first run only)
#   4. Creates convenience scripts
#
# Prerequisites:
#   - NVIDIA H200 (or A100/H100) with CUDA 12+
#   - Python 3.10+ with pip
#   - Internet access for model download (~40GB for 72B-AWQ)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "  ADE Pipeline v2 — H200 Setup"
echo "============================================================"
echo "  Project dir: $PROJECT_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Install Python dependencies
# ---------------------------------------------------------------------------
echo "[1/4] Installing Python dependencies..."

# Check if we're in an NGC container (has constraint file that blocks vLLM)
NGC_CONSTRAINT="/opt/nvidia/entrypoint.d/constraints.txt"
if [ -f "$NGC_CONSTRAINT" ]; then
    echo "  NGC container detected — bypassing torch constraint file"
    PIP_CONSTRAINT='' pip3 install 'vllm>=0.8' PyMuPDF pyyaml --no-build-isolation -q 2>&1 | tail -3
else
    pip3 install 'vllm>=0.8' PyMuPDF pyyaml -q 2>&1 | tail -3
fi

# Install training dependencies (optional, won't fail if unavailable)
echo "  Installing training dependencies (optional)..."
pip3 install transformers peft accelerate bitsandbytes datasets pillow -q 2>&1 | tail -3 || true

echo "  Dependencies installed."
echo ""

# ---------------------------------------------------------------------------
# Step 2: Fix vLLM registry bug (NGC Python 3.12 subprocess pickle issue)
# ---------------------------------------------------------------------------
echo "[2/4] Checking for vLLM registry bug..."

REGISTRY_FILE=$(python3 -c "
import importlib.util
spec = importlib.util.find_spec('vllm.model_executor.models.registry')
print(spec.origin if spec else '')
" 2>/dev/null || echo "")

if [ -n "$REGISTRY_FILE" ] && [ -f "$REGISTRY_FILE" ]; then
    if grep -q "_run_in_subprocess" "$REGISTRY_FILE" 2>/dev/null; then
        echo "  Patching vLLM registry to avoid subprocess pickle error..."
        python3 -c "
import re
with open('$REGISTRY_FILE', 'r') as f:
    content = f.read()

# Replace subprocess call with direct call
old = '_run_in_subprocess(lambda: _ModelInfo.from_model_cls(self.load_model_cls()))'
new = '_ModelInfo.from_model_cls(self.load_model_cls())'
if old in content:
    content = content.replace(old, new)
    with open('$REGISTRY_FILE', 'w') as f:
        f.write(content)
    print('  Patched successfully.')
else:
    print('  Already patched or different version.')
"
    else
        echo "  No patch needed."
    fi
else
    echo "  Could not find registry file — skipping patch."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Check flash-attn compatibility
# ---------------------------------------------------------------------------
echo "[3/4] Checking flash-attention compatibility..."

python3 -c "
import torch
try:
    import flash_attn
    print(f'  flash-attn {flash_attn.__version__} with torch {torch.__version__} — OK')
except ImportError:
    print('  flash-attn not installed — vLLM will use fallback attention')
except Exception as e:
    print(f'  flash-attn error: {e}')
    print('  You may need to reinstall flash-attn for your torch version')
    torch_ver = torch.__version__.split('+')[0].rsplit('.', 1)[0]  # e.g. 2.9
    print(f'  Try: pip install flash-attn --no-build-isolation')
" 2>&1
echo ""

# ---------------------------------------------------------------------------
# Step 4: Verify GPU and print summary
# ---------------------------------------------------------------------------
echo "[4/4] Verifying GPU..."

python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'  GPU: {gpu}')
    print(f'  VRAM: {mem:.0f} GB')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  WARNING: No GPU detected!')
"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "    1. Start vLLM:     bash deploy/start_vllm.sh"
echo "    2. Health check:   bash deploy/health_check.sh"
echo "    3. Run extraction: python3 qwen_vl_extract.py <pdf> -o output.json"
echo ""
echo "  For training:"
echo "    1. Collect data:   python3 training/collect_training_data.py --pdf-dir /path/to/pdfs"
echo "    2. Annotate:       python3 training/annotate.py"
echo "    3. Fine-tune:      python3 training/finetune_qwen_vl.py"
echo ""
