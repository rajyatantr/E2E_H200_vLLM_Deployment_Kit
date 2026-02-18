#!/bin/bash
# ============================================================================
# Health check for vLLM server + ADE pipeline
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_DIR/config.yaml"

PORT=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    print(yaml.safe_load(f).get('vllm',{}).get('port', 8000))
" 2>/dev/null || echo "8000")

echo "============================================================"
echo "  ADE Pipeline — Health Check"
echo "============================================================"
echo ""

# Check 1: vLLM server
echo -n "  [1] vLLM server (port $PORT): "
RESP=$(curl -s "http://localhost:$PORT/v1/models" 2>/dev/null)
if [ $? -eq 0 ] && echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null; then
    echo "       OK"
else
    echo "FAILED — server not responding"
fi

# Check 2: GPU
echo -n "  [2] GPU: "
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    used = torch.cuda.memory_allocated(0) / (1024**3)
    print(f'{name} ({used:.1f}/{mem:.0f} GB used)')
else:
    print('NO GPU DETECTED')
" 2>/dev/null || echo "Could not check"

# Check 3: PyMuPDF
echo -n "  [3] PyMuPDF: "
python3 -c "import fitz; print(f'v{fitz.version[0]} — OK')" 2>/dev/null || echo "NOT INSTALLED"

# Check 4: Config
echo -n "  [4] Config: "
if [ -f "$CONFIG" ]; then
    MODEL=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    print(yaml.safe_load(f).get('model',{}).get('name','?'))
" 2>/dev/null)
    echo "$CONFIG → $MODEL"
else
    echo "NOT FOUND at $CONFIG"
fi

# Check 5: Training data dir
echo -n "  [5] Training data: "
DATA_DIR=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    print(yaml.safe_load(f).get('training',{}).get('data_dir','./training/data'))
" 2>/dev/null || echo "./training/data")
if [ -d "$PROJECT_DIR/$DATA_DIR" ]; then
    COUNT=$(ls "$PROJECT_DIR/$DATA_DIR"/*.json 2>/dev/null | wc -l)
    echo "$COUNT samples in $DATA_DIR"
else
    echo "Dir not found: $DATA_DIR"
fi

echo ""
echo "============================================================"
