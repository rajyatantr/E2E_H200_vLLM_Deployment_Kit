# E2E H200 vLLM Deployment Kit

Deploy, fine-tune, and serve LLMs on [E2E Networks](https://www.e2enetworks.com/) H200 GPU instances (141GB HBM3e) using vLLM.

## Quick Start

```bash
# 1. SSH into your E2E H200 instance, then:
git clone https://github.com/rajyatantr/E2E_H200_vLLM_Deployment_Kit.git
cd E2E_H200_vLLM_Deployment_Kit

# 2. Set your HuggingFace token (required for gated models like Llama)
export HF_TOKEN="hf_your_token_here"

# 3. Run setup (installs vLLM, downloads model)
./setup.sh

# 4. Start serving
./serve.sh

# 5. Test the endpoint
python test_endpoint.py
```

## What's Included

| File | Purpose |
|---|---|
| `config.env` | Central configuration (model, paths, hyperparams) |
| `setup.sh` | Install dependencies, download model, validate GPU |
| `serve.sh` | Launch vLLM OpenAI-compatible API server |
| `finetune.sh` | Fine-tune with LoRA/QLoRA |
| `train.py` | Training script (called by finetune.sh) |
| `swap_model.sh` | Hot-swap between base and fine-tuned models |
| `build_and_push.sh` | Docker build + push to E2E Container Registry |
| `test_endpoint.py` | Test and benchmark the API endpoint |

## Workflow

### Serve a Model

```bash
# Foreground (see logs in terminal)
./serve.sh

# Background (logs to file)
./serve.sh --background

# Stop
./serve.sh --stop
```

The server exposes an OpenAI-compatible API at `http://localhost:8000/v1`.

```bash
# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Fine-Tune with LoRA

```bash
# Prepare data in chat format (JSONL)
# Each line: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

# Fine-tune
./finetune.sh examples/sample_data.jsonl

# Fine-tune with options
./finetune.sh data.jsonl --method qlora --epochs 5 --lr 1e-4

# Serve the fine-tuned model
./swap_model.sh --finetuned
```

### Swap Models

```bash
# Use latest fine-tuned adapter
./swap_model.sh --finetuned

# Use a specific adapter
./swap_model.sh --adapter /home/jovyan/adapters/my_adapter

# Revert to base model
./swap_model.sh --base

# List available adapters
./swap_model.sh --list
```

### Docker Deployment

```bash
# Build and push to E2E Container Registry
./build_and_push.sh

# Build only (no push)
./build_and_push.sh --build-only

# Run the container
docker run --gpus all -p 8000:8000 \
  -v /home/jovyan/models:/models \
  registry.e2enetworks.net/vllm-h200:latest
```

## Configuration

Edit `config.env` to customize:

- **Model**: Change `MODEL_ID` to any HuggingFace model
- **Memory**: `GPU_MEMORY_UTILIZATION` (default 0.90 for H200's 141GB)
- **Context**: `MAX_MODEL_LEN` (default 8192)
- **Quantization**: Set `QUANTIZATION` to `fp8` for larger models
- **Fine-tuning**: LoRA rank, learning rate, batch size, etc.

### Models That Fit on H200 (141GB)

| Model | Precision | Memory | Config |
|---|---|---|---|
| Llama 3.1 8B | BF16 | ~16GB | Default |
| Llama 3.1 70B | BF16 | ~140GB | `GPU_MEMORY_UTILIZATION=0.95` |
| Llama 3.1 70B | FP8 | ~70GB | `QUANTIZATION=fp8` |
| Qwen 2.5 72B | BF16 | ~140GB | `GPU_MEMORY_UTILIZATION=0.95` |
| Mistral 7B | BF16 | ~14GB | Default |

## E2E Instance Layout

```
/home/jovyan/          # Persistent storage (survives restarts)
├── models/            # Downloaded model weights
├── adapters/          # Fine-tuned LoRA adapters
└── training_outputs/  # Training checkpoints

/mnt/local/            # Fast NVMe (ephemeral, lost on shutdown)
└── vllm_logs/         # Server logs
```

## Requirements

- E2E Networks H200 GPU instance (or any NVIDIA GPU with sufficient VRAM)
- PyTorch pre-built base image (CUDA 12.4, Python 3.10)
- HuggingFace account + token for gated models
