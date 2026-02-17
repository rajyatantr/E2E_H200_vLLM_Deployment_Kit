# =============================================================================
# Dockerfile - vLLM serving container for E2E H200
# =============================================================================
# Build: ./build_and_push.sh
# Run:   docker run --gpus all -p 8000:8000 -v /home/jovyan/models:/models vllm-h200
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Install vLLM
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir vllm

# Create app directory
WORKDIR /app

# Copy config and serve script
COPY config.env serve.sh ./
RUN chmod +x serve.sh

# Model cache mount point
VOLUME ["/models"]

# Default environment
ENV MODEL_CACHE_DIR=/models
ENV SERVE_HOST=0.0.0.0
ENV SERVE_PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: serve the model
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "meta-llama/Llama-3.1-8B-Instruct", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--tensor-parallel-size", "1", \
     "--max-model-len", "8192", \
     "--gpu-memory-utilization", "0.90", \
     "--dtype", "auto", \
     "--trust-remote-code", \
     "--download-dir", "/models"]
