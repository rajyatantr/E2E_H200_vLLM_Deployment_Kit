#!/bin/bash
# Stop the vLLM server started by start_vllm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/vllm_server.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID: $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "Stopped."
    else
        echo "PID $PID not running. Cleaning up."
        rm -f "$PID_FILE"
    fi
else
    # Fallback: find vLLM process
    PIDS=$(pgrep -f "vllm.entrypoints.openai.api_server" || true)
    if [ -n "$PIDS" ]; then
        echo "Found vLLM processes: $PIDS"
        echo "Stopping..."
        kill $PIDS
        echo "Stopped."
    else
        echo "No vLLM server running."
    fi
fi
