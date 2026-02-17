#!/usr/bin/env python3
"""
test_endpoint.py - Quick test for the vLLM OpenAI-compatible endpoint.

Usage:
    python test_endpoint.py                          # Basic test
    python test_endpoint.py --base-url http://host:8000  # Custom URL
    python test_endpoint.py --model finetuned        # Test LoRA model
    python test_endpoint.py --stream                 # Test streaming
    python test_endpoint.py --benchmark              # Run throughput benchmark
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def api_request(base_url, endpoint, payload):
    """Make a request to the vLLM API."""
    url = f"{base_url}{endpoint}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code}: {body}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection failed: {e.reason}")
        print("Is the vLLM server running? Start it with: ./serve.sh")
        sys.exit(1)


def test_health(base_url):
    """Check server health."""
    try:
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"  Health: OK ({resp.status})")
            return True
    except Exception as e:
        print(f"  Health: FAILED ({e})")
        return False


def test_models(base_url):
    """List available models."""
    try:
        req = urllib.request.Request(f"{base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            models = [m["id"] for m in data["data"]]
            print(f"  Models: {', '.join(models)}")
            return models
    except Exception as e:
        print(f"  Models: FAILED ({e})")
        return []


def test_completion(base_url, model, prompt="Hello! What is 2+2?"):
    """Test a chat completion."""
    print(f"\n  Prompt: {prompt}")
    start = time.time()

    result = api_request(base_url, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.7,
    })

    elapsed = time.time() - start
    reply = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", "?")
    completion_tokens = usage.get("completion_tokens", "?")

    print(f"  Reply: {reply[:200]}{'...' if len(reply) > 200 else ''}")
    print(f"  Tokens: {prompt_tokens} in / {completion_tokens} out")
    print(f"  Latency: {elapsed:.2f}s")

    if isinstance(completion_tokens, int) and elapsed > 0:
        print(f"  Throughput: {completion_tokens / elapsed:.1f} tokens/sec")


def test_benchmark(base_url, model, num_requests=10):
    """Run a simple throughput benchmark."""
    prompts = [
        "Explain quantum computing in one sentence.",
        "Write a Python function to reverse a string.",
        "What is the capital of India?",
        "Summarize machine learning in 20 words.",
        "Write a haiku about GPUs.",
    ]

    print(f"\n  Running {num_requests} sequential requests...")
    total_tokens = 0
    start = time.time()

    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        result = api_request(base_url, "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.7,
        })
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        total_tokens += tokens
        sys.stdout.write(f"\r  Progress: {i + 1}/{num_requests}")
        sys.stdout.flush()

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg throughput: {total_tokens / elapsed:.1f} tokens/sec")
    print(f"  Avg latency: {elapsed / num_requests:.2f}s per request")


def main():
    parser = argparse.ArgumentParser(description="Test vLLM endpoint")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None, help="Model name to test")
    parser.add_argument("--prompt", default="Hello! What is 2+2?")
    parser.add_argument("--stream", action="store_true", help="Test streaming")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--num-requests", type=int, default=10)
    args = parser.parse_args()

    print("=" * 50)
    print("  vLLM Endpoint Test")
    print("=" * 50)
    print(f"  Server: {args.base_url}")

    # Health check
    if not test_health(args.base_url):
        sys.exit(1)

    # List models
    models = test_models(args.base_url)
    if not models:
        sys.exit(1)

    model = args.model or models[0]
    print(f"  Testing model: {model}")

    # Chat completion
    test_completion(args.base_url, model, args.prompt)

    # Benchmark
    if args.benchmark:
        test_benchmark(args.base_url, model, args.num_requests)

    print("\n" + "=" * 50)
    print("  All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
