from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the processor module BEFORE importing app
sys.modules['src.vits_extractor'] = MagicMock()
sys.modules['src.vits_extractor'].OlmOCRProcessor = MagicMock()

from app import app
from fastapi.testclient import TestClient

def test_api_routes():
    with TestClient(app) as client:
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        print("Health check passed")

        # Test statistics endpoint
        response = client.get("/api/statistics")
        assert response.status_code == 200
        print("/api/statistics passed")

        # Test results endpoint
        response = client.get("/api/results")
        assert response.status_code == 200
        print("/api/results passed")

        # Test search endpoint (should return empty list or results)
        response = client.get("/api/search?q=test")
        assert response.status_code == 200
        print("/api/search passed")

if __name__ == "__main__":
    try:
        test_api_routes()
        print("All API tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
