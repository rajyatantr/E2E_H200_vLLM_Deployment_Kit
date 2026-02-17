"""
Download all required models for OCR Module 3 (OlmOCR).
Run this script ONCE before building the Docker image.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model repositories on Hugging Face
# MODELS = {
#     "olmOCR-2-7B-1025": "allenai/olmOCR-2-7B-1025",
#     "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
# }

MODELS = {
    "olmOCR-2-7B-1025-FP8": "allenai/olmOCR-2-7B-1025-FP8", # Quantized Version for faster inference
}


def download_model(repo_id: str, local_dir: Path) -> bool:
    """Download a model from Hugging Face Hub."""
    try:
        logger.info(f"Downloading {repo_id}...")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        logger.info(f"Successfully downloaded {repo_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {repo_id}: {e}")
        return False


def download_all_models(models_dir: Path = None) -> dict:
    """Download all required models."""
    if models_dir is None:
        models_dir = settings.models_dir
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading models to: {models_dir}")
    logger.info(f"Total models to download: {len(MODELS)}")
    
    results = {}
    
    for model_name, repo_id in MODELS.items():
        # Convert repo_id to directory name
        local_dir_name = repo_id.replace("/", "--")
        local_dir = models_dir / local_dir_name
        
        # Check if already exists
        if local_dir.exists() and any(local_dir.iterdir()):
            logger.info(f"⏭Skipping {model_name} (already exists)")
            results[model_name] = "exists"
            continue
        
        # Download
        success = download_model(repo_id, local_dir)
        results[model_name] = "success" if success else "failed"
    
    return results


def print_summary(results: dict):
    """Print download summary."""
    total = len(results)
    success = sum(1 for v in results.values() if v == "success")
    exists = sum(1 for v in results.values() if v == "exists")
    failed = sum(1 for v in results.values() if v == "failed")
    
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total models: {total}")
    logger.info(f"Successfully downloaded: {success}")
    logger.info(f"Already existed: {exists}")
    logger.info(f"Failed: {failed}")
    
    if failed > 0:
        logger.warning("\nFailed models:")
        for model, status in results.items():
            if status == "failed":
                logger.warning(f"  - {model}")
    
    # Calculate total size
    try:
        total_size = sum(
            f.stat().st_size 
            for f in settings.models_dir.rglob("*") 
            if f.is_file()
        )
        logger.info(f"\nTotal size: {total_size / (1024**3):.2f} GB")
    except Exception as e:
        logger.warning(f"Could not calculate total size: {e}")
    
    logger.info("=" * 60)


def main():
    """Main execution function."""
    logger.info("Starting model download for OCR Module 3 (OlmOCR)")
    logger.info(f"Models will be saved to: {settings.models_dir}")
    
    # Download all models
    results = download_all_models()
    
    # Print summary
    print_summary(results)
    
    if any(v == "failed" for v in results.values()):
        logger.error("\nSome models failed to download. Please re-run the script.")
        sys.exit(1)
    else:
        logger.info("\nAll models downloaded successfully!")
        logger.info("\nYou can now run the application:")
        logger.info("   python app.py")


if __name__ == "__main__":
    main()