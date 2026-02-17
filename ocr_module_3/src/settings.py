from pathlib import Path
from typing import Optional
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings:
    """Configuration settings for OCR Module 3 (OlmOCR)."""
    
    def __init__(self):
        # Base directories
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / "models"
        self.database_dir = self.base_dir / "database"
        self.templates_dir = self.base_dir / "templates"
        
        # Environment variables for Hugging Face and model caching
        self.hf_home = os.getenv("HF_HOME", str(self.models_dir))
        self.transformers_cache = os.getenv(
            "TRANSFORMERS_CACHE", 
            str(self.models_dir / "transformers")
        )
        
        # Set environment variables
        os.environ["HF_HOME"] = self.hf_home
        os.environ["TRANSFORMERS_CACHE"] = self.transformers_cache
        
        # API settings
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8093"))  # Different port from Module 2
        
        # Processing settings
        self.max_workers = int(os.getenv("MAX_WORKERS", "1"))  # OlmOCR is memory-intensive
        self.target_image_dim = int(os.getenv("TARGET_IMAGE_DIM", "1288"))
        
        # Model settings
        # self.model_name = os.getenv("MODEL_NAME", "allenai/olmOCR-2-7B-1025")
        self.model_name = os.getenv("MODEL_NAME", "allenai/olmOCR-2-7B-1025-FP8") # Quantized Version
        self.processor_name = os.getenv("PROCESSOR_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
        
        # GPU settings
        self.use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
        self.cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")

        # vLLM engine settings (v2 tuned for H200)
        self.vllm_tensor_parallel_size = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
        self.vllm_gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.95"))
        self.vllm_max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
        self.vllm_max_num_seqs = int(os.getenv("VLLM_MAX_NUM_SEQS", "25"))

        # v2: Retry settings (matches OlmOCR production pipeline)
        self.max_page_retries = int(os.getenv("MAX_PAGE_RETRIES", "5"))
        self.initial_temperature = float(os.getenv("INITIAL_TEMPERATURE", "0.1"))
        self.temperature_step = float(os.getenv("TEMPERATURE_STEP", "0.15"))
        self.max_temperature = float(os.getenv("MAX_TEMPERATURE", "0.8"))

        # v2: File validation
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "100"))

        # v2: ADE schemas directory
        self.schemas_dir = self.base_dir / "src" / "schemas"
        
        # Database
        self.db_path = self.database_dir / "extraction_results.db"
        
        # Create directories
        self._create_directories()
        
        logger.info(f"OCR Module 3 (OlmOCR) Settings initialized")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Database directory: {self.database_dir}")
        logger.info(f"API Port: {self.api_port}")
        logger.info(f"Use GPU: {self.use_gpu}")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Directories created/verified")
    
    @property
    def cache_dir(self) -> Path:
        """Return the cache directory for models."""
        return self.models_dir
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the path for a specific model."""
        # Convert repo format to directory name
        model_dir = model_name.replace("/", "--")
        return self.models_dir / model_dir
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists locally."""
        model_path = self.get_model_path(model_name)
        return model_path.exists() and any(model_path.iterdir())


# Global settings instance
settings = Settings()


if __name__ == "__main__":
    print(f"Base Directory: {settings.base_dir}")
    print(f"Models Directory: {settings.models_dir}")
    print(f"Database Path: {settings.db_path}")
    print(f"API Port: {settings.api_port}")
    print(f"Max Workers: {settings.max_workers}")
    print(f"Model Name: {settings.model_name}")
    print(f"Processor Name: {settings.processor_name}")