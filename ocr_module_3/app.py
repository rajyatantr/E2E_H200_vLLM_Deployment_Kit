from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager
import shutil
import tempfile
import uuid
import json
import uvicorn
import logging

from src.vits_extractor import OlmOCRProcessor
from src.db_manager import DatabaseManager
from src.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
templates = Jinja2Templates(directory=str(settings.templates_dir))
db_manager = DatabaseManager(str(settings.db_path))
processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown)."""
    # Startup
    global processor
    logger.info("Initializing OlmOCR processor (v2 + ADE)...")
    processor = OlmOCRProcessor(
        max_workers=settings.max_workers
    )
    logger.info("OCR Module 3 v2 started successfully")

    yield

    # Shutdown
    if processor:
        processor.cleanup()
    logger.info("OCR Module 3 shutdown complete")


app = FastAPI(
    title="OCR Module 3 - OlmOCR API v2",
    description="Document OCR with Agentic Document Extraction (ADE) using OlmOCR-2-7B",
    version="3.2.0",
    lifespan=lifespan
)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render dashboard."""
    stats = db_manager.get_statistics()
    recent_results = db_manager.get_all_results(limit=10)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "stats": stats,
            "recent_results": recent_results
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import torch

    return {
        "status": "healthy",
        "module": "OCR Module 3 - OlmOCR v2 (ADE)",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


# API Router
router = APIRouter(prefix="/api")


# ──────────────────────────────────────────────
# File Validation Helper
# ──────────────────────────────────────────────

async def _validate_file_size(file: UploadFile) -> bytes:
    """Read and validate file size. Returns file content bytes."""
    content = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_file_size_mb}MB limit ({len(content) / (1024*1024):.1f}MB)"
        )
    await file.seek(0)
    return content


# ──────────────────────────────────────────────
# Raw OCR Endpoints (Enhanced with v2 features)
# ──────────────────────────────────────────────

@router.post("/process/single")
async def process_single_document(
    file: UploadFile = File(...),
    conversion_type: str = Form("yaml")
):
    """Process a single document with raw OCR extraction."""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    await _validate_file_size(file)
    request_id = str(uuid.uuid4())

    temp_dir = tempfile.mkdtemp()
    temp_file_path = Path(temp_dir) / file.filename

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = processor.process_document(
            file_path=temp_file_path,
            conversion_type=conversion_type,
            request_id=request_id
        )

        db_manager.insert_result(result)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"[{request_id[:8]}] Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/process/batch")
async def process_batch_documents(
    files: List[UploadFile] = File(...),
    conversion_type: str = Form("yaml")
):
    """Process multiple documents."""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files allowed")

    # Validate all file sizes
    for f in files:
        await _validate_file_size(f)

    temp_dir = tempfile.mkdtemp()
    temp_file_paths = []

    try:
        for file in files:
            temp_file_path = Path(temp_dir) / file.filename
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_file_paths.append(temp_file_path)

        results = processor.process_batch(
            file_paths=temp_file_paths,
            conversion_type=conversion_type
        )

        for result in results:
            db_manager.insert_result(result)

        stats = processor.get_stats(results)

        return JSONResponse(content={
            "results": results,
            "statistics": stats
        })

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ──────────────────────────────────────────────
# ADE Endpoint
# ──────────────────────────────────────────────

@router.post("/extract")
async def extract_document(
    file: UploadFile = File(...),
    schema_name: str = Form(None),
    schema_json: str = Form(None),
):
    """Agentic Document Extraction: schema-driven, multi-pass, self-validating.

    Provide either `schema_name` (loads from built-in schemas) or `schema_json` (inline JSON).
    """
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    await _validate_file_size(file)
    request_id = str(uuid.uuid4())

    # Resolve schema
    schema = None
    if schema_json:
        try:
            schema = json.loads(schema_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid schema_json: must be valid JSON")
    elif schema_name:
        from src.schemas.schema_loader import load_schema
        schema = load_schema(schema_name)
        if not schema:
            from src.schemas.schema_loader import list_schemas
            available = list_schemas()
            raise HTTPException(
                status_code=400,
                detail=f"Schema '{schema_name}' not found. Available: {available}"
            )
    else:
        from src.schemas.schema_loader import list_schemas
        available = list_schemas()
        raise HTTPException(
            status_code=400,
            detail=f"Provide schema_name or schema_json. Available schemas: {available}"
        )

    temp_dir = tempfile.mkdtemp()
    temp_file_path = Path(temp_dir) / file.filename

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = processor.extract_with_schema(
            file_path=temp_file_path,
            schema=schema,
            request_id=request_id
        )

        db_manager.insert_result(result)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"[{request_id[:8]}] ADE extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ──────────────────────────────────────────────
# Schema & Model Status Endpoints
# ──────────────────────────────────────────────

@router.get("/schemas")
async def get_schemas():
    """List available extraction schemas."""
    from src.schemas.schema_loader import list_schemas, load_schema
    names = list_schemas()
    return JSONResponse(content={
        "schemas": [{"name": n, "schema": load_schema(n)} for n in names]
    })


@router.get("/models/status")
async def models_status():
    """Return model status and GPU info."""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    return JSONResponse(content=processor.get_model_status())


# ──────────────────────────────────────────────
# Results & Statistics Endpoints (unchanged)
# ──────────────────────────────────────────────

@router.get("/results")
async def get_all_results(limit: Optional[int] = 100):
    """Get all extraction results."""
    results = db_manager.get_all_results(limit=limit)
    return JSONResponse(content={"results": results})


@router.get("/results/{result_id}")
async def get_result_by_id(result_id: int):
    """Get specific result by ID."""
    result = db_manager.get_result_by_id(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return JSONResponse(content=result)


@router.get("/statistics")
async def get_statistics():
    """Get database statistics."""
    stats = db_manager.get_statistics()
    return JSONResponse(content=stats)


@router.get("/search")
async def search_results(q: str):
    """Search results by document name."""
    results = db_manager.search_results(q)
    return JSONResponse(content={"results": results})


@router.post("/reset-database")
async def reset_database():
    """Reset the database."""
    try:
        db_manager.reset_database()
        return JSONResponse(content={"message": "Database reset successfully"})
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount router
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        workers=1
    )
