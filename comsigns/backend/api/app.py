"""
FastAPI application for ComSigns inference.

Provides HTTP endpoints for running model inference on .pkl samples and videos.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import routes
from backend.api.routes import inference_router, video_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ComSigns Inference API",
    description="API para inferencia de señas LSP-AEC",
    version="0.3.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(inference_router)
app.include_router(video_router)

# ============================================================
# Configuration
# ============================================================

# Default paths - can be overridden via environment variables
DEFAULT_EXPERIMENT = "run_20260122_010532"
BASE_DIR = Path(__file__).parent.parent.parent  # comsigns/

CHECKPOINT_PATH = Path(os.getenv(
    "COMSIGNS_CHECKPOINT",
    BASE_DIR / f"experiments/{DEFAULT_EXPERIMENT}/checkpoints/best.pt"
))
CLASS_MAPPING_PATH = Path(os.getenv(
    "COMSIGNS_CLASS_MAPPING", 
    BASE_DIR / f"experiments/{DEFAULT_EXPERIMENT}/class_mapping.json"
))
DICT_PATH = Path(os.getenv(
    "COMSIGNS_DICT",
    BASE_DIR.parent / "data/raw/lsp_aec/dict.json"
))
DEVICE = os.getenv("COMSIGNS_DEVICE", "cpu")

# Service instance (lazy loaded)
_service = None
_evaluator = None


def get_service():
    """Get or create the inference service."""
    global _service
    if _service is None:
        from backend.services.inference_service import InferenceService
        
        logger.info(f"Initializing InferenceService...")
        logger.info(f"  Checkpoint: {CHECKPOINT_PATH}")
        logger.info(f"  Class mapping: {CLASS_MAPPING_PATH}")
        logger.info(f"  Dict: {DICT_PATH}")
        logger.info(f"  Device: {DEVICE}")
        
        _service = InferenceService(
            checkpoint_path=CHECKPOINT_PATH,
            class_mapping_path=CLASS_MAPPING_PATH,
            dict_path=DICT_PATH if DICT_PATH.exists() else None,
            device=DEVICE,
            lazy_load=False
        )
    return _service


def get_decision_evaluator():
    """Get or create the decision evaluator."""
    global _evaluator
    if _evaluator is None:
        from backend.decision_engine import DecisionEvaluator
        logger.info("Initializing DecisionEvaluator...")
        _evaluator = DecisionEvaluator()
    return _evaluator


# ============================================================
# Response Models
# ============================================================

class Top1Prediction(BaseModel):
    """Top-1 prediction response."""
    gloss: str
    confidence: float
    bucket: str
    is_other: bool
    new_class_id: Optional[int] = None
    old_class_id: Optional[int] = None


class TopKPrediction(BaseModel):
    """Top-K prediction item."""
    rank: int
    gloss: str
    confidence: float
    bucket: str
    is_other: bool = False


class InferenceMeta(BaseModel):
    """Inference metadata."""
    model: str
    num_classes: int
    device: str
    topk_requested: int


class InferenceResult(BaseModel):
    """Full inference response."""
    top1: Top1Prediction
    topk: list[TopKPrediction]
    meta: InferenceMeta


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    num_classes: Optional[int] = None


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str


# ============================================================
# Endpoints
# ============================================================

@app.get("/", tags=["info"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "ComSigns Inference API",
        "version": "0.3.0",
        "endpoints": {
            "pkl_inference": {
                "POST /infer": "Run inference on .pkl sample",
                "POST /infer/evaluate": "Run inference with decision engine",
                "POST /infer/batch/evaluate": "Batch inference with decision engine"
            },
            "video_inference": {
                "POST /api/video/infer": "Run inference on video file(s)",
                "POST /api/video/info": "Get video metadata",
                "GET /api/video/config": "Get video processing config"
            },
            "batch_sequence": {
                "POST /api/inference/batch": "Batch inference with semantic sequence",
                "GET /api/inference/sequence": "Get semantic sequence",
                "POST /api/inference/sequence/reset": "Reset semantic sequence"
            },
            "sequence": {
                "GET /sequence": "Get current accepted sequence",
                "POST /sequence/reset": "Reset sequence state"
            },
            "system": {
                "GET /health": "Health check",
                "GET /info": "Model information"
            }
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
async def health_check():
    """Health check endpoint."""
    try:
        service = get_service()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            num_classes=service.num_classes
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False
        )


@app.get("/info", tags=["info"])
async def model_info():
    """Get model and configuration information."""
    try:
        service = get_service()
        stats = service.resolver.loader.statistics
        
        return {
            "model": {
                "checkpoint": str(CHECKPOINT_PATH),
                "num_classes": service.num_classes,
                "device": DEVICE
            },
            "mapping": {
                "original_classes": stats.num_classes_original,
                "remapped_classes": stats.num_classes_remapped,
                "head_count": stats.head_count,
                "mid_count": stats.mid_count,
                "tail_count": stats.tail_count,
                "other_class_id": stats.other_class_id
            }
        }
    except Exception as e:
        logger.error(f"Info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/infer",
    response_model=InferenceResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Inference error"}
    },
    tags=["inference"]
)
async def infer(
    file: UploadFile = File(..., description="Sample .pkl file"),
    topk: int = Query(5, ge=1, le=20, description="Number of top predictions")
):
    """Run inference on a .pkl sample.
    
    Accepts a .pkl file containing extracted features and returns
    semantic predictions with gloss, confidence, and bucket information.
    
    The .pkl file should contain a dictionary with keys:
    - hand: Hand keypoints tensor [T, hand_dim]
    - body: Body keypoints tensor [T, body_dim]  
    - face: Face keypoints tensor [T, face_dim]
    - lengths: (optional) Sequence length
    """
    # Validate file extension
    if not file.filename.endswith(".pkl"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Expected .pkl, got: {file.filename}"
        )
    
    try:
        # Read file contents
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        logger.info(f"Processing file: {file.filename} ({len(contents)} bytes)")
        
        # Run inference
        service = get_service()
        result = service.infer_from_bytes(contents, topk=topk)
        
        logger.info(
            f"Inference complete: top1={result.top1['gloss']} "
            f"({result.top1['confidence']:.2%})"
        )
        
        return result.to_dict()
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid sample: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post(
    "/infer/evaluate",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Inference error"}
    },
    tags=["inference", "decision"]
)
async def infer_with_evaluation(
    file: UploadFile = File(..., description="Sample .pkl file"),
    topk: int = Query(5, ge=1, le=20, description="Number of top predictions")
):
    """Run inference with decision engine evaluation.
    
    Processes the sample through:
    1. Model inference (same as /infer)
    2. Decision engine (acceptance rules + sequence tracking)
    
    Returns prediction with acceptance decision and current sequence state.
    """
    if not file.filename.endswith(".pkl"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Expected .pkl, got: {file.filename}"
        )
    
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        logger.info(f"Processing with evaluation: {file.filename}")
        
        # Run inference
        service = get_service()
        inference_result = service.infer_from_bytes(contents, topk=topk)
        
        # Run through decision engine
        evaluator = get_decision_evaluator()
        decision_result = evaluator.process_from_inference_result(
            inference_result.to_dict()
        )
        
        # Combine responses
        response = {
            "inference": inference_result.to_dict(),
            "prediction": decision_result["prediction"],
            "sequence": decision_result["sequence"]
        }
        
        logger.info(
            f"Evaluation complete: {decision_result['prediction']['gloss']} "
            f"- {'ACCEPTED' if decision_result['prediction']['accepted'] else 'REJECTED'}"
        )
        
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid sample: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Inference with evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/sequence", tags=["decision"])
async def get_sequence():
    """Get the current accepted sequence state."""
    evaluator = get_decision_evaluator()
    return {
        "sequence": evaluator.get_sequence_state(),
        "config": evaluator.get_config()
    }


@app.post("/sequence/reset", tags=["decision"])
async def reset_sequence():
    """Reset the sequence state.
    
    Clears all accepted and rejected items.
    """
    evaluator = get_decision_evaluator()
    evaluator.reset_sequence()
    logger.info("Sequence state reset")
    return {
        "message": "Sequence reset successfully",
        "sequence": evaluator.get_sequence_state()
    }


@app.get("/decision/config", tags=["decision"])
async def get_decision_config():
    """Get the decision engine configuration."""
    evaluator = get_decision_evaluator()
    return {
        "config": evaluator.get_config(),
        "rules": {
            "1": "Reject if bucket == OTHER",
            "2": "Reject if confidence < threshold (HEAD: 0.45, MID: 0.55)",
            "3": "Reject if margin (top1 - top2) < 0.10",
            "4": "Otherwise ACCEPT"
        }
    }


@app.post("/infer/batch", tags=["inference"])
async def infer_batch(
    files: list[UploadFile] = File(..., description="Multiple .pkl files"),
    topk: int = Query(5, ge=1, le=20)
):
    """Run inference on multiple .pkl samples.
    
    Returns a list of inference results, one per file.
    """
    results = []
    errors = []
    
    for file in files:
        try:
            if not file.filename.endswith(".pkl"):
                errors.append({
                    "filename": file.filename,
                    "error": "Invalid file type"
                })
                continue
            
            contents = await file.read()
            service = get_service()
            result = service.infer_from_bytes(contents, topk=topk)
            
            results.append({
                "filename": file.filename,
                "result": result.to_dict()
            })
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "results": results,
        "errors": errors,
        "total": len(files),
        "success": len(results),
        "failed": len(errors)
    }


@app.post(
    "/infer/batch/evaluate",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Inference error"}
    },
    tags=["inference", "decision"]
)
async def infer_batch_with_evaluation(
    files: list[UploadFile] = File(..., description="Multiple .pkl files"),
    topk: int = Query(5, ge=1, le=20)
):
    """Run batch inference with decision engine evaluation.
    
    Processes multiple samples through:
    1. Model inference for each file
    2. Decision engine evaluation (acceptance rules + sequence tracking)
    
    Returns per-file results with acceptance decisions and final sequence state.
    Files are processed in order - sequence accumulates across the batch.
    """
    results = []
    errors = []
    
    service = get_service()
    evaluator = get_decision_evaluator()
    
    for file in files:
        try:
            if not file.filename.endswith(".pkl"):
                errors.append({
                    "file_name": file.filename,
                    "error": "Invalid file type. Expected .pkl"
                })
                continue
            
            contents = await file.read()
            if len(contents) == 0:
                errors.append({
                    "file_name": file.filename,
                    "error": "Empty file"
                })
                continue
            
            # Run inference
            inference_result = service.infer_from_bytes(contents, topk=topk)
            
            # Run through decision engine
            decision_result = evaluator.process_from_inference_result(
                inference_result.to_dict()
            )
            
            results.append({
                "file_name": file.filename,
                "prediction": decision_result["prediction"]
            })
            
            logger.info(
                f"Batch item {file.filename}: {decision_result['prediction']['gloss']} "
                f"- {'ACCEPTED' if decision_result['prediction']['accepted'] else 'REJECTED'}"
            )
            
        except Exception as e:
            logger.error(f"Batch inference failed for {file.filename}: {e}")
            errors.append({
                "file_name": file.filename,
                "error": str(e)
            })
    
    # Get final sequence state after processing all files
    final_sequence = evaluator.get_sequence_state()
    
    return {
        "results": results,
        "errors": errors,
        "sequence": final_sequence,
        "summary": {
            "total": len(files),
            "processed": len(results),
            "failed": len(errors),
            "accepted": sum(1 for r in results if r["prediction"]["accepted"]),
            "rejected": sum(1 for r in results if not r["prediction"]["accepted"])
        }
    }


# ============================================================
# Startup/Shutdown Events
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("Starting ComSigns Inference API...")
    try:
        # Pre-load the model
        get_service()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Don't fail startup - allow lazy loading


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down ComSigns Inference API...")


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
