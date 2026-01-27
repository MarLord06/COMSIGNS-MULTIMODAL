"""
Video inference routes for the ComSigns API.

Provides endpoints for video upload, keypoint extraction,
and sign language inference.
"""

import logging
from pathlib import Path
from typing import List, Optional
import os

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["video"])


# ============================================================
# Response Models
# ============================================================

class VideoPredictionResponse(BaseModel):
    """Prediction result for a single video."""
    video: str = Field(..., description="Original video filename")
    class_id: int = Field(..., description="Model output class ID")
    class_name: str = Field(..., description="Internal class name (e.g., HEAD_259)")
    gloss: str = Field(..., description="Human-readable sign gloss")
    score: float = Field(..., description="Confidence score")
    accepted: bool = Field(..., description="Whether prediction was accepted")
    reason: str = Field("", description="Reason for acceptance/rejection")


class VideoInferenceResponse(BaseModel):
    """Response for video inference endpoint."""
    results: List[VideoPredictionResponse] = Field(
        default_factory=list,
        description="Per-video prediction results"
    )
    errors: List[dict] = Field(
        default_factory=list,
        description="Per-video errors"
    )


class VideoInfoResponse(BaseModel):
    """Video metadata response."""
    filename: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


# ============================================================
# Configuration
# ============================================================

# Allowed video extensions
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}

# Video constraints
MAX_VIDEO_SIZE_MB = 100
MIN_DURATION_SEC = 0.1  # 100ms minimum
MAX_DURATION_SEC = 30.0  # 30 seconds maximum

# Default paths
DEFAULT_EXPERIMENT = "run_20260122_010532"
BASE_DIR = Path(__file__).parent.parent.parent.parent  # comsigns/

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


# ============================================================
# Service Initialization
# ============================================================

_inference_service = None
_video_preprocessor = None
_decision_evaluator = None


def get_inference_service():
    """Get or create the inference service."""
    global _inference_service
    
    if _inference_service is None:
        from backend.services.inference_service import InferenceService
        
        logger.info(f"Initializing InferenceService for video...")
        logger.info(f"  Checkpoint: {CHECKPOINT_PATH}")
        logger.info(f"  Class mapping: {CLASS_MAPPING_PATH}")
        logger.info(f"  Dict path: {DICT_PATH}")
        
        _inference_service = InferenceService(
            checkpoint_path=CHECKPOINT_PATH,
            class_mapping_path=CLASS_MAPPING_PATH,
            dict_path=DICT_PATH,
            device=DEVICE,
            lazy_load=True
        )
    
    return _inference_service


def get_video_preprocessor():
    """Get or create the video preprocessor."""
    global _video_preprocessor
    
    if _video_preprocessor is None:
        from backend.services.video_preprocess import VideoPreprocessor
        _video_preprocessor = VideoPreprocessor(
            max_frames=150,
            min_frames=5
        )
    
    return _video_preprocessor


def get_decision_evaluator():
    """Get or create the decision evaluator."""
    global _decision_evaluator
    
    if _decision_evaluator is None:
        from backend.decision_engine import DecisionEvaluator
        _decision_evaluator = DecisionEvaluator()
    
    return _decision_evaluator


# ============================================================
# Validation Helpers
# ============================================================

def validate_video_file(filename: str, file_size: int) -> None:
    """Validate video file constraints.
    
    Args:
        filename: Original filename
        file_size: File size in bytes
    
    Raises:
        HTTPException: If validation fails
    """
    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    max_size_bytes = MAX_VIDEO_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB (max: {MAX_VIDEO_SIZE_MB}MB)"
        )


# ============================================================
# Endpoints
# ============================================================

@router.post("/infer", response_model=VideoInferenceResponse)
async def infer_from_videos(
    files: List[UploadFile] = File(..., description="Video files (.mp4, .mov, etc.)"),
    topk: int = Query(5, ge=1, le=10, description="Number of top predictions")
):
    """Run inference on one or more video files.
    
    Each video should contain a single, already-segmented sign.
    The endpoint:
    1. Extracts frames from video
    2. Runs MediaPipe keypoint extraction
    3. Runs model inference
    4. Applies acceptance rules
    
    Returns predictions per video with acceptance status.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Get services
    inference_service = get_inference_service()
    preprocessor = get_video_preprocessor()
    evaluator = get_decision_evaluator()
    
    results = []
    errors = []
    
    for file in files:
        filename = file.filename or "unknown.mp4"
        
        try:
            # Read file content
            content = await file.read()
            
            # Validate
            validate_video_file(filename, len(content))
            
            # Preprocess video -> tensors
            logger.info(f"Processing video: {filename}")
            features = preprocessor.process_video(content)
            
            # Run inference
            inference_result = inference_service.infer(features, topk=topk)
            inference_dict = inference_result.to_dict()
            
            # Apply decision rules
            decision_result = evaluator.process_from_inference_result(inference_dict)
            
            # Extract top-1 info
            top1 = inference_dict.get("top1", {})
            prediction_info = decision_result.get("prediction", {})
            
            # Build class_name from bucket and class_id
            bucket = top1.get("bucket", "OTHER")
            class_id = top1.get("new_class_id", 0)
            class_name = f"{bucket}_{class_id}"
            
            results.append(VideoPredictionResponse(
                video=filename,
                class_id=class_id,
                class_name=class_name,
                gloss=top1.get("gloss", "UNKNOWN"),
                score=round(top1.get("confidence", 0.0), 4),
                accepted=prediction_info.get("accepted", False),
                reason=prediction_info.get("reason", "")
            ))
            
        except HTTPException:
            raise
        except ValueError as e:
            # Validation errors (e.g., video too short)
            logger.warning(f"Validation error for {filename}: {e}")
            errors.append({
                "video": filename,
                "error": str(e)
            })
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}", exc_info=True)
            errors.append({
                "video": filename,
                "error": f"Processing failed: {str(e)}"
            })
    
    return VideoInferenceResponse(results=results, errors=errors)


@router.post("/info", response_model=List[VideoInfoResponse])
async def get_video_info(
    files: List[UploadFile] = File(..., description="Video files to inspect")
):
    """Get metadata for video files without running inference.
    
    Useful for validating videos before processing.
    """
    import tempfile
    import os
    
    preprocessor = get_video_preprocessor()
    results = []
    
    for file in files:
        filename = file.filename or "unknown.mp4"
        
        try:
            content = await file.read()
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                info = preprocessor.get_video_info(tmp_path)
                results.append(VideoInfoResponse(
                    filename=filename,
                    fps=round(info["fps"], 2),
                    frame_count=info["frame_count"],
                    width=info["width"],
                    height=info["height"],
                    duration_sec=round(info["duration_sec"], 2)
                ))
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"Error getting info for {filename}: {e}")
            results.append(VideoInfoResponse(
                filename=filename,
                fps=0,
                frame_count=0,
                width=0,
                height=0,
                duration_sec=0
            ))
    
    return results


@router.get("/config")
async def get_video_config():
    """Get current video processing configuration."""
    return {
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "max_video_size_mb": MAX_VIDEO_SIZE_MB,
        "min_duration_sec": MIN_DURATION_SEC,
        "max_duration_sec": MAX_DURATION_SEC,
        "max_frames": 150,
        "min_frames": 5,
        "target_fps": 30
    }
