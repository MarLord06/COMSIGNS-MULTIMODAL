"""
Inference routes for the ComSigns API.

Provides endpoints for batch inference with semantic resolution
and word acceptance rules.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])


# ============================================================
# Response Models
# ============================================================

class PredictionResponse(BaseModel):
    """Prediction result for a single file."""
    class_id: int = Field(..., description="Model output class ID")
    gloss: str = Field(..., description="Human-readable class name")
    bucket: str = Field(..., description="Class bucket (HEAD, MID, OTHER)")
    confidence: float = Field(..., description="Top-1 confidence score")
    accepted: bool = Field(..., description="Whether word was accepted")
    reason: str = Field(..., description="Reason for acceptance/rejection")


class FileResultResponse(BaseModel):
    """Result for a single file in a batch."""
    file_name: str = Field(..., description="Original file name")
    prediction: PredictionResponse = Field(..., description="Prediction result")


class FileErrorResponse(BaseModel):
    """Error for a single file in a batch."""
    file_name: str = Field(..., description="Original file name")
    error: str = Field(..., description="Error message")


class SequenceWordResponse(BaseModel):
    """An accepted word in the semantic sequence."""
    gloss: str = Field(..., description="Human-readable sign name")
    confidence: float = Field(..., description="Confidence score")


class SequenceResponse(BaseModel):
    """Semantic sequence of accepted words."""
    accepted: List[SequenceWordResponse] = Field(
        default_factory=list,
        description="List of accepted words"
    )
    length: int = Field(0, description="Number of accepted words")


class BatchInferenceResponse(BaseModel):
    """Response for batch inference endpoint."""
    results: List[FileResultResponse] = Field(
        default_factory=list,
        description="Per-file prediction results"
    )
    sequence: SequenceResponse = Field(
        default_factory=SequenceResponse,
        description="Semantic sequence of accepted words"
    )


class ErrorDetail(BaseModel):
    """Error detail response."""
    detail: str


# ============================================================
# Service Initialization
# ============================================================

_batch_service = None


def get_batch_service():
    """Get or create the batch inference service.
    
    Lazy loads the service on first call.
    """
    global _batch_service
    
    if _batch_service is None:
        import os
        from pathlib import Path
        
        from backend.services.inference_service import InferenceService
        from backend.services.batch_service import BatchInferenceService
        from backend.decision_engine import DecisionEvaluator
        
        # Configuration (same as main app)
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
        
        logger.info("Initializing BatchInferenceService...")
        logger.info(f"  Checkpoint: {CHECKPOINT_PATH}")
        logger.info(f"  Class mapping: {CLASS_MAPPING_PATH}")
        
        # Create dependencies
        inference_service = InferenceService(
            checkpoint_path=CHECKPOINT_PATH,
            class_mapping_path=CLASS_MAPPING_PATH,
            dict_path=DICT_PATH if DICT_PATH.exists() else None,
            device=DEVICE,
            lazy_load=False
        )
        
        evaluator = DecisionEvaluator()
        
        _batch_service = BatchInferenceService(
            inference_service=inference_service,
            decision_evaluator=evaluator
        )
        
        logger.info("BatchInferenceService initialized successfully")
    
    return _batch_service


# ============================================================
# Endpoints
# ============================================================

@router.post(
    "/batch",
    response_model=BatchInferenceResponse,
    responses={
        400: {"model": ErrorDetail, "description": "Invalid input"},
        500: {"model": ErrorDetail, "description": "Server error"}
    },
    summary="Batch inference with semantic sequence",
    description="""
Process multiple `.pkl` files in a single request.

For each file:
1. Load preprocessed keypoints
2. Run model inference (Top-K)
3. Select Top-1 prediction
4. Resolve class mapping (model_class_id → gloss, bucket)
5. Apply word acceptance rules
6. Add accepted words to semantic sequence

**Word Acceptance Rules:**
- confidence >= threshold (HEAD: 0.45, MID: 0.55)
- class_id is not OTHER
- margin between top-1 and top-2 >= 0.10

**Response:**
- Per-file prediction results with acceptance decisions
- Semantic sequence containing only accepted words
- Order preserved as submitted

**Error Handling:**
- One file failing does NOT cancel others
- Errors reported per-file in response
"""
)
async def batch_inference(
    files: List[UploadFile] = File(
        ...,
        description="One or more .pkl files containing preprocessed keypoints"
    ),
    topk: int = Query(
        5,
        ge=1,
        le=20,
        description="Number of top predictions to retrieve"
    ),
    reset_sequence: bool = Query(
        True,
        description="Reset sequence state before processing batch"
    )
):
    """Run batch inference with semantic resolution and word acceptance."""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file extensions
    invalid_files = [f.filename for f in files if not f.filename.endswith(".pkl")]
    if invalid_files:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file types. Expected .pkl: {invalid_files}"
        )
    
    try:
        batch_service = get_batch_service()
        
        # Reset sequence state if requested (default: True)
        if reset_sequence:
            batch_service.reset_sequence_state()
        
        # Collect file contents
        file_data = []
        for file in files:
            contents = await file.read()
            if len(contents) == 0:
                logger.warning(f"Empty file: {file.filename}")
                continue
            file_data.append((file.filename, contents))
        
        if not file_data:
            raise HTTPException(status_code=400, detail="All files are empty")
        
        # Process batch
        logger.info(f"Processing batch of {len(file_data)} files")
        result = batch_service.process_batch(file_data, topk=topk)
        
        # Log summary
        accepted_count = sum(
            1 for r in result.results 
            if r.prediction and r.prediction.accepted
        )
        logger.info(
            f"Batch complete: {len(result.results)} processed, "
            f"{accepted_count} accepted, "
            f"sequence length = {result.sequence.length}"
        )
        
        return result.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


@router.get(
    "/sequence",
    response_model=SequenceResponse,
    summary="Get current semantic sequence",
    description="Returns the current state of the semantic sequence."
)
async def get_sequence():
    """Get the current semantic sequence state."""
    batch_service = get_batch_service()
    evaluator = batch_service.decision_evaluator
    
    state = evaluator.get_sequence_state()
    
    return {
        "accepted": [
            {"gloss": item.get("gloss", ""), "confidence": item.get("confidence", 0)}
            for item in state.get("accepted", [])
        ],
        "length": state.get("length", 0)
    }


@router.post(
    "/sequence/reset",
    response_model=SequenceResponse,
    summary="Reset semantic sequence",
    description="Clears the semantic sequence and returns empty state."
)
async def reset_sequence():
    """Reset the semantic sequence state."""
    batch_service = get_batch_service()
    batch_service.reset_sequence_state()
    
    logger.info("Semantic sequence reset")
    
    return {
        "accepted": [],
        "length": 0
    }
