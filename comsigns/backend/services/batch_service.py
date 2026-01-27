"""
Batch inference service for processing multiple samples.

Handles the business logic for batch inference with semantic
resolution and word acceptance rules.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BatchPrediction:
    """Prediction result for a single file in a batch.
    
    Attributes:
        class_id: Model output class ID
        gloss: Human-readable class name
        bucket: Class bucket (HEAD, MID, OTHER)
        confidence: Top-1 confidence score
        accepted: Whether the word was accepted
        reason: Reason for acceptance/rejection
    """
    class_id: int
    gloss: str
    bucket: str
    confidence: float
    accepted: bool
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "class_id": self.class_id,
            "gloss": self.gloss,
            "bucket": self.bucket,
            "confidence": round(self.confidence, 4),
            "accepted": self.accepted,
            "reason": self.reason
        }


@dataclass 
class BatchFileResult:
    """Result for a single file in a batch.
    
    Attributes:
        file_name: Original file name
        prediction: Prediction result
        error: Error message if processing failed
    """
    file_name: str
    prediction: Optional[BatchPrediction] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        if self.error:
            return {
                "file_name": self.file_name,
                "error": self.error
            }
        return {
            "file_name": self.file_name,
            "prediction": self.prediction.to_dict() if self.prediction else None
        }


@dataclass
class SequenceWord:
    """An accepted word in the semantic sequence.
    
    Attributes:
        gloss: Human-readable sign name
        confidence: Confidence score when accepted
    """
    gloss: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gloss": self.gloss,
            "confidence": round(self.confidence, 4)
        }


@dataclass
class SemanticSequence:
    """Semantic sequence of accepted words.
    
    Attributes:
        accepted: List of accepted words
    """
    accepted: List[SequenceWord] = field(default_factory=list)
    
    @property
    def length(self) -> int:
        """Get sequence length."""
        return len(self.accepted)
    
    def append(self, gloss: str, confidence: float) -> None:
        """Append an accepted word to the sequence."""
        self.accepted.append(SequenceWord(gloss=gloss, confidence=confidence))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accepted": [w.to_dict() for w in self.accepted],
            "length": self.length
        }


@dataclass
class BatchInferenceResult:
    """Result of batch inference operation.
    
    Attributes:
        results: Per-file prediction results
        sequence: Semantic sequence of accepted words
    """
    results: List[BatchFileResult] = field(default_factory=list)
    sequence: SemanticSequence = field(default_factory=SemanticSequence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "results": [r.to_dict() for r in self.results if r.error is None],
            "errors": [r.to_dict() for r in self.results if r.error is not None],
            "sequence": self.sequence.to_dict()
        }


class BatchInferenceService:
    """Service for processing batch inference with semantic resolution.
    
    Combines:
    - Model inference per file
    - Semantic class resolution
    - Word acceptance rules
    - Sequence construction
    
    Example:
        >>> from backend.services.inference_service import InferenceService
        >>> from backend.decision_engine import DecisionEvaluator
        >>> 
        >>> inference_service = InferenceService(...)
        >>> evaluator = DecisionEvaluator()
        >>> 
        >>> batch_service = BatchInferenceService(inference_service, evaluator)
        >>> result = batch_service.process_files(file_contents_list)
    """
    
    def __init__(
        self,
        inference_service,
        decision_evaluator
    ):
        """Initialize the batch service.
        
        Args:
            inference_service: InferenceService instance for model inference
            decision_evaluator: DecisionEvaluator instance for acceptance rules
        """
        self.inference_service = inference_service
        self.decision_evaluator = decision_evaluator
    
    def process_single_file(
        self,
        file_name: str,
        file_contents: bytes,
        topk: int = 5
    ) -> BatchFileResult:
        """Process a single file through inference + decision engine.
        
        Args:
            file_name: Name of the file for identification
            file_contents: Raw bytes of the .pkl file
            topk: Number of top predictions to retrieve
        
        Returns:
            BatchFileResult with prediction or error
        """
        try:
            # Step 1: Run model inference
            inference_result = self.inference_service.infer_from_bytes(
                file_contents, 
                topk=topk
            )
            inference_dict = inference_result.to_dict()
            
            # Step 2: Extract top-1 prediction info
            top1 = inference_dict.get("top1", {})
            class_id = top1.get("new_class_id", 0)
            gloss = top1.get("gloss", "UNKNOWN")
            bucket = top1.get("bucket", "OTHER")
            confidence = top1.get("confidence", 0.0)
            is_other = top1.get("is_other", False)
            
            # Step 3: Apply acceptance rules via decision engine
            decision_result = self.decision_evaluator.process_from_inference_result(
                inference_dict
            )
            
            prediction_info = decision_result.get("prediction", {})
            accepted = prediction_info.get("accepted", False)
            reason = prediction_info.get("reason", "unknown")
            
            # Build prediction result
            prediction = BatchPrediction(
                class_id=class_id,
                gloss=gloss,
                bucket=bucket,
                confidence=confidence,
                accepted=accepted,
                reason=reason
            )
            
            return BatchFileResult(file_name=file_name, prediction=prediction)
            
        except Exception as e:
            logger.error(f"Failed to process {file_name}: {e}")
            return BatchFileResult(file_name=file_name, error=str(e))
    
    def process_batch(
        self,
        files: List[tuple],  # List of (file_name, file_contents)
        topk: int = 5
    ) -> BatchInferenceResult:
        """Process multiple files and build a semantic sequence.
        
        Files are processed in order. The sequence accumulates
        only accepted words, preserving submission order.
        
        Args:
            files: List of (file_name, file_contents) tuples
            topk: Number of top predictions per file
        
        Returns:
            BatchInferenceResult with all results and sequence
        """
        result = BatchInferenceResult()
        
        for file_name, file_contents in files:
            # Process individual file
            file_result = self.process_single_file(
                file_name=file_name,
                file_contents=file_contents,
                topk=topk
            )
            result.results.append(file_result)
            
            # Add to sequence if accepted
            if file_result.prediction and file_result.prediction.accepted:
                result.sequence.append(
                    gloss=file_result.prediction.gloss,
                    confidence=file_result.prediction.confidence
                )
        
        return result
    
    def reset_sequence_state(self) -> None:
        """Reset the decision evaluator's sequence state.
        
        Call this between batch requests if needed.
        """
        self.decision_evaluator.reset_sequence()
