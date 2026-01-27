"""
Evaluator for the decision engine.

Glue logic that converts model outputs to PredictionInput,
applies rules, and updates the sequence manager.
"""

import logging
from typing import Dict, Any, Optional, List

from .types import (
    PredictionInput,
    AcceptanceResult,
    DecisionEngineConfig
)
from .rules import RuleEngine
from .sequence import SequenceManager

logger = logging.getLogger(__name__)


class DecisionEvaluator:
    """Evaluates predictions and manages the sequence.
    
    Combines:
    - Rule engine for acceptance decisions
    - Sequence manager for state tracking
    
    Example:
        >>> evaluator = DecisionEvaluator()
        >>> 
        >>> # Process a model output
        >>> result = evaluator.process_prediction(
        ...     class_id=28,
        ...     class_name="yo",
        ...     bucket="HEAD",
        ...     confidence=0.65,
        ...     topk_scores=[0.65, 0.20],
        ...     topk_class_ids=[28, 45]
        ... )
        >>> 
        >>> print(result["prediction"]["accepted"])  # True
        >>> print(result["sequence"]["length"])  # 1
    """
    
    def __init__(self, config: Optional[DecisionEngineConfig] = None):
        """Initialize the evaluator.
        
        Args:
            config: Configuration for rule thresholds
        """
        self.config = config or DecisionEngineConfig()
        self.rule_engine = RuleEngine(self.config)
        self.sequence_manager = SequenceManager()
    
    def process_prediction(
        self,
        class_id: int,
        class_name: str,
        bucket: str,
        confidence: float,
        topk_scores: Optional[List[float]] = None,
        topk_class_ids: Optional[List[int]] = None,
        is_other: bool = False
    ) -> Dict[str, Any]:
        """Process a single prediction through the decision engine.
        
        Args:
            class_id: Model output class ID
            class_name: Human-readable class name (gloss)
            bucket: Class bucket (HEAD, MID, OTHER)
            confidence: Top-1 confidence score
            topk_scores: List of top-k scores (optional)
            topk_class_ids: List of top-k class IDs (optional)
            is_other: Whether this is the OTHER class
        
        Returns:
            Dictionary with prediction result and sequence state
        """
        # Build PredictionInput
        prediction = PredictionInput(
            class_id=class_id,
            class_name=class_name,
            bucket=bucket,
            confidence=confidence,
            topk_scores=topk_scores or [confidence],
            topk_class_ids=topk_class_ids or [class_id],
            is_other=is_other
        )
        
        # Apply rules
        acceptance = self.rule_engine.evaluate(prediction)
        
        # Update sequence
        self.sequence_manager.add_prediction(acceptance, prediction)
        
        # Build response
        return {
            "prediction": {
                "gloss": class_name,
                "accepted": acceptance.accepted,
                "confidence": round(confidence, 4),
                "bucket": bucket,
                "reason": acceptance.reason,
                "rule_applied": acceptance.rule_applied
            },
            "sequence": self.sequence_manager.get_summary()
        }
    
    def process_from_inference_result(
        self,
        inference_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process an inference result from the API.
        
        Converts the standard inference response format
        to PredictionInput and processes it.
        
        Args:
            inference_result: Result from /infer endpoint
        
        Returns:
            Dictionary with prediction and sequence
        """
        top1 = inference_result.get("top1", {})
        topk = inference_result.get("topk", [])
        
        # Extract top-k scores and IDs
        topk_scores = [p.get("confidence", 0) for p in topk]
        topk_class_ids = [p.get("new_class_id", 0) for p in topk]
        
        return self.process_prediction(
            class_id=top1.get("new_class_id", 0),
            class_name=top1.get("gloss", ""),
            bucket=top1.get("bucket", "OTHER"),
            confidence=top1.get("confidence", 0),
            topk_scores=topk_scores,
            topk_class_ids=topk_class_ids,
            is_other=top1.get("is_other", False)
        )
    
    def get_sequence_state(self) -> Dict[str, Any]:
        """Get the current sequence state.
        
        Returns:
            Dictionary with sequence info
        """
        return self.sequence_manager.get_summary()
    
    def get_accepted_glosses(self) -> List[str]:
        """Get list of accepted glosses.
        
        Returns:
            List of gloss strings in order
        """
        return self.sequence_manager.get_accepted_glosses()
    
    def reset_sequence(self) -> None:
        """Reset the sequence state."""
        self.sequence_manager.reset()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration.
        
        Returns:
            Dictionary with config values
        """
        return self.config.to_dict()


# Global evaluator instance for API use
_evaluator_instance: Optional[DecisionEvaluator] = None


def get_evaluator(
    config: Optional[DecisionEngineConfig] = None
) -> DecisionEvaluator:
    """Get or create the global evaluator instance.
    
    Args:
        config: Configuration (only used on first call)
    
    Returns:
        DecisionEvaluator instance
    """
    global _evaluator_instance
    
    if _evaluator_instance is None:
        _evaluator_instance = DecisionEvaluator(config)
    
    return _evaluator_instance


def reset_evaluator() -> None:
    """Reset the global evaluator instance."""
    global _evaluator_instance
    _evaluator_instance = None
