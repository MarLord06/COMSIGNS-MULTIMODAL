"""
Type definitions for inference pipeline.

Provides structured output types for predictions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TopKPrediction:
    """Single prediction in Top-K results.
    
    Attributes:
        rank: Position in Top-K (1-indexed)
        class_id: Predicted class ID
        class_name: Human-readable class name
        score: Softmax probability score
    """
    rank: int
    class_id: int
    class_name: str
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rank": self.rank,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": self.score
        }


@dataclass
class PredictionResult:
    """Complete prediction result from inference.
    
    Attributes:
        top1_class_id: Top-1 predicted class ID
        top1_class_name: Top-1 predicted class name
        top1_score: Top-1 softmax score
        topk: List of Top-K predictions
        is_other: True if prediction is the OTHER class
        raw_logits: Optional raw model output (before softmax)
        all_scores: Optional full softmax distribution
    
    Example:
        >>> result = predictor.predict(features)
        >>> print(f"Predicted: {result.top1_class_name} ({result.top1_score:.2%})")
        >>> if result.is_other:
        ...     print("  (Collapsed TAIL class)")
    """
    top1_class_id: int
    top1_class_name: str
    top1_score: float
    topk: List[TopKPrediction]
    is_other: bool
    raw_logits: Optional[List[float]] = None
    all_scores: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "top1_class_id": self.top1_class_id,
            "top1_class_name": self.top1_class_name,
            "top1_score": self.top1_score,
            "topk": [p.to_dict() for p in self.topk],
            "is_other": self.is_other
        }
    
    def format_output(self) -> str:
        """Format result as human-readable string."""
        lines = [
            "=" * 50,
            "Inference Result",
            "=" * 50,
            "",
            "Top-1 Prediction:",
            f"  Class ID   : {self.top1_class_id}",
            f"  Class Name : {self.top1_class_name}",
            f"  Score      : {self.top1_score:.4f}",
            "",
            f"Top-{len(self.topk)} Predictions:"
        ]
        
        for pred in self.topk:
            lines.append(
                f"  [{pred.rank}] {pred.class_name:<15} ({pred.score:.4f})"
            )
        
        lines.extend([
            "",
            "Prediction Summary:",
            f"  Is OTHER: {self.is_other}",
            "=" * 50
        ])
        
        return "\n".join(lines)


@dataclass
class InferenceConfig:
    """Configuration for inference.
    
    Attributes:
        device: Device to run inference on ("cpu" or "cuda")
        topk: Number of top predictions to return
        return_logits: Whether to include raw logits in result
        return_all_scores: Whether to include full score distribution
    """
    device: str = "cpu"
    topk: int = 5
    return_logits: bool = False
    return_all_scores: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device": self.device,
            "topk": self.topk,
            "return_logits": self.return_logits,
            "return_all_scores": self.return_all_scores
        }


@dataclass
class ModelInfo:
    """Information about loaded model.
    
    Attributes:
        checkpoint_path: Path to loaded checkpoint
        epoch: Training epoch of checkpoint
        num_classes: Number of output classes
        other_class_id: ID of OTHER class (-1 if not applicable)
        tail_to_other: Whether TAIL→OTHER was used
        metrics: Training metrics at checkpoint
    """
    checkpoint_path: str
    epoch: int
    num_classes: int
    other_class_id: int = -1
    tail_to_other: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "epoch": self.epoch,
            "num_classes": self.num_classes,
            "other_class_id": self.other_class_id,
            "tail_to_other": self.tail_to_other,
            "metrics": self.metrics
        }
