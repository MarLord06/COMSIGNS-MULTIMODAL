"""
Type definitions for semantic resolution.

Pure dataclasses without PyTorch dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class SemanticClassInfo:
    """Information about a semantic class.
    
    Attributes:
        new_class_id: Model output class ID (after remapping)
        old_class_id: Original dataset class ID (before remapping)
        bucket: Class bucket (HEAD, MID, or OTHER)
        gloss: Human-readable sign name
        is_other: True if this is the collapsed OTHER class
    """
    new_class_id: int
    old_class_id: Optional[int]
    bucket: str  # "HEAD" | "MID" | "OTHER"
    gloss: str
    is_other: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "new_class_id": self.new_class_id,
            "old_class_id": self.old_class_id,
            "bucket": self.bucket,
            "gloss": self.gloss,
            "is_other": self.is_other
        }


@dataclass
class SemanticPrediction:
    """Resolved semantic prediction from model output.
    
    Transforms raw model output (new_class_id, score) into
    human-readable prediction with gloss and metadata.
    
    Attributes:
        gloss: Human-readable sign name (e.g., "YO", "HOLA")
        confidence: Model confidence score (0.0 to 1.0)
        bucket: Class bucket (HEAD, MID, or OTHER)
        old_class_id: Original dataset class ID (None for OTHER)
        new_class_id: Model output class ID
        is_other: True if prediction is the collapsed OTHER class
    
    Example:
        >>> pred = SemanticPrediction(
        ...     gloss="YO",
        ...     confidence=0.85,
        ...     bucket="HEAD",
        ...     old_class_id=319,
        ...     new_class_id=28,
        ...     is_other=False
        ... )
    """
    gloss: str
    confidence: float
    bucket: str
    old_class_id: Optional[int]
    new_class_id: int
    is_other: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "gloss": self.gloss,
            "confidence": self.confidence,
            "bucket": self.bucket,
            "old_class_id": self.old_class_id,
            "new_class_id": self.new_class_id,
            "is_other": self.is_other
        }
    
    def __repr__(self) -> str:
        return (
            f"SemanticPrediction("
            f"gloss='{self.gloss}', "
            f"confidence={self.confidence:.4f}, "
            f"bucket='{self.bucket}', "
            f"is_other={self.is_other})"
        )


@dataclass
class SemanticTopK:
    """Top-K semantic predictions.
    
    Attributes:
        predictions: List of SemanticPrediction in descending confidence order
        top1: Convenience access to top-1 prediction
    """
    predictions: List[SemanticPrediction]
    
    @property
    def top1(self) -> Optional[SemanticPrediction]:
        """Get top-1 prediction."""
        return self.predictions[0] if self.predictions else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "top1": self.top1.to_dict() if self.top1 else None
        }


@dataclass
class SemanticMappingStats:
    """Statistics about the semantic mapping.
    
    Attributes:
        num_classes_original: Original number of classes in dataset
        num_classes_remapped: Number of classes after remapping
        head_count: Number of HEAD classes
        mid_count: Number of MID classes
        tail_count: Number of TAIL classes (collapsed to OTHER)
        other_class_id: The new_class_id for OTHER
    """
    num_classes_original: int
    num_classes_remapped: int
    head_count: int
    mid_count: int
    tail_count: int
    other_class_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_classes_original": self.num_classes_original,
            "num_classes_remapped": self.num_classes_remapped,
            "head_count": self.head_count,
            "mid_count": self.mid_count,
            "tail_count": self.tail_count,
            "other_class_id": self.other_class_id
        }
