"""
Type definitions for the decision engine.

Defines dataclasses for predictions, acceptance results,
and sequence state management.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Dict, Any, Optional


BucketType = Literal["HEAD", "MID", "OTHER"]


@dataclass
class PredictionInput:
    """Input for the decision engine from model inference.
    
    Attributes:
        class_id: Top-1 predicted class ID
        class_name: Human-readable class name (gloss)
        bucket: Class bucket (HEAD, MID, or OTHER)
        confidence: Top-1 confidence score
        topk_scores: List of top-k confidence scores
        topk_class_ids: List of top-k class IDs
        is_other: Whether this is the OTHER class
    """
    class_id: int
    class_name: str
    bucket: BucketType
    confidence: float
    topk_scores: List[float] = field(default_factory=list)
    topk_class_ids: List[int] = field(default_factory=list)
    is_other: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bucket": self.bucket,
            "confidence": self.confidence,
            "topk_scores": self.topk_scores,
            "topk_class_ids": self.topk_class_ids,
            "is_other": self.is_other
        }


@dataclass
class AcceptanceResult:
    """Result of word acceptance evaluation.
    
    Attributes:
        accepted: Whether the word was accepted
        reason: Human-readable reason for acceptance/rejection
        confidence: Confidence score of the prediction
        bucket: Class bucket
        rule_applied: Name of the rule that determined the result
    """
    accepted: bool
    reason: str
    confidence: float
    bucket: str
    rule_applied: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "confidence": self.confidence,
            "bucket": self.bucket,
            "rule_applied": self.rule_applied
        }


@dataclass
class SequenceItem:
    """An accepted word in the sequence.
    
    Attributes:
        gloss: Human-readable sign name
        class_id: Model class ID
        confidence: Confidence score when accepted
        bucket: Class bucket (HEAD or MID)
        position: Position in sequence (0-indexed)
    """
    gloss: str
    class_id: int
    confidence: float
    bucket: str
    position: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gloss": self.gloss,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bucket": self.bucket,
            "position": self.position
        }


@dataclass
class RejectedItem:
    """A rejected prediction with reason.
    
    Attributes:
        gloss: Predicted sign name
        class_id: Model class ID
        confidence: Confidence score
        bucket: Class bucket
        reason: Reason for rejection
        rule_applied: Which rule caused rejection
    """
    gloss: str
    class_id: int
    confidence: float
    bucket: str
    reason: str
    rule_applied: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gloss": self.gloss,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bucket": self.bucket,
            "reason": self.reason,
            "rule_applied": self.rule_applied
        }


@dataclass
class SequenceState:
    """Current state of the accepted sequence.
    
    Attributes:
        accepted: List of accepted words in order
        rejected: List of rejected predictions with reasons
    """
    accepted: List[SequenceItem] = field(default_factory=list)
    rejected: List[RejectedItem] = field(default_factory=list)
    
    @property
    def length(self) -> int:
        """Number of accepted words."""
        return len(self.accepted)
    
    @property
    def glosses(self) -> List[str]:
        """List of accepted glosses in order."""
        return [item.gloss for item in self.accepted]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accepted": [item.to_dict() for item in self.accepted],
            "rejected": [item.to_dict() for item in self.rejected],
            "length": self.length,
            "glosses": self.glosses
        }


@dataclass
class DecisionEngineConfig:
    """Configuration for the decision engine.
    
    Attributes:
        head_threshold: Minimum confidence for HEAD class
        mid_threshold: Minimum confidence for MID class
        margin_threshold: Minimum gap between top1 and top2
        reject_other: Whether to reject OTHER class
    
    Note:
        Current thresholds are set LOW for debugging/testing
        with a model that has ~20% accuracy. For production,
        consider: HEAD=0.45, MID=0.55, margin=0.10
    """
    # DEBUG: Low thresholds for testing with ~20% accuracy model
    head_threshold: float = 0.45   # Accept HEAD if confidence >= 5%
    mid_threshold: float = 0.55     # Accept MID if confidence >= 5%
    margin_threshold: float = 0.10  # Accept if top1-top2 >= 2%
    reject_other: bool = True      # Don't auto-reject OTHER class
    
    def get_threshold(self, bucket: str) -> float:
        """Get confidence threshold for a bucket."""
        if bucket == "HEAD":
            return self.head_threshold
        elif bucket == "MID":
            return self.mid_threshold
        else:
            return 1.0  # OTHER always rejected by threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "head_threshold": self.head_threshold,
            "mid_threshold": self.mid_threshold,
            "margin_threshold": self.margin_threshold,
            "reject_other": self.reject_other
        }
