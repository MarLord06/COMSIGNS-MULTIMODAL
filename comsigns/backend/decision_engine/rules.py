"""
Word acceptance rules for the decision engine.

Implements deterministic rules for accepting/rejecting predictions
based on confidence, bucket, and margin criteria.
"""

import logging
from typing import Optional

from .types import (
    PredictionInput,
    AcceptanceResult,
    DecisionEngineConfig
)

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_CONFIG = DecisionEngineConfig()


class RuleEngine:
    """Evaluates predictions against acceptance rules.
    
    Rules are applied in order:
    1. Reject if bucket == "OTHER" (when reject_other=True)
    2. Reject if confidence < threshold_by_bucket
       - HEAD: configurable (default 0.10)
       - MID: configurable (default 0.10)
    3. Reject if margin (top1 - top2) < margin_threshold
    4. Otherwise ACCEPT
    
    Example:
        >>> engine = RuleEngine()
        >>> pred = PredictionInput(
        ...     class_id=28,
        ...     class_name="yo",
        ...     bucket="HEAD",
        ...     confidence=0.65,
        ...     topk_scores=[0.65, 0.20, 0.10]
        ... )
        >>> result = engine.evaluate(pred)
        >>> print(result.accepted)  # True
    """
    
    def __init__(self, config: Optional[DecisionEngineConfig] = None):
        """Initialize the rule engine.
        
        Args:
            config: Configuration for thresholds. Uses defaults if None.
        """
        self.config = config or DEFAULT_CONFIG
    
    def evaluate(self, prediction: PredictionInput) -> AcceptanceResult:
        """Evaluate a prediction against all rules.
        
        Args:
            prediction: The prediction to evaluate
        
        Returns:
            AcceptanceResult with acceptance decision and reason
        """
        # Rule 1: Reject OTHER bucket (if configured)
        if self.config.reject_other and (prediction.bucket == "OTHER" or prediction.is_other):
            return AcceptanceResult(
                accepted=False,
                reason="Rejected: OTHER class (collapsed tail)",
                confidence=prediction.confidence,
                bucket=prediction.bucket,
                rule_applied="reject_other"
            )
        
        # Rule 2: Check confidence threshold by bucket
        threshold = self.config.get_threshold(prediction.bucket)
        if prediction.confidence < threshold:
            return AcceptanceResult(
                accepted=False,
                reason=f"Rejected: Confidence {prediction.confidence:.2%} < {threshold:.0%} threshold for {prediction.bucket}",
                confidence=prediction.confidence,
                bucket=prediction.bucket,
                rule_applied="low_confidence"
            )
        
        # Rule 3: Check margin between top1 and top2
        if len(prediction.topk_scores) >= 2:
            top1 = prediction.topk_scores[0]
            top2 = prediction.topk_scores[1]
            margin = top1 - top2
            
            if margin < self.config.margin_threshold:
                return AcceptanceResult(
                    accepted=False,
                    reason=f"Rejected: Margin {margin:.2%} < {self.config.margin_threshold:.0%} (ambiguous prediction)",
                    confidence=prediction.confidence,
                    bucket=prediction.bucket,
                    rule_applied="low_margin"
                )
        
        # All rules passed - ACCEPT
        return AcceptanceResult(
            accepted=True,
            reason=f"Accepted: {prediction.bucket} class with {prediction.confidence:.2%} confidence",
            confidence=prediction.confidence,
            bucket=prediction.bucket,
            rule_applied="accepted"
        )


def evaluate_prediction(
    prediction: PredictionInput,
    config: Optional[DecisionEngineConfig] = None
) -> AcceptanceResult:
    """Convenience function to evaluate a single prediction.
    
    Args:
        prediction: The prediction to evaluate
        config: Optional configuration for thresholds
    
    Returns:
        AcceptanceResult with acceptance decision
    """
    engine = RuleEngine(config)
    return engine.evaluate(prediction)


# Rule names for reference
RULE_NAMES = {
    "reject_other": "OTHER class rejection",
    "low_confidence": "Low confidence rejection",
    "low_margin": "Low margin (ambiguous) rejection",
    "accepted": "All rules passed"
}
