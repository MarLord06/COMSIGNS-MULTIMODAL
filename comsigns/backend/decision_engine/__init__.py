"""
Decision Engine for ComSigns.

Implements word acceptance rules and sequence-aware responses
for validating multiple isolated-word videos as a coherent sequence.

Usage:
    from backend.decision_engine import DecisionEvaluator, get_evaluator
    
    # Option 1: Create evaluator directly
    evaluator = DecisionEvaluator()
    result = evaluator.process_prediction(
        class_id=28,
        class_name="yo",
        bucket="HEAD",
        confidence=0.65,
        topk_scores=[0.65, 0.20]
    )
    
    # Option 2: Use global singleton
    evaluator = get_evaluator()
    result = evaluator.process_from_inference_result(inference_response)

Response format:
    {
        "prediction": {
            "gloss": "yo",
            "accepted": true,
            "confidence": 0.65,
            "bucket": "HEAD",
            "reason": "Accepted: HEAD class with 65% confidence"
        },
        "sequence": {
            "accepted": [{"gloss": "yo", "confidence": 0.65, "bucket": "HEAD"}],
            "length": 1,
            "glosses": ["yo"]
        }
    }
"""

from .types import (
    PredictionInput,
    AcceptanceResult,
    SequenceItem,
    RejectedItem,
    SequenceState,
    DecisionEngineConfig
)

from .rules import (
    RuleEngine,
    evaluate_prediction,
    RULE_NAMES
)

from .sequence import SequenceManager

from .evaluator import (
    DecisionEvaluator,
    get_evaluator,
    reset_evaluator
)


__all__ = [
    # Types
    "PredictionInput",
    "AcceptanceResult",
    "SequenceItem",
    "RejectedItem",
    "SequenceState",
    "DecisionEngineConfig",
    # Rules
    "RuleEngine",
    "evaluate_prediction",
    "RULE_NAMES",
    # Sequence
    "SequenceManager",
    # Evaluator
    "DecisionEvaluator",
    "get_evaluator",
    "reset_evaluator",
]
