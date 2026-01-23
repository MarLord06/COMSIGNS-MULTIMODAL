"""
Inference pipeline for ComSigns.

Provides minimal, robust inference without training dependencies.

Usage:
    >>> from inference import InferenceLoader, Predictor
    >>> 
    >>> # Load model
    >>> loader = InferenceLoader(
    ...     checkpoint_path=Path("experiments/run_001/checkpoints/best.pt")
    ... )
    >>> model, class_names, other_id = loader.load_all()
    >>> 
    >>> # Create predictor
    >>> predictor = Predictor(model, class_names, other_id)
    >>> 
    >>> # Run inference
    >>> result = predictor.predict(hand, body, face, lengths)
    >>> print(result.format_output())

Or use convenience function:
    >>> from inference import load_and_predict
    >>> result = load_and_predict(checkpoint_path, hand, body, face)
"""

from .types import (
    PredictionResult,
    TopKPrediction,
    InferenceConfig,
    ModelInfo
)

from .loader import (
    InferenceLoader,
    load_checkpoint_for_inference
)

from .predictor import (
    Predictor,
    create_predictor
)


__all__ = [
    # Types
    "PredictionResult",
    "TopKPrediction",
    "InferenceConfig",
    "ModelInfo",
    # Loader
    "InferenceLoader",
    "load_checkpoint_for_inference",
    # Predictor
    "Predictor",
    "create_predictor",
]
