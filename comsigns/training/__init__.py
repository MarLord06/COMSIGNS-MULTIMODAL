"""
Training module for ComSigns sign language recognition.

Provides a minimal, functional trainer for end-to-end training
of the MultimodalEncoder + classifier pipeline.

Example:
    from comsigns.training import Trainer, TrainerConfig, SignLanguageClassifier
    
    model = SignLanguageClassifier(encoder, num_classes=100)
    config = TrainerConfig(epochs=10, device="cuda")
    trainer = Trainer(model, config)
    history = trainer.fit(train_loader)
"""

from .config import TrainerConfig
from .trainer import Trainer
from .classifier import SignLanguageClassifier
from .loops import train_one_epoch, train, validate_one_epoch
from .metrics import MetricsTracker, compute_accuracy, compute_topk_accuracy

__all__ = [
    "TrainerConfig",
    "Trainer",
    "SignLanguageClassifier",
    "train_one_epoch",
    "train",
    "validate_one_epoch",
    "MetricsTracker",
    "compute_accuracy",
    "compute_topk_accuracy",
]
