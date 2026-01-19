"""
Training configuration dataclass.

Defines all hyperparameters and settings for the training loop
in a single, type-safe dataclass.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import torch


@dataclass
class TrainerConfig:
    """
    Configuration for the Trainer.
    
    All training hyperparameters and settings are defined here
    to ensure reproducibility and easy experimentation.
    
    Attributes:
        batch_size: Number of samples per batch (used for reference only,
                    actual batch size is determined by DataLoader)
        learning_rate: Initial learning rate for optimizer
        weight_decay: L2 regularization factor for AdamW
        epochs: Number of training epochs
        device: Device to train on ("cuda", "cpu", "mps", or "auto")
        log_every_n_steps: Log loss every N training steps
        overfit_single_batch: If True, train on same batch repeatedly (debug mode)
        gradient_clip_val: Max gradient norm for clipping (None to disable)
        seed: Random seed for reproducibility (None for no seeding)
    
    Example:
        >>> config = TrainerConfig(epochs=20, learning_rate=3e-4)
        >>> config.device
        'auto'
    """
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 10
    device: str = "auto"
    log_every_n_steps: int = 10
    overfit_single_batch: bool = False
    gradient_clip_val: Optional[float] = 1.0
    seed: Optional[int] = 42
    
    def __post_init__(self):
        """Validate and resolve configuration."""
        if self.device == "auto":
            self.device = self._detect_device()
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
    
    @staticmethod
    def _detect_device() -> str:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def get_torch_device(self) -> torch.device:
        """Return torch.device object."""
        return torch.device(self.device)
