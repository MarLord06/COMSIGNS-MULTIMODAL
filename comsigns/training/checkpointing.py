"""
Checkpointing and Best Model Selection for ComSigns Training.

This module provides robust, deterministic checkpoint management with:
- Periodic checkpoint saving (per epoch)
- Best model selection based on learned_words_count
- Training resumption support
- Compatible with baseline and TAIL→OTHER experiments

Selection criteria (in order of priority):
1. Maximize learned_words_count (semantic learning)
2. Maximize f1_macro (tiebreaker)
3. Minimize val_loss (final tiebreaker)

Example:
    >>> manager = CheckpointManager(output_dir=Path("experiments/run_001"))
    >>> 
    >>> for epoch in range(num_epochs):
    ...     train_loss = train_one_epoch(...)
    ...     val_metrics = validate(...)
    ...     
    ...     # Save periodic checkpoint
    ...     manager.save_checkpoint(
    ...         epoch=epoch,
    ...         model=model,
    ...         optimizer=optimizer,
    ...         metrics=val_metrics
    ...     )
    ...     
    ...     # Check and save if best
    ...     if manager.is_best(val_metrics):
    ...         manager.save_best(model, val_metrics)
    >>> 
    >>> # Load best model for inference
    >>> best_state = manager.load_best()
    >>> model.load_state_dict(best_state["model_state"])
"""

import json
import logging
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

PRIMARY_METRIC = "learned_words_count"
SECONDARY_METRIC = "f1_macro"
TERTIARY_METRIC = "val_loss"

# Metrics to maximize (vs minimize)
MAXIMIZE_METRICS = {"learned_words_count", "f1_macro", "accuracy", "accuracy_top5"}
MINIMIZE_METRICS = {"val_loss", "train_loss"}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CheckpointMetrics:
    """
    Metrics snapshot for checkpoint comparison.
    
    Attributes:
        epoch: Training epoch (0-indexed)
        val_loss: Validation loss
        f1_macro: Macro-averaged F1 score
        learned_words_count: Number of classes meeting learned criteria
        accuracy: Optional top-1 accuracy
        accuracy_top5: Optional top-5 accuracy
    """
    epoch: int
    val_loss: float
    f1_macro: float
    learned_words_count: int
    accuracy: Optional[float] = None
    accuracy_top5: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetrics":
        """Create from dictionary, handling missing optional fields."""
        return cls(
            epoch=data["epoch"],
            val_loss=data["val_loss"],
            f1_macro=data["f1_macro"],
            learned_words_count=data["learned_words_count"],
            accuracy=data.get("accuracy"),
            accuracy_top5=data.get("accuracy_top5")
        )


@dataclass
class BestModelInfo:
    """
    Information about the best model selection.
    
    Attributes:
        best_epoch: Epoch of the best model
        selection_criteria: List of metrics used for selection
        metrics: The metrics of the best model
        selected_at: Timestamp when best was selected
    """
    best_epoch: int
    selection_criteria: List[str]
    metrics: Dict[str, Any]
    selected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "selection_criteria": self.selection_criteria,
            "metrics": self.metrics,
            "selected_at": self.selected_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BestModelInfo":
        return cls(**data)
    
    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "BestModelInfo":
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """
    Manages checkpoint saving, best model selection, and training resumption.
    
    This class provides:
    - Periodic checkpoint saving (epoch_001.pt, epoch_002.pt, etc.)
    - Best model tracking and saving (best.pt)
    - Training state persistence
    - Resume functionality
    
    Selection criteria for best model (in priority order):
    1. Maximize learned_words_count
    2. Maximize f1_macro (tiebreaker)
    3. Minimize val_loss (final tiebreaker)
    
    Attributes:
        output_dir: Directory for all checkpoint files
        checkpoints_dir: Subdirectory for epoch checkpoints
        primary_metric: Main metric for selection (default: learned_words_count)
        secondary_metric: Tiebreaker metric (default: f1_macro)
        keep_last_n: Number of recent checkpoints to keep (0 = keep all)
    
    Example:
        >>> manager = CheckpointManager(Path("experiments/run_001"))
        >>> 
        >>> # During training
        >>> manager.save_checkpoint(epoch, model, optimizer, metrics)
        >>> if manager.is_best(metrics):
        ...     manager.save_best(model, metrics)
        >>> 
        >>> # For resumption
        >>> state = manager.load_checkpoint(epoch=5)
        >>> model.load_state_dict(state["model_state"])
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        primary_metric: str = PRIMARY_METRIC,
        secondary_metric: str = SECONDARY_METRIC,
        keep_last_n: int = 0
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            output_dir: Base directory for experiment outputs
            primary_metric: Main metric for best model selection
            secondary_metric: Tiebreaker metric
            keep_last_n: Number of recent checkpoints to keep (0 = keep all)
        """
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.primary_metric = primary_metric
        self.secondary_metric = secondary_metric
        self.keep_last_n = keep_last_n
        
        # Create directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best metrics
        self._best_metrics: Optional[CheckpointMetrics] = None
        self._best_info: Optional[BestModelInfo] = None
        
        # Load existing best if resuming
        self._load_existing_best()
    
    def _load_existing_best(self) -> None:
        """Load existing best model info if available."""
        best_info_path = self.output_dir / "best_model.json"
        if best_info_path.exists():
            try:
                self._best_info = BestModelInfo.load(best_info_path)
                self._best_metrics = CheckpointMetrics.from_dict({
                    "epoch": self._best_info.best_epoch,
                    **self._best_info.metrics
                })
                logger.info(f"Loaded existing best model info: epoch {self._best_info.best_epoch}")
            except Exception as e:
                logger.warning(f"Could not load existing best model info: {e}")
    
    # =========================================================================
    # Checkpoint Saving
    # =========================================================================
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        metrics: Dict[str, Any],
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a checkpoint for the given epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            model: PyTorch model
            optimizer: Optimizer
            metrics: Dictionary of metrics (must include val_loss, f1_macro, learned_words_count)
            scheduler: Optional learning rate scheduler
            extra_state: Optional additional state to save
        
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_path = self.checkpoints_dir / f"epoch_{epoch:03d}.pt"
        
        # Build checkpoint dict
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add scheduler state if present
        if scheduler is not None:
            checkpoint["scheduler_state"] = scheduler.state_dict()
        
        # Add extra state
        if extra_state is not None:
            checkpoint.update(extra_state)
        
        # Save
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint: {checkpoint_path}")
        
        # Clean up old checkpoints if needed
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if keep_last_n > 0."""
        if self.keep_last_n <= 0:
            return
        
        # Get all epoch checkpoints sorted by epoch number
        checkpoints = sorted(self.checkpoints_dir.glob("epoch_*.pt"))
        
        # Remove old ones, keeping best.pt untouched
        while len(checkpoints) > self.keep_last_n:
            oldest = checkpoints.pop(0)
            if oldest.name != "best.pt":
                oldest.unlink()
                logger.debug(f"Removed old checkpoint: {oldest}")
    
    # =========================================================================
    # Best Model Selection
    # =========================================================================
    
    def is_best(self, metrics: Union[Dict[str, Any], CheckpointMetrics]) -> bool:
        """
        Check if the given metrics represent the best model so far.
        
        Selection criteria (in order):
        1. Maximize learned_words_count
        2. Maximize f1_macro (tiebreaker)
        3. Minimize val_loss (final tiebreaker)
        
        Args:
            metrics: Current metrics (dict or CheckpointMetrics)
        
        Returns:
            True if this is the best model so far
        """
        # Convert dict to CheckpointMetrics if needed
        if isinstance(metrics, dict):
            current = self._dict_to_checkpoint_metrics(metrics)
        else:
            current = metrics
        
        # First model is always best
        if self._best_metrics is None:
            return True
        
        # Compare using selection criteria
        return self._compare_metrics(current, self._best_metrics)
    
    def _dict_to_checkpoint_metrics(self, metrics: Dict[str, Any]) -> CheckpointMetrics:
        """Convert metrics dict to CheckpointMetrics."""
        return CheckpointMetrics(
            epoch=metrics.get("epoch", 0),
            val_loss=metrics.get("val_loss", float("inf")),
            f1_macro=metrics.get("f1_macro", 0.0),
            learned_words_count=metrics.get("learned_words_count", 0),
            accuracy=metrics.get("accuracy"),
            accuracy_top5=metrics.get("accuracy_top5")
        )
    
    def _compare_metrics(
        self, 
        current: CheckpointMetrics, 
        best: CheckpointMetrics
    ) -> bool:
        """
        Compare two CheckpointMetrics using selection criteria.
        
        Returns True if current is better than best.
        """
        # Primary: learned_words_count (maximize)
        if current.learned_words_count > best.learned_words_count:
            return True
        if current.learned_words_count < best.learned_words_count:
            return False
        
        # Secondary: f1_macro (maximize)
        if current.f1_macro > best.f1_macro:
            return True
        if current.f1_macro < best.f1_macro:
            return False
        
        # Tertiary: val_loss (minimize)
        if current.val_loss < best.val_loss:
            return True
        
        return False
    
    def save_best(
        self,
        model: nn.Module,
        metrics: Union[Dict[str, Any], CheckpointMetrics],
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save the model as the best checkpoint.
        
        Args:
            model: PyTorch model
            metrics: Current metrics (dict or CheckpointMetrics)
            optimizer: Optional optimizer (for full checkpoint)
            scheduler: Optional scheduler
            extra_state: Optional additional state
        
        Returns:
            Path to best.pt
        """
        # Convert to CheckpointMetrics
        if isinstance(metrics, dict):
            ckpt_metrics = self._dict_to_checkpoint_metrics(metrics)
            metrics_dict = metrics
        else:
            ckpt_metrics = metrics
            metrics_dict = metrics.to_dict()
        
        # Update best tracking
        self._best_metrics = ckpt_metrics
        
        # Build best checkpoint
        best_checkpoint = {
            "epoch": ckpt_metrics.epoch,
            "model_state": model.state_dict(),
            "metrics": metrics_dict,
            "timestamp": datetime.now().isoformat()
        }
        
        if optimizer is not None:
            best_checkpoint["optimizer_state"] = optimizer.state_dict()
        
        if scheduler is not None:
            best_checkpoint["scheduler_state"] = scheduler.state_dict()
        
        if extra_state is not None:
            best_checkpoint.update(extra_state)
        
        # Save best.pt
        best_path = self.checkpoints_dir / "best.pt"
        torch.save(best_checkpoint, best_path)
        
        # Save best_model.json
        self._best_info = BestModelInfo(
            best_epoch=ckpt_metrics.epoch,
            selection_criteria=[
                self.primary_metric,
                self.secondary_metric,
                TERTIARY_METRIC
            ],
            metrics={
                "learned_words_count": ckpt_metrics.learned_words_count,
                "f1_macro": ckpt_metrics.f1_macro,
                "val_loss": ckpt_metrics.val_loss,
                "accuracy": ckpt_metrics.accuracy,
                "accuracy_top5": ckpt_metrics.accuracy_top5
            }
        )
        self._best_info.save(self.output_dir / "best_model.json")
        
        logger.info(
            f"New best model at epoch {ckpt_metrics.epoch}: "
            f"learned_words={ckpt_metrics.learned_words_count}, "
            f"f1_macro={ckpt_metrics.f1_macro:.4f}, "
            f"val_loss={ckpt_metrics.val_loss:.4f}"
        )
        
        return best_path
    
    def update_best_if_needed(
        self,
        model: nn.Module,
        metrics: Union[Dict[str, Any], CheckpointMetrics],
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if metrics are best and save if so.
        
        Convenience method combining is_best() and save_best().
        
        Returns:
            True if this was a new best model
        """
        if self.is_best(metrics):
            self.save_best(model, metrics, optimizer, scheduler, extra_state)
            return True
        return False
    
    # =========================================================================
    # Loading
    # =========================================================================
    
    def load_checkpoint(
        self, 
        epoch: Optional[int] = None,
        path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            epoch: Epoch number to load (loads epoch_XXX.pt)
            path: Direct path to checkpoint file
        
        Returns:
            Checkpoint dictionary
        
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if path is not None:
            checkpoint_path = Path(path)
        elif epoch is not None:
            checkpoint_path = self.checkpoints_dir / f"epoch_{epoch:03d}.pt"
        else:
            raise ValueError("Must specify either epoch or path")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint
    
    def load_best(self) -> Dict[str, Any]:
        """
        Load the best checkpoint.
        
        Returns:
            Best checkpoint dictionary
        
        Raises:
            FileNotFoundError: If best.pt doesn't exist
        """
        best_path = self.checkpoints_dir / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
        
        checkpoint = torch.load(best_path, map_location="cpu")
        logger.info(f"Loaded best checkpoint (epoch {checkpoint['epoch']})")
        
        return checkpoint
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent epoch checkpoint.
        
        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        checkpoints = sorted(self.checkpoints_dir.glob("epoch_*.pt"))
        if not checkpoints:
            return None
        
        latest_path = checkpoints[-1]
        return self.load_checkpoint(path=latest_path)
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def get_best_metrics(self) -> Optional[CheckpointMetrics]:
        """Get the metrics of the current best model."""
        return self._best_metrics
    
    def get_best_info(self) -> Optional[BestModelInfo]:
        """Get full info about the best model selection."""
        return self._best_info
    
    def has_checkpoints(self) -> bool:
        """Check if any checkpoints exist."""
        return len(list(self.checkpoints_dir.glob("epoch_*.pt"))) > 0
    
    def has_best(self) -> bool:
        """Check if a best checkpoint exists."""
        return (self.checkpoints_dir / "best.pt").exists()
    
    def list_checkpoints(self) -> List[Path]:
        """List all epoch checkpoints sorted by epoch."""
        return sorted(self.checkpoints_dir.glob("epoch_*.pt"))
    
    def get_latest_epoch(self) -> Optional[int]:
        """Get the latest checkpoint epoch number."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        
        # Parse epoch from filename (epoch_XXX.pt)
        latest = checkpoints[-1].stem  # "epoch_XXX"
        return int(latest.split("_")[1])
    
    def save_training_state(
        self,
        state: Dict[str, Any],
        filename: str = "training_state.json"
    ) -> Path:
        """
        Save training state metadata.
        
        Args:
            state: Dictionary with training configuration and final state
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        return path
    
    def get_summary(self) -> str:
        """Get a human-readable summary of checkpoint state."""
        lines = [
            "=" * 50,
            "CHECKPOINT SUMMARY",
            "=" * 50,
            f"Output dir: {self.output_dir}",
            f"Checkpoints: {len(self.list_checkpoints())}",
        ]
        
        if self._best_info is not None:
            lines.extend([
                f"Best epoch: {self._best_info.best_epoch}",
                f"Best learned_words: {self._best_info.metrics.get('learned_words_count', 'N/A')}",
                f"Best f1_macro: {self._best_info.metrics.get('f1_macro', 'N/A')}",
                f"Best val_loss: {self._best_info.metrics.get('val_loss', 'N/A')}",
            ])
        else:
            lines.append("Best model: Not yet selected")
        
        lines.append("=" * 50)
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def load_checkpoint_for_inference(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    device: Union[str, torch.device] = "cpu"
) -> nn.Module:
    """
    Load a checkpoint for inference.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        device: Device to load model to
    
    Returns:
        Model with loaded weights, in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path} for inference")
    return model


def load_checkpoint_for_training(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu"
) -> int:
    """
    Load a checkpoint for training resumption.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        optimizer: Optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load model to
    
    Returns:
        Next epoch number (checkpoint epoch + 1)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    
    start_epoch = checkpoint["epoch"] + 1
    
    logger.info(
        f"Resumed training from {checkpoint_path}, "
        f"starting at epoch {start_epoch}"
    )
    
    return start_epoch
