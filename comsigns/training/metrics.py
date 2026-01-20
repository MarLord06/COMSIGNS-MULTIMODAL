"""
Metrics module for sign language classification evaluation.

This module provides a decoupled, reusable metrics tracking system
that accumulates predictions across batches and computes comprehensive
classification metrics at the end of each epoch.

Design Principles:
- Decoupled from trainer (only receives tensors)
- Accumulates data across batches
- Computes metrics lazily on demand
- Handles edge cases gracefully

Example:
    >>> metrics = MetricsTracker(num_classes=505, topk=(1, 5, 10))
    >>> for batch in val_loader:
    ...     logits = model(batch)
    ...     metrics.update(logits, batch["labels"])
    >>> results = metrics.compute()
    >>> print(f"Top-5 Accuracy: {results['top5_acc']:.2%}")
    >>> metrics.reset()  # Ready for next epoch
"""

from typing import Dict, List, Optional, Tuple, Union
import logging

import torch
import numpy as np

try:
    from sklearn.metrics import precision_recall_fscore_support
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Accumulates model predictions and computes classification metrics.
    
    This class is designed to be used in a validation loop where predictions
    are accumulated batch by batch, and metrics are computed at the end
    of the epoch.
    
    Metrics computed:
    - Accuracy (global)
    - Top-K Accuracy (configurable K values)
    - Precision (macro average)
    - Recall (macro average)
    - F1-Score (macro average)
    
    Attributes:
        num_classes: Number of classes in the classification task
        topk: Tuple of K values for Top-K accuracy computation
        device: Device to store accumulated tensors
    
    Example:
        >>> tracker = MetricsTracker(num_classes=100, topk=(1, 5, 10))
        >>> 
        >>> # During validation
        >>> for batch in val_loader:
        ...     with torch.no_grad():
        ...         logits = model(batch["input"])
        ...         tracker.update(logits, batch["labels"])
        >>> 
        >>> # At end of epoch
        >>> results = tracker.compute()
        >>> print(results)
        {'accuracy': 0.85, 'top1_acc': 0.85, 'top5_acc': 0.95, ...}
        >>> 
        >>> # Reset for next epoch
        >>> tracker.reset()
    """
    
    def __init__(
        self,
        num_classes: int,
        topk: Tuple[int, ...] = (1, 5, 10),
        device: Union[str, torch.device] = "cpu"
    ):
        """
        Initialize the metrics tracker.
        
        Args:
            num_classes: Total number of classes in the classification task.
                        Used to validate K values and compute per-class metrics.
            topk: Tuple of K values for Top-K accuracy. Each K must be
                 positive and <= num_classes. Default: (1, 5, 10)
            device: Device for storing accumulated tensors. Use "cpu" for
                   memory efficiency or match model device for speed.
        
        Raises:
            ValueError: If num_classes <= 0 or any K value is invalid
        """
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        
        self.num_classes = num_classes
        self.device = torch.device(device)
        
        # Validate and store topk values (clamp to num_classes)
        self.topk = tuple(
            min(k, num_classes) for k in topk if k > 0
        )
        if not self.topk:
            self.topk = (1,)  # Default to at least top-1
        
        # Accumulators for predictions and labels
        self._all_logits: List[torch.Tensor] = []
        self._all_labels: List[torch.Tensor] = []
        self._total_samples: int = 0
        
        logger.debug(
            f"MetricsTracker initialized: {num_classes} classes, "
            f"topk={self.topk}, device={self.device}"
        )
    
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> None:
        """
        Accumulate a batch of predictions and labels.
        
        This method should be called for each batch during validation.
        The logits and labels are stored for later metric computation.
        
        Args:
            logits: Model output logits of shape [batch_size, num_classes].
                   Raw logits (before softmax) are expected.
            labels: Ground truth labels of shape [batch_size].
                   Integer class indices (0 to num_classes-1).
        
        Note:
            - Tensors are moved to the tracker's device
            - Tensors are detached from computation graph
            - This method should be called within torch.no_grad()
        
        Example:
            >>> with torch.no_grad():
            ...     logits = model(inputs)
            ...     tracker.update(logits, labels)
        """
        # Validate shapes
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D [B, C], got shape {logits.shape}")
        if labels.dim() != 1:
            raise ValueError(f"labels must be 1D [B], got shape {labels.shape}")
        if logits.size(0) != labels.size(0):
            raise ValueError(
                f"Batch size mismatch: logits={logits.size(0)}, labels={labels.size(0)}"
            )
        
        # Detach and move to tracker device (CPU to save GPU memory)
        logits = logits.detach().to(self.device)
        labels = labels.detach().to(self.device)
        
        self._all_logits.append(logits)
        self._all_labels.append(labels)
        self._total_samples += labels.size(0)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions.
        
        This method should be called at the end of the validation epoch
        after all batches have been processed via update().
        
        Returns:
            Dictionary containing all computed metrics:
            - "accuracy": Global accuracy (same as top1_acc)
            - "top{K}_acc": Top-K accuracy for each K in self.topk
            - "precision_macro": Macro-averaged precision
            - "recall_macro": Macro-averaged recall
            - "f1_macro": Macro-averaged F1 score
            
            All values are floats in range [0.0, 1.0].
            Returns zeros for all metrics if no data accumulated.
        
        Example:
            >>> results = tracker.compute()
            >>> print(f"Accuracy: {results['accuracy']:.2%}")
            >>> print(f"F1 Score: {results['f1_macro']:.4f}")
        """
        # Handle empty accumulator
        if self._total_samples == 0:
            logger.warning("compute() called with no accumulated data")
            return self._empty_results()
        
        # Concatenate all batches
        all_logits = torch.cat(self._all_logits, dim=0)  # [N, C]
        all_labels = torch.cat(self._all_labels, dim=0)  # [N]
        
        # Get predictions (argmax of logits)
        predictions = all_logits.argmax(dim=1)  # [N]
        
        results: Dict[str, float] = {}
        
        # === Top-K Accuracy (PyTorch) ===
        for k in self.topk:
            topk_acc = self._compute_topk_accuracy(all_logits, all_labels, k)
            results[f"top{k}_acc"] = topk_acc
        
        # Accuracy is same as top-1
        results["accuracy"] = results.get("top1_acc", results[f"top{self.topk[0]}_acc"])
        
        # === Precision / Recall / F1 (sklearn) ===
        if HAS_SKLEARN:
            labels_np = all_labels.cpu().numpy()
            preds_np = predictions.cpu().numpy()
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=labels_np,
                y_pred=preds_np,
                average='macro',
                zero_division=0,  # Handle classes with no predictions
                labels=range(self.num_classes)  # Include all classes
            )
            
            results["precision_macro"] = float(precision)
            results["recall_macro"] = float(recall)
            results["f1_macro"] = float(f1)
        else:
            logger.warning(
                "sklearn not available, precision/recall/f1 will be 0.0. "
                "Install with: pip install scikit-learn"
            )
            results["precision_macro"] = 0.0
            results["recall_macro"] = 0.0
            results["f1_macro"] = 0.0
        
        return results
    
    def reset(self) -> None:
        """
        Reset the tracker for a new epoch.
        
        Clears all accumulated logits and labels. Should be called
        at the start of each validation epoch or after compute().
        
        Example:
            >>> results = tracker.compute()
            >>> # Log or save results...
            >>> tracker.reset()  # Ready for next epoch
        """
        self._all_logits.clear()
        self._all_labels.clear()
        self._total_samples = 0
        
        logger.debug("MetricsTracker reset")
    
    def _compute_topk_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        k: int
    ) -> float:
        """
        Compute Top-K accuracy using PyTorch operations.
        
        Top-K accuracy measures whether the true label is among
        the K highest-scoring predictions.
        
        Args:
            logits: Logits tensor of shape [N, C]
            labels: Labels tensor of shape [N]
            k: Number of top predictions to consider
        
        Returns:
            Top-K accuracy as a float in [0.0, 1.0]
        """
        # Clamp k to valid range
        k = min(k, logits.size(1))
        
        # Get top-k predictions [N, k]
        _, topk_indices = logits.topk(k, dim=1, largest=True, sorted=False)
        
        # Check if true label is in top-k
        # Expand labels to [N, 1] for comparison
        labels_expanded = labels.unsqueeze(1)  # [N, 1]
        correct = topk_indices.eq(labels_expanded).any(dim=1)  # [N]
        
        # Compute accuracy
        accuracy = correct.float().mean().item()
        
        return accuracy
    
    def _empty_results(self) -> Dict[str, float]:
        """Return zero-valued results dictionary."""
        results = {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0
        }
        for k in self.topk:
            results[f"top{k}_acc"] = 0.0
        return results
    
    @property
    def num_samples(self) -> int:
        """Number of samples accumulated so far."""
        return self._total_samples
    
    def __repr__(self) -> str:
        return (
            f"MetricsTracker(num_classes={self.num_classes}, "
            f"topk={self.topk}, samples={self._total_samples})"
        )


# =============================================================================
# Standalone Functions (for direct use without tracker)
# =============================================================================

def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        logits: Model logits [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
    
    Returns:
        Accuracy as float in [0.0, 1.0]
    """
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).float()
    return correct.mean().item()


def compute_topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int
) -> float:
    """
    Compute Top-K accuracy.
    
    Args:
        logits: Model logits [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        k: Number of top predictions to consider
    
    Returns:
        Top-K accuracy as float in [0.0, 1.0]
    """
    k = min(k, logits.size(1))
    _, topk_indices = logits.topk(k, dim=1, largest=True, sorted=False)
    correct = topk_indices.eq(labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()
