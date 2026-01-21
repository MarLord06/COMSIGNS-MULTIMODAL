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
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
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

    # =========================================================================
    # Per-Class Metrics Methods
    # =========================================================================
    
    def compute_per_class(
        self,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics (precision, recall, F1, support, accuracy).
        
        Args:
            class_names: Optional list of class names. If provided, keys will be
                        class names instead of class indices.
        
        Returns:
            Dictionary mapping class ID (or name) to metrics dict with keys:
            - precision: Precision for this class
            - recall: Recall for this class
            - f1: F1-score for this class
            - support: Number of true samples for this class
            - accuracy: Per-class accuracy (correct / support)
            - top5_acc: Top-5 accuracy for this class (if 5 in topk)
        
        Example:
            >>> tracker.compute_per_class()
            {
                0: {"precision": 0.8, "recall": 0.75, "f1": 0.77, "support": 20, "accuracy": 0.75},
                1: {"precision": 0.6, "recall": 0.5, "f1": 0.55, "support": 10, "accuracy": 0.5},
                ...
            }
        """
        if self._total_samples == 0:
            return {}
        
        if not HAS_SKLEARN:
            logger.warning("sklearn not available, returning empty per-class metrics")
            return {}
        
        # Concatenate accumulated tensors
        logits = torch.cat(self._all_logits, dim=0)
        labels = torch.cat(self._all_labels, dim=0)
        
        # Convert to numpy
        labels_np = labels.cpu().numpy()
        preds_np = logits.argmax(dim=1).cpu().numpy()
        
        # Get unique classes that appear in labels
        unique_classes = np.unique(labels_np)
        
        # Compute precision, recall, f1 per class
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_np,
            preds_np,
            labels=unique_classes,
            average=None,
            zero_division=0.0
        )
        
        # Compute per-class accuracy
        per_class_accuracy = {}
        for cls in unique_classes:
            mask = labels_np == cls
            if mask.sum() > 0:
                per_class_accuracy[cls] = (preds_np[mask] == cls).mean()
            else:
                per_class_accuracy[cls] = 0.0
        
        # Compute top-k per class if requested
        topk_per_class = {}
        for k in self.topk:
            topk_per_class[k] = self._compute_topk_per_class(logits, labels, k)
        
        # Build results dictionary
        results = {}
        for i, cls in enumerate(unique_classes):
            cls_key = class_names[cls] if class_names and cls < len(class_names) else int(cls)
            
            metrics_dict = {
                "class_id": int(cls),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "accuracy": float(per_class_accuracy[cls])
            }
            
            # Add top-k accuracies
            for k in self.topk:
                metrics_dict[f"top{k}_acc"] = topk_per_class[k].get(int(cls), 0.0)
            
            results[cls_key] = metrics_dict
        
        return results
    
    def _compute_topk_per_class(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        k: int
    ) -> Dict[int, float]:
        """
        Compute Top-K accuracy per class.
        
        Args:
            logits: All logits [N, num_classes]
            labels: All labels [N]
            k: Value of K for top-k accuracy
        
        Returns:
            Dict mapping class_id to top-k accuracy for that class
        """
        k = min(k, logits.size(1))
        _, topk_indices = logits.topk(k, dim=1, largest=True, sorted=False)
        correct = topk_indices.eq(labels.unsqueeze(1)).any(dim=1)
        
        labels_np = labels.cpu().numpy()
        correct_np = correct.cpu().numpy()
        
        unique_classes = np.unique(labels_np)
        topk_per_class = {}
        
        for cls in unique_classes:
            mask = labels_np == cls
            if mask.sum() > 0:
                topk_per_class[int(cls)] = float(correct_np[mask].mean())
            else:
                topk_per_class[int(cls)] = 0.0
        
        return topk_per_class
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix from accumulated predictions.
        
        Returns:
            Confusion matrix as numpy array [num_classes, num_classes].
            Entry [i, j] = number of samples with true label i predicted as j.
            Returns empty array if no samples or sklearn not available.
        """
        if self._total_samples == 0:
            return np.array([])
        
        if not HAS_SKLEARN:
            logger.warning("sklearn not available for confusion matrix")
            return np.array([])
        
        labels = torch.cat(self._all_labels, dim=0).cpu().numpy()
        logits = torch.cat(self._all_logits, dim=0)
        preds = logits.argmax(dim=1).cpu().numpy()
        
        # Use all class labels for full matrix
        return confusion_matrix(
            labels,
            preds,
            labels=list(range(self.num_classes))
        )
    
    def get_worst_classes(
        self,
        k: int = 10,
        metric: str = "f1",
        class_names: Optional[List[str]] = None,
        min_support: int = 1
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Get K classes with worst performance on specified metric.
        
        Args:
            k: Number of worst classes to return
            metric: Metric to sort by ("f1", "precision", "recall", "accuracy")
            class_names: Optional class name mapping
            min_support: Minimum samples required to be considered
        
        Returns:
            List of dicts with class info, sorted from worst to best.
            Each dict contains: class_id, name (if available), metric value, support
        """
        per_class = self.compute_per_class(class_names)
        
        if not per_class:
            return []
        
        # Filter by minimum support and valid metric
        valid_classes = []
        for cls_key, metrics in per_class.items():
            if metrics["support"] >= min_support and metric in metrics:
                valid_classes.append({
                    "class_key": cls_key,
                    "class_id": metrics["class_id"],
                    "name": cls_key if isinstance(cls_key, str) else None,
                    metric: metrics[metric],
                    "support": metrics["support"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"]
                })
        
        # Sort by metric (ascending = worst first)
        valid_classes.sort(key=lambda x: x[metric])
        
        return valid_classes[:k]
    
    def get_best_classes(
        self,
        k: int = 10,
        metric: str = "f1",
        class_names: Optional[List[str]] = None,
        min_support: int = 1
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Get K classes with best performance on specified metric.
        
        Args:
            k: Number of best classes to return
            metric: Metric to sort by ("f1", "precision", "recall", "accuracy")
            class_names: Optional class name mapping
            min_support: Minimum samples required to be considered
        
        Returns:
            List of dicts with class info, sorted from best to worst.
        """
        per_class = self.compute_per_class(class_names)
        
        if not per_class:
            return []
        
        valid_classes = []
        for cls_key, metrics in per_class.items():
            if metrics["support"] >= min_support and metric in metrics:
                valid_classes.append({
                    "class_key": cls_key,
                    "class_id": metrics["class_id"],
                    "name": cls_key if isinstance(cls_key, str) else None,
                    metric: metrics[metric],
                    "support": metrics["support"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"]
                })
        
        # Sort by metric (descending = best first)
        valid_classes.sort(key=lambda x: x[metric], reverse=True)
        
        return valid_classes[:k]
    
    def get_class_distribution_analysis(
        self,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Union[int, float, Dict]]:
        """
        Analyze class distribution and coverage in accumulated data.
        
        Returns:
            Dictionary with:
            - total_samples: Total number of samples
            - num_classes_seen: Number of unique classes in data
            - num_classes_defined: Total defined classes (num_classes)
            - coverage_ratio: Proportion of classes seen
            - samples_per_class: Dict mapping class to sample count
            - class_imbalance_ratio: max_support / min_support
        """
        if self._total_samples == 0:
            return {
                "total_samples": 0,
                "num_classes_seen": 0,
                "num_classes_defined": self.num_classes,
                "coverage_ratio": 0.0,
                "samples_per_class": {},
                "class_imbalance_ratio": 0.0,
                "min_support": 0,
                "max_support": 0,
                "mean_support": 0.0,
                "std_support": 0.0
            }
        
        labels = torch.cat(self._all_labels, dim=0).cpu().numpy()
        unique, counts = np.unique(labels, return_counts=True)
        
        samples_per_class = {}
        for cls, count in zip(unique, counts):
            cls_key = class_names[cls] if class_names and cls < len(class_names) else int(cls)
            samples_per_class[cls_key] = int(count)
        
        min_support = int(counts.min())
        max_support = int(counts.max())
        imbalance_ratio = max_support / min_support if min_support > 0 else float('inf')
        
        return {
            "total_samples": int(self._total_samples),
            "num_classes_seen": int(len(unique)),
            "num_classes_defined": self.num_classes,
            "coverage_ratio": len(unique) / self.num_classes if self.num_classes > 0 else 0.0,
            "samples_per_class": samples_per_class,
            "class_imbalance_ratio": float(imbalance_ratio),
            "min_support": min_support,
            "max_support": max_support,
            "mean_support": float(counts.mean()),
            "std_support": float(counts.std())
        }
    
    def format_per_class_report(
        self,
        class_names: Optional[List[str]] = None,
        sort_by: str = "support",
        ascending: bool = False,
        top_n: Optional[int] = None
    ) -> str:
        """
        Format per-class metrics as a human-readable report.
        
        Args:
            class_names: Optional class name mapping
            sort_by: Column to sort by (support, f1, precision, recall, accuracy)
            ascending: Sort order
            top_n: Only show top N classes (None for all)
        
        Returns:
            Formatted string report
        """
        per_class = self.compute_per_class(class_names)
        
        if not per_class:
            return "No samples accumulated yet."
        
        # Convert to list for sorting
        rows = []
        for cls_key, metrics in per_class.items():
            name = cls_key if isinstance(cls_key, str) else f"Class {cls_key}"
            rows.append({
                "name": name,
                "class_id": metrics["class_id"],
                **{k: v for k, v in metrics.items() if k != "class_id"}
            })
        
        # Sort
        rows.sort(key=lambda x: x.get(sort_by, 0), reverse=not ascending)
        
        if top_n:
            rows = rows[:top_n]
        
        # Format report
        lines = []
        lines.append("=" * 80)
        lines.append("PER-CLASS METRICS REPORT")
        lines.append("=" * 80)
        lines.append(
            f"{'Class':<30} {'Support':>8} {'Prec':>7} {'Recall':>7} "
            f"{'F1':>7} {'Acc':>7} {'Top5':>7}"
        )
        lines.append("-" * 80)
        
        for row in rows:
            name = row["name"][:28] if len(row["name"]) > 28 else row["name"]
            top5 = row.get("top5_acc", 0.0)
            lines.append(
                f"{name:<30} {row['support']:>8} {row['precision']:>7.3f} "
                f"{row['recall']:>7.3f} {row['f1']:>7.3f} {row['accuracy']:>7.3f} "
                f"{top5:>7.3f}"
            )
        
        lines.append("-" * 80)
        
        # Summary statistics
        f1_scores = [r["f1"] for r in rows]
        lines.append(f"Total classes: {len(rows)}")
        lines.append(f"Mean F1: {np.mean(f1_scores):.3f} (std: {np.std(f1_scores):.3f})")
        lines.append(f"Classes with F1 > 0.5: {sum(1 for f in f1_scores if f > 0.5)}/{len(rows)}")
        lines.append(f"Classes with F1 = 0.0: {sum(1 for f in f1_scores if f == 0.0)}/{len(rows)}")
        lines.append("=" * 80)
        
        return "\n".join(lines)


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
