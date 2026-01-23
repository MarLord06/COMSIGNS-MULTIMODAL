"""
Remapped Metrics Module for TAIL → OTHER Experiment.

This module provides metrics tracking with awareness of class remapping,
specifically designed for the TAIL → OTHER experiment.

Key Components:
- RemappedMetricsTracker: Track metrics with bucket awareness
- Prediction diagnostics (pct_predictions_other, entropy)
- Bucket confusion matrix (3×3: HEAD/MID/OTHER)

⚠️ INTERPRETATION WARNINGS:
- HEAD has only 3 classes - strong improvements may not generalize
- Accuracy can inflate falsely if model predicts mostly OTHER
- Always check pct_predictions_other and entropy_of_predictions
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    from ..remapping import ClassRemapper, Bucket
except ImportError:
    from training.remapping import ClassRemapper, Bucket

logger = logging.getLogger(__name__)


@dataclass
class BucketMetricsResult:
    """Metrics for a single bucket (HEAD, MID, or OTHER)."""
    bucket: str
    num_classes: int
    num_samples: int
    accuracy: float
    top5_accuracy: float
    precision: float
    recall: float
    f1: float
    
    def to_dict(self) -> Dict:
        return {
            "bucket": self.bucket,
            "num_classes": self.num_classes,
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "top5_accuracy": self.top5_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1
        }


@dataclass
class PredictionDiagnostics:
    """Diagnostics for prediction distribution.
    
    ⚠️ IMPORTANT: These metrics help detect false accuracy inflation.
    If the model predicts mostly OTHER, accuracy may be high but
    the model isn't learning useful semantics.
    """
    pct_predictions_other: float    # % of predictions that are OTHER
    entropy_of_predictions: float   # Diversity of predictions (higher = more diverse)
    num_unique_predictions: int     # How many distinct classes predicted
    prediction_distribution: Dict[int, int]  # class_id -> count
    
    def to_dict(self) -> Dict:
        return {
            "pct_predictions_other": self.pct_predictions_other,
            "entropy_of_predictions": self.entropy_of_predictions,
            "num_unique_predictions": self.num_unique_predictions,
            "prediction_distribution": self.prediction_distribution
        }
    
    def is_valid(self, max_pct_other: float = 0.6) -> bool:
        """Check if predictions are valid (not dominated by OTHER).
        
        Args:
            max_pct_other: Maximum acceptable percentage of OTHER predictions
        
        Returns:
            True if pct_predictions_other <= max_pct_other
        """
        return self.pct_predictions_other <= max_pct_other


@dataclass
class BucketConfusionMatrix:
    """3×3 confusion matrix at bucket level (HEAD/MID/OTHER).
    
    This is the RIGHT way to visualize confusion - NOT a 51×51 matrix.
    
    Matrix layout:
                    Predicted
                 HEAD  MID  OTHER
    Actual HEAD   [.]  [.]   [.]
    Actual MID    [.]  [.]   [.]
    Actual OTHER  [.]  [.]   [.]
    """
    matrix: np.ndarray  # 3×3 array
    labels: List[str] = field(default_factory=lambda: ["HEAD", "MID", "OTHER"])
    
    # Aggregated metrics
    head_to_other: int = 0   # HEAD actual → predicted OTHER
    mid_to_other: int = 0    # MID actual → predicted OTHER
    other_to_head: int = 0   # OTHER actual → predicted HEAD
    other_to_mid: int = 0    # OTHER actual → predicted MID
    within_head: int = 0     # HEAD actual → predicted HEAD (correct bucket)
    within_mid: int = 0      # MID actual → predicted MID (correct bucket)
    within_other: int = 0    # OTHER actual → predicted OTHER (correct bucket)
    
    def to_dict(self) -> Dict:
        return {
            "matrix": self.matrix.tolist(),
            "labels": self.labels,
            "aggregated": {
                "head_to_other": self.head_to_other,
                "mid_to_other": self.mid_to_other,
                "other_to_head": self.other_to_head,
                "other_to_mid": self.other_to_mid,
                "within_head": self.within_head,
                "within_mid": self.within_mid,
                "within_other": self.within_other
            }
        }


@dataclass
class OtherDiagnostics:
    """Diagnostics specific to the OTHER class."""
    samples_mapped_to_other: int       # Total samples in OTHER class
    pct_samples_in_other: float        # % of validation set in OTHER
    other_precision: float             # Precision for OTHER class
    other_recall: float                # Recall for OTHER class
    other_f1: float                    # F1 for OTHER class
    confusion_to_other: int            # Non-OTHER predicted as OTHER
    confusion_from_other: int          # OTHER predicted as non-OTHER
    
    def to_dict(self) -> Dict:
        return {
            "samples_mapped_to_other": self.samples_mapped_to_other,
            "pct_samples_in_other": self.pct_samples_in_other,
            "other_precision": self.other_precision,
            "other_recall": self.other_recall,
            "other_f1": self.other_f1,
            "confusion_to_other": self.confusion_to_other,
            "confusion_from_other": self.confusion_from_other
        }


class RemappedMetricsTracker:
    """Track metrics with awareness of class remapping.
    
    This tracker is specifically designed for the TAIL → OTHER experiment
    and provides:
    - Metrics by bucket (HEAD, MID, OTHER)
    - Prediction diagnostics to detect false accuracy inflation
    - 3×3 bucket confusion matrix
    - OTHER-specific diagnostics
    
    Example:
        >>> tracker = RemappedMetricsTracker(remapper)
        >>> tracker.update(predictions, targets, logits)
        >>> metrics = tracker.compute_bucket_metrics()
        >>> diagnostics = tracker.compute_prediction_diagnostics()
    """
    
    def __init__(self, remapper: ClassRemapper):
        """Initialize tracker with remapper.
        
        Args:
            remapper: Fitted ClassRemapper instance
        """
        self.remapper = remapper
        
        # Storage for predictions and targets (new class IDs)
        self.predictions: List[int] = []
        self.targets: List[int] = []
        self.logits: List[np.ndarray] = []  # For top-k metrics
        
        # Bucket indices
        self._head_indices: List[int] = []
        self._mid_indices: List[int] = []
        self._other_indices: List[int] = []
    
    def reset(self) -> None:
        """Clear all stored predictions and targets."""
        self.predictions = []
        self.targets = []
        self.logits = []
        self._head_indices = []
        self._mid_indices = []
        self._other_indices = []
    
    def update(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        logits: Optional[np.ndarray] = None
    ) -> None:
        """Update tracker with new batch of predictions.
        
        Args:
            predictions: Array of predicted class IDs (new IDs)
            targets: Array of ground truth class IDs (new IDs)
            logits: Optional logits for top-k computation
        """
        predictions = np.atleast_1d(predictions)
        targets = np.atleast_1d(targets)
        
        start_idx = len(self.predictions)
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            self.predictions.append(int(pred))
            self.targets.append(int(target))
            
            # Track bucket membership for targets
            idx = start_idx + i
            if target == self.remapper.other_class_id:
                self._other_indices.append(idx)
            else:
                bucket = self.remapper.get_new_class_bucket(int(target))
                if bucket == Bucket.HEAD:
                    self._head_indices.append(idx)
                elif bucket == Bucket.MID:
                    self._mid_indices.append(idx)
        
        if logits is not None:
            for l in logits:
                self.logits.append(l)
    
    def compute_bucket_metrics(self) -> Dict[str, BucketMetricsResult]:
        """Compute metrics for HEAD, MID, and OTHER buckets.
        
        Returns:
            Dict with 'HEAD', 'MID', 'OTHER' keys mapping to BucketMetricsResult
        """
        results = {}
        
        # HEAD metrics
        results["HEAD"] = self._compute_for_indices(
            self._head_indices, 
            "HEAD",
            len(self.remapper.bucket_to_old_ids[Bucket.HEAD])
        )
        
        # MID metrics
        results["MID"] = self._compute_for_indices(
            self._mid_indices,
            "MID",
            len(self.remapper.bucket_to_old_ids[Bucket.MID])
        )
        
        # OTHER metrics (single class)
        results["OTHER"] = self._compute_for_indices(
            self._other_indices,
            "OTHER",
            1  # OTHER is a single class
        )
        
        return results
    
    def _compute_for_indices(
        self,
        indices: List[int],
        bucket_name: str,
        num_classes: int
    ) -> BucketMetricsResult:
        """Compute metrics for a subset of samples."""
        if not indices:
            return BucketMetricsResult(
                bucket=bucket_name,
                num_classes=num_classes,
                num_samples=0,
                accuracy=0.0,
                top5_accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0
            )
        
        preds = np.array([self.predictions[i] for i in indices])
        tgts = np.array([self.targets[i] for i in indices])
        
        # Accuracy
        correct = (preds == tgts).sum()
        accuracy = correct / len(indices)
        
        # Top-5 accuracy (if logits available)
        top5_accuracy = 0.0
        if self.logits:
            logits_subset = np.array([self.logits[i] for i in indices])
            top5_preds = np.argsort(logits_subset, axis=1)[:, -5:]
            top5_correct = sum(t in top5 for t, top5 in zip(tgts, top5_preds))
            top5_accuracy = top5_correct / len(indices)
        
        # Precision, Recall, F1 (micro for this bucket)
        # For bucket-level metrics, we compute these for all classes in bucket
        tp = correct
        total_pred = len(indices)  # All predictions for these samples
        total_actual = len(indices)  # All actuals for these samples
        
        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_actual if total_actual > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return BucketMetricsResult(
            bucket=bucket_name,
            num_classes=num_classes,
            num_samples=len(indices),
            accuracy=accuracy,
            top5_accuracy=top5_accuracy,
            precision=precision,
            recall=recall,
            f1=f1
        )
    
    def compute_prediction_diagnostics(self) -> PredictionDiagnostics:
        """Compute prediction distribution diagnostics.
        
        ⚠️ IMPORTANT: These metrics help detect false accuracy inflation.
        If pct_predictions_other > 50%, accuracy may be misleading.
        
        Returns:
            PredictionDiagnostics with distribution statistics
        """
        if not self.predictions:
            return PredictionDiagnostics(
                pct_predictions_other=0.0,
                entropy_of_predictions=0.0,
                num_unique_predictions=0,
                prediction_distribution={}
            )
        
        pred_array = np.array(self.predictions)
        total_preds = len(pred_array)
        
        # Count predictions per class
        pred_counts = Counter(self.predictions)
        
        # Percentage of OTHER predictions
        other_count = pred_counts.get(self.remapper.other_class_id, 0)
        pct_other = other_count / total_preds if total_preds > 0 else 0.0
        
        # Entropy of prediction distribution
        probs = np.array(list(pred_counts.values())) / total_preds
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Number of unique predictions
        num_unique = len(pred_counts)
        
        return PredictionDiagnostics(
            pct_predictions_other=pct_other,
            entropy_of_predictions=float(entropy),
            num_unique_predictions=num_unique,
            prediction_distribution=dict(pred_counts)
        )
    
    def compute_bucket_confusion(self) -> BucketConfusionMatrix:
        """Compute 3×3 confusion matrix at bucket level.
        
        This is the RIGHT way to visualize confusion for this experiment.
        NOT a full 51×51 matrix.
        
        Returns:
            BucketConfusionMatrix with 3×3 matrix and aggregated metrics
        """
        # Initialize 3×3 matrix: [HEAD, MID, OTHER] × [HEAD, MID, OTHER]
        matrix = np.zeros((3, 3), dtype=int)
        
        # Map bucket to index
        bucket_to_idx = {"HEAD": 0, "MID": 1, "OTHER": 2}
        
        for pred, target in zip(self.predictions, self.targets):
            # Get bucket for target
            if target == self.remapper.other_class_id:
                target_bucket = "OTHER"
            else:
                bucket = self.remapper.get_new_class_bucket(target)
                target_bucket = bucket.value if bucket != Bucket.TAIL else "OTHER"
            
            # Get bucket for prediction
            if pred == self.remapper.other_class_id:
                pred_bucket = "OTHER"
            else:
                try:
                    bucket = self.remapper.get_new_class_bucket(pred)
                    pred_bucket = bucket.value if bucket != Bucket.TAIL else "OTHER"
                except ValueError:
                    pred_bucket = "OTHER"  # Unknown prediction → treat as OTHER
            
            # Update matrix
            matrix[bucket_to_idx[target_bucket], bucket_to_idx[pred_bucket]] += 1
        
        # Compute aggregated metrics
        result = BucketConfusionMatrix(matrix=matrix)
        result.head_to_other = int(matrix[0, 2])  # HEAD → OTHER
        result.mid_to_other = int(matrix[1, 2])   # MID → OTHER
        result.other_to_head = int(matrix[2, 0])  # OTHER → HEAD
        result.other_to_mid = int(matrix[2, 1])   # OTHER → MID
        result.within_head = int(matrix[0, 0])    # HEAD → HEAD
        result.within_mid = int(matrix[1, 1])     # MID → MID
        result.within_other = int(matrix[2, 2])   # OTHER → OTHER
        
        return result
    
    def compute_other_diagnostics(self) -> OtherDiagnostics:
        """Compute OTHER-specific diagnostics.
        
        Returns:
            OtherDiagnostics with precision, recall, confusion metrics
        """
        total_samples = len(self.predictions)
        other_id = self.remapper.other_class_id
        
        # Count true/false positives/negatives for OTHER
        tp = 0  # OTHER predicted, OTHER actual
        fp = 0  # OTHER predicted, non-OTHER actual
        fn = 0  # non-OTHER predicted, OTHER actual
        
        for pred, target in zip(self.predictions, self.targets):
            if target == other_id and pred == other_id:
                tp += 1
            elif target != other_id and pred == other_id:
                fp += 1
            elif target == other_id and pred != other_id:
                fn += 1
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        samples_in_other = len(self._other_indices)
        pct_in_other = samples_in_other / total_samples if total_samples > 0 else 0.0
        
        return OtherDiagnostics(
            samples_mapped_to_other=samples_in_other,
            pct_samples_in_other=pct_in_other,
            other_precision=precision,
            other_recall=recall,
            other_f1=f1,
            confusion_to_other=fp,   # Non-OTHER predicted as OTHER
            confusion_from_other=fn  # OTHER predicted as non-OTHER
        )
    
    def compute_global_metrics(self) -> Dict[str, float]:
        """Compute global (non-bucketed) metrics.
        
        Returns:
            Dict with accuracy, f1_macro, f1_weighted, etc.
        """
        if not self.predictions:
            return {
                "accuracy": 0.0,
                "top5_accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0
            }
        
        preds = np.array(self.predictions)
        tgts = np.array(self.targets)
        
        # Accuracy
        accuracy = (preds == tgts).mean()
        
        # Top-5 accuracy
        top5_accuracy = 0.0
        if self.logits:
            logits_array = np.array(self.logits)
            top5_preds = np.argsort(logits_array, axis=1)[:, -5:]
            top5_correct = sum(t in top5 for t, top5 in zip(tgts, top5_preds))
            top5_accuracy = top5_correct / len(tgts)
        
        # F1 macro and weighted
        classes = list(range(self.remapper.num_classes_remapped))
        
        # Per-class metrics
        f1_scores = []
        supports = []
        
        for c in classes:
            tp = ((preds == c) & (tgts == c)).sum()
            fp = ((preds == c) & (tgts != c)).sum()
            fn = ((preds != c) & (tgts == c)).sum()
            support = (tgts == c).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores.append(f1)
            supports.append(support)
        
        f1_macro = np.mean(f1_scores)
        total_support = sum(supports)
        f1_weighted = sum(f1 * s / total_support for f1, s in zip(f1_scores, supports)) if total_support > 0 else 0.0
        
        return {
            "accuracy": float(accuracy),
            "top5_accuracy": float(top5_accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted)
        }
    
    def get_full_report(self) -> Dict:
        """Generate complete metrics report.
        
        Returns:
            Dict with all metrics, diagnostics, and interpretation notes
        """
        bucket_metrics = self.compute_bucket_metrics()
        pred_diagnostics = self.compute_prediction_diagnostics()
        bucket_confusion = self.compute_bucket_confusion()
        other_diagnostics = self.compute_other_diagnostics()
        global_metrics = self.compute_global_metrics()
        
        return {
            "global": global_metrics,
            "by_bucket": {k: v.to_dict() for k, v in bucket_metrics.items()},
            "prediction_diagnostics": pred_diagnostics.to_dict(),
            "bucket_confusion": bucket_confusion.to_dict(),
            "other_diagnostics": other_diagnostics.to_dict(),
            "interpretation_notes": {
                "warning_if_pct_other_high": (
                    "Si pct_predictions_other > 50%, accuracy puede estar inflada. "
                    f"Actual: {pred_diagnostics.pct_predictions_other:.1%}"
                ),
                "warning_head_only_3_classes": (
                    f"HEAD tiene solo {bucket_metrics['HEAD'].num_classes} clases - "
                    "mejoras fuertes pueden no generalizar"
                ),
                "is_valid": pred_diagnostics.is_valid()
            },
            "remapper_summary": self.remapper.get_config_summary()
        }
    
    def save_report(self, path: Path) -> None:
        """Save full report to JSON file.
        
        Args:
            path: Output path for JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        report = self.get_full_report()
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Metrics report saved to {path}")


def create_comparison_report(
    baseline_metrics: Dict,
    tail_to_other_metrics: Dict,
    output_path: Optional[Path] = None
) -> Dict:
    """Create comparison report between baseline and TAIL→OTHER experiment.
    
    Args:
        baseline_metrics: Metrics from baseline run
        tail_to_other_metrics: Metrics from TAIL→OTHER run
        output_path: Optional path to save report
    
    Returns:
        Comparison report dict
    """
    def get_value(d: Dict, *keys, default: float = 0.0) -> float:
        """Safely get nested dict value."""
        for key in keys:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return default
        return d if isinstance(d, (int, float)) else default
    
    # Extract comparable metrics
    baseline = {
        "accuracy": get_value(baseline_metrics, "global", "accuracy"),
        "top5_accuracy": get_value(baseline_metrics, "global", "top5_accuracy"),
        "f1_macro": get_value(baseline_metrics, "global", "f1_macro"),
        "f1_weighted": get_value(baseline_metrics, "global", "f1_weighted"),
        "head_accuracy": get_value(baseline_metrics, "by_bucket", "HEAD", "accuracy"),
        "mid_accuracy": get_value(baseline_metrics, "by_bucket", "MID", "accuracy"),
        "tail_accuracy": get_value(baseline_metrics, "by_bucket", "TAIL", "accuracy")
    }
    
    tail_other = {
        "accuracy": get_value(tail_to_other_metrics, "global", "accuracy"),
        "top5_accuracy": get_value(tail_to_other_metrics, "global", "top5_accuracy"),
        "f1_macro": get_value(tail_to_other_metrics, "global", "f1_macro"),
        "f1_weighted": get_value(tail_to_other_metrics, "global", "f1_weighted"),
        "head_accuracy": get_value(tail_to_other_metrics, "by_bucket", "HEAD", "accuracy"),
        "mid_accuracy": get_value(tail_to_other_metrics, "by_bucket", "MID", "accuracy"),
        "other_accuracy": get_value(tail_to_other_metrics, "by_bucket", "OTHER", "accuracy"),
        "pct_predictions_other": get_value(
            tail_to_other_metrics, "prediction_diagnostics", "pct_predictions_other"
        ),
        "entropy_of_predictions": get_value(
            tail_to_other_metrics, "prediction_diagnostics", "entropy_of_predictions"
        )
    }
    
    # Compute deltas
    delta = {
        "accuracy": tail_other["accuracy"] - baseline["accuracy"],
        "top5_accuracy": tail_other["top5_accuracy"] - baseline["top5_accuracy"],
        "f1_macro": tail_other["f1_macro"] - baseline["f1_macro"],
        "f1_weighted": tail_other["f1_weighted"] - baseline["f1_weighted"],
        "head_accuracy": tail_other["head_accuracy"] - baseline["head_accuracy"],
        "mid_accuracy": tail_other["mid_accuracy"] - baseline["mid_accuracy"]
    }
    
    report = {
        "baseline": baseline,
        "tail_to_other": tail_other,
        "delta": delta,
        "interpretation_notes": {
            "warning_if_pct_other_high": (
                "Si pct_predictions_other > 50%, accuracy puede estar inflada"
            ),
            "warning_head_only_3_classes": (
                "HEAD tiene solo 3 clases - mejoras fuertes pueden no generalizar"
            ),
            "f1_note": (
                "F1 macro: cada clase pesa igual. F1 weighted: ponderado por soporte."
            )
        }
    }
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Comparison report saved to {output_path}")
    
    return report
