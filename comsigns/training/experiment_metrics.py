"""
Experiment Metrics Module for TAIL → OTHER Controlled Experiment.

This module provides comprehensive metrics tracking specifically designed
for comparing baseline (505 classes) vs TAIL→OTHER (≈50 classes) experiments.

Goal: Validate that the model learns meaningful signals and predicts
some glosses correctly - NOT seeking SOTA.

Design Principles:
- Computed from model outputs only (logits + labels)
- Deterministic and reproducible
- Compatible between baseline and tail_to_other runs
- Minimal and focused on experiment questions

Key Metrics:
1. Global Metrics: accuracy@1, accuracy@5, micro/macro/weighted F1
2. Bucket Metrics: per HEAD/MID/TAIL or OTHER
3. Coverage Metrics: class prediction coverage
4. Collapse Diagnostics: detecting prediction collapse

Example:
    >>> tracker = ExperimentMetricsTracker(
    ...     num_classes=51,
    ...     bucket_mapping={0: 'HEAD', 1: 'HEAD', 2: 'MID', ...},
    ...     other_class_id=50
    ... )
    >>> for batch in val_loader:
    ...     tracker.update(logits, labels)
    >>> results = tracker.compute_all()
    >>> tracker.export_artifacts(output_dir)
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Metrics Results
# =============================================================================

@dataclass
class GlobalMetrics:
    """Global metrics for the entire validation set.
    
    All values in [0, 1].
    """
    accuracy_at_1: float
    accuracy_at_5: float
    micro_f1: float
    macro_f1: float
    weighted_f1: float
    num_samples: int
    num_classes: int
    
    def to_dict(self) -> Dict:
        return {
            "accuracy_at_1": self.accuracy_at_1,
            "accuracy_at_5": self.accuracy_at_5,
            "micro_f1": self.micro_f1,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "num_samples": self.num_samples,
            "num_classes": self.num_classes
        }


@dataclass
class BucketMetrics:
    """Metrics for a single bucket (HEAD, MID, TAIL, or OTHER).
    
    Answers: Where does the model learn?
    """
    bucket_name: str
    accuracy_at_1: float
    accuracy_at_5: float
    macro_f1: float
    num_classes_in_bucket: int
    num_classes_predicted: int  # Classes with ≥1 correct prediction
    num_samples: int
    
    def to_dict(self) -> Dict:
        return {
            "bucket_name": self.bucket_name,
            "accuracy_at_1": self.accuracy_at_1,
            "accuracy_at_5": self.accuracy_at_5,
            "macro_f1": self.macro_f1,
            "num_classes_in_bucket": self.num_classes_in_bucket,
            "num_classes_predicted": self.num_classes_predicted,
            "num_samples": self.num_samples
        }


@dataclass
class CoverageMetrics:
    """Coverage metrics measuring class space exploration.
    
    Answers: Is the model exploring the class space or collapsing?
    """
    coverage_at_1: float  # % of classes predicted correctly ≥1 time
    coverage_at_5: float  # % of classes appearing in top-5 ≥1 time
    coverage_at_1_by_bucket: Dict[str, float] = field(default_factory=dict)
    coverage_at_5_by_bucket: Dict[str, float] = field(default_factory=dict)
    num_classes_predicted_at_1: int = 0
    num_classes_predicted_at_5: int = 0
    total_classes: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "coverage_at_1": self.coverage_at_1,
            "coverage_at_5": self.coverage_at_5,
            "coverage_at_1_by_bucket": self.coverage_at_1_by_bucket,
            "coverage_at_5_by_bucket": self.coverage_at_5_by_bucket,
            "num_classes_predicted_at_1": self.num_classes_predicted_at_1,
            "num_classes_predicted_at_5": self.num_classes_predicted_at_5,
            "total_classes": self.total_classes
        }


@dataclass
class CollapseDiagnostics:
    """Diagnostics for detecting prediction collapse.
    
    Answers: Is the model collapsing predictions to few classes?
    """
    pct_predictions_most_frequent: float  # % going to most predicted class
    most_frequent_class_id: int
    pct_predictions_other: float  # % going to OTHER (if exists)
    prediction_entropy: float  # Shannon entropy (higher = more diverse)
    num_unique_predictions: int
    prediction_distribution: Dict[int, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "pct_predictions_most_frequent": self.pct_predictions_most_frequent,
            "most_frequent_class_id": self.most_frequent_class_id,
            "pct_predictions_other": self.pct_predictions_other,
            "prediction_entropy": self.prediction_entropy,
            "num_unique_predictions": self.num_unique_predictions,
            # Don't include full distribution in JSON to keep it small
        }


# =============================================================================
# Main Experiment Metrics Tracker
# =============================================================================

class ExperimentMetricsTracker:
    """
    Comprehensive metrics tracker for TAIL → OTHER experiment.
    
    This tracker accumulates predictions across batches and computes
    all metrics required for experiment comparison.
    
    Key Features:
    - Bucket-aware metrics (HEAD/MID/TAIL or OTHER)
    - Coverage metrics for class exploration
    - Collapse diagnostics
    - Deterministic and reproducible
    - Compatible between baseline and tail_to_other runs
    
    Example:
        >>> # For tail_to_other experiment
        >>> tracker = ExperimentMetricsTracker(
        ...     num_classes=51,
        ...     bucket_mapping={0: 'HEAD', 1: 'MID', ..., 50: 'OTHER'},
        ...     other_class_id=50,
        ...     experiment_id="tail_to_other_20260121"
        ... )
        >>> 
        >>> # Accumulate predictions
        >>> for batch in val_loader:
        ...     tracker.update(logits, labels)
        >>> 
        >>> # Compute all metrics
        >>> results = tracker.compute_all()
        >>> 
        >>> # Export artifacts
        >>> tracker.export_artifacts(Path("experiments/run_xxx/"))
    """
    
    def __init__(
        self,
        num_classes: int,
        bucket_mapping: Optional[Dict[int, str]] = None,
        other_class_id: Optional[int] = None,
        experiment_id: Optional[str] = None,
        device: Union[str, torch.device] = "cpu"
    ):
        """
        Initialize the experiment metrics tracker.
        
        Args:
            num_classes: Total number of classes in the model output
            bucket_mapping: Dict mapping class_id -> bucket_name ('HEAD', 'MID', 'TAIL', 'OTHER')
                          If None, all classes treated as same bucket
            other_class_id: ID of the OTHER class (for tail_to_other experiments)
                          If None, no OTHER-specific metrics computed
            experiment_id: Unique identifier for this experiment run
            device: Device for tensor storage
        """
        self.num_classes = num_classes
        self.bucket_mapping = bucket_mapping or {}
        self.other_class_id = other_class_id
        self.experiment_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Derived: bucket to class IDs
        self.bucket_to_classes: Dict[str, Set[int]] = {}
        for class_id, bucket in self.bucket_mapping.items():
            if bucket not in self.bucket_to_classes:
                self.bucket_to_classes[bucket] = set()
            self.bucket_to_classes[bucket].add(class_id)
        
        # Storage for accumulated data
        self._predictions: List[int] = []  # Top-1 predictions
        self._targets: List[int] = []      # Ground truth labels
        self._top5_predictions: List[np.ndarray] = []  # Top-5 predictions per sample
        self._logits: List[np.ndarray] = []  # Optional: full logits for future use
        
        # Cached results
        self._global_metrics: Optional[GlobalMetrics] = None
        self._bucket_metrics: Optional[Dict[str, BucketMetrics]] = None
        self._coverage_metrics: Optional[CoverageMetrics] = None
        self._collapse_diagnostics: Optional[CollapseDiagnostics] = None
    
    @property
    def predictions(self) -> List[int]:
        """Get all accumulated top-1 predictions."""
        return self._predictions
    
    @property
    def targets(self) -> List[int]:
        """Get all accumulated ground truth labels."""
        return self._targets
    
    def reset(self) -> None:
        """Clear all accumulated data and cached results."""
        self._predictions = []
        self._targets = []
        self._top5_predictions = []
        self._logits = []
        self._global_metrics = None
        self._bucket_metrics = None
        self._coverage_metrics = None
        self._collapse_diagnostics = None
    
    def update(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        store_logits: bool = False
    ) -> None:
        """
        Accumulate predictions from a batch.
        
        Args:
            logits: Model output logits, shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size,)
            store_logits: Whether to store full logits (memory intensive)
        """
        # Convert to numpy for consistent handling
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Ensure 2D logits
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        
        # Flatten targets
        targets = np.atleast_1d(targets).flatten()
        
        # Compute predictions
        top1_preds = np.argmax(logits, axis=1)
        
        # Compute top-5 predictions (handle case where num_classes < 5)
        k = min(5, logits.shape[1])
        top5_preds = np.argsort(logits, axis=1)[:, -k:][:, ::-1]  # Descending order
        
        # Accumulate
        self._predictions.extend(top1_preds.tolist())
        self._targets.extend(targets.tolist())
        self._top5_predictions.extend(top5_preds)
        
        if store_logits:
            self._logits.extend(logits)
        
        # Invalidate cache
        self._global_metrics = None
        self._bucket_metrics = None
        self._coverage_metrics = None
        self._collapse_diagnostics = None
    
    # =========================================================================
    # Global Metrics
    # =========================================================================
    
    def compute_global_metrics(self) -> GlobalMetrics:
        """Compute global metrics for the entire validation set."""
        if self._global_metrics is not None:
            return self._global_metrics
        
        if not self._predictions:
            return GlobalMetrics(
                accuracy_at_1=0.0,
                accuracy_at_5=0.0,
                micro_f1=0.0,
                macro_f1=0.0,
                weighted_f1=0.0,
                num_samples=0,
                num_classes=self.num_classes
            )
        
        preds = np.array(self._predictions)
        targets = np.array(self._targets)
        top5_preds = np.array(self._top5_predictions)
        
        n_samples = len(targets)
        
        # Accuracy@1
        acc_at_1 = float((preds == targets).mean())
        
        # Accuracy@5
        acc_at_5 = float(np.mean([t in top5 for t, top5 in zip(targets, top5_preds)]))
        
        # F1 scores
        micro_f1, macro_f1, weighted_f1 = self._compute_f1_scores(preds, targets)
        
        self._global_metrics = GlobalMetrics(
            accuracy_at_1=acc_at_1,
            accuracy_at_5=acc_at_5,
            micro_f1=micro_f1,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            num_samples=n_samples,
            num_classes=self.num_classes
        )
        
        return self._global_metrics
    
    def _compute_f1_scores(
        self,
        preds: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute micro, macro, and weighted F1 scores."""
        # Per-class TP, FP, FN
        classes = list(range(self.num_classes))
        
        tp = np.zeros(self.num_classes)
        fp = np.zeros(self.num_classes)
        fn = np.zeros(self.num_classes)
        support = np.zeros(self.num_classes)
        
        for c in classes:
            tp[c] = ((preds == c) & (targets == c)).sum()
            fp[c] = ((preds == c) & (targets != c)).sum()
            fn[c] = ((preds != c) & (targets == c)).sum()
            support[c] = (targets == c).sum()
        
        # Micro F1 (sum then divide)
        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
            if (micro_precision + micro_recall) > 0 else 0.0
        
        # Per-class F1
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        f1_per_class = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0
        )
        
        # Macro F1 (mean of per-class F1, only for classes with support)
        classes_with_support = support > 0
        macro_f1 = float(f1_per_class[classes_with_support].mean()) if classes_with_support.any() else 0.0
        
        # Weighted F1
        total_support = support.sum()
        weighted_f1 = float((f1_per_class * support).sum() / total_support) if total_support > 0 else 0.0
        
        return float(micro_f1), macro_f1, weighted_f1
    
    # =========================================================================
    # Bucket Metrics
    # =========================================================================
    
    def compute_bucket_metrics(self) -> Dict[str, BucketMetrics]:
        """Compute metrics for each bucket (HEAD, MID, TAIL/OTHER)."""
        if self._bucket_metrics is not None:
            return self._bucket_metrics
        
        if not self._predictions or not self.bucket_mapping:
            return {}
        
        preds = np.array(self._predictions)
        targets = np.array(self._targets)
        top5_preds = np.array(self._top5_predictions)
        
        self._bucket_metrics = {}
        
        for bucket_name, class_ids in self.bucket_to_classes.items():
            # Find samples belonging to this bucket
            mask = np.isin(targets, list(class_ids))
            bucket_targets = targets[mask]
            bucket_preds = preds[mask]
            bucket_top5 = top5_preds[mask]
            
            if len(bucket_targets) == 0:
                self._bucket_metrics[bucket_name] = BucketMetrics(
                    bucket_name=bucket_name,
                    accuracy_at_1=0.0,
                    accuracy_at_5=0.0,
                    macro_f1=0.0,
                    num_classes_in_bucket=len(class_ids),
                    num_classes_predicted=0,
                    num_samples=0
                )
                continue
            
            # Accuracy@1
            acc_at_1 = float((bucket_preds == bucket_targets).mean())
            
            # Accuracy@5
            acc_at_5 = float(np.mean([
                t in top5 for t, top5 in zip(bucket_targets, bucket_top5)
            ]))
            
            # Macro F1 for bucket classes only
            macro_f1 = self._compute_bucket_macro_f1(bucket_preds, bucket_targets, class_ids)
            
            # Classes with ≥1 correct prediction
            correct_mask = bucket_preds == bucket_targets
            classes_predicted = len(set(bucket_targets[correct_mask]))
            
            self._bucket_metrics[bucket_name] = BucketMetrics(
                bucket_name=bucket_name,
                accuracy_at_1=acc_at_1,
                accuracy_at_5=acc_at_5,
                macro_f1=macro_f1,
                num_classes_in_bucket=len(class_ids),
                num_classes_predicted=classes_predicted,
                num_samples=len(bucket_targets)
            )
        
        return self._bucket_metrics
    
    def _compute_bucket_macro_f1(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        class_ids: Set[int]
    ) -> float:
        """Compute macro F1 for a subset of classes."""
        f1_scores = []
        
        for c in class_ids:
            tp = ((preds == c) & (targets == c)).sum()
            fp = ((preds == c) & (targets != c)).sum()
            fn = ((preds != c) & (targets == c)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Only include classes with support
            if (targets == c).sum() > 0:
                f1_scores.append(f1)
        
        return float(np.mean(f1_scores)) if f1_scores else 0.0
    
    # =========================================================================
    # Coverage Metrics
    # =========================================================================
    
    def compute_coverage_metrics(self) -> CoverageMetrics:
        """Compute coverage metrics measuring class space exploration."""
        if self._coverage_metrics is not None:
            return self._coverage_metrics
        
        if not self._predictions:
            return CoverageMetrics(
                coverage_at_1=0.0,
                coverage_at_5=0.0,
                total_classes=self.num_classes
            )
        
        preds = np.array(self._predictions)
        targets = np.array(self._targets)
        top5_preds = np.array(self._top5_predictions)
        
        # Classes with ≥1 correct prediction
        correct_mask = preds == targets
        classes_correct_at_1 = set(targets[correct_mask])
        
        # Classes appearing correctly in top-5
        classes_correct_at_5 = set()
        for t, top5 in zip(targets, top5_preds):
            if t in top5:
                classes_correct_at_5.add(t)
        
        # Total classes in validation set
        classes_in_val = set(targets)
        total_classes = len(classes_in_val)
        
        coverage_at_1 = len(classes_correct_at_1) / total_classes if total_classes > 0 else 0.0
        coverage_at_5 = len(classes_correct_at_5) / total_classes if total_classes > 0 else 0.0
        
        # Per-bucket coverage
        coverage_at_1_by_bucket = {}
        coverage_at_5_by_bucket = {}
        
        for bucket_name, class_ids in self.bucket_to_classes.items():
            bucket_classes_in_val = class_ids & classes_in_val
            bucket_total = len(bucket_classes_in_val)
            
            if bucket_total > 0:
                bucket_correct_at_1 = len(classes_correct_at_1 & class_ids)
                bucket_correct_at_5 = len(classes_correct_at_5 & class_ids)
                
                coverage_at_1_by_bucket[bucket_name] = bucket_correct_at_1 / bucket_total
                coverage_at_5_by_bucket[bucket_name] = bucket_correct_at_5 / bucket_total
            else:
                coverage_at_1_by_bucket[bucket_name] = 0.0
                coverage_at_5_by_bucket[bucket_name] = 0.0
        
        self._coverage_metrics = CoverageMetrics(
            coverage_at_1=coverage_at_1,
            coverage_at_5=coverage_at_5,
            coverage_at_1_by_bucket=coverage_at_1_by_bucket,
            coverage_at_5_by_bucket=coverage_at_5_by_bucket,
            num_classes_predicted_at_1=len(classes_correct_at_1),
            num_classes_predicted_at_5=len(classes_correct_at_5),
            total_classes=total_classes
        )
        
        return self._coverage_metrics
    
    # =========================================================================
    # Collapse Diagnostics
    # =========================================================================
    
    def compute_collapse_diagnostics(self) -> CollapseDiagnostics:
        """Compute diagnostics for detecting prediction collapse."""
        if self._collapse_diagnostics is not None:
            return self._collapse_diagnostics
        
        if not self._predictions:
            return CollapseDiagnostics(
                pct_predictions_most_frequent=0.0,
                most_frequent_class_id=-1,
                pct_predictions_other=0.0,
                prediction_entropy=0.0,
                num_unique_predictions=0
            )
        
        preds = np.array(self._predictions)
        n_samples = len(preds)
        
        # Prediction distribution
        pred_counts = Counter(preds.tolist())
        
        # Most frequent class
        most_frequent_class, most_frequent_count = pred_counts.most_common(1)[0]
        pct_most_frequent = most_frequent_count / n_samples
        
        # Percentage to OTHER
        pct_other = 0.0
        if self.other_class_id is not None:
            other_count = pred_counts.get(self.other_class_id, 0)
            pct_other = other_count / n_samples
        
        # Shannon entropy
        probs = np.array(list(pred_counts.values())) / n_samples
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        
        # Max possible entropy (uniform distribution)
        max_entropy = np.log(self.num_classes)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        self._collapse_diagnostics = CollapseDiagnostics(
            pct_predictions_most_frequent=pct_most_frequent,
            most_frequent_class_id=most_frequent_class,
            pct_predictions_other=pct_other,
            prediction_entropy=normalized_entropy,  # Normalized to [0, 1]
            num_unique_predictions=len(pred_counts),
            prediction_distribution=dict(pred_counts)
        )
        
        return self._collapse_diagnostics
    
    # =========================================================================
    # Combined Methods
    # =========================================================================
    
    def compute_all(self) -> Dict:
        """Compute all metrics and return as a single dictionary."""
        global_metrics = self.compute_global_metrics()
        bucket_metrics = self.compute_bucket_metrics()
        coverage_metrics = self.compute_coverage_metrics()
        collapse_diagnostics = self.compute_collapse_diagnostics()
        
        return {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "global": global_metrics.to_dict(),
            "by_bucket": {k: v.to_dict() for k, v in bucket_metrics.items()},
            "coverage": coverage_metrics.to_dict(),
            "collapse_diagnostics": collapse_diagnostics.to_dict(),
            "config": {
                "num_classes": self.num_classes,
                "other_class_id": self.other_class_id,
                "buckets": list(self.bucket_to_classes.keys())
            }
        }
    
    def export_artifacts(self, output_dir: Path) -> Dict[str, Path]:
        """
        Export all metrics to separate JSON files.
        
        Args:
            output_dir: Directory to save artifacts
        
        Returns:
            Dict mapping artifact name to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        base_info = {
            "experiment_id": self.experiment_id,
            "num_classes": self.num_classes,
            "num_samples": len(self._predictions),
            "timestamp": timestamp
        }
        
        artifacts = {}
        
        # 1. Global metrics
        global_metrics = self.compute_global_metrics()
        global_path = output_dir / "metrics_global.json"
        with open(global_path, "w", encoding="utf-8") as f:
            json.dump({**base_info, **global_metrics.to_dict()}, f, indent=2)
        artifacts["metrics_global"] = global_path
        
        # 2. Bucket metrics
        bucket_metrics = self.compute_bucket_metrics()
        bucket_path = output_dir / "metrics_by_bucket.json"
        with open(bucket_path, "w", encoding="utf-8") as f:
            data = {
                **base_info,
                "buckets": {k: v.to_dict() for k, v in bucket_metrics.items()}
            }
            json.dump(data, f, indent=2)
        artifacts["metrics_by_bucket"] = bucket_path
        
        # 3. Coverage metrics
        coverage_metrics = self.compute_coverage_metrics()
        coverage_path = output_dir / "coverage_metrics.json"
        with open(coverage_path, "w", encoding="utf-8") as f:
            json.dump({**base_info, **coverage_metrics.to_dict()}, f, indent=2)
        artifacts["coverage_metrics"] = coverage_path
        
        # 4. Collapse diagnostics
        collapse_diagnostics = self.compute_collapse_diagnostics()
        collapse_path = output_dir / "collapse_diagnostics.json"
        with open(collapse_path, "w", encoding="utf-8") as f:
            json.dump({**base_info, **collapse_diagnostics.to_dict()}, f, indent=2)
        artifacts["collapse_diagnostics"] = collapse_path
        
        logger.info(f"Exported {len(artifacts)} metric artifacts to {output_dir}")
        
        return artifacts
    
    def get_summary(self) -> str:
        """Get a human-readable summary of metrics."""
        global_m = self.compute_global_metrics()
        bucket_m = self.compute_bucket_metrics()
        coverage_m = self.compute_coverage_metrics()
        collapse_d = self.compute_collapse_diagnostics()
        
        lines = [
            f"=" * 60,
            f"Experiment Metrics Summary: {self.experiment_id}",
            f"=" * 60,
            f"",
            f"GLOBAL METRICS ({global_m.num_samples} samples, {global_m.num_classes} classes):",
            f"  Accuracy@1:  {global_m.accuracy_at_1:.2%}",
            f"  Accuracy@5:  {global_m.accuracy_at_5:.2%}",
            f"  Micro-F1:    {global_m.micro_f1:.4f}",
            f"  Macro-F1:    {global_m.macro_f1:.4f}",
            f"  Weighted-F1: {global_m.weighted_f1:.4f}",
            f"",
            f"BUCKET METRICS:",
        ]
        
        for bucket_name, bm in bucket_m.items():
            lines.append(
                f"  {bucket_name}: Acc@1={bm.accuracy_at_1:.2%}, "
                f"Acc@5={bm.accuracy_at_5:.2%}, "
                f"F1={bm.macro_f1:.4f}, "
                f"Predicted={bm.num_classes_predicted}/{bm.num_classes_in_bucket} classes"
            )
        
        lines.extend([
            f"",
            f"COVERAGE:",
            f"  Coverage@1: {coverage_m.coverage_at_1:.2%} ({coverage_m.num_classes_predicted_at_1}/{coverage_m.total_classes} classes)",
            f"  Coverage@5: {coverage_m.coverage_at_5:.2%} ({coverage_m.num_classes_predicted_at_5}/{coverage_m.total_classes} classes)",
        ])
        
        for bucket_name in coverage_m.coverage_at_1_by_bucket:
            c1 = coverage_m.coverage_at_1_by_bucket[bucket_name]
            c5 = coverage_m.coverage_at_5_by_bucket.get(bucket_name, 0.0)
            lines.append(f"  {bucket_name}: Coverage@1={c1:.2%}, Coverage@5={c5:.2%}")
        
        lines.extend([
            f"",
            f"COLLAPSE DIAGNOSTICS:",
            f"  Most frequent class: {collapse_d.most_frequent_class_id} ({collapse_d.pct_predictions_most_frequent:.1%} of predictions)",
            f"  Predictions to OTHER: {collapse_d.pct_predictions_other:.1%}",
            f"  Prediction entropy: {collapse_d.prediction_entropy:.3f} (normalized)",
            f"  Unique predictions: {collapse_d.num_unique_predictions}",
            f"=" * 60,
        ])
        
        return "\n".join(lines)


# =============================================================================
# Factory Function
# =============================================================================

def create_experiment_tracker(
    num_classes: int,
    bucket_mapping: Optional[Dict[int, str]] = None,
    remapper: Optional['ClassRemapper'] = None,
    experiment_id: Optional[str] = None
) -> ExperimentMetricsTracker:
    """
    Create an ExperimentMetricsTracker with proper configuration.
    
    This is a convenience function that can derive bucket_mapping from
    a ClassRemapper if available.
    
    Args:
        num_classes: Total number of output classes
        bucket_mapping: Direct mapping class_id -> bucket_name
        remapper: ClassRemapper instance (alternative to bucket_mapping)
        experiment_id: Experiment identifier
    
    Returns:
        Configured ExperimentMetricsTracker
    """
    other_class_id = None
    
    # Derive bucket mapping from remapper if available
    if remapper is not None and bucket_mapping is None:
        bucket_mapping = {}
        other_class_id = remapper.other_class_id
        
        for new_id in range(remapper.num_classes_remapped):
            if new_id == other_class_id:
                bucket_mapping[new_id] = "OTHER"
            else:
                # Determine bucket from original class
                try:
                    bucket = remapper.get_new_class_bucket(new_id)
                    bucket_mapping[new_id] = bucket.value
                except (ValueError, AttributeError):
                    bucket_mapping[new_id] = "UNKNOWN"
    
    return ExperimentMetricsTracker(
        num_classes=num_classes,
        bucket_mapping=bucket_mapping,
        other_class_id=other_class_id,
        experiment_id=experiment_id
    )


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_experiments(
    baseline_metrics: Dict,
    experiment_metrics: Dict,
    output_path: Optional[Path] = None
) -> Dict:
    """
    Compare baseline vs experimental condition metrics.
    
    Args:
        baseline_metrics: Results from compute_all() for baseline
        experiment_metrics: Results from compute_all() for experiment
        output_path: Optional path to save comparison
    
    Returns:
        Comparison report with deltas
    """
    def safe_get(d: Dict, *keys, default: float = 0.0) -> float:
        for key in keys:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return default
        return d if isinstance(d, (int, float)) else default
    
    # Extract comparable metrics
    baseline_global = baseline_metrics.get("global", {})
    exp_global = experiment_metrics.get("global", {})
    
    comparison = {
        "baseline": {
            "experiment_id": baseline_metrics.get("experiment_id"),
            "accuracy_at_1": safe_get(baseline_global, "accuracy_at_1"),
            "accuracy_at_5": safe_get(baseline_global, "accuracy_at_5"),
            "macro_f1": safe_get(baseline_global, "macro_f1"),
            "weighted_f1": safe_get(baseline_global, "weighted_f1"),
        },
        "experiment": {
            "experiment_id": experiment_metrics.get("experiment_id"),
            "accuracy_at_1": safe_get(exp_global, "accuracy_at_1"),
            "accuracy_at_5": safe_get(exp_global, "accuracy_at_5"),
            "macro_f1": safe_get(exp_global, "macro_f1"),
            "weighted_f1": safe_get(exp_global, "weighted_f1"),
        },
        "delta": {},
        "bucket_comparison": {},
        "key_metrics": {}
    }
    
    # Compute deltas
    for metric in ["accuracy_at_1", "accuracy_at_5", "macro_f1", "weighted_f1"]:
        baseline_val = comparison["baseline"][metric]
        exp_val = comparison["experiment"][metric]
        comparison["delta"][metric] = exp_val - baseline_val
    
    # Bucket comparison (HEAD + MID only - comparable between runs)
    baseline_buckets = baseline_metrics.get("by_bucket", {})
    exp_buckets = experiment_metrics.get("by_bucket", {})
    
    for bucket in ["HEAD", "MID"]:
        baseline_bucket = baseline_buckets.get(bucket, {})
        exp_bucket = exp_buckets.get(bucket, {})
        
        baseline_acc = baseline_bucket.get("accuracy_at_1", 0.0) if isinstance(baseline_bucket, dict) else 0.0
        exp_acc = exp_bucket.get("accuracy_at_1", 0.0) if isinstance(exp_bucket, dict) else 0.0
        
        comparison["bucket_comparison"][bucket] = {
            "baseline_acc_at_1": baseline_acc,
            "experiment_acc_at_1": exp_acc,
            "delta_acc_at_1": exp_acc - baseline_acc,
            "baseline_coverage": safe_get(baseline_metrics, "coverage", "coverage_at_5_by_bucket", bucket, default=0.0),
            "experiment_coverage": safe_get(experiment_metrics, "coverage", "coverage_at_5_by_bucket", bucket, default=0.0),
        }
    
    # Key experiment metrics (success criteria)
    head_delta = comparison["bucket_comparison"].get("HEAD", {}).get("delta_acc_at_1", 0.0)
    mid_delta = comparison["bucket_comparison"].get("MID", {}).get("delta_acc_at_1", 0.0)
    
    comparison["key_metrics"] = {
        "delta_macro_f1_head_mid": comparison["delta"]["macro_f1"],  # Proxy
        "delta_coverage_at_5_head_mid": (
            comparison["bucket_comparison"].get("HEAD", {}).get("experiment_coverage", 0.0) +
            comparison["bucket_comparison"].get("MID", {}).get("experiment_coverage", 0.0)
        ) / 2 - (
            comparison["bucket_comparison"].get("HEAD", {}).get("baseline_coverage", 0.0) +
            comparison["bucket_comparison"].get("MID", {}).get("baseline_coverage", 0.0)
        ) / 2,
        "delta_acc_at_1_head": head_delta,
        "delta_acc_at_1_mid": mid_delta,
        "experiment_improves_head": head_delta > 0,
        "experiment_improves_mid": mid_delta > 0,
    }
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison saved to {output_path}")
    
    return comparison
