"""
Final Evaluation Module.

Provides post-training evaluation that generates:
- Per-class metrics (Precision, Recall, F1, Support)
- Confusion matrix (raw counts and normalized)
- Visualization artifacts

This module is invoked AFTER training completes, not during epochs.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Container for final evaluation results.
    
    Attributes:
        y_true: Ground truth labels
        y_pred: Model predictions (argmax of logits)
        y_logits: Raw logits for top-k analysis
        num_samples: Total samples evaluated
        num_classes: Number of classes
        class_names: Optional class name mapping
        timestamp: Evaluation timestamp
    """
    y_true: np.ndarray
    y_pred: np.ndarray
    y_logits: np.ndarray
    num_samples: int
    num_classes: int
    class_names: Optional[List[str]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_confusion_matrix(self, normalize: Optional[str] = None) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            normalize: None for raw counts, 'true' for recall normalization,
                      'pred' for precision normalization, 'all' for total.
        
        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for confusion matrix")
        
        return confusion_matrix(
            self.y_true,
            self.y_pred,
            labels=list(range(self.num_classes)),
            normalize=normalize
        )
    
    def get_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        Compute per-class Precision, Recall, F1, Support.
        
        Returns:
            Dict mapping class_id to metrics dict
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for per-class metrics")
        
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true,
            self.y_pred,
            labels=list(range(self.num_classes)),
            average=None,
            zero_division=0.0
        )
        
        results = {}
        for i in range(self.num_classes):
            name = self.class_names[i] if self.class_names and i < len(self.class_names) else None
            results[i] = {
                "class_id": i,
                "class_name": name,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }
        
        return results
    
    def get_global_metrics(self) -> Dict[str, float]:
        """Get global (macro/micro averaged) metrics."""
        if not HAS_SKLEARN:
            return {}
        
        # Macro average
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='macro', zero_division=0.0
        )
        
        # Weighted average
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='weighted', zero_division=0.0
        )
        
        # Accuracy
        accuracy = (self.y_true == self.y_pred).mean()
        
        return {
            "accuracy": float(accuracy),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(p_weighted),
            "recall_weighted": float(r_weighted),
            "f1_weighted": float(f1_weighted),
            "num_samples": self.num_samples,
            "num_classes": self.num_classes
        }


class FinalEvaluator:
    """
    Performs final evaluation after training completes.
    
    This class is responsible for:
    1. Running inference on validation set (model in eval mode, no gradients)
    2. Accumulating predictions (y_true, y_pred)
    3. Computing per-class metrics
    4. Generating and saving confusion matrix
    5. Persisting all artifacts to disk
    
    Example:
        >>> evaluator = FinalEvaluator(model, val_loader, num_classes=505)
        >>> result = evaluator.evaluate()
        >>> evaluator.save_artifacts(output_dir="experiments/run_001/")
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            dataloader: Validation DataLoader
            num_classes: Total number of classes
            class_names: Optional list of class names (indexed by class_id)
            device: Device to run evaluation on
        """
        self.model = model
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device or next(model.parameters()).device
        
        self._result: Optional[EvaluationResult] = None
    
    def evaluate(self) -> EvaluationResult:
        """
        Run final evaluation on validation set.
        
        Accumulates all predictions in a single pass.
        Model is set to eval() mode with no gradients.
        
        Returns:
            EvaluationResult with y_true, y_pred, and computed metrics
        """
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION - Post-Training")
        logger.info("=" * 60)
        
        self.model.eval()
        
        all_labels = []
        all_preds = []
        all_logits = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                # Move batch to device
                hand = batch["hand"].to(self.device)
                body = batch["body"].to(self.device)
                face = batch["face"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                mask = batch.get("mask")
                if mask is not None and mask.numel() > 0:
                    mask = mask.to(self.device)
                else:
                    mask = None
                
                # Forward pass
                logits = self.model(hand, body, face, lengths=lengths, mask=mask)
                preds = logits.argmax(dim=1)
                
                # Accumulate
                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())
                all_logits.append(logits.cpu())
                
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"  Evaluated {batch_idx + 1}/{len(self.dataloader)} batches")
        
        # Concatenate all
        y_true = torch.cat(all_labels, dim=0).numpy()
        y_pred = torch.cat(all_preds, dim=0).numpy()
        y_logits = torch.cat(all_logits, dim=0).numpy()
        
        logger.info(f"Evaluation complete: {len(y_true)} samples")
        
        self._result = EvaluationResult(
            y_true=y_true,
            y_pred=y_pred,
            y_logits=y_logits,
            num_samples=len(y_true),
            num_classes=self.num_classes,
            class_names=self.class_names
        )
        
        return self._result
    
    @property
    def result(self) -> Optional[EvaluationResult]:
        """Get evaluation result (None if evaluate() not called)."""
        return self._result
    
    def save_artifacts(
        self,
        output_dir: Union[str, Path],
        prefix: str = "",
        save_metrics_json: bool = True,
        save_metrics_csv: bool = True,
        save_confusion_matrix_png: bool = True,
        save_confusion_matrix_csv: bool = True,
        dataset_name: str = "validation",
        epoch: Optional[int] = None
    ) -> Dict[str, Path]:
        """
        Save all evaluation artifacts to disk.
        
        Args:
            output_dir: Directory to save artifacts
            prefix: Optional prefix for filenames
            save_metrics_json: Save per-class metrics as JSON
            save_metrics_csv: Save per-class metrics as CSV
            save_confusion_matrix_png: Save confusion matrix image
            save_confusion_matrix_csv: Save confusion matrix as CSV
            dataset_name: Name for titles (e.g., "validation", "test")
            epoch: Final epoch number for titles
        
        Returns:
            Dict mapping artifact type to saved path
        """
        if self._result is None:
            raise RuntimeError("Must call evaluate() before save_artifacts()")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        prefix = f"{prefix}_" if prefix else ""
        
        # 1. Per-class metrics JSON
        if save_metrics_json:
            path = output_dir / f"{prefix}metrics_by_class.json"
            self._save_metrics_json(path)
            saved_paths["metrics_json"] = path
        
        # 2. Per-class metrics CSV
        if save_metrics_csv:
            path = output_dir / f"{prefix}metrics_by_class.csv"
            self._save_metrics_csv(path)
            saved_paths["metrics_csv"] = path
        
        # 3. Confusion matrix PNG
        if save_confusion_matrix_png:
            path = output_dir / f"{prefix}confusion_matrix.png"
            title = f"Confusion Matrix - {dataset_name}"
            if epoch is not None:
                title += f" (Epoch {epoch})"
            self._save_confusion_matrix_png(path, title=title)
            saved_paths["confusion_matrix_png"] = path
        
        # 4. Confusion matrix CSV
        if save_confusion_matrix_csv:
            path = output_dir / f"{prefix}confusion_matrix.csv"
            self._save_confusion_matrix_csv(path)
            saved_paths["confusion_matrix_csv"] = path
        
        # 5. Summary JSON (global metrics)
        summary_path = output_dir / f"{prefix}evaluation_summary.json"
        self._save_summary_json(summary_path, dataset_name, epoch)
        saved_paths["summary_json"] = summary_path
        
        logger.info(f"Artifacts saved to {output_dir}/")
        for name, path in saved_paths.items():
            logger.info(f"  - {name}: {path.name}")
        
        return saved_paths
    
    def _save_metrics_json(self, path: Path) -> None:
        """Save per-class metrics as JSON."""
        per_class = self._result.get_per_class_metrics()
        
        # Convert to serializable format
        data = {
            "timestamp": self._result.timestamp,
            "num_samples": self._result.num_samples,
            "num_classes": self._result.num_classes,
            "metrics_by_class": per_class
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_metrics_csv(self, path: Path) -> None:
        """Save per-class metrics as CSV."""
        import csv
        
        per_class = self._result.get_per_class_metrics()
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "support"])
            
            for class_id in sorted(per_class.keys()):
                m = per_class[class_id]
                writer.writerow([
                    m["class_id"],
                    m["class_name"] or "",
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1']:.4f}",
                    m["support"]
                ])
    
    def _save_confusion_matrix_csv(self, path: Path) -> None:
        """Save confusion matrix as CSV."""
        import csv
        
        cm = self._result.get_confusion_matrix()
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Header
            if self.class_names:
                header = ["true\\pred"] + [self.class_names[i] if i < len(self.class_names) else f"C{i}" 
                                           for i in range(self.num_classes)]
            else:
                header = ["true\\pred"] + [f"C{i}" for i in range(self.num_classes)]
            writer.writerow(header)
            
            # Data rows
            for i in range(self.num_classes):
                row_name = self.class_names[i] if self.class_names and i < len(self.class_names) else f"C{i}"
                row = [row_name] + [int(cm[i, j]) for j in range(self.num_classes)]
                writer.writerow(row)
    
    def _save_confusion_matrix_png(self, path: Path, title: str = "Confusion Matrix") -> None:
        """Save confusion matrix as PNG image."""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available, skipping confusion matrix PNG")
            return
        
        cm = self._result.get_confusion_matrix()
        cm_normalized = self._result.get_confusion_matrix(normalize='true')
        
        # Determine figure size based on number of classes
        n = self.num_classes
        if n <= 20:
            figsize = (12, 10)
            show_labels = True
            show_values = True
        elif n <= 50:
            figsize = (16, 14)
            show_labels = True
            show_values = False
        else:
            figsize = (20, 18)
            show_labels = False
            show_values = False
        
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
        
        # Raw counts
        self._plot_cm(axes[0], cm, f"{title} (Raw Counts)", show_labels, show_values)
        
        # Normalized (recall)
        self._plot_cm(axes[1], cm_normalized, f"{title} (Normalized by True Class)", 
                     show_labels, show_values, is_normalized=True)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix image saved: {path}")
    
    def _plot_cm(
        self,
        ax,
        cm: np.ndarray,
        title: str,
        show_labels: bool,
        show_values: bool,
        is_normalized: bool = False
    ) -> None:
        """Plot confusion matrix on given axis."""
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        
        n = cm.shape[0]
        
        if show_labels and self.class_names:
            # Show class names
            tick_marks = np.arange(n)
            labels = [self.class_names[i][:15] if i < len(self.class_names) else f"{i}" 
                     for i in range(n)]
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
        elif show_labels:
            # Show indices for smaller matrices
            if n <= 30:
                ax.set_xticks(np.arange(n))
                ax.set_yticks(np.arange(n))
                ax.set_xticklabels([str(i) for i in range(n)], rotation=90, fontsize=6)
                ax.set_yticklabels([str(i) for i in range(n)], fontsize=6)
        
        # Show values in cells
        if show_values and n <= 20:
            thresh = cm.max() / 2.
            fmt = '.2f' if is_normalized else 'd'
            for i in range(n):
                for j in range(n):
                    val = cm[i, j]
                    if val > 0 or not is_normalized:
                        ax.text(j, i, format(val, fmt),
                               ha="center", va="center",
                               color="white" if val > thresh else "black",
                               fontsize=7)
    
    def _save_summary_json(self, path: Path, dataset_name: str, epoch: Optional[int]) -> None:
        """Save evaluation summary as JSON."""
        global_metrics = self._result.get_global_metrics()
        
        # Add metadata
        global_metrics["dataset"] = dataset_name
        global_metrics["epoch"] = epoch
        global_metrics["timestamp"] = self._result.timestamp
        
        # Class distribution
        per_class = self._result.get_per_class_metrics()
        f1_scores = [m["f1"] for m in per_class.values()]
        supports = [m["support"] for m in per_class.values()]
        
        global_metrics["class_statistics"] = {
            "classes_with_support": sum(1 for s in supports if s > 0),
            "classes_with_f1_above_0.5": sum(1 for f in f1_scores if f > 0.5),
            "classes_with_f1_zero": sum(1 for f in f1_scores if f == 0.0),
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "min_support": int(min(s for s in supports if s > 0)) if any(s > 0 for s in supports) else 0,
            "max_support": int(max(supports)),
            "mean_support": float(np.mean([s for s in supports if s > 0])) if any(s > 0 for s in supports) else 0
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(global_metrics, f, indent=2)
    
    def print_summary(self) -> None:
        """Print evaluation summary to console."""
        if self._result is None:
            print("No evaluation results. Call evaluate() first.")
            return
        
        global_metrics = self._result.get_global_metrics()
        per_class = self._result.get_per_class_metrics()
        
        print("\n" + "=" * 70)
        print("FINAL EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Samples evaluated: {global_metrics['num_samples']}")
        print(f"Classes: {global_metrics['num_classes']}")
        print()
        print("GLOBAL METRICS:")
        print(f"  Accuracy:           {global_metrics['accuracy']:.4f}")
        print(f"  Precision (macro):  {global_metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):     {global_metrics['recall_macro']:.4f}")
        print(f"  F1 (macro):         {global_metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted):      {global_metrics['f1_weighted']:.4f}")
        print()
        
        # F1 distribution
        f1_scores = [m["f1"] for m in per_class.values()]
        supports = [m["support"] for m in per_class.values()]
        classes_with_support = sum(1 for s in supports if s > 0)
        
        print("CLASS DISTRIBUTION:")
        print(f"  Classes with support:   {classes_with_support}/{len(per_class)}")
        print(f"  Classes with F1 > 0.5:  {sum(1 for f in f1_scores if f > 0.5)}")
        print(f"  Classes with F1 = 0:    {sum(1 for f in f1_scores if f == 0.0)}")
        print(f"  Mean F1:                {np.mean(f1_scores):.4f} (std: {np.std(f1_scores):.4f})")
        print("=" * 70)


def run_final_evaluation(
    model: nn.Module,
    val_loader: DataLoader,
    num_classes: int,
    output_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    dataset_name: str = "validation",
    epoch: Optional[int] = None,
    prefix: str = ""
) -> Tuple[EvaluationResult, Dict[str, Path]]:
    """
    Convenience function to run complete final evaluation.
    
    Args:
        model: Trained model
        val_loader: Validation DataLoader
        num_classes: Number of classes
        output_dir: Directory to save artifacts
        class_names: Optional class name mapping
        device: Evaluation device
        dataset_name: Name for titles
        epoch: Final epoch number
        prefix: Filename prefix
    
    Returns:
        Tuple of (EvaluationResult, dict of saved artifact paths)
    
    Example:
        >>> result, paths = run_final_evaluation(
        ...     model, val_loader, 505,
        ...     output_dir="experiments/run_001/",
        ...     class_names=gloss_names,
        ...     epoch=10
        ... )
        >>> print(f"Accuracy: {result.get_global_metrics()['accuracy']:.2%}")
    """
    evaluator = FinalEvaluator(
        model=model,
        dataloader=val_loader,
        num_classes=num_classes,
        class_names=class_names,
        device=device
    )
    
    result = evaluator.evaluate()
    evaluator.print_summary()
    
    paths = evaluator.save_artifacts(
        output_dir=output_dir,
        prefix=prefix,
        dataset_name=dataset_name,
        epoch=epoch
    )
    
    return result, paths
