"""
Coverage Metrics Module.

Computes semantic coverage metrics for classification models,
especially useful for long-tail scenarios where traditional
accuracy may be misleading.

Coverage metrics answer: "How well does the model cover the vocabulary?"
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CoverageResult:
    """Coverage analysis results."""
    
    # Global coverage
    coverage_at_1: float
    coverage_at_5: float
    coverage_at_10: float
    coverage_at_20: float
    
    # Per-class coverage (% of samples where class appears in top-k)
    per_class_coverage_5: Dict[int, float]
    per_class_coverage_10: Dict[int, float]
    
    # Vocabulary coverage (% of classes that appear at least once in top-k predictions)
    vocab_coverage_1: float  # % of classes predicted as top-1 at least once
    vocab_coverage_5: float  # % of classes appearing in any top-5
    vocab_coverage_10: float
    
    # Zero-hit classes (classes never in top-k)
    zero_hit_classes_5: List[int]
    zero_hit_classes_10: List[int]
    
    # Sample-level stats
    total_samples: int
    num_classes: int
    
    def to_dict(self) -> Dict:
        return {
            "coverage_at_1": self.coverage_at_1,
            "coverage_at_5": self.coverage_at_5,
            "coverage_at_10": self.coverage_at_10,
            "coverage_at_20": self.coverage_at_20,
            "vocab_coverage_1": self.vocab_coverage_1,
            "vocab_coverage_5": self.vocab_coverage_5,
            "vocab_coverage_10": self.vocab_coverage_10,
            "zero_hit_classes_5": self.zero_hit_classes_5,
            "zero_hit_classes_10": self.zero_hit_classes_10,
            "num_zero_hit_5": len(self.zero_hit_classes_5),
            "num_zero_hit_10": len(self.zero_hit_classes_10),
            "total_samples": self.total_samples,
            "num_classes": self.num_classes
        }


class CoverageAnalyzer:
    """
    Analyzes coverage metrics for classification.
    
    Coverage@K: % of samples where the true class appears in top-K predictions.
    This is equivalent to Recall@K but framed as "coverage".
    
    Vocabulary Coverage: % of classes that the model predicts at least once
    in its top-K predictions across the entire validation set.
    
    Zero-Hit Classes: Classes that never appear in top-K predictions,
    indicating the model effectively "forgot" them.
    """
    
    def __init__(self):
        self._y_true: Optional[np.ndarray] = None
        self._y_logits: Optional[np.ndarray] = None
        self._y_pred_topk: Dict[int, np.ndarray] = {}
        self._class_names: Dict[int, str] = {}
        self._result: Optional[CoverageResult] = None
    
    def load_data(
        self,
        y_true: np.ndarray,
        y_logits: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> None:
        """
        Load evaluation data.
        
        Args:
            y_true: Ground truth labels [N]
            y_logits: Model logits [N, C]
            class_names: Optional list of class names
        """
        self._y_true = np.asarray(y_true)
        self._y_logits = np.asarray(y_logits)
        
        # Precompute top-k predictions for efficiency
        for k in [1, 5, 10, 20]:
            self._y_pred_topk[k] = np.argsort(self._y_logits, axis=1)[:, -k:]
        
        if class_names:
            self._class_names = {i: name for i, name in enumerate(class_names)}
        
        logger.info(f"Loaded {len(y_true)} samples, {y_logits.shape[1]} classes")
    
    def load_from_files(
        self,
        logits_file: Union[str, Path],
        labels_file: Union[str, Path]
    ) -> None:
        """
        Load data from numpy files.
        
        Args:
            logits_file: Path to .npy file with logits [N, C]
            labels_file: Path to .npy file with labels [N]
        """
        logits = np.load(logits_file)
        labels = np.load(labels_file)
        self.load_data(labels, logits)
    
    def analyze(self) -> CoverageResult:
        """Perform coverage analysis."""
        if self._y_true is None or self._y_logits is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        n_samples = len(self._y_true)
        n_classes = self._y_logits.shape[1]
        
        # Coverage@K (% of samples with true class in top-K)
        cov_1 = self._compute_coverage(1)
        cov_5 = self._compute_coverage(5)
        cov_10 = self._compute_coverage(10)
        cov_20 = self._compute_coverage(20)
        
        # Per-class coverage
        per_class_5 = self._compute_per_class_coverage(5)
        per_class_10 = self._compute_per_class_coverage(10)
        
        # Vocabulary coverage (classes that appear in any prediction)
        vocab_1, zero_1 = self._compute_vocab_coverage(1)
        vocab_5, zero_5 = self._compute_vocab_coverage(5)
        vocab_10, zero_10 = self._compute_vocab_coverage(10)
        
        self._result = CoverageResult(
            coverage_at_1=cov_1,
            coverage_at_5=cov_5,
            coverage_at_10=cov_10,
            coverage_at_20=cov_20,
            per_class_coverage_5=per_class_5,
            per_class_coverage_10=per_class_10,
            vocab_coverage_1=vocab_1,
            vocab_coverage_5=vocab_5,
            vocab_coverage_10=vocab_10,
            zero_hit_classes_5=zero_5,
            zero_hit_classes_10=zero_10,
            total_samples=n_samples,
            num_classes=n_classes
        )
        
        return self._result
    
    def _compute_coverage(self, k: int) -> float:
        """Compute Coverage@K."""
        topk = self._y_pred_topk[k]
        hits = np.array([self._y_true[i] in topk[i] for i in range(len(self._y_true))])
        return hits.mean()
    
    def _compute_per_class_coverage(self, k: int) -> Dict[int, float]:
        """Compute coverage for each class."""
        topk = self._y_pred_topk[k]
        n_classes = self._y_logits.shape[1]
        
        per_class = {}
        for c in range(n_classes):
            mask = self._y_true == c
            if mask.sum() == 0:
                per_class[c] = 0.0
                continue
            
            class_samples = topk[mask]
            hits = np.array([c in class_samples[i] for i in range(len(class_samples))])
            per_class[c] = hits.mean()
        
        return per_class
    
    def _compute_vocab_coverage(self, k: int) -> Tuple[float, List[int]]:
        """
        Compute vocabulary coverage.
        
        Returns:
            (coverage_ratio, list_of_zero_hit_classes)
        """
        topk = self._y_pred_topk[k]
        n_classes = self._y_logits.shape[1]
        
        # Classes that appear in any top-k prediction
        predicted_classes = set(topk.flatten())
        
        # All classes
        all_classes = set(range(n_classes))
        
        # Zero-hit classes
        zero_hit = sorted(all_classes - predicted_classes)
        
        vocab_coverage = len(predicted_classes) / n_classes
        
        return vocab_coverage, zero_hit
    
    def print_report(self) -> None:
        """Print coverage report."""
        if self._result is None:
            print("No results. Run analyze() first.")
            return
        
        r = self._result
        
        print("\n" + "=" * 60)
        print("COVERAGE METRICS REPORT")
        print("=" * 60)
        
        print(f"\nTotal Samples: {r.total_samples}")
        print(f"Total Classes: {r.num_classes}")
        
        print("\n" + "-" * 60)
        print("COVERAGE@K (% samples with true class in top-K)")
        print("-" * 60)
        print(f"  Coverage@1:  {r.coverage_at_1:>8.2%}")
        print(f"  Coverage@5:  {r.coverage_at_5:>8.2%}")
        print(f"  Coverage@10: {r.coverage_at_10:>8.2%}")
        print(f"  Coverage@20: {r.coverage_at_20:>8.2%}")
        
        print("\n" + "-" * 60)
        print("VOCABULARY COVERAGE (% classes appearing in predictions)")
        print("-" * 60)
        print(f"  Vocab Coverage@1:  {r.vocab_coverage_1:>8.2%} ({int(r.vocab_coverage_1 * r.num_classes)} classes)")
        print(f"  Vocab Coverage@5:  {r.vocab_coverage_5:>8.2%} ({int(r.vocab_coverage_5 * r.num_classes)} classes)")
        print(f"  Vocab Coverage@10: {r.vocab_coverage_10:>8.2%} ({int(r.vocab_coverage_10 * r.num_classes)} classes)")
        
        print("\n" + "-" * 60)
        print("ZERO-HIT CLASSES (never appear in top-K)")
        print("-" * 60)
        print(f"  Zero-Hit@5:  {len(r.zero_hit_classes_5)} classes ({len(r.zero_hit_classes_5)/r.num_classes:.1%})")
        print(f"  Zero-Hit@10: {len(r.zero_hit_classes_10)} classes ({len(r.zero_hit_classes_10)/r.num_classes:.1%})")
        
        if len(r.zero_hit_classes_5) > 0 and len(r.zero_hit_classes_5) <= 20:
            print(f"\n  Classes never in top-5:")
            for cid in r.zero_hit_classes_5[:20]:
                name = self._class_names.get(cid, f"class_{cid}")
                print(f"    - {name} (id={cid})")
    
    def export_results(self, output_path: Union[str, Path]) -> None:
        """Export results to JSON."""
        if self._result is None:
            raise RuntimeError("No results to export")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._result.to_dict(), f, indent=2)
        
        logger.info(f"Coverage results exported to {output_path}")


def compute_coverage_metrics(
    y_true: np.ndarray,
    y_logits: np.ndarray,
    class_names: Optional[List[str]] = None,
    print_report: bool = True
) -> CoverageResult:
    """
    Convenience function to compute coverage metrics.
    
    Args:
        y_true: Ground truth labels [N]
        y_logits: Model logits [N, C]
        class_names: Optional list of class names
        print_report: Whether to print report
    
    Returns:
        CoverageResult
    """
    analyzer = CoverageAnalyzer()
    analyzer.load_data(y_true, y_logits, class_names)
    result = analyzer.analyze()
    
    if print_report:
        analyzer.print_report()
    
    return result
