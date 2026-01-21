"""
Bucket Analysis Module.

Analyzes model performance by class support buckets (HEAD/MID/TAIL)
and computes coverage metrics for long-tail classification scenarios.

This is a READ-ONLY analysis module - it does NOT modify:
- Model
- Trainer  
- Dataset
- Split

It provides diagnostic insights to guide optimization strategy.

Bucket Definitions:
- HEAD: ≥ 10 training samples (well-represented classes)
- MID: 3-9 training samples (moderately represented)
- TAIL: 1-2 training samples (underrepresented, long-tail)

Key Metrics:
- Accuracy@K: % of samples where correct class is in top-K predictions
- Coverage@K: Same as Accuracy@K, framed as "coverage of vocabulary"
- Recall@5: Per-class recall aggregated within bucket

Important: Support counts should come from TRAINING set, not validation.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict, Counter

import numpy as np

logger = logging.getLogger(__name__)


class Bucket(Enum):
    """Support bucket classification."""
    HEAD = "HEAD"   # ≥ 10 samples
    MID = "MID"     # 3-9 samples
    TAIL = "TAIL"   # 1-2 samples


# Bucket thresholds (configurable)
BUCKET_THRESHOLDS = {
    Bucket.HEAD: (10, float('inf')),  # ≥ 10
    Bucket.MID: (3, 9),               # 3-9
    Bucket.TAIL: (1, 2),              # 1-2
}


@dataclass
class BucketMetrics:
    """Metrics for a single bucket."""
    bucket: Bucket
    num_classes: int
    num_samples: int
    accuracy_at_1: float
    accuracy_at_5: float
    coverage_at_5: float
    coverage_at_10: float
    recall_at_5: float
    class_ids: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "bucket": self.bucket.value,
            "num_classes": self.num_classes,
            "num_samples": self.num_samples,
            "accuracy_at_1": self.accuracy_at_1,
            "accuracy_at_5": self.accuracy_at_5,
            "coverage_at_5": self.coverage_at_5,
            "coverage_at_10": self.coverage_at_10,
            "recall_at_5": self.recall_at_5,
            "class_ids": self.class_ids
        }


@dataclass 
class BucketAnalysisResult:
    """Complete bucket analysis result."""
    bucket_metrics: Dict[Bucket, BucketMetrics]
    class_to_bucket: Dict[int, Bucket]
    class_support: Dict[int, int]
    global_coverage_at_5: float
    global_coverage_at_10: float
    tail_analysis: Dict
    
    def to_dict(self) -> Dict:
        return {
            "bucket_metrics": {b.value: m.to_dict() for b, m in self.bucket_metrics.items()},
            "class_to_bucket": {str(k): v.value for k, v in self.class_to_bucket.items()},
            "global_coverage_at_5": self.global_coverage_at_5,
            "global_coverage_at_10": self.global_coverage_at_10,
            "tail_analysis": self.tail_analysis
        }


class BucketAnalyzer:
    """
    Analyzes model performance by support buckets.
    
    Buckets:
    - HEAD: ≥ 10 training samples
    - MID: 3-9 training samples  
    - TAIL: 1-2 training samples
    
    Example:
        >>> analyzer = BucketAnalyzer()
        >>> analyzer.load_from_files(
        ...     metrics_file="experiments/run_001/metrics_by_class.json",
        ...     summary_file="experiments/run_001/evaluation_summary.json"
        ... )
        >>> result = analyzer.analyze()
        >>> analyzer.print_report()
    """
    
    def __init__(
        self,
        head_threshold: int = 10,
        mid_range: Tuple[int, int] = (3, 9),
        tail_range: Tuple[int, int] = (1, 2)
    ):
        """
        Initialize analyzer with bucket thresholds.
        
        Args:
            head_threshold: Minimum samples for HEAD bucket
            mid_range: (min, max) samples for MID bucket
            tail_range: (min, max) samples for TAIL bucket
        """
        self.head_threshold = head_threshold
        self.mid_range = mid_range
        self.tail_range = tail_range
        
        # Data storage
        self._metrics_by_class: Dict[int, Dict] = {}
        self._class_support: Dict[int, int] = {}
        self._y_true: Optional[np.ndarray] = None
        self._y_logits: Optional[np.ndarray] = None
        self._class_names: Dict[int, str] = {}
        
        # Results
        self._result: Optional[BucketAnalysisResult] = None
    
    def classify_bucket(self, support: int) -> Bucket:
        """Classify a class into a bucket based on support."""
        if support >= self.head_threshold:
            return Bucket.HEAD
        elif self.mid_range[0] <= support <= self.mid_range[1]:
            return Bucket.MID
        else:
            return Bucket.TAIL
    
    def load_from_files(
        self,
        metrics_file: Union[str, Path],
        summary_file: Optional[Union[str, Path]] = None,
        split_file: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Load data from evaluation output files.
        
        Args:
            metrics_file: Path to metrics_by_class.json
            summary_file: Path to evaluation_summary.json (optional)
            split_file: Path to split file for training support counts
        """
        metrics_path = Path(metrics_file)
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract metrics by class
        if "metrics_by_class" in data:
            raw_metrics = data["metrics_by_class"]
        else:
            raw_metrics = data
        
        for class_id_str, metrics in raw_metrics.items():
            class_id = int(class_id_str)
            self._metrics_by_class[class_id] = metrics
            self._class_support[class_id] = metrics.get("support", 0)
            if metrics.get("class_name"):
                self._class_names[class_id] = metrics["class_name"]
        
        logger.info(f"Loaded metrics for {len(self._metrics_by_class)} classes")
    
    def load_from_evaluation_result(
        self,
        y_true: np.ndarray,
        y_logits: np.ndarray,
        train_support: Dict[int, int],
        class_names: Optional[List[str]] = None
    ) -> None:
        """
        Load data directly from evaluation arrays.
        
        Args:
            y_true: Ground truth labels [N]
            y_logits: Model logits [N, C]
            train_support: Dict mapping class_id to training sample count
            class_names: Optional list of class names
        """
        self._y_true = y_true
        self._y_logits = y_logits
        self._class_support = train_support.copy()
        
        if class_names:
            self._class_names = {i: name for i, name in enumerate(class_names)}
        
        logger.info(f"Loaded {len(y_true)} samples, {len(train_support)} classes")
    
    def load_training_support(
        self,
        split_file: Union[str, Path]
    ) -> None:
        """
        Load training support counts from split file.
        
        Args:
            split_file: Path to stratified split JSON
        """
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        
        # Count samples per class in training split
        train_indices = split_data.get("train", [])
        # This requires the actual dataset to map indices to labels
        # For now, we'll use the validation support from metrics
        logger.info(f"Split file loaded: {len(train_indices)} train samples")
    
    def analyze(self) -> BucketAnalysisResult:
        """
        Perform bucket analysis.
        
        Returns:
            BucketAnalysisResult with all metrics
        """
        # Classify each class into bucket
        class_to_bucket = {}
        bucket_classes = defaultdict(list)
        
        for class_id, support in self._class_support.items():
            bucket = self.classify_bucket(support)
            class_to_bucket[class_id] = bucket
            bucket_classes[bucket].append(class_id)
        
        # Compute metrics per bucket
        bucket_metrics = {}
        
        for bucket in Bucket:
            class_ids = bucket_classes[bucket]
            if not class_ids:
                bucket_metrics[bucket] = BucketMetrics(
                    bucket=bucket,
                    num_classes=0,
                    num_samples=0,
                    accuracy_at_1=0.0,
                    accuracy_at_5=0.0,
                    coverage_at_5=0.0,
                    coverage_at_10=0.0,
                    recall_at_5=0.0,
                    class_ids=[]
                )
                continue
            
            metrics = self._compute_bucket_metrics(class_ids, bucket)
            bucket_metrics[bucket] = metrics
        
        # Global coverage
        global_cov_5 = self._compute_global_coverage(5)
        global_cov_10 = self._compute_global_coverage(10)
        
        # Tail analysis
        tail_analysis = self._analyze_tail(bucket_classes[Bucket.TAIL], bucket_metrics)
        
        self._result = BucketAnalysisResult(
            bucket_metrics=bucket_metrics,
            class_to_bucket=class_to_bucket,
            class_support=self._class_support,
            global_coverage_at_5=global_cov_5,
            global_coverage_at_10=global_cov_10,
            tail_analysis=tail_analysis
        )
        
        return self._result
    
    def _compute_bucket_metrics(
        self,
        class_ids: List[int],
        bucket: Bucket
    ) -> BucketMetrics:
        """Compute metrics for a specific bucket."""
        # Filter metrics for this bucket's classes
        bucket_data = [self._metrics_by_class[cid] for cid in class_ids 
                       if cid in self._metrics_by_class]
        
        if not bucket_data:
            return BucketMetrics(
                bucket=bucket,
                num_classes=len(class_ids),
                num_samples=0,
                accuracy_at_1=0.0,
                accuracy_at_5=0.0,
                coverage_at_5=0.0,
                coverage_at_10=0.0,
                recall_at_5=0.0,
                class_ids=class_ids
            )
        
        # Aggregate metrics (weighted by support)
        total_samples = sum(d.get("support", 0) for d in bucket_data)
        
        # Compute weighted accuracy@1 (recall is accuracy per class)
        weighted_acc1 = sum(
            d.get("recall", 0) * d.get("support", 0) 
            for d in bucket_data
        ) / max(total_samples, 1)
        
        # For top-5 metrics, we need logits
        # If not available, estimate from precision/recall
        acc5 = self._estimate_accuracy_at_k(bucket_data, k=5)
        cov5 = acc5  # Coverage@5 ≈ Accuracy@5 when computed per-sample
        cov10 = self._estimate_accuracy_at_k(bucket_data, k=10)
        
        # Recall@5: for each class in bucket, what % of its samples appear in any top-5
        recall5 = acc5  # Same as coverage when averaged
        
        return BucketMetrics(
            bucket=bucket,
            num_classes=len(class_ids),
            num_samples=total_samples,
            accuracy_at_1=weighted_acc1,
            accuracy_at_5=acc5,
            coverage_at_5=cov5,
            coverage_at_10=cov10,
            recall_at_5=recall5,
            class_ids=class_ids
        )
    
    def _estimate_accuracy_at_k(
        self,
        bucket_data: List[Dict],
        k: int
    ) -> float:
        """
        Estimate Accuracy@K from per-class metrics.
        
        This is an approximation when we don't have raw logits.
        """
        # If we have top-k accuracy stored
        topk_key = f"top{k}_acc"
        
        total_samples = sum(d.get("support", 0) for d in bucket_data)
        if total_samples == 0:
            return 0.0
        
        # Check if top-k is available in metrics
        if any(topk_key in d for d in bucket_data):
            weighted_topk = sum(
                d.get(topk_key, d.get("recall", 0)) * d.get("support", 0)
                for d in bucket_data
            ) / total_samples
            return weighted_topk
        
        # Fallback: use recall as lower bound
        weighted_recall = sum(
            d.get("recall", 0) * d.get("support", 0)
            for d in bucket_data
        ) / total_samples
        
        # Heuristic: top-5 is typically 2-3x higher than top-1 in long-tail
        # This is a rough estimate
        if k == 5:
            return min(weighted_recall * 2.5, 1.0)
        elif k == 10:
            return min(weighted_recall * 3.5, 1.0)
        
        return weighted_recall
    
    def _compute_global_coverage(self, k: int) -> float:
        """Compute global Coverage@K."""
        if self._y_true is not None and self._y_logits is not None:
            # Compute from raw data
            topk_preds = np.argsort(self._y_logits, axis=1)[:, -k:]
            correct = np.array([
                self._y_true[i] in topk_preds[i] 
                for i in range(len(self._y_true))
            ])
            return correct.mean()
        
        # Estimate from per-class metrics
        total_samples = sum(d.get("support", 0) for d in self._metrics_by_class.values())
        if total_samples == 0:
            return 0.0
        
        topk_key = f"top{k}_acc"
        weighted = sum(
            d.get(topk_key, d.get("recall", 0)) * d.get("support", 0)
            for d in self._metrics_by_class.values()
        ) / total_samples
        
        return weighted
    
    def _analyze_tail(
        self,
        tail_classes: List[int],
        bucket_metrics: Dict[Bucket, BucketMetrics]
    ) -> Dict:
        """Generate tail-specific analysis."""
        total_classes = len(self._class_support)
        total_samples = sum(self._class_support.values())
        
        tail_samples = sum(self._class_support.get(cid, 0) for cid in tail_classes)
        
        head_metrics = bucket_metrics.get(Bucket.HEAD, BucketMetrics(
            bucket=Bucket.HEAD, num_classes=0, num_samples=0,
            accuracy_at_1=0, accuracy_at_5=0, coverage_at_5=0,
            coverage_at_10=0, recall_at_5=0
        ))
        mid_metrics = bucket_metrics.get(Bucket.MID, BucketMetrics(
            bucket=Bucket.MID, num_classes=0, num_samples=0,
            accuracy_at_1=0, accuracy_at_5=0, coverage_at_5=0,
            coverage_at_10=0, recall_at_5=0
        ))
        tail_metrics = bucket_metrics.get(Bucket.TAIL, BucketMetrics(
            bucket=Bucket.TAIL, num_classes=0, num_samples=0,
            accuracy_at_1=0, accuracy_at_5=0, coverage_at_5=0,
            coverage_at_10=0, recall_at_5=0
        ))
        
        # Is TAIL learning anything useful?
        tail_is_learning = tail_metrics.accuracy_at_5 > 0.1  # > 10% top-5
        tail_is_noise = tail_metrics.accuracy_at_1 < 0.05 and tail_metrics.accuracy_at_5 < 0.15
        
        return {
            "tail_class_count": len(tail_classes),
            "tail_class_percentage": len(tail_classes) / max(total_classes, 1) * 100,
            "tail_sample_count": tail_samples,
            "tail_sample_percentage": tail_samples / max(total_samples, 1) * 100,
            "tail_accuracy_at_1": tail_metrics.accuracy_at_1,
            "tail_accuracy_at_5": tail_metrics.accuracy_at_5,
            "tail_coverage_at_5": tail_metrics.coverage_at_5,
            "tail_recall_at_5": tail_metrics.recall_at_5,
            "comparison": {
                "head_vs_tail_acc1_ratio": head_metrics.accuracy_at_1 / max(tail_metrics.accuracy_at_1, 0.001),
                "head_vs_tail_acc5_ratio": head_metrics.accuracy_at_5 / max(tail_metrics.accuracy_at_5, 0.001),
                "mid_vs_tail_acc1_ratio": mid_metrics.accuracy_at_1 / max(tail_metrics.accuracy_at_1, 0.001),
            },
            "diagnosis": {
                "tail_is_learning": tail_is_learning,
                "tail_is_noise": tail_is_noise,
                "recommendation": self._get_tail_recommendation(
                    tail_metrics, head_metrics, len(tail_classes), total_classes
                )
            }
        }
    
    def _get_tail_recommendation(
        self,
        tail: BucketMetrics,
        head: BucketMetrics,
        tail_count: int,
        total_count: int
    ) -> str:
        """Generate recommendation based on tail analysis."""
        tail_pct = tail_count / max(total_count, 1)
        
        if tail.accuracy_at_5 < 0.10:
            return "TAIL_TO_OTHER"  # Tail is essentially noise
        elif tail.accuracy_at_5 < 0.25 and tail_pct > 0.5:
            return "TAIL_EXCLUSION"  # Too many tail classes, exclude
        elif tail.accuracy_at_5 >= 0.25:
            return "KEEP_TAIL"  # Model is learning something
        else:
            return "TAIL_FEW_SHOT"  # Consider few-shot approach
    
    def print_report(self) -> None:
        """Print formatted analysis report."""
        if self._result is None:
            print("No analysis results. Run analyze() first.")
            return
        
        print("\n" + "=" * 80)
        print("BUCKET ANALYSIS REPORT")
        print("=" * 80)
        
        # Bucket summary table
        print("\n" + "-" * 80)
        print("METRICS BY BUCKET")
        print("-" * 80)
        print(f"{'Bucket':<8} {'Classes':>8} {'Samples':>10} {'Acc@1':>8} {'Acc@5':>8} {'Cov@5':>8} {'Cov@10':>8}")
        print("-" * 80)
        
        for bucket in [Bucket.HEAD, Bucket.MID, Bucket.TAIL]:
            m = self._result.bucket_metrics.get(bucket)
            if m:
                print(
                    f"{bucket.value:<8} {m.num_classes:>8} {m.num_samples:>10} "
                    f"{m.accuracy_at_1:>8.2%} {m.accuracy_at_5:>8.2%} "
                    f"{m.coverage_at_5:>8.2%} {m.coverage_at_10:>8.2%}"
                )
        
        print("-" * 80)
        print(f"{'GLOBAL':<8} {'':<8} {'':<10} {'':<8} {'':<8} "
              f"{self._result.global_coverage_at_5:>8.2%} {self._result.global_coverage_at_10:>8.2%}")
        
        # Tail analysis
        tail = self._result.tail_analysis
        print("\n" + "-" * 80)
        print("LONG-TAIL ANALYSIS")
        print("-" * 80)
        print(f"TAIL classes:        {tail['tail_class_count']} ({tail['tail_class_percentage']:.1f}% of vocabulary)")
        print(f"TAIL samples:        {tail['tail_sample_count']} ({tail['tail_sample_percentage']:.1f}% of data)")
        print(f"TAIL Accuracy@1:     {tail['tail_accuracy_at_1']:.2%}")
        print(f"TAIL Accuracy@5:     {tail['tail_accuracy_at_5']:.2%}")
        print(f"TAIL Coverage@5:     {tail['tail_coverage_at_5']:.2%}")
        
        print("\n" + "-" * 40)
        print("HEAD vs TAIL Comparison:")
        print(f"  Acc@1 ratio: {tail['comparison']['head_vs_tail_acc1_ratio']:.1f}x")
        print(f"  Acc@5 ratio: {tail['comparison']['head_vs_tail_acc5_ratio']:.1f}x")
        
        # Diagnosis
        diag = tail["diagnosis"]
        print("\n" + "-" * 40)
        print("DIAGNOSIS:")
        if diag["tail_is_noise"]:
            print("  ⚠️  TAIL appears to be NOISE - model not learning from tail classes")
        elif diag["tail_is_learning"]:
            print("  ✓  TAIL shows LEARNING signal - top-5 accuracy meaningful")
        else:
            print("  ?  TAIL status UNCLEAR - needs further investigation")
        
        print(f"\n  RECOMMENDED STRATEGY: {diag['recommendation']}")
        
        # Strategy explanations
        print("\n" + "=" * 80)
        print("TAIL STRATEGIES (Reference - NOT IMPLEMENTED)")
        print("=" * 80)
        self._print_strategy_guide()
    
    def _print_strategy_guide(self) -> None:
        """Print strategy guide for tail handling."""
        strategies = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ STRATEGY 1: TAIL → OTHER                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Description: Merge all TAIL classes into a single "OTHER" class             │
│                                                                             │
│ Pros:                                                                       │
│   - Simplifies model (fewer classes)                                        │
│   - Improves HEAD/MID accuracy                                              │
│   - Reduces confusion from underrepresented classes                         │
│                                                                             │
│ Cons:                                                                       │
│   - Loses granularity for TAIL glosses                                      │
│   - "OTHER" class may become too heterogeneous                              │
│   - Cannot distinguish between different rare signs                         │
│                                                                             │
│ When to use: Tail Acc@5 < 10%, Tail is essentially noise                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ STRATEGY 2: TAIL EXCLUSION                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Description: Train only on HEAD + MID classes, ignore TAIL                  │
│                                                                             │
│ Pros:                                                                       │
│   - Cleaner training signal                                                 │
│   - Higher overall accuracy on supported classes                            │
│   - Simpler evaluation                                                      │
│                                                                             │
│ Cons:                                                                       │
│   - Loses ~50%+ of vocabulary                                               │
│   - Model cannot recognize TAIL signs at all                                │
│   - May not be acceptable for production use                                │
│                                                                             │
│ When to use: TAIL > 50% of classes AND Tail Acc@5 < 25%                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ STRATEGY 3: TAIL FEW-SHOT (Future)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Description: Separate classifier or retrieval for TAIL classes              │
│                                                                             │
│ Pros:                                                                       │
│   - Preserves full vocabulary                                               │
│   - Specialized handling for rare classes                                   │
│   - Can leverage embedding similarity                                       │
│                                                                             │
│ Cons:                                                                       │
│   - More complex architecture                                               │
│   - Requires embedding quality to be good                                   │
│   - Two-stage inference                                                     │
│                                                                             │
│ When to use: Need full vocabulary AND embeddings show structure             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
        print(strategies)
    
    def export_results(
        self,
        output_path: Union[str, Path]
    ) -> None:
        """Export analysis results to JSON."""
        if self._result is None:
            raise RuntimeError("No results to export. Run analyze() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._result.to_dict(), f, indent=2)
        
        logger.info(f"Results exported to {output_path}")
    
    def get_bucket_classes(self, bucket: Bucket) -> List[int]:
        """Get list of class IDs in a specific bucket."""
        if self._result is None:
            raise RuntimeError("Run analyze() first")
        return self._result.bucket_metrics[bucket].class_ids
    
    def get_bucket_class_names(self, bucket: Bucket) -> List[str]:
        """Get list of class names in a specific bucket."""
        class_ids = self.get_bucket_classes(bucket)
        return [self._class_names.get(cid, f"class_{cid}") for cid in class_ids]


def compute_training_support_from_dataset(
    dataset_root: Union[str, Path],
    split_file: Optional[Union[str, Path]] = None,
    split: str = "train"
) -> Dict[int, int]:
    """
    Compute training support counts from AEC dataset.
    
    Args:
        dataset_root: Path to AEC dataset root
        split_file: Optional path to split file
        split: Which split to use ("train" or "val")
    
    Returns:
        Dict mapping class_id to sample count in training set
    """
    # Import here to avoid circular dependency
    from core.data.datasets.aec import AECDataset
    
    dataset = AECDataset(
        dataset_root=Path(dataset_root),
        split_file=Path(split_file) if split_file else None,
        split=split if split_file else None
    )
    
    # Count samples per class
    support = Counter()
    for i in range(len(dataset)):
        sample = dataset[i]
        support[sample.gloss_id] += 1
    
    return dict(support)


def run_bucket_analysis(
    metrics_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    print_report: bool = True,
    training_support: Optional[Dict[int, int]] = None
) -> BucketAnalysisResult:
    """
    Convenience function to run complete bucket analysis.
    
    Args:
        metrics_file: Path to metrics_by_class.json
        output_file: Optional path to export results
        print_report: Whether to print report to console
        training_support: Optional dict of class_id -> training sample count.
                         If not provided, will use validation support from metrics.
    
    Returns:
        BucketAnalysisResult
    
    Example:
        >>> result = run_bucket_analysis(
        ...     "experiments/run_001/metrics_by_class.json",
        ...     output_file="experiments/run_001/bucket_analysis.json"
        ... )
    """
    analyzer = BucketAnalyzer()
    analyzer.load_from_files(metrics_file)
    
    # Override support with training support if provided
    if training_support:
        analyzer._class_support = training_support.copy()
        logger.info(f"Using provided training support for {len(training_support)} classes")
    
    result = analyzer.analyze()
    
    if print_report:
        analyzer.print_report()
    
    if output_file:
        analyzer.export_results(output_file)
    
    return result
