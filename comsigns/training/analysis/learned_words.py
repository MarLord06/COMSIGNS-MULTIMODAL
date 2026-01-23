"""
Learned Words Analysis Module.

This module provides metrics for evaluating semantic learning effectiveness,
answering the key question: "How many words does the model reliably recognize?"

A class (gloss) is considered LEARNED if it meets ALL criteria:
1. Minimum support: support >= min_support (default=2)
2. Consistency: recall@1 >= recall_threshold (default=0.5)
3. Precision: precision@1 >= precision_threshold (default=0.5)
4. Separability: f1 >= f1_threshold (default=0.5)

These thresholds are configurable but have sensible defaults.

Example:
    >>> analyzer = LearnedWordsAnalyzer(
    ...     metrics_by_class=per_class_metrics,
    ...     bucket_mapping=bucket_mapping,
    ...     criteria=LearnedWordCriteria()
    ... )
    >>> report = analyzer.analyze()
    >>> print(f"Learned {report.learned_words_count} words")
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)


@dataclass
class LearnedWordCriteria:
    """
    Criteria for determining if a word (class) is learned.
    
    All thresholds must be met for a word to be considered "learned".
    
    Attributes:
        min_support: Minimum number of samples in validation set (default=2)
        min_precision: Minimum precision@1 threshold (default=0.5)
        min_recall: Minimum recall@1 threshold (default=0.5)
        min_f1: Minimum F1 score threshold (default=0.5)
    """
    min_support: int = 2
    min_precision: float = 0.5
    min_recall: float = 0.5
    min_f1: float = 0.5
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "LearnedWordCriteria":
        return cls(**data)
    
    def __post_init__(self):
        if self.min_support < 1:
            raise ValueError("min_support must be >= 1")
        if not 0.0 <= self.min_precision <= 1.0:
            raise ValueError("min_precision must be in [0, 1]")
        if not 0.0 <= self.min_recall <= 1.0:
            raise ValueError("min_recall must be in [0, 1]")
        if not 0.0 <= self.min_f1 <= 1.0:
            raise ValueError("min_f1 must be in [0, 1]")


@dataclass
class ClassMetrics:
    """
    Per-class metrics for learned words analysis.
    
    This is a lightweight dataclass for passing per-class metrics
    to the analyzer.
    """
    class_id: int
    support: int  # Number of samples in validation set
    precision: float
    recall: float
    f1: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BucketLearnedStats:
    """Statistics for learned words in a single bucket."""
    bucket_name: str
    total_classes: int
    eligible_classes: int  # Classes with support >= min_support
    learned_words: int
    pct_learned: float  # learned_words / eligible_classes
    learned_class_ids: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "bucket_name": self.bucket_name,
            "total_classes": self.total_classes,
            "eligible_classes": self.eligible_classes,
            "learned_words": self.learned_words,
            "pct_learned": self.pct_learned,
            "learned_class_ids": self.learned_class_ids
        }


@dataclass
class LearnedWordsReport:
    """
    Complete report of learned words analysis.
    
    Attributes:
        criteria: The criteria used for analysis
        total_classes: Total number of classes in the model
        eligible_classes: Classes meeting minimum support threshold
        learned_words_count: Number of classes meeting ALL criteria
        pct_vocabulary_learned: learned_words_count / eligible_classes
        by_bucket: Per-bucket statistics
        learned_class_ids: List of class IDs that are "learned"
        not_learned_reasons: Summary of why classes weren't learned
    """
    criteria: LearnedWordCriteria
    total_classes: int
    eligible_classes: int
    learned_words_count: int
    pct_vocabulary_learned: float
    by_bucket: Dict[str, BucketLearnedStats]
    learned_class_ids: List[int]
    not_learned_reasons: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "criteria": self.criteria.to_dict(),
            "total_classes": self.total_classes,
            "eligible_classes": self.eligible_classes,
            "learned_words_count": self.learned_words_count,
            "pct_vocabulary_learned": self.pct_vocabulary_learned,
            "by_bucket": {k: v.to_dict() for k, v in self.by_bucket.items()},
            "learned_class_ids": self.learned_class_ids,
            "not_learned_reasons": self.not_learned_reasons
        }
    
    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Learned words report saved to {path}")
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 60,
            "LEARNED WORDS ANALYSIS",
            "=" * 60,
            f"",
            f"Criteria:",
            f"  min_support={self.criteria.min_support}, "
            f"min_precision={self.criteria.min_precision}, "
            f"min_recall={self.criteria.min_recall}, "
            f"min_f1={self.criteria.min_f1}",
            f"",
            f"Results:",
            f"  Total classes: {self.total_classes}",
            f"  Eligible (support >= {self.criteria.min_support}): {self.eligible_classes}",
            f"  LEARNED WORDS: {self.learned_words_count}",
            f"  % Vocabulary Learned: {self.pct_vocabulary_learned:.1%}",
            f"",
            f"By Bucket:",
        ]
        
        for bucket_name, stats in self.by_bucket.items():
            lines.append(
                f"  {bucket_name}: {stats.learned_words}/{stats.eligible_classes} "
                f"learned ({stats.pct_learned:.1%})"
            )
        
        if self.not_learned_reasons:
            lines.extend([
                f"",
                f"Not Learned Reasons (for eligible classes):",
            ])
            for reason, count in sorted(self.not_learned_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  {reason}: {count}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class LearnedWordsAnalyzer:
    """
    Analyzer for determining which words the model has effectively learned.
    
    This class takes per-class metrics and determines which classes
    meet the criteria for being considered "learned".
    
    Example:
        >>> # From existing per-class metrics
        >>> metrics_by_class = {
        ...     0: ClassMetrics(0, support=10, precision=0.8, recall=0.7, f1=0.75),
        ...     1: ClassMetrics(1, support=5, precision=0.3, recall=0.2, f1=0.24),
        ... }
        >>> bucket_mapping = {0: "HEAD", 1: "MID"}
        >>> 
        >>> analyzer = LearnedWordsAnalyzer(
        ...     metrics_by_class=metrics_by_class,
        ...     bucket_mapping=bucket_mapping,
        ...     criteria=LearnedWordCriteria()
        ... )
        >>> report = analyzer.analyze()
        >>> print(report.learned_words_count)
        1
    """
    
    def __init__(
        self,
        metrics_by_class: Dict[int, ClassMetrics],
        bucket_mapping: Optional[Dict[int, str]] = None,
        criteria: Optional[LearnedWordCriteria] = None
    ):
        """
        Initialize the analyzer.
        
        Args:
            metrics_by_class: Dict mapping class_id to ClassMetrics
            bucket_mapping: Optional dict mapping class_id to bucket name
                          (HEAD, MID, TAIL, OTHER). If None, all classes
                          are treated as a single bucket "ALL".
            criteria: Criteria for determining learned words.
                     Uses defaults if None.
        """
        self.metrics_by_class = metrics_by_class
        self.bucket_mapping = bucket_mapping or {}
        self.criteria = criteria or LearnedWordCriteria()
        
        # Derive bucket to class IDs mapping
        self.bucket_to_classes: Dict[str, Set[int]] = {}
        for class_id in metrics_by_class.keys():
            bucket = self.bucket_mapping.get(class_id, "ALL")
            if bucket not in self.bucket_to_classes:
                self.bucket_to_classes[bucket] = set()
            self.bucket_to_classes[bucket].add(class_id)
    
    def is_word_learned(self, class_metrics: ClassMetrics) -> tuple[bool, str]:
        """
        Check if a word meets all criteria for being learned.
        
        Args:
            class_metrics: Metrics for the class
        
        Returns:
            Tuple of (is_learned, reason_if_not)
        """
        # Check support
        if class_metrics.support < self.criteria.min_support:
            return False, "insufficient_support"
        
        # Check precision
        if class_metrics.precision < self.criteria.min_precision:
            return False, "low_precision"
        
        # Check recall
        if class_metrics.recall < self.criteria.min_recall:
            return False, "low_recall"
        
        # Check F1
        if class_metrics.f1 < self.criteria.min_f1:
            return False, "low_f1"
        
        return True, "learned"
    
    def analyze(self) -> LearnedWordsReport:
        """
        Analyze all classes and generate learned words report.
        
        Returns:
            LearnedWordsReport with complete analysis
        """
        total_classes = len(self.metrics_by_class)
        eligible_classes = 0
        learned_class_ids: List[int] = []
        not_learned_reasons: Dict[str, int] = {}
        
        # Track per-bucket statistics
        bucket_stats: Dict[str, Dict[str, Any]] = {}
        for bucket_name in self.bucket_to_classes.keys():
            bucket_stats[bucket_name] = {
                "total": 0,
                "eligible": 0,
                "learned": 0,
                "learned_ids": []
            }
        
        # Analyze each class
        for class_id, metrics in self.metrics_by_class.items():
            bucket = self.bucket_mapping.get(class_id, "ALL")
            bucket_stats[bucket]["total"] += 1
            
            # Check if eligible (meets support threshold)
            if metrics.support >= self.criteria.min_support:
                eligible_classes += 1
                bucket_stats[bucket]["eligible"] += 1
                
                # Check if learned
                is_learned, reason = self.is_word_learned(metrics)
                
                if is_learned:
                    learned_class_ids.append(class_id)
                    bucket_stats[bucket]["learned"] += 1
                    bucket_stats[bucket]["learned_ids"].append(class_id)
                else:
                    not_learned_reasons[reason] = not_learned_reasons.get(reason, 0) + 1
        
        # Build bucket reports
        by_bucket: Dict[str, BucketLearnedStats] = {}
        for bucket_name, stats in bucket_stats.items():
            pct_learned = (
                stats["learned"] / stats["eligible"] 
                if stats["eligible"] > 0 else 0.0
            )
            by_bucket[bucket_name] = BucketLearnedStats(
                bucket_name=bucket_name,
                total_classes=stats["total"],
                eligible_classes=stats["eligible"],
                learned_words=stats["learned"],
                pct_learned=pct_learned,
                learned_class_ids=sorted(stats["learned_ids"])
            )
        
        # Calculate overall percentage
        pct_vocabulary_learned = (
            len(learned_class_ids) / eligible_classes
            if eligible_classes > 0 else 0.0
        )
        
        return LearnedWordsReport(
            criteria=self.criteria,
            total_classes=total_classes,
            eligible_classes=eligible_classes,
            learned_words_count=len(learned_class_ids),
            pct_vocabulary_learned=pct_vocabulary_learned,
            by_bucket=by_bucket,
            learned_class_ids=sorted(learned_class_ids),
            not_learned_reasons=not_learned_reasons
        )


def compute_class_metrics_from_predictions(
    predictions: List[int],
    targets: List[int],
    num_classes: int
) -> Dict[int, ClassMetrics]:
    """
    Compute per-class metrics from predictions and targets.
    
    This is a convenience function for computing ClassMetrics
    from raw model outputs.
    
    Args:
        predictions: List of predicted class IDs
        targets: List of ground truth class IDs
        num_classes: Total number of classes
    
    Returns:
        Dict mapping class_id to ClassMetrics
    """
    import numpy as np
    
    preds = np.array(predictions)
    tgts = np.array(targets)
    
    metrics_by_class = {}
    
    for class_id in range(num_classes):
        # Support = number of samples with this true class
        support = (tgts == class_id).sum()
        
        if support == 0:
            # No samples in validation set
            metrics_by_class[class_id] = ClassMetrics(
                class_id=class_id,
                support=0,
                precision=0.0,
                recall=0.0,
                f1=0.0
            )
            continue
        
        # True positives, false positives, false negatives
        tp = ((preds == class_id) & (tgts == class_id)).sum()
        fp = ((preds == class_id) & (tgts != class_id)).sum()
        fn = ((preds != class_id) & (tgts == class_id)).sum()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics_by_class[class_id] = ClassMetrics(
            class_id=class_id,
            support=int(support),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1)
        )
    
    return metrics_by_class


def analyze_learned_words(
    predictions: List[int],
    targets: List[int],
    num_classes: int,
    bucket_mapping: Optional[Dict[int, str]] = None,
    criteria: Optional[LearnedWordCriteria] = None
) -> LearnedWordsReport:
    """
    Convenience function for complete learned words analysis.
    
    This is the main entry point for learned words analysis.
    
    Args:
        predictions: List of predicted class IDs
        targets: List of ground truth class IDs
        num_classes: Total number of classes
        bucket_mapping: Optional dict mapping class_id to bucket name
        criteria: Optional criteria for determining learned words
    
    Returns:
        LearnedWordsReport with complete analysis
    
    Example:
        >>> report = analyze_learned_words(
        ...     predictions=preds,
        ...     targets=labels,
        ...     num_classes=51,
        ...     bucket_mapping=bucket_map
        ... )
        >>> print(f"Learned {report.learned_words_count} words")
    """
    # Compute per-class metrics
    metrics_by_class = compute_class_metrics_from_predictions(
        predictions, targets, num_classes
    )
    
    # Run analysis
    analyzer = LearnedWordsAnalyzer(
        metrics_by_class=metrics_by_class,
        bucket_mapping=bucket_mapping,
        criteria=criteria
    )
    
    return analyzer.analyze()
