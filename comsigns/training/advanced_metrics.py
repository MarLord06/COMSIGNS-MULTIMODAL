"""
Advanced metrics for ComSigns training.

Includes:
- Learned words analysis
- Rejection metrics (FAR, FRR)
- Per-bucket metrics (HEAD, MID, OTHER)
- Composite scoring for model selection
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LearnedWordCriteria:
    """Criteria for considering a word as "learned"."""
    min_support: int = 3
    min_precision: float = 0.6
    min_recall: float = 0.5
    min_f1: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            "min_support": self.min_support,
            "min_precision": self.min_precision,
            "min_recall": self.min_recall,
            "min_f1": self.min_f1
        }


@dataclass
class WordMetrics:
    """Per-word metrics."""
    gloss: str
    class_id: int
    bucket: str  # HEAD, MID, OTHER
    support: int
    predictions: int
    correct: int
    precision: float
    recall: float
    f1: float
    is_learned: bool
    not_learned_reason: Optional[str] = None


@dataclass
class LearnedWordsReport:
    """Report of learned words analysis."""
    criteria: LearnedWordCriteria
    total_classes: int
    learned_count: int
    not_learned_count: int
    pct_vocabulary_learned: float
    learned_by_bucket: Dict[str, int]
    learned_words: List[str]
    not_learned_words: List[Dict]  # {word, reason, metrics}
    word_metrics: List[WordMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "criteria": self.criteria.to_dict(),
            "summary": {
                "total_classes": self.total_classes,
                "learned_count": self.learned_count,
                "not_learned_count": self.not_learned_count,
                "pct_vocabulary_learned": round(self.pct_vocabulary_learned, 4)
            },
            "learned_by_bucket": self.learned_by_bucket,
            "learned_words": self.learned_words,
            "not_learned_words": self.not_learned_words
        }
    
    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    def get_summary(self) -> str:
        lines = [
            "=" * 60,
            "LEARNED WORDS REPORT",
            "=" * 60,
            f"Criteria: support≥{self.criteria.min_support}, "
            f"P≥{self.criteria.min_precision}, R≥{self.criteria.min_recall}, "
            f"F1≥{self.criteria.min_f1}",
            "",
            f"Total classes: {self.total_classes}",
            f"Learned: {self.learned_count} ({self.pct_vocabulary_learned:.1%})",
            f"Not learned: {self.not_learned_count}",
            "",
            "By bucket:",
        ]
        for bucket, count in self.learned_by_bucket.items():
            lines.append(f"  {bucket}: {count}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class RejectionMetrics:
    """Rejection/acceptance metrics for the model."""
    # Thresholds used
    confidence_threshold: float
    margin_threshold: float
    
    # Basic counts
    total_samples: int
    accepted_samples: int
    rejected_samples: int
    
    # False Accept Rate: % of OTHER/wrong predictions accepted with high confidence
    false_accepts: int
    false_accept_rate: float
    
    # False Reject Rate: % of correct predictions rejected due to low confidence
    false_rejects: int
    false_reject_rate: float
    
    # Accept@threshold metrics
    accept_at_conf: float  # Accuracy among accepted samples (by confidence)
    accept_at_margin: float  # Accuracy among accepted samples (by margin)
    
    # Predictions going to OTHER
    predictions_to_other: int
    pct_predictions_other: float
    
    def to_dict(self) -> Dict:
        return {
            "thresholds": {
                "confidence": self.confidence_threshold,
                "margin": self.margin_threshold
            },
            "counts": {
                "total": self.total_samples,
                "accepted": self.accepted_samples,
                "rejected": self.rejected_samples
            },
            "false_accept": {
                "count": self.false_accepts,
                "rate": round(self.false_accept_rate, 4)
            },
            "false_reject": {
                "count": self.false_rejects,
                "rate": round(self.false_reject_rate, 4)
            },
            "accept_accuracy": {
                "by_confidence": round(self.accept_at_conf, 4),
                "by_margin": round(self.accept_at_margin, 4)
            },
            "other_predictions": {
                "count": self.predictions_to_other,
                "percentage": round(self.pct_predictions_other, 4)
            }
        }
    
    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


@dataclass 
class BucketMetrics:
    """Metrics per bucket (HEAD, MID, OTHER)."""
    bucket: str
    num_classes: int
    total_support: int
    accuracy_at_1: float
    accuracy_at_5: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    coverage: float  # % of predictions that go to this bucket


@dataclass
class CompositeScore:
    """Composite score for model selection."""
    # Component scores
    f1_macro_no_other: float
    learned_words_count: int
    pct_predictions_other: float
    
    # Weights
    alpha: float = 0.4  # Weight for F1
    beta: float = 0.4   # Weight for learned words (normalized)
    gamma: float = 0.2  # Penalty for OTHER predictions
    
    # Normalization factors
    max_possible_words: int = 100
    
    @property
    def score(self) -> float:
        """Compute composite score."""
        normalized_learned = self.learned_words_count / max(self.max_possible_words, 1)
        return (
            self.alpha * self.f1_macro_no_other
            + self.beta * normalized_learned
            - self.gamma * self.pct_predictions_other
        )
    
    def to_dict(self) -> Dict:
        return {
            "components": {
                "f1_macro_no_other": round(self.f1_macro_no_other, 4),
                "learned_words_count": self.learned_words_count,
                "pct_predictions_other": round(self.pct_predictions_other, 4)
            },
            "weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma
            },
            "composite_score": round(self.score, 4)
        }


class AdvancedMetricsCalculator:
    """Calculator for all advanced metrics."""
    
    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        bucket_mapping: Dict[int, str],
        other_class_id: Optional[int] = None,
        learned_criteria: Optional[LearnedWordCriteria] = None
    ):
        """
        Initialize calculator.
        
        Args:
            num_classes: Total number of classes
            class_names: List of class names indexed by class_id
            bucket_mapping: Mapping from class_id to bucket (HEAD, MID, OTHER)
            other_class_id: ID of the OTHER class (for special handling)
            learned_criteria: Criteria for learned words
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.bucket_mapping = bucket_mapping
        self.other_class_id = other_class_id
        self.criteria = learned_criteria or LearnedWordCriteria()
        
        # Initialize confusion matrix
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.confidences: List[Tuple[int, int, float]] = []  # (pred, target, conf)
        self.margins: List[Tuple[int, int, float]] = []  # (pred, target, margin)
    
    def update(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        logits: Optional[np.ndarray] = None
    ):
        """
        Update metrics with batch of predictions.
        
        Args:
            predictions: Predicted class IDs (N,)
            targets: True class IDs (N,)
            logits: Raw logits (N, num_classes) for confidence calculation
        """
        for pred, target in zip(predictions, targets):
            if 0 <= pred < self.num_classes and 0 <= target < self.num_classes:
                self.confusion_matrix[target, pred] += 1
        
        if logits is not None:
            probs = self._softmax(logits)
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                conf = probs[i, pred]
                # Margin: difference between top-1 and top-2
                sorted_probs = np.sort(probs[i])[::-1]
                margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
                
                self.confidences.append((int(pred), int(target), float(conf)))
                self.margins.append((int(pred), int(target), float(margin)))
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def compute_per_class_metrics(self) -> Dict[int, Dict]:
        """Compute precision, recall, F1 per class."""
        metrics = {}
        
        for class_id in range(self.num_classes):
            tp = self.confusion_matrix[class_id, class_id]
            fp = self.confusion_matrix[:, class_id].sum() - tp
            fn = self.confusion_matrix[class_id, :].sum() - tp
            support = self.confusion_matrix[class_id, :].sum()
            predictions = self.confusion_matrix[:, class_id].sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[class_id] = {
                "support": int(support),
                "predictions": int(predictions),
                "correct": int(tp),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "bucket": self.bucket_mapping.get(class_id, "UNKNOWN")
            }
        
        return metrics
    
    def compute_learned_words(self) -> LearnedWordsReport:
        """Compute learned words report."""
        per_class = self.compute_per_class_metrics()
        
        learned_words = []
        not_learned = []
        learned_by_bucket = {"HEAD": 0, "MID": 0, "OTHER": 0}
        word_metrics = []
        
        for class_id, metrics in per_class.items():
            gloss = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            bucket = metrics["bucket"]
            
            # Skip OTHER class for learned words
            if class_id == self.other_class_id:
                continue
            
            # Check criteria
            reasons = []
            if metrics["support"] < self.criteria.min_support:
                reasons.append(f"support={metrics['support']}<{self.criteria.min_support}")
            if metrics["precision"] < self.criteria.min_precision:
                reasons.append(f"precision={metrics['precision']:.2f}<{self.criteria.min_precision}")
            if metrics["recall"] < self.criteria.min_recall:
                reasons.append(f"recall={metrics['recall']:.2f}<{self.criteria.min_recall}")
            if metrics["f1"] < self.criteria.min_f1:
                reasons.append(f"f1={metrics['f1']:.2f}<{self.criteria.min_f1}")
            
            is_learned = len(reasons) == 0
            
            wm = WordMetrics(
                gloss=gloss,
                class_id=class_id,
                bucket=bucket,
                support=metrics["support"],
                predictions=metrics["predictions"],
                correct=metrics["correct"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                is_learned=is_learned,
                not_learned_reason="; ".join(reasons) if reasons else None
            )
            word_metrics.append(wm)
            
            if is_learned:
                learned_words.append(gloss)
                if bucket in learned_by_bucket:
                    learned_by_bucket[bucket] += 1
            else:
                not_learned.append({
                    "word": gloss,
                    "reason": "; ".join(reasons),
                    "metrics": {
                        "support": metrics["support"],
                        "precision": round(metrics["precision"], 3),
                        "recall": round(metrics["recall"], 3),
                        "f1": round(metrics["f1"], 3)
                    }
                })
        
        total_classes = self.num_classes - (1 if self.other_class_id is not None else 0)
        learned_count = len(learned_words)
        
        return LearnedWordsReport(
            criteria=self.criteria,
            total_classes=total_classes,
            learned_count=learned_count,
            not_learned_count=len(not_learned),
            pct_vocabulary_learned=learned_count / total_classes if total_classes > 0 else 0,
            learned_by_bucket=learned_by_bucket,
            learned_words=learned_words,
            not_learned_words=not_learned,
            word_metrics=word_metrics
        )
    
    def compute_rejection_metrics(
        self,
        confidence_threshold: float = 0.5,
        margin_threshold: float = 0.2
    ) -> RejectionMetrics:
        """Compute rejection/acceptance metrics."""
        if not self.confidences:
            return RejectionMetrics(
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
                total_samples=0,
                accepted_samples=0,
                rejected_samples=0,
                false_accepts=0,
                false_accept_rate=0.0,
                false_rejects=0,
                false_reject_rate=0.0,
                accept_at_conf=0.0,
                accept_at_margin=0.0,
                predictions_to_other=0,
                pct_predictions_other=0.0
            )
        
        total = len(self.confidences)
        
        # By confidence threshold
        accepted_conf = [(p, t, c) for p, t, c in self.confidences if c >= confidence_threshold]
        rejected_conf = [(p, t, c) for p, t, c in self.confidences if c < confidence_threshold]
        
        # False accepts: accepted but wrong
        false_accepts = sum(1 for p, t, _ in accepted_conf if p != t)
        # False rejects: rejected but would have been correct
        false_rejects = sum(1 for p, t, _ in rejected_conf if p == t)
        
        # Accuracy among accepted
        correct_accepted_conf = sum(1 for p, t, _ in accepted_conf if p == t)
        accept_at_conf = correct_accepted_conf / len(accepted_conf) if accepted_conf else 0.0
        
        # By margin threshold
        accepted_margin = [(p, t, m) for p, t, m in self.margins if m >= margin_threshold]
        correct_accepted_margin = sum(1 for p, t, _ in accepted_margin if p == t)
        accept_at_margin = correct_accepted_margin / len(accepted_margin) if accepted_margin else 0.0
        
        # Predictions to OTHER
        predictions_to_other = sum(1 for p, _, _ in self.confidences if p == self.other_class_id)
        
        return RejectionMetrics(
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            total_samples=total,
            accepted_samples=len(accepted_conf),
            rejected_samples=len(rejected_conf),
            false_accepts=false_accepts,
            false_accept_rate=false_accepts / len(accepted_conf) if accepted_conf else 0.0,
            false_rejects=false_rejects,
            false_reject_rate=false_rejects / len(rejected_conf) if rejected_conf else 0.0,
            accept_at_conf=accept_at_conf,
            accept_at_margin=accept_at_margin,
            predictions_to_other=predictions_to_other,
            pct_predictions_other=predictions_to_other / total if total > 0 else 0.0
        )
    
    def compute_bucket_metrics(self) -> Dict[str, BucketMetrics]:
        """Compute metrics per bucket."""
        per_class = self.compute_per_class_metrics()
        
        buckets = {"HEAD": [], "MID": [], "OTHER": []}
        
        for class_id, metrics in per_class.items():
            bucket = metrics["bucket"]
            if bucket in buckets:
                buckets[bucket].append((class_id, metrics))
        
        result = {}
        total_predictions = self.confusion_matrix.sum()
        
        for bucket, class_metrics in buckets.items():
            if not class_metrics:
                continue
            
            num_classes = len(class_metrics)
            total_support = sum(m["support"] for _, m in class_metrics)
            
            # Accuracy@1: correct predictions in this bucket / total support
            total_correct = sum(m["correct"] for _, m in class_metrics)
            acc_at_1 = total_correct / total_support if total_support > 0 else 0.0
            
            # Macro metrics
            macro_p = np.mean([m["precision"] for _, m in class_metrics])
            macro_r = np.mean([m["recall"] for _, m in class_metrics])
            macro_f1 = np.mean([m["f1"] for _, m in class_metrics])
            
            # Coverage: predictions going to this bucket
            bucket_predictions = sum(m["predictions"] for _, m in class_metrics)
            coverage = bucket_predictions / total_predictions if total_predictions > 0 else 0.0
            
            result[bucket] = BucketMetrics(
                bucket=bucket,
                num_classes=num_classes,
                total_support=total_support,
                accuracy_at_1=acc_at_1,
                accuracy_at_5=0.0,  # Would need top-5 predictions
                macro_precision=macro_p,
                macro_recall=macro_r,
                macro_f1=macro_f1,
                coverage=coverage
            )
        
        return result
    
    def compute_composite_score(self, max_possible_words: int = 100) -> CompositeScore:
        """Compute composite score for model selection."""
        per_class = self.compute_per_class_metrics()
        learned = self.compute_learned_words()
        rejection = self.compute_rejection_metrics()
        
        # F1 macro excluding OTHER
        f1_scores = [
            m["f1"] for cid, m in per_class.items() 
            if cid != self.other_class_id and m["support"] > 0
        ]
        f1_macro_no_other = np.mean(f1_scores) if f1_scores else 0.0
        
        return CompositeScore(
            f1_macro_no_other=f1_macro_no_other,
            learned_words_count=learned.learned_count,
            pct_predictions_other=rejection.pct_predictions_other,
            max_possible_words=max_possible_words
        )
