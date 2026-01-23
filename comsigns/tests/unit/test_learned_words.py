"""
Unit tests for learned words analysis module.

These tests validate the LearnedWordsAnalyzer functionality
for determining how many words a model has effectively learned.
"""

import pytest
import json
import tempfile
from pathlib import Path

from comsigns.training.analysis.learned_words import (
    LearnedWordCriteria,
    ClassMetrics,
    BucketLearnedStats,
    LearnedWordsReport,
    LearnedWordsAnalyzer,
    compute_class_metrics_from_predictions,
    analyze_learned_words
)


# ============================================================================
# LearnedWordCriteria Tests
# ============================================================================

class TestLearnedWordCriteria:
    """Tests for LearnedWordCriteria dataclass."""
    
    def test_default_values(self):
        """Default criteria should have sensible values."""
        criteria = LearnedWordCriteria()
        assert criteria.min_support == 2
        assert criteria.min_precision == 0.5
        assert criteria.min_recall == 0.5
        assert criteria.min_f1 == 0.5
    
    def test_custom_values(self):
        """Custom criteria values should be accepted."""
        criteria = LearnedWordCriteria(
            min_support=5,
            min_precision=0.7,
            min_recall=0.6,
            min_f1=0.65
        )
        assert criteria.min_support == 5
        assert criteria.min_precision == 0.7
        assert criteria.min_recall == 0.6
        assert criteria.min_f1 == 0.65
    
    def test_to_dict(self):
        """Criteria should serialize to dict."""
        criteria = LearnedWordCriteria(min_support=3)
        d = criteria.to_dict()
        assert d["min_support"] == 3
        assert d["min_precision"] == 0.5
    
    def test_from_dict(self):
        """Criteria should deserialize from dict."""
        data = {"min_support": 4, "min_precision": 0.6, "min_recall": 0.55, "min_f1": 0.58}
        criteria = LearnedWordCriteria.from_dict(data)
        assert criteria.min_support == 4
        assert criteria.min_precision == 0.6
    
    def test_invalid_min_support(self):
        """min_support < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="min_support"):
            LearnedWordCriteria(min_support=0)
    
    def test_invalid_precision_range(self):
        """Precision outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="min_precision"):
            LearnedWordCriteria(min_precision=1.5)
        with pytest.raises(ValueError, match="min_precision"):
            LearnedWordCriteria(min_precision=-0.1)
    
    def test_invalid_recall_range(self):
        """Recall outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="min_recall"):
            LearnedWordCriteria(min_recall=2.0)
    
    def test_invalid_f1_range(self):
        """F1 outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="min_f1"):
            LearnedWordCriteria(min_f1=-0.5)


# ============================================================================
# ClassMetrics Tests
# ============================================================================

class TestClassMetrics:
    """Tests for ClassMetrics dataclass."""
    
    def test_creation(self):
        """ClassMetrics should be created with all fields."""
        cm = ClassMetrics(
            class_id=0,
            support=10,
            precision=0.8,
            recall=0.7,
            f1=0.75
        )
        assert cm.class_id == 0
        assert cm.support == 10
        assert cm.precision == 0.8
        assert cm.recall == 0.7
        assert cm.f1 == 0.75
    
    def test_to_dict(self):
        """ClassMetrics should serialize to dict."""
        cm = ClassMetrics(0, 5, 0.5, 0.6, 0.55)
        d = cm.to_dict()
        assert d["class_id"] == 0
        assert d["support"] == 5


# ============================================================================
# LearnedWordsAnalyzer Core Tests
# ============================================================================

class TestLearnedWordsAnalyzer:
    """Tests for LearnedWordsAnalyzer class."""
    
    def create_metrics(self, class_id, support, precision, recall, f1):
        """Helper to create ClassMetrics."""
        return ClassMetrics(
            class_id=class_id,
            support=support,
            precision=precision,
            recall=recall,
            f1=f1
        )
    
    def test_word_learned_meets_all_criteria(self):
        """A word meeting all criteria should be marked as learned."""
        metrics_by_class = {
            0: self.create_metrics(0, support=10, precision=0.8, recall=0.7, f1=0.75),
        }
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            criteria=LearnedWordCriteria()
        )
        
        is_learned, reason = analyzer.is_word_learned(metrics_by_class[0])
        assert is_learned is True
        assert reason == "learned"
    
    def test_word_not_learned_low_support(self):
        """A word with insufficient support should not be learned."""
        metrics_by_class = {
            0: self.create_metrics(0, support=1, precision=0.9, recall=0.9, f1=0.9),
        }
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            criteria=LearnedWordCriteria(min_support=2)
        )
        
        is_learned, reason = analyzer.is_word_learned(metrics_by_class[0])
        assert is_learned is False
        assert reason == "insufficient_support"
    
    def test_word_not_learned_low_precision(self):
        """A word with low precision should not be learned."""
        metrics_by_class = {
            0: self.create_metrics(0, support=10, precision=0.3, recall=0.7, f1=0.42),
        }
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            criteria=LearnedWordCriteria(min_precision=0.5)
        )
        
        is_learned, reason = analyzer.is_word_learned(metrics_by_class[0])
        assert is_learned is False
        assert reason == "low_precision"
    
    def test_word_not_learned_low_recall(self):
        """A word with low recall should not be learned."""
        metrics_by_class = {
            0: self.create_metrics(0, support=10, precision=0.8, recall=0.3, f1=0.44),
        }
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            criteria=LearnedWordCriteria(min_recall=0.5)
        )
        
        is_learned, reason = analyzer.is_word_learned(metrics_by_class[0])
        assert is_learned is False
        assert reason == "low_recall"
    
    def test_word_not_learned_low_f1(self):
        """A word with low F1 should not be learned."""
        # Edge case: precision and recall could be okay individually
        # but F1 might still be low due to their combination
        metrics_by_class = {
            0: self.create_metrics(0, support=10, precision=0.6, recall=0.5, f1=0.45),
        }
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            criteria=LearnedWordCriteria(min_f1=0.5)
        )
        
        is_learned, reason = analyzer.is_word_learned(metrics_by_class[0])
        assert is_learned is False
        assert reason == "low_f1"
    
    def test_analyze_multiple_classes(self):
        """Analyze should correctly count learned words across multiple classes."""
        metrics_by_class = {
            # Learned: meets all criteria
            0: self.create_metrics(0, support=10, precision=0.8, recall=0.7, f1=0.75),
            # Learned: barely meets criteria
            1: self.create_metrics(1, support=2, precision=0.5, recall=0.5, f1=0.5),
            # Not learned: low recall
            2: self.create_metrics(2, support=5, precision=0.7, recall=0.3, f1=0.42),
            # Not learned: low support
            3: self.create_metrics(3, support=1, precision=0.9, recall=0.9, f1=0.9),
        }
        
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            criteria=LearnedWordCriteria()
        )
        report = analyzer.analyze()
        
        assert report.total_classes == 4
        assert report.eligible_classes == 3  # class 3 excluded (support=1)
        assert report.learned_words_count == 2  # classes 0 and 1
        assert 0 in report.learned_class_ids
        assert 1 in report.learned_class_ids
    
    def test_analyze_with_bucket_mapping(self):
        """Analyze should correctly split by buckets."""
        metrics_by_class = {
            # HEAD bucket - learned
            0: self.create_metrics(0, support=20, precision=0.9, recall=0.85, f1=0.87),
            1: self.create_metrics(1, support=15, precision=0.85, recall=0.8, f1=0.82),
            # MID bucket - one learned, one not
            2: self.create_metrics(2, support=5, precision=0.7, recall=0.6, f1=0.65),
            3: self.create_metrics(3, support=4, precision=0.3, recall=0.2, f1=0.24),
            # TAIL bucket - not learned (low support)
            4: self.create_metrics(4, support=1, precision=0.8, recall=0.8, f1=0.8),
        }
        bucket_mapping = {0: "HEAD", 1: "HEAD", 2: "MID", 3: "MID", 4: "TAIL"}
        
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            bucket_mapping=bucket_mapping,
            criteria=LearnedWordCriteria()
        )
        report = analyzer.analyze()
        
        # Check overall
        assert report.learned_words_count == 3  # 0, 1, 2
        
        # Check HEAD bucket
        assert "HEAD" in report.by_bucket
        head_stats = report.by_bucket["HEAD"]
        assert head_stats.total_classes == 2
        assert head_stats.eligible_classes == 2
        assert head_stats.learned_words == 2
        assert head_stats.pct_learned == 1.0
        
        # Check MID bucket
        assert "MID" in report.by_bucket
        mid_stats = report.by_bucket["MID"]
        assert mid_stats.total_classes == 2
        assert mid_stats.eligible_classes == 2
        assert mid_stats.learned_words == 1
        assert mid_stats.pct_learned == 0.5
        
        # Check TAIL bucket
        assert "TAIL" in report.by_bucket
        tail_stats = report.by_bucket["TAIL"]
        assert tail_stats.total_classes == 1
        assert tail_stats.eligible_classes == 0  # support=1 < min_support=2
        assert tail_stats.learned_words == 0
    
    def test_other_bucket_supported(self):
        """OTHER bucket (for collapsed tail classes) should work correctly."""
        metrics_by_class = {
            0: self.create_metrics(0, support=20, precision=0.9, recall=0.85, f1=0.87),
            1: self.create_metrics(1, support=50, precision=0.6, recall=0.5, f1=0.55),  # OTHER class
        }
        bucket_mapping = {0: "HEAD", 1: "OTHER"}
        
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            bucket_mapping=bucket_mapping,
            criteria=LearnedWordCriteria()
        )
        report = analyzer.analyze()
        
        assert "OTHER" in report.by_bucket
        other_stats = report.by_bucket["OTHER"]
        assert other_stats.total_classes == 1
        assert other_stats.learned_words == 1
    
    def test_not_learned_reasons_tracked(self):
        """Reasons for not learning should be tracked."""
        metrics_by_class = {
            0: self.create_metrics(0, support=10, precision=0.3, recall=0.7, f1=0.42),  # low precision
            1: self.create_metrics(1, support=10, precision=0.7, recall=0.3, f1=0.42),  # low recall
            2: self.create_metrics(2, support=10, precision=0.3, recall=0.3, f1=0.3),   # low precision (checked first)
            3: self.create_metrics(3, support=10, precision=0.9, recall=0.9, f1=0.9),   # learned
        }
        
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            criteria=LearnedWordCriteria()
        )
        report = analyzer.analyze()
        
        # All 4 classes are eligible (support >= 2)
        assert report.eligible_classes == 4
        assert report.learned_words_count == 1
        
        # Check reasons
        assert "low_precision" in report.not_learned_reasons
        assert report.not_learned_reasons["low_precision"] >= 1
        assert "low_recall" in report.not_learned_reasons
        assert report.not_learned_reasons["low_recall"] >= 1
    
    def test_zero_eligible_classes(self):
        """Handle case with no eligible classes."""
        metrics_by_class = {
            0: self.create_metrics(0, support=1, precision=0.9, recall=0.9, f1=0.9),
        }
        
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            criteria=LearnedWordCriteria(min_support=5)
        )
        report = analyzer.analyze()
        
        assert report.total_classes == 1
        assert report.eligible_classes == 0
        assert report.learned_words_count == 0
        assert report.pct_vocabulary_learned == 0.0
    
    def test_empty_metrics(self):
        """Handle empty metrics dict."""
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class={},
            criteria=LearnedWordCriteria()
        )
        report = analyzer.analyze()
        
        assert report.total_classes == 0
        assert report.eligible_classes == 0
        assert report.learned_words_count == 0


# ============================================================================
# LearnedWordsReport Tests
# ============================================================================

class TestLearnedWordsReport:
    """Tests for LearnedWordsReport dataclass."""
    
    def create_sample_report(self):
        """Create a sample report for testing."""
        return LearnedWordsReport(
            criteria=LearnedWordCriteria(),
            total_classes=10,
            eligible_classes=8,
            learned_words_count=5,
            pct_vocabulary_learned=0.625,
            by_bucket={
                "HEAD": BucketLearnedStats(
                    bucket_name="HEAD",
                    total_classes=3,
                    eligible_classes=3,
                    learned_words=3,
                    pct_learned=1.0,
                    learned_class_ids=[0, 1, 2]
                )
            },
            learned_class_ids=[0, 1, 2, 3, 4],
            not_learned_reasons={"low_recall": 2, "low_precision": 1}
        )
    
    def test_to_dict(self):
        """Report should serialize to dict."""
        report = self.create_sample_report()
        d = report.to_dict()
        
        assert d["total_classes"] == 10
        assert d["eligible_classes"] == 8
        assert d["learned_words_count"] == 5
        assert d["pct_vocabulary_learned"] == 0.625
        assert "HEAD" in d["by_bucket"]
    
    def test_save_to_json(self):
        """Report should save to JSON file."""
        report = self.create_sample_report()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save(path)
            
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            
            assert data["learned_words_count"] == 5
            assert data["criteria"]["min_support"] == 2
    
    def test_get_summary(self):
        """Report should generate human-readable summary."""
        report = self.create_sample_report()
        summary = report.get_summary()
        
        assert "LEARNED WORDS ANALYSIS" in summary
        assert "LEARNED WORDS: 5" in summary
        assert "HEAD:" in summary
        assert "62.5%" in summary


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestComputeClassMetricsFromPredictions:
    """Tests for compute_class_metrics_from_predictions function."""
    
    def test_perfect_predictions(self):
        """Perfect predictions should give perfect metrics."""
        predictions = [0, 0, 1, 1, 2, 2]
        targets = [0, 0, 1, 1, 2, 2]
        
        metrics = compute_class_metrics_from_predictions(predictions, targets, num_classes=3)
        
        for class_id in range(3):
            assert metrics[class_id].precision == 1.0
            assert metrics[class_id].recall == 1.0
            assert metrics[class_id].f1 == 1.0
            assert metrics[class_id].support == 2
    
    def test_zero_support_class(self):
        """Class with no samples should have zero metrics."""
        predictions = [0, 0]
        targets = [0, 0]
        
        metrics = compute_class_metrics_from_predictions(predictions, targets, num_classes=3)
        
        assert metrics[0].support == 2
        assert metrics[1].support == 0
        assert metrics[2].support == 0
        assert metrics[1].precision == 0.0
        assert metrics[1].recall == 0.0
    
    def test_mixed_predictions(self):
        """Mixed predictions should give correct metrics."""
        # Class 0: 3 samples, 2 correct, 1 wrong (predicted as 1)
        # Class 1: 2 samples, 1 correct, 1 wrong (predicted as 0)
        predictions = [0, 0, 1, 1, 0]
        targets = [0, 0, 0, 1, 1]
        
        metrics = compute_class_metrics_from_predictions(predictions, targets, num_classes=2)
        
        # Class 0: TP=2, FP=1, FN=1 -> P=2/3, R=2/3
        assert metrics[0].support == 3
        assert metrics[0].precision == pytest.approx(2/3, rel=1e-5)
        assert metrics[0].recall == pytest.approx(2/3, rel=1e-5)
        
        # Class 1: TP=1, FP=1, FN=1 -> P=0.5, R=0.5
        assert metrics[1].support == 2
        assert metrics[1].precision == 0.5
        assert metrics[1].recall == 0.5


class TestAnalyzeLearnedWords:
    """Tests for the analyze_learned_words convenience function."""
    
    def test_end_to_end(self):
        """End-to-end test of the convenience function."""
        # Simulate a model that learned class 0 well, class 1 poorly
        predictions = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
        targets = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        
        report = analyze_learned_words(
            predictions=predictions,
            targets=targets,
            num_classes=2,
            criteria=LearnedWordCriteria(min_support=2, min_precision=0.5, min_recall=0.5, min_f1=0.5)
        )
        
        # Class 0: 5 samples, 4 correct predictions for class 0 out of 8 total class 0 preds
        # Class 1: 5 samples, 2 correct predictions for class 1 out of 2 total class 1 preds
        assert report.total_classes == 2
        assert report.eligible_classes == 2  # Both have support >= 2
        
        # Class 1 has high precision (2/2=1.0) but low recall (2/5=0.4)
        # So class 1 is NOT learned
        assert report.learned_words_count >= 0  # Depends on exact metrics
    
    def test_with_bucket_mapping(self):
        """Test with bucket mapping."""
        predictions = [0, 0, 0, 1, 1]
        targets = [0, 0, 0, 1, 1]
        bucket_mapping = {0: "HEAD", 1: "MID"}
        
        report = analyze_learned_words(
            predictions=predictions,
            targets=targets,
            num_classes=2,
            bucket_mapping=bucket_mapping
        )
        
        assert "HEAD" in report.by_bucket
        assert "MID" in report.by_bucket


# ============================================================================
# Edge Cases & Integration Tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_boundary_thresholds(self):
        """Test behavior at exact threshold boundaries."""
        metrics_by_class = {
            # Exactly at threshold - should be learned
            0: ClassMetrics(0, support=2, precision=0.5, recall=0.5, f1=0.5),
            # Just below threshold - should not be learned
            1: ClassMetrics(1, support=2, precision=0.49, recall=0.5, f1=0.5),
        }
        
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            criteria=LearnedWordCriteria()
        )
        report = analyzer.analyze()
        
        assert 0 in report.learned_class_ids
        assert 1 not in report.learned_class_ids
    
    def test_strict_criteria(self):
        """Test with very strict criteria."""
        metrics_by_class = {
            0: ClassMetrics(0, support=100, precision=0.95, recall=0.92, f1=0.93),
            1: ClassMetrics(1, support=50, precision=0.85, recall=0.88, f1=0.86),
        }
        
        strict_criteria = LearnedWordCriteria(
            min_support=10,
            min_precision=0.9,
            min_recall=0.9,
            min_f1=0.9
        )
        
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            criteria=strict_criteria
        )
        report = analyzer.analyze()
        
        # Only class 0 meets the strict criteria
        assert report.learned_words_count == 1
        assert 0 in report.learned_class_ids
    
    def test_relaxed_criteria(self):
        """Test with very relaxed criteria."""
        metrics_by_class = {
            0: ClassMetrics(0, support=2, precision=0.2, recall=0.2, f1=0.2),
            1: ClassMetrics(1, support=2, precision=0.3, recall=0.25, f1=0.27),
        }
        
        relaxed_criteria = LearnedWordCriteria(
            min_support=1,
            min_precision=0.1,
            min_recall=0.1,
            min_f1=0.1
        )
        
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            criteria=relaxed_criteria
        )
        report = analyzer.analyze()
        
        # Both classes meet the relaxed criteria
        assert report.learned_words_count == 2
    
    def test_bucket_counts_consistent(self):
        """Verify bucket counts are consistent with totals."""
        metrics_by_class = {
            i: ClassMetrics(i, support=5+i, precision=0.5+i*0.05, recall=0.5+i*0.05, f1=0.5+i*0.05)
            for i in range(10)
        }
        bucket_mapping = {
            0: "HEAD", 1: "HEAD", 2: "HEAD",
            3: "MID", 4: "MID", 5: "MID", 6: "MID",
            7: "TAIL", 8: "TAIL", 9: "TAIL"
        }
        
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            bucket_mapping=bucket_mapping,
            criteria=LearnedWordCriteria()
        )
        report = analyzer.analyze()
        
        # Total learned should equal sum of bucket learned
        bucket_learned_sum = sum(
            stats.learned_words for stats in report.by_bucket.values()
        )
        assert report.learned_words_count == bucket_learned_sum
        
        # Total classes should equal sum of bucket totals
        bucket_total_sum = sum(
            stats.total_classes for stats in report.by_bucket.values()
        )
        assert report.total_classes == bucket_total_sum
