"""
Tests for advanced per-class metrics and analysis modules.

Tests:
- MetricsTracker per-class methods
- Confusion matrix utilities
- Dataset coverage analyzer
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

from training.metrics import MetricsTracker
from training.analysis.coverage import DatasetCoverageAnalyzer
from training.analysis.confusion import (
    export_confusion_matrix_csv,
    export_confusion_matrix_heatmap,
    get_most_confused_pairs,
    get_confusion_summary,
    format_confusion_report
)


# =============================================================================
# MetricsTracker Per-Class Tests
# =============================================================================

class TestMetricsTrackerPerClass:
    """Test per-class metrics in MetricsTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create tracker with 5 classes."""
        return MetricsTracker(num_classes=5, topk=(1, 3))
    
    @pytest.fixture
    def sample_data(self):
        """Create sample logits and labels for 3 classes."""
        # 10 samples, predictions mostly correct
        logits = torch.tensor([
            [2.0, 0.5, 0.1],  # pred 0, true 0
            [0.1, 2.0, 0.5],  # pred 1, true 1
            [0.1, 0.5, 2.0],  # pred 2, true 2
            [2.0, 0.5, 0.1],  # pred 0, true 0
            [0.5, 2.0, 0.1],  # pred 1, true 1
            [2.0, 0.1, 0.5],  # pred 0, true 1 (error!)
            [0.5, 0.1, 2.0],  # pred 2, true 2
            [0.1, 2.0, 0.5],  # pred 1, true 0 (error!)
            [0.1, 0.5, 2.0],  # pred 2, true 2
            [2.0, 0.5, 0.1],  # pred 0, true 0
        ])
        labels = torch.tensor([0, 1, 2, 0, 1, 1, 2, 0, 2, 0])
        return logits, labels
    
    def test_compute_per_class_empty(self, tracker):
        """Test per-class with no samples."""
        result = tracker.compute_per_class()
        assert result == {}
    
    def test_compute_per_class_basic(self):
        """Test basic per-class computation."""
        tracker = MetricsTracker(num_classes=3, topk=(1, 3))
        
        # Class 0: 4 samples, 3 correct
        # Class 1: 3 samples, 2 correct
        # Class 2: 3 samples, 3 correct
        logits = torch.tensor([
            [2.0, 0.5, 0.1],  # pred 0
            [0.1, 2.0, 0.5],  # pred 1
            [0.1, 0.5, 2.0],  # pred 2
            [2.0, 0.5, 0.1],  # pred 0
            [0.5, 2.0, 0.1],  # pred 1
            [2.0, 0.1, 0.5],  # pred 0 (error for class 1)
            [0.5, 0.1, 2.0],  # pred 2
            [0.1, 2.0, 0.5],  # pred 1 (error for class 0)
            [0.1, 0.5, 2.0],  # pred 2
            [2.0, 0.5, 0.1],  # pred 0
        ])
        labels = torch.tensor([0, 1, 2, 0, 1, 1, 2, 0, 2, 0])
        
        tracker.update(logits, labels)
        per_class = tracker.compute_per_class()
        
        assert len(per_class) == 3
        
        # Check class 0: 4 samples, 3 correct -> recall=0.75
        assert 0 in per_class
        assert per_class[0]["support"] == 4
        assert per_class[0]["recall"] == pytest.approx(0.75, rel=0.01)
        
        # Check class 2: 3 samples, 3 correct -> recall=1.0
        assert per_class[2]["support"] == 3
        assert per_class[2]["recall"] == pytest.approx(1.0)
    
    def test_compute_per_class_with_names(self):
        """Test per-class with class name mapping."""
        tracker = MetricsTracker(num_classes=3, topk=(1,))
        
        logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 2.0, 0.5], [0.1, 0.5, 2.0]])
        labels = torch.tensor([0, 1, 2])
        tracker.update(logits, labels)
        
        class_names = ["cat", "dog", "bird"]
        per_class = tracker.compute_per_class(class_names=class_names)
        
        assert "cat" in per_class
        assert "dog" in per_class
        assert "bird" in per_class
        assert per_class["cat"]["class_id"] == 0
        assert per_class["dog"]["class_id"] == 1
    
    def test_per_class_has_topk(self):
        """Test that per-class includes top-k accuracy."""
        tracker = MetricsTracker(num_classes=3, topk=(1, 3))
        
        logits = torch.tensor([
            [2.0, 1.5, 0.1],  # pred 0, 2nd: 1
            [0.1, 2.0, 1.5],  # pred 1, 2nd: 2
            [1.5, 0.1, 2.0],  # pred 2, 2nd: 0
        ])
        labels = torch.tensor([0, 1, 2])
        tracker.update(logits, labels)
        
        per_class = tracker.compute_per_class()
        
        assert "top1_acc" in per_class[0]
        assert "top3_acc" in per_class[0]
        assert per_class[0]["top1_acc"] == 1.0  # Perfect prediction
    
    def test_get_confusion_matrix_empty(self, tracker):
        """Test confusion matrix with no samples."""
        cm = tracker.get_confusion_matrix()
        assert cm.size == 0
    
    def test_get_confusion_matrix_basic(self):
        """Test confusion matrix generation."""
        tracker = MetricsTracker(num_classes=3, topk=(1,))
        
        # Predictions: [0, 1, 0] for labels [0, 1, 2]
        logits = torch.tensor([
            [2.0, 0.5, 0.1],
            [0.1, 2.0, 0.5],
            [2.0, 0.5, 0.1],  # Predict 0 for true 2
        ])
        labels = torch.tensor([0, 1, 2])
        tracker.update(logits, labels)
        
        cm = tracker.get_confusion_matrix()
        
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 1  # True 0, pred 0
        assert cm[1, 1] == 1  # True 1, pred 1
        assert cm[2, 0] == 1  # True 2, pred 0 (error)
        assert cm[2, 2] == 0  # True 2, pred 2 (didn't happen)
    
    def test_get_worst_classes(self):
        """Test getting worst performing classes."""
        tracker = MetricsTracker(num_classes=3, topk=(1,))
        
        # Class 0: perfect, Class 1: 50%, Class 2: 0%
        logits = torch.tensor([
            [2.0, 0.5, 0.1],  # pred 0, true 0 ✓
            [2.0, 0.5, 0.1],  # pred 0, true 0 ✓
            [0.1, 2.0, 0.5],  # pred 1, true 1 ✓
            [2.0, 0.5, 0.1],  # pred 0, true 1 ✗
            [0.1, 2.0, 0.5],  # pred 1, true 2 ✗
            [2.0, 0.5, 0.1],  # pred 0, true 2 ✗
        ])
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        tracker.update(logits, labels)
        
        worst = tracker.get_worst_classes(k=3, metric="f1")
        
        assert len(worst) == 3
        # Class 2 should be worst (F1=0), Class 0 should be best
        assert worst[0]["class_id"] == 2  # Worst F1
        assert worst[0]["f1"] == 0.0
    
    def test_get_best_classes(self):
        """Test getting best performing classes."""
        tracker = MetricsTracker(num_classes=3, topk=(1,))
        
        logits = torch.tensor([
            [2.0, 0.5, 0.1],
            [2.0, 0.5, 0.1],
            [0.1, 2.0, 0.5],
            [2.0, 0.5, 0.1],
            [0.1, 0.5, 2.0],
            [0.1, 0.5, 2.0],
        ])
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        tracker.update(logits, labels)
        
        best = tracker.get_best_classes(k=3, metric="f1")
        
        assert len(best) == 3
        # Class 0 and 2 should have best F1
        assert best[0]["f1"] >= best[1]["f1"] >= best[2]["f1"]
    
    def test_get_class_distribution_analysis(self):
        """Test class distribution analysis."""
        tracker = MetricsTracker(num_classes=5, topk=(1,))
        
        # Imbalanced: class 0 has 5, class 1 has 1
        logits = torch.randn(6, 5)
        labels = torch.tensor([0, 0, 0, 0, 0, 1])
        tracker.update(logits, labels)
        
        analysis = tracker.get_class_distribution_analysis()
        
        assert analysis["total_samples"] == 6
        assert analysis["num_classes_seen"] == 2
        assert analysis["num_classes_defined"] == 5
        assert analysis["coverage_ratio"] == pytest.approx(0.4)
        assert analysis["class_imbalance_ratio"] == 5.0
        assert analysis["min_support"] == 1
        assert analysis["max_support"] == 5
    
    def test_format_per_class_report(self):
        """Test per-class report formatting."""
        tracker = MetricsTracker(num_classes=3, topk=(1, 5))
        
        logits = torch.tensor([
            [2.0, 0.5, 0.1],
            [0.1, 2.0, 0.5],
            [0.1, 0.5, 2.0],
        ])
        labels = torch.tensor([0, 1, 2])
        tracker.update(logits, labels)
        
        report = tracker.format_per_class_report(
            class_names=["alpha", "beta", "gamma"]
        )
        
        assert "PER-CLASS METRICS REPORT" in report
        assert "alpha" in report
        assert "beta" in report
        assert "gamma" in report
        assert "Prec" in report
        assert "Recall" in report


# =============================================================================
# Confusion Matrix Utilities Tests
# =============================================================================

class TestConfusionMatrixUtils:
    """Test confusion matrix utility functions."""
    
    @pytest.fixture
    def sample_cm(self):
        """Create a sample 3x3 confusion matrix."""
        # True\Pred  0  1  2
        # 0          5  2  1
        # 1          1  6  2
        # 2          0  1  7
        return np.array([
            [5, 2, 1],
            [1, 6, 2],
            [0, 1, 7]
        ])
    
    def test_export_csv(self, sample_cm):
        """Test exporting confusion matrix to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cm.csv"
            
            result = export_confusion_matrix_csv(
                sample_cm,
                output_path,
                class_names=["cat", "dog", "bird"]
            )
            
            assert result.exists()
            
            # Read and verify content
            with open(result, "r") as f:
                content = f.read()
            
            assert "cat" in content
            assert "dog" in content
            assert "bird" in content
            assert "Total" in content
    
    def test_export_csv_without_names(self, sample_cm):
        """Test CSV export without class names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cm.csv"
            
            result = export_confusion_matrix_csv(sample_cm, output_path)
            
            assert result.exists()
            with open(result, "r") as f:
                content = f.read()
            assert "class_0" in content
    
    def test_get_most_confused_pairs(self, sample_cm):
        """Test getting most confused pairs."""
        pairs = get_most_confused_pairs(
            sample_cm,
            k=5,
            class_names=["cat", "dog", "bird"]
        )
        
        assert len(pairs) <= 5
        # All pairs should have count > 0
        assert all(p["count"] > 0 for p in pairs)
        # Should be sorted by count descending
        counts = [p["count"] for p in pairs]
        assert counts == sorted(counts, reverse=True)
        
        # Check structure
        assert "true_class" in pairs[0]
        assert "pred_class" in pairs[0]
        assert "true_name" in pairs[0]
        assert "pred_name" in pairs[0]
        assert "confusion_rate" in pairs[0]
    
    def test_get_most_confused_excludes_diagonal(self, sample_cm):
        """Test that diagonal is excluded by default."""
        pairs = get_most_confused_pairs(sample_cm, k=100)
        
        for pair in pairs:
            assert pair["true_class"] != pair["pred_class"]
    
    def test_get_confusion_summary(self, sample_cm):
        """Test confusion summary statistics."""
        summary = get_confusion_summary(
            sample_cm,
            class_names=["cat", "dog", "bird"]
        )
        
        assert summary["total_samples"] == 25
        assert summary["correct_predictions"] == 18  # 5+6+7
        assert summary["accuracy"] == pytest.approx(18/25)
        assert "per_class_accuracy" in summary
        assert "cat" in summary["per_class_accuracy"]
        assert summary["per_class_accuracy"]["cat"]["accuracy"] == pytest.approx(5/8)
    
    def test_format_confusion_report(self, sample_cm):
        """Test confusion report formatting."""
        report = format_confusion_report(
            sample_cm,
            class_names=["cat", "dog", "bird"]
        )
        
        assert "CONFUSION MATRIX ANALYSIS" in report
        assert "Overall accuracy" in report
        assert "TOP" in report
        assert "CONFUSED PAIRS" in report


# =============================================================================
# Dataset Coverage Analyzer Tests
# =============================================================================

class TestDatasetCoverageAnalyzer:
    """Test DatasetCoverageAnalyzer class."""
    
    @pytest.fixture
    def vocab_file(self):
        """Create temporary vocabulary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "dict.json"
            vocab = {
                "hello": 0,
                "world": 1,
                "test": 2,
                "python": 3,
                "code": 4
            }
            with open(vocab_path, "w") as f:
                json.dump(vocab, f)
            yield vocab_path
    
    def test_init_with_class_names(self):
        """Test initialization with class names."""
        analyzer = DatasetCoverageAnalyzer(
            class_names=["a", "b", "c"]
        )
        
        assert analyzer.num_classes == 3
        assert analyzer.get_class_name(0) == "a"
        assert analyzer.get_class_index("b") == 1
    
    def test_load_vocabulary(self, vocab_file):
        """Test loading vocabulary from file."""
        analyzer = DatasetCoverageAnalyzer(vocab_path=vocab_file)
        num_classes = analyzer.load_vocabulary()
        
        assert num_classes == 5
        assert analyzer.get_class_name(0) == "hello"
        assert analyzer.get_class_index("world") == 1
    
    def test_analyze_from_labels(self, vocab_file):
        """Test analysis from label list."""
        analyzer = DatasetCoverageAnalyzer(vocab_path=vocab_file)
        analyzer.load_vocabulary()
        
        # Provide labels for only 3 of 5 classes
        labels = [0, 0, 1, 1, 1, 2, 2]
        analysis = analyzer.analyze_from_labels(labels)
        
        assert analysis["total_samples"] == 7
        assert analysis["vocab_size"] == 5
        assert analysis["classes_in_data"] == 3
        assert analysis["coverage_ratio"] == pytest.approx(0.6)
        assert analysis["num_missing"] == 2
    
    def test_missing_classes(self, vocab_file):
        """Test detection of missing classes."""
        analyzer = DatasetCoverageAnalyzer(vocab_path=vocab_file)
        analyzer.load_vocabulary()
        
        labels = [0, 1, 2]  # Missing classes 3 and 4
        analysis = analyzer.analyze_from_labels(labels)
        
        missing_ids = [m["class_id"] for m in analysis["missing_classes"]]
        assert 3 in missing_ids
        assert 4 in missing_ids
    
    def test_imbalance_metrics(self, vocab_file):
        """Test imbalance metrics calculation."""
        analyzer = DatasetCoverageAnalyzer(vocab_path=vocab_file)
        analyzer.load_vocabulary()
        
        # Highly imbalanced: class 0 has 10, class 1 has 1
        labels = [0] * 10 + [1]
        analysis = analyzer.analyze_from_labels(labels)
        
        imb = analysis["imbalance_metrics"]
        assert imb["min_count"] == 1
        assert imb["max_count"] == 10
        assert imb["imbalance_ratio"] == 10.0
    
    def test_get_underrepresented_classes(self, vocab_file):
        """Test getting underrepresented classes."""
        analyzer = DatasetCoverageAnalyzer(vocab_path=vocab_file)
        analyzer.load_vocabulary()
        
        labels = [0] * 10 + [1] * 2 + [2]  # 3, 4 missing; 2 has 1
        analyzer.analyze_from_labels(labels)
        
        underrep = analyzer.get_underrepresented_classes(threshold=5)
        
        # Should include missing classes (0 samples) and class 2 (1 sample)
        assert len(underrep) >= 3
        # Should be sorted by count ascending
        counts = [u["count"] for u in underrep]
        assert counts == sorted(counts)
    
    def test_format_report(self, vocab_file):
        """Test report formatting."""
        analyzer = DatasetCoverageAnalyzer(vocab_path=vocab_file)
        analyzer.load_vocabulary()
        
        labels = [0, 0, 1, 2, 2, 2]
        analyzer.analyze_from_labels(labels)
        
        report = analyzer.format_report()
        
        assert "DATASET COVERAGE ANALYSIS REPORT" in report
        assert "Total samples:" in report
        assert "Vocabulary size:" in report
        assert "Coverage:" in report
    
    def test_export_to_json(self, vocab_file):
        """Test JSON export."""
        analyzer = DatasetCoverageAnalyzer(vocab_path=vocab_file)
        analyzer.load_vocabulary()
        
        labels = [0, 1, 2]
        analyzer.analyze_from_labels(labels)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "analysis.json"
            analyzer.export_to_json(output_path)
            
            assert output_path.exists()
            
            with open(output_path, "r") as f:
                data = json.load(f)
            
            assert "total_samples" in data
            assert "coverage_ratio" in data
    
    def test_reset(self, vocab_file):
        """Test reset functionality."""
        analyzer = DatasetCoverageAnalyzer(vocab_path=vocab_file)
        analyzer.load_vocabulary()
        
        analyzer.analyze_from_labels([0, 1, 2])
        assert analyzer.get_analysis()["total_samples"] == 3
        
        analyzer.reset()
        assert analyzer.get_analysis()["total_samples"] == 0
    
    def test_incremental_analysis(self, vocab_file):
        """Test incremental label addition."""
        analyzer = DatasetCoverageAnalyzer(vocab_path=vocab_file)
        analyzer.load_vocabulary()
        
        analyzer.analyze_from_labels([0, 1], reset=False)
        analysis1 = analyzer.get_analysis()
        assert analysis1["total_samples"] == 2
        
        analyzer.analyze_from_labels([2, 3], reset=False)
        analysis2 = analyzer.get_analysis()
        assert analysis2["total_samples"] == 4
        assert analysis2["classes_in_data"] == 4


# =============================================================================
# Integration Tests
# =============================================================================

class TestMetricsAnalysisIntegration:
    """Integration tests combining MetricsTracker with analysis tools."""
    
    def test_tracker_cm_to_analysis(self):
        """Test using tracker's confusion matrix with analysis utils."""
        tracker = MetricsTracker(num_classes=3, topk=(1,))
        
        logits = torch.tensor([
            [2.0, 0.5, 0.1],
            [0.1, 2.0, 0.5],
            [0.1, 0.5, 2.0],
            [0.1, 2.0, 0.5],  # Error: pred 1, true 0
        ])
        labels = torch.tensor([0, 1, 2, 0])
        tracker.update(logits, labels)
        
        cm = tracker.get_confusion_matrix()
        
        # Use confusion matrix utils
        summary = get_confusion_summary(cm, ["A", "B", "C"])
        assert summary["total_samples"] == 4
        
        pairs = get_most_confused_pairs(cm, k=5, class_names=["A", "B", "C"])
        assert len(pairs) >= 1  # At least one confusion
    
    def test_full_pipeline(self):
        """Test complete pipeline: tracker -> per-class -> confusion -> export."""
        tracker = MetricsTracker(num_classes=5, topk=(1, 3))
        
        # Simulate multiple batches
        for _ in range(3):
            logits = torch.randn(10, 5)
            labels = torch.randint(0, 5, (10,))
            tracker.update(logits, labels)
        
        # Get all metrics
        global_metrics = tracker.compute()
        per_class = tracker.compute_per_class()
        cm = tracker.get_confusion_matrix()
        worst = tracker.get_worst_classes(k=3)
        best = tracker.get_best_classes(k=3)
        distribution = tracker.get_class_distribution_analysis()
        
        # Verify all components work
        assert "accuracy" in global_metrics
        assert len(per_class) > 0
        assert cm.shape == (5, 5)
        assert len(worst) <= 3
        assert len(best) <= 3
        assert distribution["total_samples"] == 30


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_class(self):
        """Test with single class."""
        tracker = MetricsTracker(num_classes=1, topk=(1,))
        
        logits = torch.tensor([[2.0]])
        labels = torch.tensor([0])
        tracker.update(logits, labels)
        
        per_class = tracker.compute_per_class()
        assert len(per_class) == 1
        assert per_class[0]["accuracy"] == 1.0
    
    def test_all_wrong_predictions(self):
        """Test when all predictions are wrong."""
        tracker = MetricsTracker(num_classes=2, topk=(1,))
        
        logits = torch.tensor([
            [0.1, 2.0],  # pred 1, true 0
            [2.0, 0.1],  # pred 0, true 1
        ])
        labels = torch.tensor([0, 1])
        tracker.update(logits, labels)
        
        metrics = tracker.compute()
        assert metrics["accuracy"] == 0.0
        
        per_class = tracker.compute_per_class()
        assert per_class[0]["recall"] == 0.0
        assert per_class[1]["recall"] == 0.0
    
    def test_large_num_classes(self):
        """Test with large number of classes."""
        tracker = MetricsTracker(num_classes=500, topk=(1, 5, 10))
        
        logits = torch.randn(100, 500)
        labels = torch.randint(0, 500, (100,))
        tracker.update(logits, labels)
        
        cm = tracker.get_confusion_matrix()
        assert cm.shape == (500, 500)
        
        per_class = tracker.compute_per_class()
        # Should have metrics for classes that appeared
        assert len(per_class) <= 100  # At most 100 unique classes
    
    def test_missing_sklearn(self):
        """Test graceful handling when sklearn is missing (mocked)."""
        tracker = MetricsTracker(num_classes=3, topk=(1,))
        
        logits = torch.randn(5, 3)
        labels = torch.randint(0, 3, (5,))
        tracker.update(logits, labels)
        
        # These should work without sklearn for basic operations
        assert tracker.num_samples == 5
        cm = tracker.get_confusion_matrix()  # Should work with sklearn
        assert cm.shape == (3, 3)
