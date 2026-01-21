"""
Tests for Coverage Metrics module.

Tests cover:
- Coverage@K computation
- Per-class coverage
- Vocabulary coverage
- Zero-hit class detection
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from training.analysis.coverage_metrics import (
    CoverageResult,
    CoverageAnalyzer,
    compute_coverage_metrics
)


class TestCoverageResult:
    """Tests for CoverageResult dataclass."""
    
    def test_to_dict(self):
        """to_dict should convert all fields."""
        result = CoverageResult(
            coverage_at_1=0.25,
            coverage_at_5=0.50,
            coverage_at_10=0.65,
            coverage_at_20=0.80,
            per_class_coverage_5={0: 0.8, 1: 0.3},
            per_class_coverage_10={0: 0.9, 1: 0.5},
            vocab_coverage_1=0.6,
            vocab_coverage_5=0.8,
            vocab_coverage_10=0.9,
            zero_hit_classes_5=[5, 6, 7],
            zero_hit_classes_10=[7],
            total_samples=100,
            num_classes=10
        )
        
        d = result.to_dict()
        
        assert d["coverage_at_1"] == 0.25
        assert d["coverage_at_5"] == 0.50
        assert d["vocab_coverage_5"] == 0.8
        assert d["num_zero_hit_5"] == 3
        assert d["num_zero_hit_10"] == 1


class TestCoverageAnalyzer:
    """Tests for CoverageAnalyzer class."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        # 10 samples, 5 classes
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        
        # Logits where higher = more likely
        # Each row: logits for classes 0-4
        y_logits = np.array([
            [2.0, 1.0, 0.5, 0.3, 0.1],  # sample 0: predicts 0 ✓
            [1.5, 2.0, 0.5, 0.3, 0.1],  # sample 1: predicts 1, true=0 ✗
            [0.5, 2.0, 1.5, 0.3, 0.1],  # sample 2: predicts 1 ✓
            [0.5, 1.5, 2.0, 0.3, 0.1],  # sample 3: predicts 2, true=1 ✗
            [0.5, 0.3, 2.0, 1.5, 0.1],  # sample 4: predicts 2 ✓
            [0.5, 0.3, 1.5, 2.0, 0.1],  # sample 5: predicts 3, true=2 ✗
            [0.5, 0.3, 0.1, 2.0, 1.5],  # sample 6: predicts 3 ✓
            [0.5, 0.3, 0.1, 1.5, 2.0],  # sample 7: predicts 4, true=3 ✗
            [0.5, 0.3, 0.1, 0.2, 2.0],  # sample 8: predicts 4 ✓
            [2.0, 0.3, 0.1, 0.2, 1.5],  # sample 9: predicts 0, true=4 ✗
        ])
        
        return y_true, y_logits
    
    def test_load_data(self, simple_data):
        """Should load data correctly."""
        y_true, y_logits = simple_data
        
        analyzer = CoverageAnalyzer()
        analyzer.load_data(y_true, y_logits)
        
        assert len(analyzer._y_true) == 10
        assert analyzer._y_logits.shape == (10, 5)
        assert 1 in analyzer._y_pred_topk  # top-1 precomputed
        assert 5 in analyzer._y_pred_topk  # top-5 precomputed
    
    def test_coverage_at_1(self, simple_data):
        """Coverage@1 should be accuracy."""
        y_true, y_logits = simple_data
        
        analyzer = CoverageAnalyzer()
        analyzer.load_data(y_true, y_logits)
        result = analyzer.analyze()
        
        # 5 correct predictions out of 10
        assert result.coverage_at_1 == 0.5
    
    def test_coverage_at_5(self, simple_data):
        """Coverage@5 should include all classes in top-5."""
        y_true, y_logits = simple_data
        
        analyzer = CoverageAnalyzer()
        analyzer.load_data(y_true, y_logits)
        result = analyzer.analyze()
        
        # With 5 classes, top-5 includes everything
        assert result.coverage_at_5 == 1.0
    
    def test_per_class_coverage(self, simple_data):
        """Per-class coverage should be computed."""
        y_true, y_logits = simple_data
        
        analyzer = CoverageAnalyzer()
        analyzer.load_data(y_true, y_logits)
        result = analyzer.analyze()
        
        # All 5 classes should have coverage info
        assert len(result.per_class_coverage_5) == 5
        
        # Class 0: 2 samples, both should be in top-5
        assert result.per_class_coverage_5[0] == 1.0
    
    def test_vocab_coverage(self, simple_data):
        """Vocabulary coverage should count predicted classes."""
        y_true, y_logits = simple_data
        
        analyzer = CoverageAnalyzer()
        analyzer.load_data(y_true, y_logits)
        result = analyzer.analyze()
        
        # All classes should be predicted at least once in top-5
        assert result.vocab_coverage_5 == 1.0
    
    def test_zero_hit_classes(self):
        """Should detect classes never predicted in top-k."""
        # 5 samples, 10 classes (some never predicted)
        y_true = np.array([0, 1, 2, 3, 4])
        
        # Logits: strongly predict own class, second choice also in 0-4
        # Classes 7-9 should never appear in top-5
        y_logits = np.zeros((5, 10))
        for i in range(5):
            y_logits[i, i] = 10.0  # Strong prediction for own class
            y_logits[i, (i+1) % 5] = 5.0  # Second choice in 0-4
            y_logits[i, (i+2) % 5] = 4.0  # Third choice in 0-4
            y_logits[i, (i+3) % 5] = 3.0  # Fourth choice in 0-4
            y_logits[i, (i+4) % 5] = 2.0  # Fifth choice in 0-4
            # Classes 5-9 have low scores, but some may leak into top-5
            y_logits[i, 5] = 0.1
            y_logits[i, 6] = 0.1
            y_logits[i, 7] = 0.05
            y_logits[i, 8] = 0.05
            y_logits[i, 9] = 0.05
        
        analyzer = CoverageAnalyzer()
        analyzer.load_data(y_true, y_logits)
        result = analyzer.analyze()
        
        # Classes 5-9 should mostly be zero-hit in top-5
        # At least some classes should never appear in top-5
        assert len(result.zero_hit_classes_5) >= 3
        # Classes 7, 8, 9 should definitely be zero-hit
        for c in [7, 8, 9]:
            assert c in result.zero_hit_classes_5


class TestComputeCoverageMetrics:
    """Tests for convenience function."""
    
    def test_compute_coverage_metrics(self):
        """Should compute metrics end-to-end."""
        y_true = np.array([0, 0, 1, 1])
        y_logits = np.array([
            [2.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ])
        
        result = compute_coverage_metrics(
            y_true, y_logits,
            print_report=False
        )
        
        assert isinstance(result, CoverageResult)
        assert result.coverage_at_1 == 1.0  # All correct


class TestCoverageEdgeCases:
    """Tests for edge cases."""
    
    def test_single_sample(self):
        """Should handle single sample."""
        y_true = np.array([0])
        y_logits = np.array([[2.0, 1.0, 0.5]])
        
        analyzer = CoverageAnalyzer()
        analyzer.load_data(y_true, y_logits)
        result = analyzer.analyze()
        
        assert result.coverage_at_1 == 1.0
        assert result.total_samples == 1
    
    def test_single_class(self):
        """Should handle single class."""
        y_true = np.array([0, 0, 0])
        y_logits = np.array([
            [2.0],
            [1.5],
            [1.8],
        ])
        
        analyzer = CoverageAnalyzer()
        analyzer.load_data(y_true, y_logits)
        result = analyzer.analyze()
        
        assert result.coverage_at_1 == 1.0
        assert result.num_classes == 1
    
    def test_all_wrong_predictions(self):
        """Should handle 0% accuracy."""
        y_true = np.array([0, 0, 0])
        y_logits = np.array([
            [0.5, 2.0],  # predicts 1, true=0
            [0.5, 2.0],  # predicts 1, true=0
            [0.5, 2.0],  # predicts 1, true=0
        ])
        
        analyzer = CoverageAnalyzer()
        analyzer.load_data(y_true, y_logits)
        result = analyzer.analyze()
        
        assert result.coverage_at_1 == 0.0
    
    def test_analyze_without_load_raises(self):
        """analyze() without load should raise."""
        analyzer = CoverageAnalyzer()
        
        with pytest.raises(RuntimeError):
            analyzer.analyze()
    
    def test_export_results(self, tmp_path):
        """Should export to JSON."""
        y_true = np.array([0, 1])
        y_logits = np.array([[2.0, 1.0], [1.0, 2.0]])
        
        analyzer = CoverageAnalyzer()
        analyzer.load_data(y_true, y_logits)
        analyzer.analyze()
        
        output_path = tmp_path / "coverage.json"
        analyzer.export_results(output_path)
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert "coverage_at_5" in data


class TestCoverageWithClassNames:
    """Tests for coverage with class names."""
    
    def test_load_with_class_names(self):
        """Should store class names."""
        y_true = np.array([0, 1, 2])
        y_logits = np.array([
            [2.0, 1.0, 0.5],
            [0.5, 2.0, 1.0],
            [0.5, 1.0, 2.0],
        ])
        class_names = ["alpha", "beta", "gamma"]
        
        analyzer = CoverageAnalyzer()
        analyzer.load_data(y_true, y_logits, class_names)
        
        assert analyzer._class_names[0] == "alpha"
        assert analyzer._class_names[1] == "beta"
        assert analyzer._class_names[2] == "gamma"
