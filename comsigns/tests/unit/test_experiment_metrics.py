"""
Unit tests for ExperimentMetricsTracker.

Tests the metrics computation for the TAIL → OTHER experiment:
- Global metrics (accuracy, F1 scores)
- Bucket metrics (HEAD, MID, TAIL/OTHER)
- Coverage metrics
- Collapse diagnostics
- Artifact export
- Comparison utilities
"""

import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

from training.experiment_metrics import (
    GlobalMetrics,
    BucketMetrics,
    CoverageMetrics,
    CollapseDiagnostics,
    ExperimentMetricsTracker,
    create_experiment_tracker,
    compare_experiments
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_bucket_mapping() -> Dict[int, str]:
    """Simple bucket mapping for 10 classes."""
    return {
        0: "HEAD", 1: "HEAD",                    # 2 HEAD classes
        2: "MID", 3: "MID", 4: "MID",           # 3 MID classes
        5: "TAIL", 6: "TAIL", 7: "TAIL",        # 3 TAIL classes
        8: "TAIL", 9: "TAIL"                     # 2 more TAIL classes
    }


@pytest.fixture
def tail_to_other_mapping() -> Dict[int, str]:
    """Bucket mapping for tail_to_other experiment (6 classes)."""
    return {
        0: "HEAD", 1: "HEAD",    # 2 HEAD classes
        2: "MID", 3: "MID",      # 2 MID classes
        4: "MID",                # 1 more MID
        5: "OTHER"               # Collapsed TAIL
    }


@pytest.fixture
def perfect_predictions():
    """Logits and targets for perfect predictions."""
    num_samples = 20
    num_classes = 10
    
    targets = np.array([i % num_classes for i in range(num_samples)])
    logits = np.zeros((num_samples, num_classes))
    for i, t in enumerate(targets):
        logits[i, t] = 10.0  # High confidence on correct class
    
    return torch.tensor(logits), torch.tensor(targets)


@pytest.fixture
def random_predictions():
    """Random logits and uniform targets."""
    num_samples = 100
    num_classes = 10
    
    np.random.seed(42)
    targets = np.random.randint(0, num_classes, num_samples)
    logits = np.random.randn(num_samples, num_classes)
    
    return torch.tensor(logits, dtype=torch.float32), torch.tensor(targets)


@pytest.fixture
def collapsed_predictions():
    """Predictions that collapse to a single class."""
    num_samples = 50
    num_classes = 10
    
    targets = np.array([i % num_classes for i in range(num_samples)])
    logits = np.zeros((num_samples, num_classes))
    logits[:, 0] = 10.0  # All predictions go to class 0
    
    return torch.tensor(logits), torch.tensor(targets)


# =============================================================================
# GlobalMetrics Tests
# =============================================================================

class TestGlobalMetrics:
    """Tests for global metrics computation."""
    
    def test_perfect_accuracy(self, simple_bucket_mapping, perfect_predictions):
        """Test metrics with perfect predictions."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        metrics = tracker.compute_global_metrics()
        
        assert metrics.accuracy_at_1 == 1.0
        assert metrics.accuracy_at_5 == 1.0
        assert metrics.micro_f1 == 1.0
        assert metrics.num_samples == 20
    
    def test_random_accuracy_reasonable(self, simple_bucket_mapping, random_predictions):
        """Test metrics with random predictions are in valid range."""
        logits, targets = random_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        metrics = tracker.compute_global_metrics()
        
        assert 0.0 <= metrics.accuracy_at_1 <= 1.0
        assert 0.0 <= metrics.accuracy_at_5 <= 1.0
        assert metrics.accuracy_at_5 >= metrics.accuracy_at_1  # Top-5 >= Top-1
        assert 0.0 <= metrics.micro_f1 <= 1.0
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert 0.0 <= metrics.weighted_f1 <= 1.0
    
    def test_empty_tracker(self):
        """Test metrics with no data."""
        tracker = ExperimentMetricsTracker(num_classes=10)
        metrics = tracker.compute_global_metrics()
        
        assert metrics.accuracy_at_1 == 0.0
        assert metrics.num_samples == 0
    
    def test_to_dict(self, simple_bucket_mapping, perfect_predictions):
        """Test GlobalMetrics serialization."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        metrics = tracker.compute_global_metrics()
        d = metrics.to_dict()
        
        assert "accuracy_at_1" in d
        assert "accuracy_at_5" in d
        assert "micro_f1" in d
        assert "macro_f1" in d
        assert "weighted_f1" in d


# =============================================================================
# BucketMetrics Tests
# =============================================================================

class TestBucketMetrics:
    """Tests for bucket-aware metrics."""
    
    def test_bucket_separation(self, simple_bucket_mapping, perfect_predictions):
        """Test that metrics are computed per bucket."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        bucket_metrics = tracker.compute_bucket_metrics()
        
        assert "HEAD" in bucket_metrics
        assert "MID" in bucket_metrics
        assert "TAIL" in bucket_metrics
        
        assert bucket_metrics["HEAD"].num_classes_in_bucket == 2
        assert bucket_metrics["MID"].num_classes_in_bucket == 3
        assert bucket_metrics["TAIL"].num_classes_in_bucket == 5
    
    def test_head_metrics_perfect(self, simple_bucket_mapping, perfect_predictions):
        """Test HEAD bucket with perfect predictions."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        bucket_metrics = tracker.compute_bucket_metrics()
        
        # HEAD should have perfect metrics for its samples
        head_metrics = bucket_metrics["HEAD"]
        assert head_metrics.accuracy_at_1 == 1.0
        assert head_metrics.accuracy_at_5 == 1.0
    
    def test_classes_predicted_count(self, simple_bucket_mapping, perfect_predictions):
        """Test num_classes_predicted count."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        bucket_metrics = tracker.compute_bucket_metrics()
        
        # With perfect predictions, all classes should be predicted
        for bucket_name, bm in bucket_metrics.items():
            assert bm.num_classes_predicted > 0
    
    def test_tail_to_other_bucket(self, tail_to_other_mapping):
        """Test OTHER bucket in tail_to_other experiment."""
        num_samples = 30
        num_classes = 6
        
        targets = np.array([i % num_classes for i in range(num_samples)])
        logits = np.zeros((num_samples, num_classes))
        for i, t in enumerate(targets):
            logits[i, t] = 10.0
        
        tracker = ExperimentMetricsTracker(
            num_classes=num_classes,
            bucket_mapping=tail_to_other_mapping,
            other_class_id=5
        )
        tracker.update(torch.tensor(logits), torch.tensor(targets))
        
        bucket_metrics = tracker.compute_bucket_metrics()
        
        assert "OTHER" in bucket_metrics
        assert bucket_metrics["OTHER"].num_classes_in_bucket == 1
        assert bucket_metrics["OTHER"].accuracy_at_1 == 1.0


# =============================================================================
# CoverageMetrics Tests
# =============================================================================

class TestCoverageMetrics:
    """Tests for coverage metrics."""
    
    def test_perfect_coverage(self, simple_bucket_mapping, perfect_predictions):
        """Test coverage with perfect predictions."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        coverage = tracker.compute_coverage_metrics()
        
        assert coverage.coverage_at_1 == 1.0
        assert coverage.coverage_at_5 == 1.0
    
    def test_collapsed_coverage(self, simple_bucket_mapping, collapsed_predictions):
        """Test coverage when predictions collapse."""
        logits, targets = collapsed_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        coverage = tracker.compute_coverage_metrics()
        
        # Only class 0 is predicted correctly (if it appears in targets)
        assert coverage.coverage_at_1 < 0.5
        assert coverage.num_classes_predicted_at_1 <= 1
    
    def test_coverage_by_bucket(self, simple_bucket_mapping, perfect_predictions):
        """Test per-bucket coverage."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        coverage = tracker.compute_coverage_metrics()
        
        assert "HEAD" in coverage.coverage_at_1_by_bucket
        assert "MID" in coverage.coverage_at_1_by_bucket
        assert "TAIL" in coverage.coverage_at_1_by_bucket


# =============================================================================
# CollapseDiagnostics Tests
# =============================================================================

class TestCollapseDiagnostics:
    """Tests for collapse diagnostics."""
    
    def test_collapsed_detection(self, simple_bucket_mapping, collapsed_predictions):
        """Test detection of collapsed predictions."""
        logits, targets = collapsed_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        diagnostics = tracker.compute_collapse_diagnostics()
        
        assert diagnostics.pct_predictions_most_frequent == 1.0
        assert diagnostics.most_frequent_class_id == 0
        assert diagnostics.num_unique_predictions == 1
        # Entropy should be ~0 when all predictions are same class (allow small floating point error)
        assert diagnostics.prediction_entropy == pytest.approx(0.0, abs=1e-9)
    
    def test_diverse_predictions(self, simple_bucket_mapping, perfect_predictions):
        """Test diagnostics with diverse predictions."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        diagnostics = tracker.compute_collapse_diagnostics()
        
        # Should have multiple unique predictions
        assert diagnostics.num_unique_predictions > 1
        # Entropy should be > 0 with diverse predictions
        assert diagnostics.prediction_entropy > 0.0
    
    def test_other_class_percentage(self, tail_to_other_mapping):
        """Test OTHER class percentage calculation."""
        num_samples = 30
        num_classes = 6
        
        # All predictions go to OTHER (class 5)
        targets = np.array([i % num_classes for i in range(num_samples)])
        logits = np.zeros((num_samples, num_classes))
        logits[:, 5] = 10.0  # All predict OTHER
        
        tracker = ExperimentMetricsTracker(
            num_classes=num_classes,
            bucket_mapping=tail_to_other_mapping,
            other_class_id=5
        )
        tracker.update(torch.tensor(logits), torch.tensor(targets))
        
        diagnostics = tracker.compute_collapse_diagnostics()
        
        assert diagnostics.pct_predictions_other == 1.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestExperimentMetricsIntegration:
    """Integration tests for full metrics pipeline."""
    
    def test_compute_all(self, simple_bucket_mapping, perfect_predictions):
        """Test compute_all returns complete results."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping,
            experiment_id="test_exp_001"
        )
        tracker.update(logits, targets)
        
        results = tracker.compute_all()
        
        assert "experiment_id" in results
        assert "timestamp" in results
        assert "global" in results
        assert "by_bucket" in results
        assert "coverage" in results
        assert "collapse_diagnostics" in results
        assert "config" in results
    
    def test_export_artifacts(self, simple_bucket_mapping, perfect_predictions, tmp_path):
        """Test artifact export."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping,
            experiment_id="test_export"
        )
        tracker.update(logits, targets)
        
        artifacts = tracker.export_artifacts(tmp_path)
        
        assert len(artifacts) == 4
        assert (tmp_path / "metrics_global.json").exists()
        assert (tmp_path / "metrics_by_bucket.json").exists()
        assert (tmp_path / "coverage_metrics.json").exists()
        assert (tmp_path / "collapse_diagnostics.json").exists()
        
        # Verify JSON content
        with open(tmp_path / "metrics_global.json") as f:
            data = json.load(f)
            assert "experiment_id" in data
            assert "accuracy_at_1" in data
    
    def test_reset(self, simple_bucket_mapping, perfect_predictions):
        """Test tracker reset."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        # Verify data exists
        assert len(tracker._predictions) > 0
        
        tracker.reset()
        
        # Verify data cleared
        assert len(tracker._predictions) == 0
        assert len(tracker._targets) == 0
    
    def test_multiple_batches(self, simple_bucket_mapping):
        """Test accumulation across multiple batches."""
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        
        # Add multiple batches
        for _ in range(5):
            logits = torch.randn(10, 10)
            targets = torch.randint(0, 10, (10,))
            tracker.update(logits, targets)
        
        assert len(tracker._predictions) == 50
        
        metrics = tracker.compute_global_metrics()
        assert metrics.num_samples == 50
    
    def test_caching(self, simple_bucket_mapping, perfect_predictions):
        """Test that metrics are cached."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits, targets)
        
        # First call computes
        metrics1 = tracker.compute_global_metrics()
        # Second call returns cached
        metrics2 = tracker.compute_global_metrics()
        
        assert metrics1 is metrics2
    
    def test_cache_invalidation(self, simple_bucket_mapping, perfect_predictions):
        """Test that cache is invalidated on update."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        tracker.update(logits[:10], targets[:10])
        
        metrics1 = tracker.compute_global_metrics()
        
        tracker.update(logits[10:], targets[10:])
        
        metrics2 = tracker.compute_global_metrics()
        
        # Should be different objects (recalculated)
        assert metrics1 is not metrics2
    
    def test_get_summary(self, simple_bucket_mapping, perfect_predictions):
        """Test human-readable summary."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping,
            experiment_id="summary_test"
        )
        tracker.update(logits, targets)
        
        summary = tracker.get_summary()
        
        assert "Experiment Metrics Summary" in summary
        assert "GLOBAL METRICS" in summary
        assert "BUCKET METRICS" in summary
        assert "COVERAGE" in summary
        assert "COLLAPSE DIAGNOSTICS" in summary


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateExperimentTracker:
    """Tests for factory function."""
    
    def test_basic_creation(self):
        """Test basic tracker creation."""
        tracker = create_experiment_tracker(
            num_classes=50,
            experiment_id="test_001"
        )
        
        assert tracker.num_classes == 50
        assert tracker.experiment_id == "test_001"
    
    def test_with_bucket_mapping(self, simple_bucket_mapping):
        """Test creation with bucket mapping."""
        tracker = create_experiment_tracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        
        assert "HEAD" in tracker.bucket_to_classes
        assert "MID" in tracker.bucket_to_classes
        assert "TAIL" in tracker.bucket_to_classes


# =============================================================================
# Comparison Tests
# =============================================================================

class TestCompareExperiments:
    """Tests for experiment comparison."""
    
    def test_comparison_structure(self):
        """Test comparison output structure."""
        baseline = {
            "experiment_id": "baseline",
            "global": {
                "accuracy_at_1": 0.09,
                "accuracy_at_5": 0.25,
                "macro_f1": 0.05,
                "weighted_f1": 0.08
            },
            "by_bucket": {
                "HEAD": {"accuracy_at_1": 0.60},
                "MID": {"accuracy_at_1": 0.10}
            },
            "coverage": {
                "coverage_at_5_by_bucket": {"HEAD": 0.8, "MID": 0.3}
            }
        }
        
        experiment = {
            "experiment_id": "tail_to_other",
            "global": {
                "accuracy_at_1": 0.15,
                "accuracy_at_5": 0.35,
                "macro_f1": 0.10,
                "weighted_f1": 0.12
            },
            "by_bucket": {
                "HEAD": {"accuracy_at_1": 0.75},
                "MID": {"accuracy_at_1": 0.20}
            },
            "coverage": {
                "coverage_at_5_by_bucket": {"HEAD": 0.9, "MID": 0.5}
            }
        }
        
        comparison = compare_experiments(baseline, experiment)
        
        assert "baseline" in comparison
        assert "experiment" in comparison
        assert "delta" in comparison
        assert "bucket_comparison" in comparison
        assert "key_metrics" in comparison
    
    def test_delta_calculation(self):
        """Test delta calculation is correct."""
        baseline = {
            "global": {"accuracy_at_1": 0.10, "accuracy_at_5": 0.20, "macro_f1": 0.05, "weighted_f1": 0.08}
        }
        experiment = {
            "global": {"accuracy_at_1": 0.20, "accuracy_at_5": 0.35, "macro_f1": 0.15, "weighted_f1": 0.18}
        }
        
        comparison = compare_experiments(baseline, experiment)
        
        assert comparison["delta"]["accuracy_at_1"] == pytest.approx(0.10)
        assert comparison["delta"]["accuracy_at_5"] == pytest.approx(0.15)
        assert comparison["delta"]["macro_f1"] == pytest.approx(0.10)
    
    def test_improvement_detection(self):
        """Test improvement detection in key_metrics."""
        baseline = {
            "global": {"accuracy_at_1": 0.10, "accuracy_at_5": 0.20, "macro_f1": 0.05, "weighted_f1": 0.08},
            "by_bucket": {
                "HEAD": {"accuracy_at_1": 0.50},
                "MID": {"accuracy_at_1": 0.10}
            },
            "coverage": {"coverage_at_5_by_bucket": {"HEAD": 0.8, "MID": 0.3}}
        }
        experiment = {
            "global": {"accuracy_at_1": 0.20, "accuracy_at_5": 0.35, "macro_f1": 0.15, "weighted_f1": 0.18},
            "by_bucket": {
                "HEAD": {"accuracy_at_1": 0.70},
                "MID": {"accuracy_at_1": 0.25}
            },
            "coverage": {"coverage_at_5_by_bucket": {"HEAD": 0.9, "MID": 0.5}}
        }
        
        comparison = compare_experiments(baseline, experiment)
        
        # Check that deltas are computed correctly
        assert comparison["bucket_comparison"]["HEAD"]["delta_acc_at_1"] == pytest.approx(0.20)
        assert comparison["bucket_comparison"]["MID"]["delta_acc_at_1"] == pytest.approx(0.15)
        
        # These should show improvement
        assert comparison["key_metrics"]["delta_acc_at_1_head"] > 0
        assert comparison["key_metrics"]["delta_acc_at_1_mid"] > 0
    
    def test_save_comparison(self, tmp_path):
        """Test saving comparison to file."""
        baseline = {"global": {"accuracy_at_1": 0.10, "accuracy_at_5": 0.2, "macro_f1": 0.05, "weighted_f1": 0.08}}
        experiment = {"global": {"accuracy_at_1": 0.20, "accuracy_at_5": 0.3, "macro_f1": 0.10, "weighted_f1": 0.12}}
        
        output_path = tmp_path / "comparison.json"
        compare_experiments(baseline, experiment, output_path)
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = json.load(f)
            assert "delta" in data


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_sample(self, simple_bucket_mapping):
        """Test with single sample."""
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        
        logits = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        targets = torch.tensor([0])
        
        tracker.update(logits, targets)
        metrics = tracker.compute_global_metrics()
        
        assert metrics.num_samples == 1
        assert metrics.accuracy_at_1 == 1.0
    
    def test_all_same_class(self, simple_bucket_mapping):
        """Test when all samples are same class."""
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        
        logits = torch.randn(20, 10)
        targets = torch.zeros(20, dtype=torch.long)  # All class 0
        
        tracker.update(logits, targets)
        coverage = tracker.compute_coverage_metrics()
        
        assert coverage.total_classes == 1  # Only 1 class in validation
    
    def test_numpy_input(self, simple_bucket_mapping):
        """Test with numpy arrays as input."""
        tracker = ExperimentMetricsTracker(
            num_classes=10,
            bucket_mapping=simple_bucket_mapping
        )
        
        logits = np.random.randn(20, 10)
        targets = np.random.randint(0, 10, 20)
        
        tracker.update(logits, targets)
        metrics = tracker.compute_global_metrics()
        
        assert metrics.num_samples == 20
    
    def test_no_bucket_mapping(self, perfect_predictions):
        """Test without bucket mapping."""
        logits, targets = perfect_predictions
        
        tracker = ExperimentMetricsTracker(num_classes=10)
        tracker.update(logits, targets)
        
        # Global metrics should work
        metrics = tracker.compute_global_metrics()
        assert metrics.accuracy_at_1 == 1.0
        
        # Bucket metrics should be empty
        bucket_metrics = tracker.compute_bucket_metrics()
        assert bucket_metrics == {}
    
    def test_few_classes(self):
        """Test with very few classes (edge case for top-5)."""
        tracker = ExperimentMetricsTracker(num_classes=3)
        
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        
        tracker.update(logits, targets)
        metrics = tracker.compute_global_metrics()
        
        # Should handle k > num_classes gracefully
        assert 0.0 <= metrics.accuracy_at_5 <= 1.0
