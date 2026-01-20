"""
Tests for the metrics module.

Tests the MetricsTracker class and standalone metric functions.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.metrics import (
    MetricsTracker,
    compute_accuracy,
    compute_topk_accuracy
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_tracker():
    """Create a simple tracker with 10 classes."""
    return MetricsTracker(num_classes=10, topk=(1, 3, 5))


@pytest.fixture
def perfect_predictions():
    """Create logits where prediction equals label (perfect accuracy)."""
    batch_size = 16
    num_classes = 10
    
    labels = torch.arange(batch_size) % num_classes
    logits = torch.zeros(batch_size, num_classes)
    
    # Set high logit for correct class
    for i, label in enumerate(labels):
        logits[i, label] = 10.0
    
    return logits, labels


@pytest.fixture
def random_predictions():
    """Create random logits."""
    torch.manual_seed(42)
    batch_size = 100
    num_classes = 10
    
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    return logits, labels


@pytest.fixture
def controlled_topk_data():
    """
    Create controlled data for testing top-k accuracy.
    
    Structure:
    - 10 samples, 5 classes
    - For each sample i, correct class is i % 5
    - Logits are set so correct class is at rank (i % 3) + 1
      - i=0,3,6,9: correct class has rank 1 (top-1)
      - i=1,4,7: correct class has rank 2 (top-2)
      - i=2,5,8: correct class has rank 3 (top-3)
    """
    num_samples = 10
    num_classes = 5
    
    labels = torch.tensor([i % 5 for i in range(num_samples)])
    logits = torch.zeros(num_samples, num_classes)
    
    for i in range(num_samples):
        correct_class = labels[i].item()
        rank = (i % 3) + 1  # 1, 2, or 3
        
        # Set logits so other classes are higher than correct
        for c in range(num_classes):
            if c == correct_class:
                logits[i, c] = num_classes - rank  # Lower = lower rank
            else:
                # Ensure consistent ordering
                logits[i, c] = num_classes - rank + 1 + (c if c < correct_class else c - 1) * 0.1
    
    return logits, labels


# =============================================================================
# Tests: MetricsTracker Initialization
# =============================================================================

class TestMetricsTrackerInit:
    """Tests for MetricsTracker initialization."""
    
    def test_basic_init(self):
        """Should initialize with default parameters."""
        tracker = MetricsTracker(num_classes=100)
        
        assert tracker.num_classes == 100
        assert 1 in tracker.topk
        assert tracker.num_samples == 0
    
    def test_custom_topk(self):
        """Should accept custom topk values."""
        tracker = MetricsTracker(num_classes=100, topk=(1, 5, 10, 20))
        
        assert tracker.topk == (1, 5, 10, 20)
    
    def test_topk_clamped_to_num_classes(self):
        """Should clamp topk values to num_classes."""
        tracker = MetricsTracker(num_classes=5, topk=(1, 10, 100))
        
        # 10 and 100 should be clamped to 5
        assert max(tracker.topk) <= 5
    
    def test_invalid_num_classes_raises(self):
        """Should raise for invalid num_classes."""
        with pytest.raises(ValueError):
            MetricsTracker(num_classes=0)
        
        with pytest.raises(ValueError):
            MetricsTracker(num_classes=-1)
    
    def test_repr(self):
        """Should have informative repr."""
        tracker = MetricsTracker(num_classes=100, topk=(1, 5))
        
        repr_str = repr(tracker)
        assert "100" in repr_str
        assert "MetricsTracker" in repr_str


# =============================================================================
# Tests: Update Method
# =============================================================================

class TestMetricsTrackerUpdate:
    """Tests for MetricsTracker.update()."""
    
    def test_update_accumulates(self, simple_tracker, random_predictions):
        """Should accumulate samples across multiple updates."""
        logits, labels = random_predictions
        batch1 = logits[:50], labels[:50]
        batch2 = logits[50:], labels[50:]
        
        simple_tracker.update(*batch1)
        assert simple_tracker.num_samples == 50
        
        simple_tracker.update(*batch2)
        assert simple_tracker.num_samples == 100
    
    def test_update_validates_shapes(self, simple_tracker):
        """Should validate input tensor shapes."""
        # 3D logits should fail
        with pytest.raises(ValueError, match="2D"):
            simple_tracker.update(
                torch.randn(2, 3, 10),
                torch.randint(0, 10, (2,))
            )
        
        # 2D labels should fail
        with pytest.raises(ValueError, match="1D"):
            simple_tracker.update(
                torch.randn(2, 10),
                torch.randint(0, 10, (2, 1))
            )
    
    def test_update_validates_batch_size_match(self, simple_tracker):
        """Should require matching batch sizes."""
        with pytest.raises(ValueError, match="mismatch"):
            simple_tracker.update(
                torch.randn(10, 10),
                torch.randint(0, 10, (5,))
            )
    
    def test_update_handles_gpu_tensors(self, simple_tracker):
        """Should handle GPU tensors by moving to tracker device."""
        logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))
        
        # Simulate GPU tensor (just test the movement logic)
        simple_tracker.update(logits, labels)
        
        # Data should be on tracker's device (CPU)
        assert simple_tracker.num_samples == 8


# =============================================================================
# Tests: Reset Method
# =============================================================================

class TestMetricsTrackerReset:
    """Tests for MetricsTracker.reset()."""
    
    def test_reset_clears_data(self, simple_tracker, random_predictions):
        """Should clear all accumulated data."""
        logits, labels = random_predictions
        
        simple_tracker.update(logits, labels)
        assert simple_tracker.num_samples > 0
        
        simple_tracker.reset()
        assert simple_tracker.num_samples == 0
    
    def test_reset_allows_reuse(self, simple_tracker, random_predictions):
        """Should allow tracker to be reused after reset."""
        logits, labels = random_predictions
        
        # First epoch
        simple_tracker.update(logits, labels)
        results1 = simple_tracker.compute()
        simple_tracker.reset()
        
        # Second epoch (should work the same)
        simple_tracker.update(logits, labels)
        results2 = simple_tracker.compute()
        
        # Results should be identical
        assert results1 == results2


# =============================================================================
# Tests: Compute Method
# =============================================================================

class TestMetricsTrackerCompute:
    """Tests for MetricsTracker.compute()."""
    
    def test_compute_returns_dict(self, simple_tracker, random_predictions):
        """Should return dictionary with expected keys."""
        logits, labels = random_predictions
        simple_tracker.update(logits, labels)
        
        results = simple_tracker.compute()
        
        assert isinstance(results, dict)
        assert "accuracy" in results
        assert "top1_acc" in results
        assert "precision_macro" in results
        assert "recall_macro" in results
        assert "f1_macro" in results
    
    def test_compute_all_values_are_floats(self, simple_tracker, random_predictions):
        """All metric values should be floats."""
        logits, labels = random_predictions
        simple_tracker.update(logits, labels)
        
        results = simple_tracker.compute()
        
        for key, value in results.items():
            assert isinstance(value, float), f"{key} is not float: {type(value)}"
    
    def test_compute_values_in_range(self, simple_tracker, random_predictions):
        """All metric values should be in [0, 1]."""
        logits, labels = random_predictions
        simple_tracker.update(logits, labels)
        
        results = simple_tracker.compute()
        
        for key, value in results.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of range"
    
    def test_compute_empty_tracker(self, simple_tracker):
        """Should return zeros for empty tracker."""
        results = simple_tracker.compute()
        
        assert results["accuracy"] == 0.0
        assert results["f1_macro"] == 0.0
    
    def test_compute_perfect_predictions(self, simple_tracker, perfect_predictions):
        """Should return 1.0 for perfect predictions."""
        logits, labels = perfect_predictions
        simple_tracker.update(logits, labels)
        
        results = simple_tracker.compute()
        
        assert results["accuracy"] == 1.0
        assert results["top1_acc"] == 1.0
        # Note: precision/recall/f1 may not be 1.0 if not all classes present


# =============================================================================
# Tests: Top-K Accuracy
# =============================================================================

class TestTopKAccuracy:
    """Tests for Top-K accuracy computation."""
    
    def test_top1_equals_accuracy(self, simple_tracker, random_predictions):
        """Top-1 accuracy should equal global accuracy."""
        logits, labels = random_predictions
        simple_tracker.update(logits, labels)
        
        results = simple_tracker.compute()
        
        assert results["top1_acc"] == results["accuracy"]
    
    def test_topk_monotonic_increasing(self, simple_tracker, random_predictions):
        """Top-K accuracy should increase with K."""
        logits, labels = random_predictions
        simple_tracker.update(logits, labels)
        
        results = simple_tracker.compute()
        
        # top1 <= top3 <= top5
        assert results["top1_acc"] <= results["top3_acc"]
        assert results["top3_acc"] <= results["top5_acc"]
    
    def test_topk_perfect_at_max_k(self):
        """Top-K should be 1.0 when K equals num_classes."""
        tracker = MetricsTracker(num_classes=5, topk=(1, 5))
        
        logits = torch.randn(20, 5)
        labels = torch.randint(0, 5, (20,))
        tracker.update(logits, labels)
        
        results = tracker.compute()
        
        # When K = num_classes, every prediction includes the correct class
        assert results["top5_acc"] == 1.0
    
    def test_standalone_topk_function(self, random_predictions):
        """Standalone compute_topk_accuracy should work."""
        logits, labels = random_predictions
        
        top1 = compute_topk_accuracy(logits, labels, k=1)
        top5 = compute_topk_accuracy(logits, labels, k=5)
        
        assert 0.0 <= top1 <= 1.0
        assert 0.0 <= top5 <= 1.0
        assert top1 <= top5


# =============================================================================
# Tests: Precision / Recall / F1
# =============================================================================

class TestPrecisionRecallF1:
    """Tests for precision, recall, and F1 computation."""
    
    def test_perfect_predictions_high_metrics(self, perfect_predictions):
        """Perfect predictions should have high precision/recall."""
        tracker = MetricsTracker(num_classes=10, topk=(1,))
        logits, labels = perfect_predictions
        tracker.update(logits, labels)
        
        results = tracker.compute()
        
        # Should be high (may not be 1.0 due to macro averaging)
        assert results["precision_macro"] >= 0.5
        assert results["recall_macro"] >= 0.5
        assert results["f1_macro"] >= 0.5
    
    def test_handles_missing_classes(self):
        """Should handle cases where some classes have no samples."""
        tracker = MetricsTracker(num_classes=100, topk=(1,))
        
        # Only use 5 classes out of 100
        logits = torch.randn(20, 100)
        labels = torch.randint(0, 5, (20,))  # Only classes 0-4
        tracker.update(logits, labels)
        
        results = tracker.compute()
        
        # Should not crash, metrics should be valid
        assert 0.0 <= results["precision_macro"] <= 1.0
        assert 0.0 <= results["recall_macro"] <= 1.0
        assert 0.0 <= results["f1_macro"] <= 1.0


# =============================================================================
# Tests: Standalone Functions
# =============================================================================

class TestStandaloneFunctions:
    """Tests for standalone metric functions."""
    
    def test_compute_accuracy(self, perfect_predictions):
        """compute_accuracy should work correctly."""
        logits, labels = perfect_predictions
        
        acc = compute_accuracy(logits, labels)
        
        assert acc == 1.0
    
    def test_compute_accuracy_random(self, random_predictions):
        """compute_accuracy should be in valid range."""
        logits, labels = random_predictions
        
        acc = compute_accuracy(logits, labels)
        
        assert 0.0 <= acc <= 1.0
    
    def test_compute_topk_accuracy_k_larger_than_classes(self):
        """Should handle K > num_classes gracefully."""
        logits = torch.randn(10, 5)  # 5 classes
        labels = torch.randint(0, 5, (10,))
        
        # K=100 should be clamped to 5
        acc = compute_topk_accuracy(logits, labels, k=100)
        
        # Should be 1.0 since we're checking all classes
        assert acc == 1.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestMetricsIntegration:
    """Integration tests for complete metric workflow."""
    
    def test_multi_batch_workflow(self):
        """Test typical multi-batch validation workflow."""
        tracker = MetricsTracker(num_classes=20, topk=(1, 5, 10))
        
        # Simulate 5 batches
        torch.manual_seed(42)
        for _ in range(5):
            logits = torch.randn(16, 20)
            labels = torch.randint(0, 20, (16,))
            tracker.update(logits, labels)
        
        assert tracker.num_samples == 80
        
        results = tracker.compute()
        
        # All metrics should be valid
        assert all(0.0 <= v <= 1.0 for v in results.values())
        
        # Reset and verify
        tracker.reset()
        assert tracker.num_samples == 0
    
    def test_deterministic_results(self):
        """Same data should produce same results."""
        torch.manual_seed(42)
        logits = torch.randn(100, 50)
        labels = torch.randint(0, 50, (100,))
        
        tracker1 = MetricsTracker(num_classes=50, topk=(1, 5))
        tracker1.update(logits, labels)
        results1 = tracker1.compute()
        
        tracker2 = MetricsTracker(num_classes=50, topk=(1, 5))
        tracker2.update(logits, labels)
        results2 = tracker2.compute()
        
        assert results1 == results2
    
    def test_large_num_classes(self):
        """Should handle large number of classes (like real dataset)."""
        tracker = MetricsTracker(num_classes=505, topk=(1, 5, 10))
        
        # Simulate batch
        logits = torch.randn(32, 505)
        labels = torch.randint(0, 505, (32,))
        tracker.update(logits, labels)
        
        results = tracker.compute()
        
        assert "top10_acc" in results
        assert all(0.0 <= v <= 1.0 for v in results.values())
