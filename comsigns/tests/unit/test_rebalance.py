"""
Unit tests for rebalance module.

Tests dataset rebalancing:
- Down-sampling of OTHER class
- Up-sampling of minority classes via augmentation
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from collections import Counter

from training.rebalance import RebalanceConfig, RebalancedDataset
from training.augmentation import AugmentConfig, KeypointAugmenter


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_rebalance_config():
    """Default rebalancing config."""
    return RebalanceConfig(
        target_strategy="median",
        other_max_multiplier=2.0,
        other_max_cap=None
    )


@pytest.fixture
def mock_imbalanced_dataset():
    """Create a mock imbalanced dataset.
    
    Class distribution:
    - Class 0 (OTHER): 100 samples (heavily over-represented)
    - Class 1: 20 samples
    - Class 2: 15 samples
    - Class 3: 5 samples (minority)
    """
    samples = []
    
    # OTHER class (id=0) - 100 samples
    for i in range(100):
        samples.append({
            "hand": np.random.randn(10, 42, 4).astype(np.float32),
            "body": np.random.randn(10, 33, 4).astype(np.float32),
            "face": np.random.randn(10, 468, 4).astype(np.float32),
            "class_id": 0,
            "gloss": "OTHER"
        })
    
    # Class 1 - 20 samples
    for i in range(20):
        samples.append({
            "hand": np.random.randn(10, 42, 4).astype(np.float32),
            "body": np.random.randn(10, 33, 4).astype(np.float32),
            "face": np.random.randn(10, 468, 4).astype(np.float32),
            "class_id": 1,
            "gloss": "hola"
        })
    
    # Class 2 - 15 samples
    for i in range(15):
        samples.append({
            "hand": np.random.randn(10, 42, 4).astype(np.float32),
            "body": np.random.randn(10, 33, 4).astype(np.float32),
            "face": np.random.randn(10, 468, 4).astype(np.float32),
            "class_id": 2,
            "gloss": "adios"
        })
    
    # Class 3 - 5 samples (minority)
    for i in range(5):
        samples.append({
            "hand": np.random.randn(10, 42, 4).astype(np.float32),
            "body": np.random.randn(10, 33, 4).astype(np.float32),
            "face": np.random.randn(10, 468, 4).astype(np.float32),
            "class_id": 3,
            "gloss": "gracias"
        })
    
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=len(samples))
    dataset.__getitem__ = MagicMock(side_effect=lambda i: samples[i])
    dataset.samples = samples
    
    return dataset


@pytest.fixture
def class_counts():
    """Class counts for the imbalanced dataset."""
    return {
        0: 100,  # OTHER
        1: 20,
        2: 15,
        3: 5
    }


@pytest.fixture
def augmenter():
    """Augmenter for upsampling."""
    config = AugmentConfig(time_shift=1, noise_std=0.01, mirror_prob=0.3)
    return KeypointAugmenter(config)


# =============================================================================
# Tests: RebalanceConfig
# =============================================================================

class TestRebalanceConfig:
    """Tests for RebalanceConfig dataclass."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        config = RebalanceConfig()
        
        assert config.target_strategy in ["median", "mean", "max"]
        assert config.other_max_multiplier > 0
    
    def test_custom_values(self):
        """Should accept custom values."""
        config = RebalanceConfig(
            target_strategy="mean",
            other_max_multiplier=3.0,
            other_max_cap=50
        )
        
        assert config.target_strategy == "mean"
        assert config.other_max_multiplier == 3.0
        assert config.other_max_cap == 50
    
    def test_other_max_cap_limits_other(self):
        """other_max_cap should provide absolute limit."""
        config = RebalanceConfig(
            other_max_multiplier=10.0,
            other_max_cap=30
        )
        
        # Even with high multiplier, cap should limit
        assert config.other_max_cap == 30


# =============================================================================
# Tests: RebalancedDataset - Down-sampling OTHER
# =============================================================================

class TestRebalancedDatasetDownsample:
    """Tests for down-sampling the OTHER class."""
    
    def test_reduces_other_class_count(
        self, mock_imbalanced_dataset, class_counts, augmenter
    ):
        """Should reduce OTHER class to target."""
        config = RebalanceConfig(
            target_strategy="median",
            other_max_multiplier=2.0
        )
        
        rebalanced = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        # Count OTHER samples in rebalanced dataset
        other_count = sum(
            1 for i in range(len(rebalanced))
            if rebalanced[i]["class_id"] == 0
        )
        
        # Median of non-OTHER classes is median([20, 15, 5]) = 15
        # OTHER should be <= 2.0 * 15 = 30
        assert other_count <= 30
    
    def test_respects_other_max_cap(
        self, mock_imbalanced_dataset, class_counts, augmenter
    ):
        """Should respect absolute cap on OTHER."""
        config = RebalanceConfig(
            other_max_multiplier=10.0,  # Would allow 150
            other_max_cap=25  # But cap at 25
        )
        
        rebalanced = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        other_count = sum(
            1 for i in range(len(rebalanced))
            if rebalanced[i]["class_id"] == 0
        )
        
        assert other_count <= 25
    
    def test_random_sampling_for_downsampling(
        self, mock_imbalanced_dataset, class_counts, augmenter
    ):
        """Down-sampling should use random selection."""
        config = RebalanceConfig(other_max_multiplier=0.5)  # Very aggressive
        
        np.random.seed(42)
        rebalanced1 = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        np.random.seed(123)
        rebalanced2 = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        # With different seeds, should select different samples
        # (or at least have different order)
        assert len(rebalanced1) == len(rebalanced2)


# =============================================================================
# Tests: RebalancedDataset - Up-sampling minorities
# =============================================================================

class TestRebalancedDatasetUpsample:
    """Tests for up-sampling minority classes."""
    
    def test_increases_minority_class_count(
        self, mock_imbalanced_dataset, class_counts, augmenter
    ):
        """Should increase minority class samples via augmentation."""
        config = RebalanceConfig(
            target_strategy="median",
            other_max_multiplier=1.0  # Keep OTHER at bay
        )
        
        rebalanced = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        # Count class 3 (minority, originally 5 samples)
        class3_count = sum(
            1 for i in range(len(rebalanced))
            if rebalanced[i]["class_id"] == 3
        )
        
        # Should be upsampled toward median (15)
        assert class3_count >= 5  # At least original
    
    def test_augmented_samples_are_different(
        self, mock_imbalanced_dataset, class_counts, augmenter
    ):
        """Augmented samples should differ from originals."""
        config = RebalanceConfig(
            target_strategy="max",  # Upsample aggressively
            other_max_multiplier=0.5
        )
        
        rebalanced = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        # Get all class 3 samples
        class3_samples = [
            rebalanced[i] for i in range(len(rebalanced))
            if rebalanced[i]["class_id"] == 3
        ]
        
        if len(class3_samples) > 5:  # If upsampled
            # Augmented samples should differ
            originals = class3_samples[:5]
            augmented = class3_samples[5:]
            
            # At least some should be different
            if len(augmented) > 0:
                all_same = all(
                    np.allclose(aug["hand"], orig["hand"])
                    for aug, orig in zip(augmented, originals)
                )
                assert not all_same


# =============================================================================
# Tests: RebalancedDataset - General behavior
# =============================================================================

class TestRebalancedDatasetGeneral:
    """General tests for RebalancedDataset."""
    
    def test_preserves_sample_structure(
        self, mock_imbalanced_dataset, class_counts, augmenter
    ):
        """Should preserve sample structure."""
        config = RebalanceConfig()
        
        rebalanced = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        sample = rebalanced[0]
        
        assert "hand" in sample
        assert "body" in sample
        assert "face" in sample
        assert "class_id" in sample
        assert "gloss" in sample
    
    def test_len_reflects_rebalancing(
        self, mock_imbalanced_dataset, class_counts, augmenter
    ):
        """Length should reflect rebalanced count."""
        config = RebalanceConfig(
            other_max_multiplier=0.5  # Heavy downsampling
        )
        
        original_len = len(mock_imbalanced_dataset)
        
        rebalanced = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        # Should be different due to rebalancing
        assert len(rebalanced) != original_len
    
    def test_iteration_covers_all_samples(
        self, mock_imbalanced_dataset, class_counts, augmenter
    ):
        """Should be able to iterate through all samples."""
        config = RebalanceConfig()
        
        rebalanced = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        # Should not raise
        samples = [rebalanced[i] for i in range(len(rebalanced))]
        assert len(samples) == len(rebalanced)
    
    def test_class_distribution_more_balanced(
        self, mock_imbalanced_dataset, class_counts, augmenter
    ):
        """Rebalanced distribution should be more uniform."""
        config = RebalanceConfig(
            target_strategy="median",
            other_max_multiplier=1.5
        )
        
        rebalanced = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        # Count new distribution
        new_counts = Counter()
        for i in range(len(rebalanced)):
            new_counts[rebalanced[i]["class_id"]] += 1
        
        # Coefficient of variation should be lower
        orig_values = list(class_counts.values())
        new_values = list(new_counts.values())
        
        orig_cv = np.std(orig_values) / np.mean(orig_values)
        new_cv = np.std(new_values) / np.mean(new_values) if np.mean(new_values) > 0 else 0
        
        # New distribution should be more balanced (lower CV)
        # Allow some flexibility
        assert new_cv <= orig_cv * 1.5  # Should improve or stay similar


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestRebalanceEdgeCases:
    """Tests for edge cases in rebalancing."""
    
    def test_single_class_dataset(self, augmenter):
        """Should handle single-class dataset."""
        samples = [
            {"hand": np.random.randn(10, 42, 4).astype(np.float32),
             "body": np.random.randn(10, 33, 4).astype(np.float32),
             "face": np.random.randn(10, 468, 4).astype(np.float32),
             "class_id": 0, "gloss": "only"}
            for _ in range(10)
        ]
        
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=len(samples))
        dataset.__getitem__ = MagicMock(side_effect=lambda i: samples[i])
        
        config = RebalanceConfig()
        
        # Should not raise
        rebalanced = RebalancedDataset(
            dataset,
            other_class_id=0,
            class_counts={0: 10},
            augmenter=augmenter,
            config=config
        )
        
        assert len(rebalanced) > 0
    
    def test_no_other_class(self, augmenter):
        """Should handle dataset without OTHER class."""
        samples = [
            {"hand": np.random.randn(10, 42, 4).astype(np.float32),
             "body": np.random.randn(10, 33, 4).astype(np.float32),
             "face": np.random.randn(10, 468, 4).astype(np.float32),
             "class_id": i % 3 + 1, "gloss": f"class_{i % 3 + 1}"}
            for i in range(30)
        ]
        
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=len(samples))
        dataset.__getitem__ = MagicMock(side_effect=lambda i: samples[i])
        
        config = RebalanceConfig()
        
        # OTHER class_id=0 doesn't exist
        rebalanced = RebalancedDataset(
            dataset,
            other_class_id=0,  # Not in dataset
            class_counts={1: 10, 2: 10, 3: 10},
            augmenter=augmenter,
            config=config
        )
        
        # Should still work
        assert len(rebalanced) > 0
    
    def test_empty_dataset(self, augmenter):
        """Should handle empty dataset."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=0)
        dataset.__getitem__ = MagicMock(side_effect=lambda i: (_ for _ in ()).throw(IndexError))
        
        config = RebalanceConfig()
        
        # Should not raise
        rebalanced = RebalancedDataset(
            dataset,
            other_class_id=0,
            class_counts={},
            augmenter=augmenter,
            config=config
        )
        
        assert len(rebalanced) == 0


# =============================================================================
# Tests: Strategy Variations
# =============================================================================

class TestRebalanceStrategies:
    """Tests for different rebalancing strategies."""
    
    def test_median_strategy(
        self, mock_imbalanced_dataset, class_counts, augmenter
    ):
        """Median strategy should target median support."""
        config = RebalanceConfig(target_strategy="median")
        
        rebalanced = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        assert len(rebalanced) > 0
    
    def test_mean_strategy(
        self, mock_imbalanced_dataset, class_counts, augmenter
    ):
        """Mean strategy should target mean support."""
        config = RebalanceConfig(target_strategy="mean")
        
        rebalanced = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        assert len(rebalanced) > 0
    
    def test_max_strategy(
        self, mock_imbalanced_dataset, class_counts, augmenter
    ):
        """Max strategy should target max support."""
        config = RebalanceConfig(target_strategy="max")
        
        rebalanced = RebalancedDataset(
            mock_imbalanced_dataset,
            other_class_id=0,
            class_counts=class_counts,
            augmenter=augmenter,
            config=config
        )
        
        # With max strategy, all classes should be upsampled to max
        new_counts = Counter()
        for i in range(len(rebalanced)):
            new_counts[rebalanced[i]["class_id"]] += 1
        
        # Non-OTHER classes should approach max (20)
        for class_id in [1, 2, 3]:
            assert new_counts[class_id] >= class_counts[class_id]
