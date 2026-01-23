"""
Unit tests for ClassRemapper and RemappedDataset.

Tests the TAIL → OTHER experiment infrastructure:
- ClassRemapper: mapping old class IDs to new collapsed IDs
- RemappedDataset: dataset wrapper with on-the-fly remapping
- Determinism and reproducibility
- Serialization (save/load)
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from training.remapping import (
    Bucket,
    RemapConfig,
    ClassRemapper,
    RemappedDataset,
    compute_class_support,
    create_remapped_datasets
)


# =============================================================================
# Fixtures
# =============================================================================

@dataclass
class MockSample:
    """Mock sample with gloss_id."""
    gloss_id: int
    features: np.ndarray


class MockDataset(Dataset):
    """Mock dataset for testing remapping."""
    
    def __init__(self, samples: List[MockSample], gloss_to_id: Dict[str, int]):
        self.samples = samples
        self.gloss_to_id = gloss_to_id
    
    def __getitem__(self, idx: int) -> MockSample:
        # Return a copy to avoid mutation
        sample = self.samples[idx]
        return MockSample(
            gloss_id=sample.gloss_id,
            features=sample.features.copy()
        )
    
    def __len__(self) -> int:
        return len(self.samples)


@pytest.fixture
def class_support_505() -> Dict[int, int]:
    """Create class support matching real scenario: 505 classes.
    
    Distribution:
    - HEAD (≥10 samples): 3 classes
    - MID (3-9 samples): 47 classes
    - TAIL (1-2 samples): 455 classes
    """
    support = {}
    
    # HEAD: 3 classes with 10+ samples
    support[100] = 15
    support[200] = 12
    support[300] = 10
    
    # MID: 47 classes with 3-9 samples
    for i in range(47):
        support[i + 400] = 3 + (i % 7)  # 3-9 samples
    
    # TAIL: 455 classes with 1-2 samples
    for i in range(455):
        support[i + 500] = 1 + (i % 2)  # 1-2 samples
    
    return support


@pytest.fixture
def class_support_simple() -> Dict[int, int]:
    """Simple class support for basic tests."""
    return {
        0: 15,   # HEAD
        1: 12,   # HEAD
        2: 5,    # MID
        3: 4,    # MID
        4: 3,    # MID
        5: 2,    # TAIL
        6: 1,    # TAIL
        7: 1,    # TAIL
    }


@pytest.fixture
def mock_dataset() -> MockDataset:
    """Create mock dataset with known class distribution."""
    samples = []
    gloss_to_id = {}
    
    # HEAD classes: 2 classes, 10+ samples each
    for i in range(12):
        samples.append(MockSample(gloss_id=0, features=np.random.randn(10)))
    for i in range(11):
        samples.append(MockSample(gloss_id=1, features=np.random.randn(10)))
    
    # MID classes: 3 classes, 3-9 samples each
    for i in range(5):
        samples.append(MockSample(gloss_id=2, features=np.random.randn(10)))
    for i in range(4):
        samples.append(MockSample(gloss_id=3, features=np.random.randn(10)))
    for i in range(3):
        samples.append(MockSample(gloss_id=4, features=np.random.randn(10)))
    
    # TAIL classes: 5 classes, 1-2 samples each
    for i in range(2):
        samples.append(MockSample(gloss_id=5, features=np.random.randn(10)))
    for i in range(1):
        samples.append(MockSample(gloss_id=6, features=np.random.randn(10)))
    for i in range(2):
        samples.append(MockSample(gloss_id=7, features=np.random.randn(10)))
    for i in range(1):
        samples.append(MockSample(gloss_id=8, features=np.random.randn(10)))
    for i in range(1):
        samples.append(MockSample(gloss_id=9, features=np.random.randn(10)))
    
    gloss_to_id = {f"gloss_{i}": i for i in range(10)}
    
    return MockDataset(samples, gloss_to_id)


# =============================================================================
# RemapConfig Tests
# =============================================================================

class TestRemapConfig:
    """Tests for RemapConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = RemapConfig()
        assert config.strategy == "tail_to_other"
        assert config.head_threshold == 10
        assert config.mid_range == (3, 9)
        assert config.other_class_name == "OTHER"
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = RemapConfig(
            strategy="tail_exclude",
            head_threshold=20,
            mid_range=(5, 15),
            other_class_name="UNKNOWN"
        )
        assert config.strategy == "tail_exclude"
        assert config.head_threshold == 20
        assert config.mid_range == (5, 15)
        assert config.other_class_name == "UNKNOWN"
    
    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        config = RemapConfig(
            strategy="tail_to_other",
            head_threshold=15,
            mid_range=(4, 10)
        )
        d = config.to_dict()
        restored = RemapConfig.from_dict(d)
        
        assert restored.strategy == config.strategy
        assert restored.head_threshold == config.head_threshold
        assert restored.mid_range == config.mid_range


# =============================================================================
# ClassRemapper Tests
# =============================================================================

class TestClassRemapper:
    """Tests for ClassRemapper."""
    
    def test_fit_creates_mapping(self, class_support_simple):
        """Test that fit() creates valid mapping."""
        remapper = ClassRemapper()
        remapper.fit(class_support_simple)
        
        assert remapper._fitted
        assert len(remapper.old_to_new) == len(class_support_simple)
        assert remapper.num_classes_original == 8
        # HEAD(2) + MID(3) + OTHER(1) = 6
        assert remapper.num_classes_remapped == 6
    
    def test_fit_with_505_classes(self, class_support_505):
        """Test fit with 505 classes (real scenario)."""
        remapper = ClassRemapper()
        remapper.fit(class_support_505)
        
        assert remapper.num_classes_original == 505
        # HEAD(3) + MID(47) + OTHER(1) = 51
        assert remapper.num_classes_remapped == 51
        assert remapper.classes_collapsed == 455
        assert len(remapper.bucket_to_old_ids[Bucket.HEAD]) == 3
        assert len(remapper.bucket_to_old_ids[Bucket.MID]) == 47
        assert len(remapper.bucket_to_old_ids[Bucket.TAIL]) == 455
    
    def test_deterministic_mapping(self, class_support_simple):
        """Test that same input produces same output."""
        remapper1 = ClassRemapper()
        remapper1.fit(class_support_simple)
        
        remapper2 = ClassRemapper()
        remapper2.fit(class_support_simple)
        
        assert remapper1.old_to_new == remapper2.old_to_new
        assert remapper1.new_to_old == remapper2.new_to_old
    
    def test_head_preserved(self, class_support_simple):
        """Test that HEAD classes maintain separate identities."""
        remapper = ClassRemapper()
        remapper.fit(class_support_simple)
        
        # HEAD classes (0, 1) should map to different new IDs
        new_id_0 = remapper.transform(0)
        new_id_1 = remapper.transform(1)
        
        assert new_id_0 != new_id_1
        # Should be first indices
        assert new_id_0 in [0, 1]
        assert new_id_1 in [0, 1]
    
    def test_mid_preserved(self, class_support_simple):
        """Test that MID classes maintain separate identities."""
        remapper = ClassRemapper()
        remapper.fit(class_support_simple)
        
        # MID classes (2, 3, 4) should map to different new IDs
        new_ids = [remapper.transform(i) for i in [2, 3, 4]]
        
        assert len(set(new_ids)) == 3  # All different
    
    def test_tail_collapsed(self, class_support_simple):
        """Test that all TAIL classes map to OTHER."""
        remapper = ClassRemapper()
        remapper.fit(class_support_simple)
        
        # TAIL classes (5, 6, 7) should all map to same OTHER class
        tail_new_ids = [remapper.transform(i) for i in [5, 6, 7]]
        
        assert len(set(tail_new_ids)) == 1  # All same
        assert tail_new_ids[0] == remapper.other_class_id
    
    def test_transform_returns_correct_id(self, class_support_simple):
        """Test transform returns correct new class ID."""
        remapper = ClassRemapper()
        remapper.fit(class_support_simple)
        
        for old_id in class_support_simple.keys():
            new_id = remapper.transform(old_id)
            assert new_id in remapper.new_to_old
    
    def test_transform_raises_on_unknown_id(self, class_support_simple):
        """Test transform raises on unknown class ID."""
        remapper = ClassRemapper()
        remapper.fit(class_support_simple)
        
        with pytest.raises(ValueError, match="Unknown class ID"):
            remapper.transform(999)
    
    def test_transform_raises_if_not_fitted(self):
        """Test transform raises if not fitted."""
        remapper = ClassRemapper()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            remapper.transform(0)
    
    def test_inverse_transform(self, class_support_simple):
        """Test inverse_transform returns correct old class IDs."""
        remapper = ClassRemapper()
        remapper.fit(class_support_simple)
        
        # OTHER should return all TAIL classes
        other_originals = remapper.inverse_transform(remapper.other_class_id)
        assert set(other_originals) == {5, 6, 7}
        
        # HEAD classes should return single class
        head_new_id = remapper.transform(0)
        head_originals = remapper.inverse_transform(head_new_id)
        assert head_originals == [0]
    
    def test_save_load(self, class_support_simple, tmp_path):
        """Test persistence to JSON."""
        remapper = ClassRemapper()
        remapper.fit(class_support_simple)
        
        save_path = tmp_path / "mapping.json"
        remapper.save(save_path)
        
        assert save_path.exists()
        
        # Load and verify
        loaded = ClassRemapper.load(save_path)
        
        assert loaded.num_classes_original == remapper.num_classes_original
        assert loaded.num_classes_remapped == remapper.num_classes_remapped
        assert loaded.old_to_new == remapper.old_to_new
        assert loaded.new_to_old == remapper.new_to_old
        assert loaded.other_class_id == remapper.other_class_id
    
    def test_other_class_is_last(self, class_support_simple):
        """Test that OTHER is always the last class index."""
        remapper = ClassRemapper()
        remapper.fit(class_support_simple)
        
        # OTHER should be num_classes - 1
        assert remapper.other_class_id == remapper.num_classes_remapped - 1
    
    def test_no_class_id_collision(self, class_support_505):
        """Test no collisions in new class IDs."""
        remapper = ClassRemapper()
        remapper.fit(class_support_505)
        
        # All new class IDs should be unique for HEAD and MID
        head_mid_new_ids = []
        for bucket in [Bucket.HEAD, Bucket.MID]:
            for old_id in remapper.bucket_to_old_ids[bucket]:
                head_mid_new_ids.append(remapper.transform(old_id))
        
        assert len(head_mid_new_ids) == len(set(head_mid_new_ids))
    
    def test_get_bucket(self, class_support_simple):
        """Test get_bucket returns correct bucket."""
        remapper = ClassRemapper()
        remapper.fit(class_support_simple)
        
        assert remapper.get_bucket(0) == Bucket.HEAD
        assert remapper.get_bucket(1) == Bucket.HEAD
        assert remapper.get_bucket(2) == Bucket.MID
        assert remapper.get_bucket(5) == Bucket.TAIL
    
    def test_is_other(self, class_support_simple):
        """Test is_other correctly identifies OTHER class."""
        remapper = ClassRemapper()
        remapper.fit(class_support_simple)
        
        other_id = remapper.other_class_id
        head_id = remapper.transform(0)
        
        assert remapper.is_other(other_id) == True
        assert remapper.is_other(head_id) == False
    
    def test_tail_exclude_strategy(self, class_support_simple):
        """Test tail_exclude strategy maps TAIL to -1."""
        config = RemapConfig(strategy="tail_exclude")
        remapper = ClassRemapper(config)
        remapper.fit(class_support_simple)
        
        # TAIL classes should map to -1
        assert remapper.transform(5) == -1
        assert remapper.transform(6) == -1
        assert remapper.transform(7) == -1
        
        # No OTHER class
        assert remapper.other_class_id == -1
        
        # Only HEAD + MID classes
        assert remapper.num_classes_remapped == 5  # 2 HEAD + 3 MID
    
    def test_get_config_summary(self, class_support_505):
        """Test get_config_summary returns correct info."""
        remapper = ClassRemapper()
        remapper.fit(class_support_505)
        
        summary = remapper.get_config_summary()
        
        assert summary["strategy"] == "tail_to_other"
        assert summary["num_classes_original"] == 505
        assert summary["num_classes_remapped"] == 51
        assert summary["head_classes"] == 3
        assert summary["mid_classes"] == 47
        assert summary["tail_classes"] == 455


# =============================================================================
# RemappedDataset Tests
# =============================================================================

class TestRemappedDataset:
    """Tests for RemappedDataset."""
    
    def test_num_classes_property(self, mock_dataset):
        """Test num_classes returns remapped count."""
        support = compute_class_support(mock_dataset)
        remapper = ClassRemapper()
        remapper.fit(support)
        
        wrapped = RemappedDataset(mock_dataset, remapper)
        
        # 2 HEAD + 3 MID + 1 OTHER = 6
        assert wrapped.num_classes == 6
    
    def test_getitem_remaps_gloss_id(self, mock_dataset):
        """Test __getitem__ returns remapped gloss_id."""
        support = compute_class_support(mock_dataset)
        remapper = ClassRemapper()
        remapper.fit(support)
        
        wrapped = RemappedDataset(mock_dataset, remapper)
        
        # Get a sample and verify gloss_id is remapped
        sample = wrapped[0]
        
        # Original was class 0 (HEAD), should be remapped to 0 or 1
        assert sample.gloss_id in [0, 1]
        assert sample.gloss_id != remapper.other_class_id
    
    def test_len_unchanged(self, mock_dataset):
        """Test __len__ returns same as base dataset (for tail_to_other)."""
        support = compute_class_support(mock_dataset)
        remapper = ClassRemapper()
        remapper.fit(support)
        
        wrapped = RemappedDataset(mock_dataset, remapper)
        
        assert len(wrapped) == len(mock_dataset)
    
    def test_tail_samples_map_to_other(self, mock_dataset):
        """Test samples from TAIL classes have gloss_id=OTHER."""
        support = compute_class_support(mock_dataset)
        remapper = ClassRemapper()
        remapper.fit(support)
        
        wrapped = RemappedDataset(mock_dataset, remapper)
        
        # Find a TAIL sample (original class 5-9)
        tail_indices = []
        for i, sample in enumerate(mock_dataset.samples):
            if sample.gloss_id >= 5:  # TAIL classes
                tail_indices.append(i)
        
        # Verify all map to OTHER
        for idx in tail_indices:
            sample = wrapped[idx]
            assert sample.gloss_id == remapper.other_class_id
    
    def test_does_not_modify_base_dataset(self, mock_dataset):
        """Test wrapper does not modify original dataset."""
        support = compute_class_support(mock_dataset)
        remapper = ClassRemapper()
        remapper.fit(support)
        
        original_id = mock_dataset[0].gloss_id
        
        wrapped = RemappedDataset(mock_dataset, remapper)
        _ = wrapped[0]  # Access wrapped sample
        
        # Original should be unchanged
        assert mock_dataset[0].gloss_id == original_id
    
    def test_id_to_gloss_property(self, mock_dataset):
        """Test id_to_gloss returns new class names."""
        support = compute_class_support(mock_dataset)
        remapper = ClassRemapper()
        remapper.fit(support)
        
        wrapped = RemappedDataset(mock_dataset, remapper)
        
        id_to_gloss = wrapped.id_to_gloss
        assert remapper.other_class_id in id_to_gloss
        assert id_to_gloss[remapper.other_class_id] == "OTHER"
    
    def test_tail_exclude_filters_samples(self, mock_dataset):
        """Test tail_exclude strategy filters TAIL samples."""
        support = compute_class_support(mock_dataset)
        config = RemapConfig(strategy="tail_exclude")
        remapper = ClassRemapper(config)
        remapper.fit(support)
        
        wrapped = RemappedDataset(mock_dataset, remapper, filter_excluded=True)
        
        # Should have fewer samples (TAIL filtered out)
        # TAIL has 7 samples total
        expected_len = len(mock_dataset) - 7
        assert len(wrapped) == expected_len
        
        # All samples should have valid (non -1) gloss_id
        for i in range(len(wrapped)):
            sample = wrapped[i]
            assert sample.gloss_id >= 0


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestComputeClassSupport:
    """Tests for compute_class_support helper."""
    
    def test_computes_correct_counts(self, mock_dataset):
        """Test support counts are correct."""
        support = compute_class_support(mock_dataset)
        
        # HEAD
        assert support[0] == 12
        assert support[1] == 11
        # MID
        assert support[2] == 5
        assert support[3] == 4
        assert support[4] == 3
        # TAIL
        assert support[5] == 2
        assert support[6] == 1
        assert support[7] == 2
        assert support[8] == 1
        assert support[9] == 1
    
    def test_handles_empty_dataset(self):
        """Test handles empty dataset."""
        empty_dataset = MockDataset([], {})
        support = compute_class_support(empty_dataset)
        assert support == {}


class TestCreateRemappedDatasets:
    """Tests for create_remapped_datasets convenience function."""
    
    def test_creates_both_datasets(self, mock_dataset):
        """Test creates wrapped train and val datasets."""
        # Use same dataset for simplicity
        train_wrapped, val_wrapped, remapper = create_remapped_datasets(
            mock_dataset,
            mock_dataset
        )
        
        assert isinstance(train_wrapped, RemappedDataset)
        assert isinstance(val_wrapped, RemappedDataset)
        assert isinstance(remapper, ClassRemapper)
        assert remapper._fitted
    
    def test_uses_train_support(self, mock_dataset):
        """Test remapper is fitted on train dataset support."""
        train_wrapped, val_wrapped, remapper = create_remapped_datasets(
            mock_dataset,
            mock_dataset
        )
        
        # Remapper should reflect train dataset distribution
        assert remapper.num_classes_original == 10


# =============================================================================
# Integration Tests
# =============================================================================

class TestRemappingIntegration:
    """Integration tests for remapping pipeline."""
    
    def test_full_pipeline(self, mock_dataset, tmp_path):
        """Test full remapping pipeline."""
        # 1. Compute support
        support = compute_class_support(mock_dataset)
        
        # 2. Create and fit remapper
        config = RemapConfig(strategy="tail_to_other")
        remapper = ClassRemapper(config)
        remapper.fit(support)
        
        # 3. Wrap dataset
        wrapped = RemappedDataset(mock_dataset, remapper)
        
        # 4. Verify all samples have valid new class IDs
        seen_classes = set()
        for i in range(len(wrapped)):
            sample = wrapped[i]
            assert 0 <= sample.gloss_id < remapper.num_classes_remapped
            seen_classes.add(sample.gloss_id)
        
        # 5. Save and reload remapper
        save_path = tmp_path / "mapping.json"
        remapper.save(save_path)
        loaded = ClassRemapper.load(save_path)
        
        # 6. Verify loaded remapper works
        wrapped2 = RemappedDataset(mock_dataset, loaded)
        
        for i in range(len(wrapped)):
            assert wrapped[i].gloss_id == wrapped2[i].gloss_id
    
    def test_dataloader_compatibility(self, mock_dataset):
        """Test wrapped dataset works with DataLoader."""
        from torch.utils.data import DataLoader
        
        support = compute_class_support(mock_dataset)
        remapper = ClassRemapper()
        remapper.fit(support)
        
        wrapped = RemappedDataset(mock_dataset, remapper)
        
        # Simple collate for mock samples
        def collate(samples):
            return {
                'gloss_ids': torch.tensor([s.gloss_id for s in samples]),
                'features': torch.stack([torch.from_numpy(s.features) for s in samples])
            }
        
        loader = DataLoader(wrapped, batch_size=4, collate_fn=collate)
        
        batch = next(iter(loader))
        
        assert batch['gloss_ids'].shape[0] == 4
        assert all(0 <= gid < remapper.num_classes_remapped for gid in batch['gloss_ids'])
