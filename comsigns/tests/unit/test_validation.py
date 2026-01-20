"""
Unit tests for validation and data splitting.

Tests verify:
- Train/val split correctness
- Split reproducibility with seed
- validate_one_epoch behavior
- No gradients during validation
- Trainer integration with validation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader

from comsigns.core.data.splits import (
    create_train_val_split,
    create_train_val_test_split,
    get_split_indices,
)
from comsigns.training.config import TrainerConfig
from comsigns.training.loops import validate_one_epoch
from comsigns.training.trainer import Trainer
from comsigns.training.classifier import SignLanguageClassifier


# =============================================================================
# Mock Components (same as test_trainer.py)
# =============================================================================

class MockEncoder(nn.Module):
    """Simple encoder for testing."""
    
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.output_dim = output_dim
        self.hand_proj = nn.Linear(168, 128)
        self.body_proj = nn.Linear(132, 128)
        self.face_proj = nn.Linear(1872, 128)
        self.fusion = nn.Linear(128 * 3, output_dim)
    
    def forward(self, hand, body, face):
        h = self.hand_proj(hand)
        b = self.body_proj(body)
        f = self.face_proj(face)
        fused = torch.cat([h, b, f], dim=-1)
        return self.fusion(fused)


@dataclass
class MockSample:
    gloss: str
    gloss_id: int
    hand_keypoints: np.ndarray
    body_keypoints: np.ndarray
    face_keypoints: np.ndarray
    
    @property
    def num_frames(self) -> int:
        return self.hand_keypoints.shape[0]


class MockDataset(Dataset):
    """Mock dataset for testing splits."""
    
    def __init__(self, num_samples: int = 100, num_classes: int = 10):
        self.samples = []
        for i in range(num_samples):
            T = np.random.randint(10, 30)
            self.samples.append(MockSample(
                gloss=f"gloss_{i % num_classes}",
                gloss_id=i % num_classes,
                hand_keypoints=np.random.randn(T, 168).astype(np.float32),
                body_keypoints=np.random.randn(T, 132).astype(np.float32),
                face_keypoints=np.random.randn(T, 1872).astype(np.float32),
            ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def mock_collate_fn(batch):
    """Simple collate function."""
    hand_list = [s.hand_keypoints for s in batch]
    body_list = [s.body_keypoints for s in batch]
    face_list = [s.face_keypoints for s in batch]
    labels = [s.gloss_id for s in batch]
    lengths = [s.num_frames for s in batch]
    
    max_len = max(lengths)
    B = len(batch)
    
    def pad_sequence(arrays, max_len):
        D = arrays[0].shape[1]
        padded = np.zeros((B, max_len, D), dtype=np.float32)
        for i, arr in enumerate(arrays):
            padded[i, :len(arr)] = arr
        return torch.from_numpy(padded)
    
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return {
        "hand": pad_sequence(hand_list, max_len),
        "body": pad_sequence(body_list, max_len),
        "face": pad_sequence(face_list, max_len),
        "labels": torch.tensor(labels, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "mask": mask,
    }


# =============================================================================
# Test Data Splits
# =============================================================================

class TestCreateTrainValSplit:
    """Tests for create_train_val_split function."""
    
    def test_split_sizes_correct(self):
        """Test that split respects the ratio."""
        dataset = MockDataset(num_samples=100)
        train_set, val_set = create_train_val_split(dataset, val_ratio=0.2)
        
        assert len(train_set) == 80
        assert len(val_set) == 20
        assert len(train_set) + len(val_set) == len(dataset)
    
    def test_split_reproducibility(self):
        """Test that same seed produces same split."""
        dataset = MockDataset(num_samples=100)
        
        train1, val1 = create_train_val_split(dataset, val_ratio=0.2, seed=42)
        train2, val2 = create_train_val_split(dataset, val_ratio=0.2, seed=42)
        
        # Same indices
        assert train1.indices == train2.indices
        assert val1.indices == val2.indices
    
    def test_different_seed_different_split(self):
        """Test that different seeds produce different splits."""
        dataset = MockDataset(num_samples=100)
        
        train1, val1 = create_train_val_split(dataset, val_ratio=0.2, seed=42)
        train2, val2 = create_train_val_split(dataset, val_ratio=0.2, seed=123)
        
        # Different indices (very unlikely to be the same)
        assert train1.indices != train2.indices
    
    def test_invalid_ratio_raises(self):
        """Test that invalid ratios raise errors."""
        dataset = MockDataset(num_samples=100)
        
        with pytest.raises(ValueError):
            create_train_val_split(dataset, val_ratio=0.0)
        
        with pytest.raises(ValueError):
            create_train_val_split(dataset, val_ratio=1.0)
        
        with pytest.raises(ValueError):
            create_train_val_split(dataset, val_ratio=-0.1)
    
    def test_subsets_are_disjoint(self):
        """Test that train and val have no overlap."""
        dataset = MockDataset(num_samples=100)
        train_set, val_set = create_train_val_split(dataset, val_ratio=0.2)
        
        train_indices = set(train_set.indices)
        val_indices = set(val_set.indices)
        
        assert len(train_indices & val_indices) == 0


class TestCreateTrainValTestSplit:
    """Tests for 3-way split."""
    
    def test_three_way_split_sizes(self):
        """Test that 3-way split respects ratios."""
        dataset = MockDataset(num_samples=100)
        train, val, test = create_train_val_test_split(
            dataset, val_ratio=0.1, test_ratio=0.1
        )
        
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
    
    def test_invalid_combined_ratio_raises(self):
        """Test that val_ratio + test_ratio >= 1 raises error."""
        dataset = MockDataset(num_samples=100)
        
        with pytest.raises(ValueError):
            create_train_val_test_split(dataset, val_ratio=0.5, test_ratio=0.5)


class TestGetSplitIndices:
    """Tests for get_split_indices function."""
    
    def test_indices_correct_size(self):
        """Test that indices have correct sizes."""
        train_idx, val_idx = get_split_indices(100, val_ratio=0.2, seed=42)
        
        assert len(train_idx) == 80
        assert len(val_idx) == 20
    
    def test_indices_no_overlap(self):
        """Test that train and val indices don't overlap."""
        train_idx, val_idx = get_split_indices(100, val_ratio=0.2, seed=42)
        
        assert len(set(train_idx) & set(val_idx)) == 0
    
    def test_indices_cover_all(self):
        """Test that all indices are covered."""
        train_idx, val_idx = get_split_indices(100, val_ratio=0.2, seed=42)
        
        all_indices = set(train_idx) | set(val_idx)
        assert all_indices == set(range(100))


# =============================================================================
# Test Validation Loop
# =============================================================================

class TestValidateOneEpoch:
    """Tests for validate_one_epoch function."""
    
    @pytest.fixture
    def setup(self):
        """Create model and dataloader for testing."""
        encoder = MockEncoder(output_dim=512)
        model = SignLanguageClassifier(encoder, num_classes=5)
        
        dataset = MockDataset(num_samples=20, num_classes=5)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=mock_collate_fn
        )
        
        return model, loader
    
    def test_returns_loss(self, setup):
        """Test that validation returns a loss value."""
        model, loader = setup
        device = torch.device("cpu")
        loss_fn = nn.CrossEntropyLoss()
        
        metrics = validate_one_epoch(model, loader, loss_fn, device)
        
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] > 0
    
    def test_no_gradients(self, setup):
        """Test that validation doesn't compute gradients."""
        model, loader = setup
        device = torch.device("cpu")
        loss_fn = nn.CrossEntropyLoss()
        
        # Run validation
        validate_one_epoch(model, loader, loss_fn, device)
        
        # Check no gradients were computed
        for param in model.parameters():
            assert param.grad is None
    
    def test_model_in_eval_mode(self, setup):
        """Test that model is in eval mode during validation."""
        model, loader = setup
        device = torch.device("cpu")
        loss_fn = nn.CrossEntropyLoss()
        
        # Start in train mode
        model.train()
        assert model.training
        
        # After validation, model should be in eval mode
        validate_one_epoch(model, loader, loss_fn, device)
        assert not model.training
    
    def test_deterministic_output(self, setup):
        """Test that validation is deterministic."""
        model, loader = setup
        device = torch.device("cpu")
        loss_fn = nn.CrossEntropyLoss()
        
        # Run twice
        metrics1 = validate_one_epoch(model, loader, loss_fn, device)
        metrics2 = validate_one_epoch(model, loader, loss_fn, device)
        
        # Should be identical (no dropout in eval mode)
        assert metrics1["loss"] == metrics2["loss"]


# =============================================================================
# Test Trainer with Validation
# =============================================================================

class TestTrainerWithValidation:
    """Tests for Trainer with validation loop."""
    
    @pytest.fixture
    def full_setup(self):
        """Create complete training setup with validation."""
        encoder = MockEncoder(output_dim=512)
        model = SignLanguageClassifier(encoder, num_classes=5)
        
        dataset = MockDataset(num_samples=50, num_classes=5)
        train_set, val_set = create_train_val_split(dataset, val_ratio=0.2)
        
        train_loader = DataLoader(
            train_set,
            batch_size=4,
            shuffle=True,
            collate_fn=mock_collate_fn
        )
        
        val_loader = DataLoader(
            val_set,
            batch_size=4,
            shuffle=False,
            collate_fn=mock_collate_fn
        )
        
        config = TrainerConfig(
            epochs=3,
            learning_rate=1e-3,
            device="cpu",
            log_every_n_steps=100,
            validate=True,
            seed=42
        )
        
        return model, train_loader, val_loader, config
    
    def test_trainer_logs_both_losses(self, full_setup):
        """Test that trainer returns both train and val losses."""
        model, train_loader, val_loader, config = full_setup
        trainer = Trainer(model, config)
        
        history = trainer.fit(train_loader, val_loader=val_loader)
        
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == config.epochs
        assert len(history["val_loss"]) == config.epochs
    
    def test_trainer_without_val_loader(self, full_setup):
        """Test that trainer works without validation loader."""
        model, train_loader, _, config = full_setup
        trainer = Trainer(model, config)
        
        history = trainer.fit(train_loader)  # No val_loader
        
        assert "train_loss" in history
        assert len(history["train_loss"]) == config.epochs
        assert len(history["val_loss"]) == 0  # Empty, no validation
    
    def test_validate_false_skips_validation(self, full_setup):
        """Test that validate=False skips validation even with val_loader."""
        model, train_loader, val_loader, config = full_setup
        config.validate = False
        
        trainer = Trainer(model, config)
        history = trainer.fit(train_loader, val_loader=val_loader)
        
        # val_loss should be empty since validate=False
        assert len(history["val_loss"]) == 0
    
    def test_overfit_mode_disables_validation(self):
        """Test that overfit mode automatically disables validation."""
        config = TrainerConfig(
            overfit_single_batch=True,
            validate=True  # Should be overridden
        )
        
        # Config should auto-disable validation in overfit mode
        assert config.validate is False


# =============================================================================
# Integration Test
# =============================================================================

class TestValidationIntegration:
    """Integration tests for the validation pipeline."""
    
    def test_end_to_end_with_validation(self):
        """Test complete training pipeline with validation."""
        # Setup
        encoder = MockEncoder(output_dim=512)
        model = SignLanguageClassifier(encoder, num_classes=5)
        
        dataset = MockDataset(num_samples=100, num_classes=5)
        train_set, val_set = create_train_val_split(dataset, val_ratio=0.2, seed=42)
        
        train_loader = DataLoader(
            train_set, batch_size=8, shuffle=True, collate_fn=mock_collate_fn
        )
        val_loader = DataLoader(
            val_set, batch_size=8, shuffle=False, collate_fn=mock_collate_fn
        )
        
        config = TrainerConfig(
            epochs=5,
            learning_rate=1e-3,
            device="cpu",
            log_every_n_steps=100,
            validate=True,
            seed=42
        )
        
        # Train
        trainer = Trainer(model, config)
        history = trainer.fit(train_loader, val_loader=val_loader)
        
        # Verify
        assert len(history["train_loss"]) == 5
        assert len(history["val_loss"]) == 5
        assert all(isinstance(l, float) for l in history["train_loss"])
        assert all(isinstance(l, float) for l in history["val_loss"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
