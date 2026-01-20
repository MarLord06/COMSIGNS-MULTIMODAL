"""
Unit tests for the training module.

Tests verify:
- TrainerConfig validation and defaults
- SignLanguageClassifier forward pass
- Masked pooling correctness
- Training loop execution
- Gradient flow
- Overfit mode functionality
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader

from comsigns.training.config import TrainerConfig
from comsigns.training.classifier import SignLanguageClassifier
from comsigns.training.trainer import Trainer
from comsigns.training.loops import train_one_epoch, validate_gradients


# =============================================================================
# Mock Components
# =============================================================================

class MockEncoder(nn.Module):
    """Simple encoder for testing that mimics MultimodalEncoder interface."""
    
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.output_dim = output_dim
        # Simple linear layers to make it trainable
        self.hand_proj = nn.Linear(168, 128)
        self.body_proj = nn.Linear(132, 128)
        self.face_proj = nn.Linear(1872, 128)
        self.fusion = nn.Linear(128 * 3, output_dim)
    
    def forward(self, hand, body, face):
        B, T, _ = hand.shape
        h = self.hand_proj(hand)
        b = self.body_proj(body)
        f = self.face_proj(face)
        fused = torch.cat([h, b, f], dim=-1)
        return self.fusion(fused)


@dataclass
class MockSample:
    """Mock sample mimicking EncoderReadySample."""
    gloss: str
    gloss_id: int
    hand_keypoints: np.ndarray
    body_keypoints: np.ndarray
    face_keypoints: np.ndarray
    
    @property
    def num_frames(self) -> int:
        return self.hand_keypoints.shape[0]


class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, num_samples: int = 20, num_classes: int = 5):
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
    """Simple collate that mimics encoder_collate_fn."""
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
# Test TrainerConfig
# =============================================================================

class TestTrainerConfig:
    """Tests for TrainerConfig."""
    
    def test_default_values(self):
        """Test that defaults are sensible."""
        config = TrainerConfig()
        assert config.batch_size == 16
        assert config.learning_rate == 1e-4
        assert config.epochs == 10
        assert config.log_every_n_steps == 10
        assert config.overfit_single_batch is False
    
    def test_auto_device_detection(self):
        """Test device auto-detection."""
        config = TrainerConfig(device="auto")
        # Should resolve to cpu, cuda, or mps
        assert config.device in ["cpu", "cuda", "mps"]
    
    def test_invalid_batch_size_raises(self):
        """Test validation catches invalid batch_size."""
        with pytest.raises(ValueError, match="batch_size"):
            TrainerConfig(batch_size=0)
    
    def test_invalid_learning_rate_raises(self):
        """Test validation catches invalid learning_rate."""
        with pytest.raises(ValueError, match="learning_rate"):
            TrainerConfig(learning_rate=-0.001)
    
    def test_invalid_epochs_raises(self):
        """Test validation catches invalid epochs."""
        with pytest.raises(ValueError, match="epochs"):
            TrainerConfig(epochs=0)
    
    def test_get_torch_device(self):
        """Test torch device conversion."""
        config = TrainerConfig(device="cpu")
        device = config.get_torch_device()
        assert isinstance(device, torch.device)
        assert device.type == "cpu"


# =============================================================================
# Test SignLanguageClassifier
# =============================================================================

class TestSignLanguageClassifier:
    """Tests for SignLanguageClassifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create a test classifier."""
        encoder = MockEncoder(output_dim=512)
        return SignLanguageClassifier(encoder, num_classes=10)
    
    def test_forward_shape(self, classifier):
        """Test output shape is correct."""
        B, T = 4, 20
        hand = torch.randn(B, T, 168)
        body = torch.randn(B, T, 132)
        face = torch.randn(B, T, 1872)
        lengths = torch.tensor([20, 15, 18, 10])
        
        logits = classifier(hand, body, face, lengths=lengths)
        
        assert logits.shape == (B, 10)  # [batch, num_classes]
    
    def test_forward_without_lengths(self, classifier):
        """Test forward works without lengths (uses simple mean)."""
        B, T = 4, 20
        hand = torch.randn(B, T, 168)
        body = torch.randn(B, T, 132)
        face = torch.randn(B, T, 1872)
        
        logits = classifier(hand, body, face)
        
        assert logits.shape == (B, 10)
    
    def test_masked_mean_pooling(self, classifier):
        """Test that masked mean pooling ignores padding."""
        B, T = 2, 10
        # Create embeddings where valid positions are 1.0, padding is 0.0
        embeddings = torch.zeros(B, T, 512)
        
        # Sample 0: length 5, fill with 1s
        embeddings[0, :5, :] = 1.0
        # Sample 1: length 8, fill with 2s
        embeddings[1, :8, :] = 2.0
        
        lengths = torch.tensor([5, 8])
        
        pooled = classifier._masked_mean_pool(embeddings, lengths, mask=None)
        
        # Sample 0: mean of 5 ones = 1.0
        assert torch.allclose(pooled[0], torch.ones(512), atol=1e-5)
        # Sample 1: mean of 8 twos = 2.0
        assert torch.allclose(pooled[1], torch.ones(512) * 2, atol=1e-5)
    
    def test_max_pooling(self, classifier):
        """Test max pooling."""
        classifier.pooling = "max"
        
        B, T = 4, 20
        hand = torch.randn(B, T, 168)
        body = torch.randn(B, T, 132)
        face = torch.randn(B, T, 1872)
        
        logits = classifier(hand, body, face)
        
        assert logits.shape == (B, 10)
    
    def test_last_pooling(self, classifier):
        """Test last-timestep pooling."""
        classifier.pooling = "last"
        
        B, T = 4, 20
        hand = torch.randn(B, T, 168)
        body = torch.randn(B, T, 132)
        face = torch.randn(B, T, 1872)
        lengths = torch.tensor([20, 15, 18, 10])
        
        logits = classifier(hand, body, face, lengths=lengths)
        
        assert logits.shape == (B, 10)
    
    def test_get_embeddings(self, classifier):
        """Test embedding extraction without classification."""
        B, T = 4, 20
        hand = torch.randn(B, T, 168)
        body = torch.randn(B, T, 132)
        face = torch.randn(B, T, 1872)
        
        embeddings = classifier.get_embeddings(hand, body, face)
        
        assert embeddings.shape == (B, 512)  # [batch, encoder_dim]
    
    def test_invalid_num_classes_raises(self):
        """Test validation catches invalid num_classes."""
        encoder = MockEncoder()
        with pytest.raises(ValueError, match="num_classes"):
            SignLanguageClassifier(encoder, num_classes=0)


# =============================================================================
# Test Training Loops
# =============================================================================

class TestTrainingLoops:
    """Tests for training loop functions."""
    
    @pytest.fixture
    def training_setup(self):
        """Create complete training setup."""
        encoder = MockEncoder(output_dim=512)
        model = SignLanguageClassifier(encoder, num_classes=5)
        
        dataset = MockDataset(num_samples=20, num_classes=5)
        loader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            collate_fn=mock_collate_fn
        )
        
        config = TrainerConfig(
            epochs=2,
            learning_rate=1e-3,
            device="cpu",
            log_every_n_steps=5,
            overfit_single_batch=False
        )
        
        return model, loader, config
    
    def test_train_one_epoch_runs(self, training_setup):
        """Test that one epoch completes without error."""
        model, loader, config = training_setup
        device = config.get_torch_device()
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        metrics = train_one_epoch(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=1,
            config=config
        )
        
        assert "loss" in metrics
        assert "num_steps" in metrics
        assert metrics["num_steps"] > 0
        assert isinstance(metrics["loss"], float)
    
    def test_validate_gradients(self, training_setup):
        """Test gradient validation function."""
        model, loader, config = training_setup
        device = config.get_torch_device()
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        # Do one forward-backward
        batch = next(iter(loader))
        hand = batch["hand"].to(device)
        body = batch["body"].to(device)
        face = batch["face"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        logits = model(hand, body, face)
        loss = loss_fn(logits, labels)
        loss.backward()
        
        # Validate gradients
        grad_info = validate_gradients(model)
        
        assert grad_info["has_gradients"] is True
        assert grad_info["non_zero_params"] > 0
        assert grad_info["total_params"] > 0


# =============================================================================
# Test Trainer Class
# =============================================================================

class TestTrainer:
    """Tests for the Trainer class."""
    
    @pytest.fixture
    def trainer_setup(self):
        """Create Trainer with mock components."""
        encoder = MockEncoder(output_dim=512)
        model = SignLanguageClassifier(encoder, num_classes=5)
        
        dataset = MockDataset(num_samples=20, num_classes=5)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=mock_collate_fn
        )
        
        config = TrainerConfig(
            epochs=2,
            learning_rate=1e-3,
            device="cpu",
            log_every_n_steps=100,  # Reduce logging noise
            seed=42
        )
        
        return model, loader, config
    
    def test_trainer_init(self, trainer_setup):
        """Test Trainer initialization."""
        model, loader, config = trainer_setup
        trainer = Trainer(model, config)
        
        assert trainer.model is not None
        assert trainer.config is config
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
    
    def test_trainer_fit(self, trainer_setup):
        """Test full training run."""
        model, loader, config = trainer_setup
        trainer = Trainer(model, config)
        
        history = trainer.fit(loader)
        
        assert "train_loss" in history
        assert "epoch" in history
        assert len(history["train_loss"]) == config.epochs
        assert trainer.current_epoch == config.epochs
    
    def test_trainer_single_step(self, trainer_setup):
        """Test single training step."""
        model, loader, config = trainer_setup
        trainer = Trainer(model, config)
        
        batch = next(iter(loader))
        metrics = trainer.train_step(batch)
        
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert trainer.global_step == 1
    
    def test_trainer_validation(self, trainer_setup):
        """Test training validation helper."""
        model, loader, config = trainer_setup
        trainer = Trainer(model, config)
        
        batch = next(iter(loader))
        results = trainer.validate_training(batch)
        
        assert "loss" in results
        assert "has_gradients" in results
        assert results["has_gradients"] is True
        assert results["non_zero_params"] > 0
    
    def test_overfit_mode_loss_decreases(self, trainer_setup):
        """Test that loss decreases in overfit mode."""
        model, loader, config = trainer_setup
        config.overfit_single_batch = True
        config.epochs = 50  # More epochs to see convergence
        config.learning_rate = 1e-2  # Higher LR for faster convergence
        
        trainer = Trainer(model, config)
        history = trainer.fit(loader)
        
        # Loss should decrease significantly
        initial_loss = history["train_loss"][0]
        final_loss = history["train_loss"][-1]
        
        assert final_loss < initial_loss, (
            f"Loss did not decrease in overfit mode: "
            f"{initial_loss:.4f} -> {final_loss:.4f}"
        )
    
    def test_state_dict(self, trainer_setup):
        """Test state dict for checkpointing."""
        model, loader, config = trainer_setup
        trainer = Trainer(model, config)
        
        # Do some training
        trainer.fit(loader, epochs=1)
        
        state = trainer.state_dict()
        
        assert "model_state_dict" in state
        assert "optimizer_state_dict" in state
        assert "epoch" in state
        assert "global_step" in state
        assert "history" in state


# =============================================================================
# Integration Test
# =============================================================================

class TestIntegration:
    """Integration tests for the complete training pipeline."""
    
    def test_end_to_end_training(self):
        """Test complete training pipeline."""
        # Setup
        encoder = MockEncoder(output_dim=512)
        model = SignLanguageClassifier(encoder, num_classes=5)
        
        dataset = MockDataset(num_samples=20, num_classes=5)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=mock_collate_fn
        )
        
        config = TrainerConfig(
            epochs=3,
            learning_rate=1e-3,
            device="cpu",
            log_every_n_steps=100,
            seed=42
        )
        
        # Train
        trainer = Trainer(model, config)
        history = trainer.fit(loader)
        
        # Verify
        assert len(history["train_loss"]) == 3
        assert all(isinstance(l, float) for l in history["train_loss"])
        assert trainer.global_step > 0
        
        # Model should be trainable
        batch = next(iter(loader))
        with torch.no_grad():
            logits = model(
                batch["hand"],
                batch["body"],
                batch["face"]
            )
        assert logits.shape == (4, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
