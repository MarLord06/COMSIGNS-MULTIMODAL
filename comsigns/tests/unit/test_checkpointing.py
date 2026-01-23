"""
Unit tests for checkpointing module.

These tests validate:
- CheckpointMetrics dataclass
- CheckpointManager checkpoint saving/loading
- Best model selection criteria
- Training resumption
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from comsigns.training.checkpointing import (
    CheckpointMetrics,
    BestModelInfo,
    CheckpointManager,
    load_checkpoint_for_inference,
    load_checkpoint_for_training,
    PRIMARY_METRIC,
    SECONDARY_METRIC,
)


# =============================================================================
# Fixtures
# =============================================================================

class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel(input_dim=10, output_dim=5)


@pytest.fixture
def optimizer(simple_model):
    """Create an optimizer for testing."""
    return optim.Adam(simple_model.parameters(), lr=0.001)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def checkpoint_manager(temp_dir):
    """Create a CheckpointManager instance."""
    return CheckpointManager(output_dir=temp_dir)


# =============================================================================
# CheckpointMetrics Tests
# =============================================================================

class TestCheckpointMetrics:
    """Tests for CheckpointMetrics dataclass."""
    
    def test_creation(self):
        """Basic creation with required fields."""
        metrics = CheckpointMetrics(
            epoch=5,
            val_loss=0.5,
            f1_macro=0.65,
            learned_words_count=10
        )
        assert metrics.epoch == 5
        assert metrics.val_loss == 0.5
        assert metrics.f1_macro == 0.65
        assert metrics.learned_words_count == 10
        assert metrics.accuracy is None
    
    def test_creation_with_optional_fields(self):
        """Creation with optional fields."""
        metrics = CheckpointMetrics(
            epoch=10,
            val_loss=0.3,
            f1_macro=0.75,
            learned_words_count=15,
            accuracy=0.82,
            accuracy_top5=0.95
        )
        assert metrics.accuracy == 0.82
        assert metrics.accuracy_top5 == 0.95
    
    def test_to_dict(self):
        """Conversion to dictionary."""
        metrics = CheckpointMetrics(
            epoch=5,
            val_loss=0.5,
            f1_macro=0.65,
            learned_words_count=10
        )
        d = metrics.to_dict()
        assert d["epoch"] == 5
        assert d["val_loss"] == 0.5
        # Optional None values should be excluded
        assert "accuracy" not in d or d["accuracy"] is None
    
    def test_from_dict(self):
        """Creation from dictionary."""
        data = {
            "epoch": 7,
            "val_loss": 0.4,
            "f1_macro": 0.7,
            "learned_words_count": 12
        }
        metrics = CheckpointMetrics.from_dict(data)
        assert metrics.epoch == 7
        assert metrics.learned_words_count == 12


# =============================================================================
# BestModelInfo Tests
# =============================================================================

class TestBestModelInfo:
    """Tests for BestModelInfo dataclass."""
    
    def test_creation(self):
        """Basic creation."""
        info = BestModelInfo(
            best_epoch=37,
            selection_criteria=["learned_words_count", "f1_macro", "val_loss"],
            metrics={"learned_words_count": 15, "f1_macro": 0.165, "val_loss": 4.02}
        )
        assert info.best_epoch == 37
        assert len(info.selection_criteria) == 3
    
    def test_save_and_load(self, temp_dir):
        """Save and load from JSON."""
        info = BestModelInfo(
            best_epoch=10,
            selection_criteria=["learned_words_count", "f1_macro"],
            metrics={"learned_words_count": 8, "f1_macro": 0.5}
        )
        
        path = temp_dir / "best_model.json"
        info.save(path)
        
        loaded = BestModelInfo.load(path)
        assert loaded.best_epoch == 10
        assert loaded.metrics["learned_words_count"] == 8


# =============================================================================
# CheckpointManager - Basic Operations
# =============================================================================

class TestCheckpointManagerBasic:
    """Basic CheckpointManager tests."""
    
    def test_initialization(self, temp_dir):
        """Manager creates directories on init."""
        manager = CheckpointManager(output_dir=temp_dir)
        assert manager.checkpoints_dir.exists()
        assert manager.primary_metric == PRIMARY_METRIC
    
    def test_has_checkpoints_empty(self, checkpoint_manager):
        """has_checkpoints returns False when empty."""
        assert checkpoint_manager.has_checkpoints() is False
    
    def test_has_best_empty(self, checkpoint_manager):
        """has_best returns False when no best saved."""
        assert checkpoint_manager.has_best() is False
    
    def test_get_latest_epoch_empty(self, checkpoint_manager):
        """get_latest_epoch returns None when empty."""
        assert checkpoint_manager.get_latest_epoch() is None


# =============================================================================
# CheckpointManager - Saving Tests
# =============================================================================

class TestCheckpointManagerSaving:
    """Tests for checkpoint saving."""
    
    def test_save_checkpoint(self, checkpoint_manager, simple_model, optimizer):
        """Save a single checkpoint."""
        metrics = {"val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        
        path = checkpoint_manager.save_checkpoint(
            epoch=0,
            model=simple_model,
            optimizer=optimizer,
            metrics=metrics
        )
        
        assert path.exists()
        assert path.name == "epoch_000.pt"
        assert checkpoint_manager.has_checkpoints()
    
    def test_save_multiple_checkpoints(self, checkpoint_manager, simple_model, optimizer):
        """Save multiple checkpoints."""
        for epoch in range(5):
            metrics = {"val_loss": 0.5 - epoch * 0.05, "f1_macro": 0.5 + epoch * 0.05, "learned_words_count": epoch}
            checkpoint_manager.save_checkpoint(epoch, simple_model, optimizer, metrics)
        
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 5
        assert checkpoint_manager.get_latest_epoch() == 4
    
    def test_save_checkpoint_with_scheduler(self, checkpoint_manager, simple_model, optimizer):
        """Save checkpoint including scheduler state."""
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        metrics = {"val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        
        path = checkpoint_manager.save_checkpoint(
            epoch=0,
            model=simple_model,
            optimizer=optimizer,
            metrics=metrics,
            scheduler=scheduler
        )
        
        # Verify scheduler state is saved
        checkpoint = torch.load(path)
        assert "scheduler_state" in checkpoint
    
    def test_save_checkpoint_with_extra_state(self, checkpoint_manager, simple_model, optimizer):
        """Save checkpoint with extra custom state."""
        metrics = {"val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        extra = {"num_classes": 51, "tail_to_other": True}
        
        path = checkpoint_manager.save_checkpoint(
            epoch=0,
            model=simple_model,
            optimizer=optimizer,
            metrics=metrics,
            extra_state=extra
        )
        
        checkpoint = torch.load(path)
        assert checkpoint["num_classes"] == 51
        assert checkpoint["tail_to_other"] is True


# =============================================================================
# CheckpointManager - Best Model Selection
# =============================================================================

class TestBestModelSelection:
    """Tests for best model selection criteria."""
    
    def test_first_model_is_always_best(self, checkpoint_manager):
        """First model should always be considered best."""
        metrics = {"val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        assert checkpoint_manager.is_best(metrics) is True
    
    def test_higher_learned_words_is_best(self, checkpoint_manager, simple_model):
        """Higher learned_words_count should be considered best."""
        # First model
        metrics1 = {"epoch": 0, "val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        checkpoint_manager.save_best(simple_model, metrics1)
        
        # Higher learned_words_count should be best
        metrics2 = {"epoch": 1, "val_loss": 0.6, "f1_macro": 0.5, "learned_words_count": 7}
        assert checkpoint_manager.is_best(metrics2) is True
        
        # Lower learned_words_count should not be best
        metrics3 = {"epoch": 2, "val_loss": 0.3, "f1_macro": 0.8, "learned_words_count": 4}
        assert checkpoint_manager.is_best(metrics3) is False
    
    def test_f1_macro_tiebreaker(self, checkpoint_manager, simple_model):
        """When learned_words equal, f1_macro should break tie."""
        metrics1 = {"epoch": 0, "val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        checkpoint_manager.save_best(simple_model, metrics1)
        
        # Same learned_words, higher f1_macro should be best
        metrics2 = {"epoch": 1, "val_loss": 0.5, "f1_macro": 0.7, "learned_words_count": 5}
        assert checkpoint_manager.is_best(metrics2) is True
        
        # Same learned_words, lower f1_macro should not be best
        metrics3 = {"epoch": 2, "val_loss": 0.3, "f1_macro": 0.5, "learned_words_count": 5}
        assert checkpoint_manager.is_best(metrics3) is False
    
    def test_val_loss_final_tiebreaker(self, checkpoint_manager, simple_model):
        """When learned_words and f1_macro equal, val_loss should break tie."""
        metrics1 = {"epoch": 0, "val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        checkpoint_manager.save_best(simple_model, metrics1)
        
        # Same learned_words and f1_macro, lower val_loss should be best
        metrics2 = {"epoch": 1, "val_loss": 0.4, "f1_macro": 0.6, "learned_words_count": 5}
        assert checkpoint_manager.is_best(metrics2) is True
        
        # Same learned_words and f1_macro, higher val_loss should not be best
        checkpoint_manager.save_best(simple_model, metrics2)
        metrics3 = {"epoch": 2, "val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        assert checkpoint_manager.is_best(metrics3) is False
    
    def test_save_best_creates_files(self, checkpoint_manager, simple_model):
        """save_best should create best.pt and best_model.json."""
        metrics = {"epoch": 5, "val_loss": 0.3, "f1_macro": 0.7, "learned_words_count": 10}
        
        path = checkpoint_manager.save_best(simple_model, metrics)
        
        assert path.exists()
        assert path.name == "best.pt"
        assert (checkpoint_manager.output_dir / "best_model.json").exists()
        assert checkpoint_manager.has_best()
    
    def test_update_best_if_needed(self, checkpoint_manager, simple_model):
        """update_best_if_needed convenience method."""
        # First should be best
        metrics1 = {"epoch": 0, "val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        was_best = checkpoint_manager.update_best_if_needed(simple_model, metrics1)
        assert was_best is True
        
        # Better should update
        metrics2 = {"epoch": 1, "val_loss": 0.4, "f1_macro": 0.7, "learned_words_count": 8}
        was_best = checkpoint_manager.update_best_if_needed(simple_model, metrics2)
        assert was_best is True
        
        # Worse should not update
        metrics3 = {"epoch": 2, "val_loss": 0.6, "f1_macro": 0.5, "learned_words_count": 3}
        was_best = checkpoint_manager.update_best_if_needed(simple_model, metrics3)
        assert was_best is False


# =============================================================================
# CheckpointManager - Loading Tests
# =============================================================================

class TestCheckpointManagerLoading:
    """Tests for checkpoint loading."""
    
    def test_load_checkpoint_by_epoch(self, checkpoint_manager, simple_model, optimizer):
        """Load checkpoint by epoch number."""
        metrics = {"val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        checkpoint_manager.save_checkpoint(3, simple_model, optimizer, metrics)
        
        loaded = checkpoint_manager.load_checkpoint(epoch=3)
        assert loaded["epoch"] == 3
        assert "model_state" in loaded
        assert "optimizer_state" in loaded
    
    def test_load_checkpoint_by_path(self, checkpoint_manager, simple_model, optimizer):
        """Load checkpoint by direct path."""
        metrics = {"val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        path = checkpoint_manager.save_checkpoint(0, simple_model, optimizer, metrics)
        
        loaded = checkpoint_manager.load_checkpoint(path=path)
        assert loaded["epoch"] == 0
    
    def test_load_best(self, checkpoint_manager, simple_model):
        """Load the best checkpoint."""
        metrics = {"epoch": 10, "val_loss": 0.3, "f1_macro": 0.7, "learned_words_count": 15}
        checkpoint_manager.save_best(simple_model, metrics)
        
        loaded = checkpoint_manager.load_best()
        assert loaded["epoch"] == 10
        assert loaded["metrics"]["learned_words_count"] == 15
    
    def test_load_latest(self, checkpoint_manager, simple_model, optimizer):
        """Load the most recent checkpoint."""
        for epoch in range(5):
            metrics = {"val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": epoch}
            checkpoint_manager.save_checkpoint(epoch, simple_model, optimizer, metrics)
        
        loaded = checkpoint_manager.load_latest()
        assert loaded["epoch"] == 4
    
    def test_load_nonexistent_raises(self, checkpoint_manager):
        """Loading nonexistent checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint(epoch=999)
        
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_best()


# =============================================================================
# CheckpointManager - Cleanup Tests
# =============================================================================

class TestCheckpointCleanup:
    """Tests for checkpoint cleanup (keep_last_n)."""
    
    def test_keep_all_by_default(self, temp_dir, simple_model, optimizer):
        """By default, all checkpoints are kept."""
        manager = CheckpointManager(output_dir=temp_dir, keep_last_n=0)
        
        for epoch in range(10):
            metrics = {"val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": epoch}
            manager.save_checkpoint(epoch, simple_model, optimizer, metrics)
        
        assert len(manager.list_checkpoints()) == 10
    
    def test_keep_last_n(self, temp_dir, simple_model, optimizer):
        """Only keep last N checkpoints."""
        manager = CheckpointManager(output_dir=temp_dir, keep_last_n=3)
        
        for epoch in range(10):
            metrics = {"val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": epoch}
            manager.save_checkpoint(epoch, simple_model, optimizer, metrics)
        
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3
        # Should have latest 3
        epochs = [int(c.stem.split("_")[1]) for c in checkpoints]
        assert epochs == [7, 8, 9]


# =============================================================================
# Resume Training Tests
# =============================================================================

class TestResumeTraining:
    """Tests for training resumption."""
    
    def test_resume_from_checkpoint(self, checkpoint_manager, simple_model, optimizer):
        """Resume training from a checkpoint."""
        # Save initial checkpoint
        metrics = {"val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        checkpoint_manager.save_checkpoint(5, simple_model, optimizer, metrics)
        
        # Create new model and optimizer
        new_model = SimpleModel(input_dim=10, output_dim=5)
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        
        # Load checkpoint
        checkpoint = checkpoint_manager.load_checkpoint(epoch=5)
        new_model.load_state_dict(checkpoint["model_state"])
        new_optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        
        assert start_epoch == 6
        
        # Verify model weights match
        for p1, p2 in zip(simple_model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_load_checkpoint_for_inference_helper(self, checkpoint_manager, simple_model):
        """Test load_checkpoint_for_inference helper function."""
        metrics = {"epoch": 0, "val_loss": 0.3, "f1_macro": 0.7, "learned_words_count": 10}
        checkpoint_manager.save_best(simple_model, metrics)
        
        new_model = SimpleModel(input_dim=10, output_dim=5)
        best_path = checkpoint_manager.checkpoints_dir / "best.pt"
        
        loaded_model = load_checkpoint_for_inference(best_path, new_model)
        
        assert not loaded_model.training  # Should be in eval mode
    
    def test_load_checkpoint_for_training_helper(self, checkpoint_manager, simple_model, optimizer):
        """Test load_checkpoint_for_training helper function."""
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        metrics = {"val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5}
        path = checkpoint_manager.save_checkpoint(
            10, simple_model, optimizer, metrics, scheduler=scheduler
        )
        
        new_model = SimpleModel(input_dim=10, output_dim=5)
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=1)
        
        start_epoch = load_checkpoint_for_training(
            path, new_model, new_optimizer, new_scheduler
        )
        
        assert start_epoch == 11


# =============================================================================
# Integration Tests
# =============================================================================

class TestCheckpointingIntegration:
    """Integration tests simulating real training loop."""
    
    def test_full_training_simulation(self, temp_dir):
        """Simulate a full training loop with checkpointing."""
        model = SimpleModel(input_dim=10, output_dim=5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        manager = CheckpointManager(output_dir=temp_dir)
        
        # Simulate training
        best_epoch = None
        for epoch in range(10):
            # Simulate metrics that improve then plateau
            learned_words = min(epoch + 1, 7)  # Caps at 7
            f1_macro = 0.3 + epoch * 0.05
            val_loss = 0.8 - epoch * 0.05
            
            metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "f1_macro": f1_macro,
                "learned_words_count": learned_words
            }
            
            # Save checkpoint
            manager.save_checkpoint(epoch, model, optimizer, metrics)
            
            # Update best if needed
            if manager.update_best_if_needed(model, metrics, optimizer):
                best_epoch = epoch
        
        # Verify best model selection
        assert manager.has_best()
        best_info = manager.get_best_info()
        
        # Best should be when learned_words peaked (epoch 6 when it reaches 7)
        # or later if f1_macro continues improving with same learned_words
        assert best_info.metrics["learned_words_count"] == 7
        
        # Verify we can load best
        best_checkpoint = manager.load_best()
        assert best_checkpoint["epoch"] == best_info.best_epoch
    
    def test_resume_and_continue(self, temp_dir):
        """Test resuming training and continuing to improve."""
        model = SimpleModel(input_dim=10, output_dim=5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        manager = CheckpointManager(output_dir=temp_dir)
        
        # Initial training (epochs 0-4)
        for epoch in range(5):
            metrics = {
                "epoch": epoch,
                "val_loss": 0.8 - epoch * 0.1,
                "f1_macro": 0.3 + epoch * 0.05,
                "learned_words_count": epoch + 1
            }
            manager.save_checkpoint(epoch, model, optimizer, metrics)
            manager.update_best_if_needed(model, metrics, optimizer)
        
        # Simulate interruption and resume
        new_model = SimpleModel(input_dim=10, output_dim=5)
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        
        # New manager should load existing best
        new_manager = CheckpointManager(output_dir=temp_dir)
        assert new_manager.get_best_metrics().learned_words_count == 5
        
        # Continue training (epochs 5-9)
        latest = new_manager.load_latest()
        start_epoch = latest["epoch"] + 1
        new_model.load_state_dict(latest["model_state"])
        new_optimizer.load_state_dict(latest["optimizer_state"])
        
        for epoch in range(start_epoch, 10):
            metrics = {
                "epoch": epoch,
                "val_loss": 0.3 - (epoch - 5) * 0.02,
                "f1_macro": 0.5 + (epoch - 5) * 0.03,
                "learned_words_count": 5 + (epoch - 5)  # Continues improving
            }
            new_manager.save_checkpoint(epoch, new_model, new_optimizer, metrics)
            new_manager.update_best_if_needed(new_model, metrics, new_optimizer)
        
        # Best should now be from continued training
        assert new_manager.get_best_metrics().learned_words_count == 9
        assert new_manager.get_best_info().best_epoch == 9


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_metrics_with_checkpoint_metrics_object(self, checkpoint_manager, simple_model):
        """is_best and save_best work with CheckpointMetrics object."""
        metrics = CheckpointMetrics(
            epoch=0,
            val_loss=0.5,
            f1_macro=0.6,
            learned_words_count=5
        )
        
        assert checkpoint_manager.is_best(metrics) is True
        checkpoint_manager.save_best(simple_model, metrics)
        
        assert checkpoint_manager.get_best_metrics().epoch == 0
    
    def test_zero_learned_words(self, checkpoint_manager, simple_model):
        """Handle case where learned_words_count is 0."""
        metrics1 = {"epoch": 0, "val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 0}
        checkpoint_manager.save_best(simple_model, metrics1)
        
        # Even 1 learned word is better
        metrics2 = {"epoch": 1, "val_loss": 0.9, "f1_macro": 0.1, "learned_words_count": 1}
        assert checkpoint_manager.is_best(metrics2) is True
    
    def test_deterministic_selection(self, temp_dir):
        """Same metrics sequence should produce same best selection."""
        metrics_sequence = [
            {"epoch": 0, "val_loss": 0.8, "f1_macro": 0.3, "learned_words_count": 2},
            {"epoch": 1, "val_loss": 0.7, "f1_macro": 0.4, "learned_words_count": 3},
            {"epoch": 2, "val_loss": 0.6, "f1_macro": 0.5, "learned_words_count": 5},
            {"epoch": 3, "val_loss": 0.5, "f1_macro": 0.6, "learned_words_count": 5},
            {"epoch": 4, "val_loss": 0.4, "f1_macro": 0.55, "learned_words_count": 4},
        ]
        
        # Run twice
        for run in range(2):
            run_dir = temp_dir / f"run_{run}"
            model = SimpleModel()
            optimizer = optim.Adam(model.parameters())
            manager = CheckpointManager(output_dir=run_dir)
            
            for metrics in metrics_sequence:
                manager.update_best_if_needed(model, metrics, optimizer)
        
        # Both runs should select same best epoch
        info1 = BestModelInfo.load(temp_dir / "run_0" / "best_model.json")
        info2 = BestModelInfo.load(temp_dir / "run_1" / "best_model.json")
        
        assert info1.best_epoch == info2.best_epoch
        # Best should be epoch 3 (learned_words=5, higher f1_macro than epoch 2)
        assert info1.best_epoch == 3
    
    def test_get_summary(self, checkpoint_manager, simple_model, optimizer):
        """get_summary returns meaningful text."""
        # Before any checkpoints
        summary = checkpoint_manager.get_summary()
        assert "CHECKPOINT SUMMARY" in summary
        assert "Not yet selected" in summary
        
        # After saving best
        metrics = {"epoch": 5, "val_loss": 0.3, "f1_macro": 0.7, "learned_words_count": 10}
        checkpoint_manager.save_checkpoint(5, simple_model, optimizer, metrics)
        checkpoint_manager.save_best(simple_model, metrics)
        
        summary = checkpoint_manager.get_summary()
        assert "Best epoch: 5" in summary
        assert "learned_words: 10" in summary
