"""
Tests for final evaluation module.

Tests:
- FinalEvaluator class
- EvaluationResult dataclass
- Artifact generation (JSON, CSV, PNG)
- Integration with Trainer
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.evaluation import (
    FinalEvaluator,
    EvaluationResult,
    run_final_evaluation
)


# =============================================================================
# Fixtures
# =============================================================================

class SimpleClassifier(nn.Module):
    """Simple classifier for testing."""
    
    def __init__(self, input_dim: int = 10, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dim * 3, num_classes)  # hand + body + face
    
    def forward(self, hand, body, face, lengths=None, mask=None):
        # Simple concat and classify
        x = torch.cat([hand.mean(dim=1), body.mean(dim=1), face.mean(dim=1)], dim=-1)
        return self.fc(x)


@pytest.fixture
def model():
    """Create simple classifier."""
    return SimpleClassifier(input_dim=10, num_classes=5)


@pytest.fixture
def val_loader():
    """Create validation DataLoader."""
    batch_size = 4
    num_samples = 20
    seq_len = 8
    
    # Create batches as dicts
    def collate_fn(batch):
        hand = torch.stack([b[0] for b in batch])
        body = torch.stack([b[1] for b in batch])
        face = torch.stack([b[2] for b in batch])
        labels = torch.stack([b[3] for b in batch])
        lengths = torch.full((len(batch),), seq_len, dtype=torch.long)
        return {
            "hand": hand,
            "body": body,
            "face": face,
            "labels": labels,
            "lengths": lengths,
            "mask": None
        }
    
    # Generate data
    data = [
        (
            torch.randn(seq_len, 10),  # hand
            torch.randn(seq_len, 10),  # body
            torch.randn(seq_len, 10),  # face
            torch.tensor(i % 5)        # label
        )
        for i in range(num_samples)
    ]
    
    return DataLoader(data, batch_size=batch_size, collate_fn=collate_fn)


@pytest.fixture
def class_names():
    """Sample class names."""
    return ["alpha", "beta", "gamma", "delta", "epsilon"]


# =============================================================================
# EvaluationResult Tests
# =============================================================================

class TestEvaluationResult:
    """Test EvaluationResult dataclass."""
    
    def test_get_confusion_matrix(self):
        """Test confusion matrix generation."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 0, 0, 1, 1])  # Some errors
        y_logits = np.random.randn(6, 3)
        
        result = EvaluationResult(
            y_true=y_true,
            y_pred=y_pred,
            y_logits=y_logits,
            num_samples=6,
            num_classes=3
        )
        
        cm = result.get_confusion_matrix()
        
        assert cm.shape == (3, 3)
        assert cm.sum() == 6
        assert cm[0, 0] == 2  # True 0, pred 0
        assert cm[2, 0] == 1  # True 2, pred 0 (error)
        assert cm[2, 1] == 1  # True 2, pred 1 (error)
    
    def test_get_confusion_matrix_normalized(self):
        """Test normalized confusion matrix."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        y_logits = np.random.randn(4, 2)
        
        result = EvaluationResult(
            y_true=y_true,
            y_pred=y_pred,
            y_logits=y_logits,
            num_samples=4,
            num_classes=2
        )
        
        cm_norm = result.get_confusion_matrix(normalize='true')
        
        # Row 0: 2/2 = 1.0 correct
        assert cm_norm[0, 0] == pytest.approx(1.0)
        # Row 1: 1/2 = 0.5 correct
        assert cm_norm[1, 1] == pytest.approx(0.5)
    
    def test_get_per_class_metrics(self):
        """Test per-class metrics computation."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 2])
        y_logits = np.random.randn(6, 3)
        
        result = EvaluationResult(
            y_true=y_true,
            y_pred=y_pred,
            y_logits=y_logits,
            num_samples=6,
            num_classes=3,
            class_names=["A", "B", "C"]
        )
        
        metrics = result.get_per_class_metrics()
        
        assert len(metrics) == 3
        assert metrics[0]["class_name"] == "A"
        assert metrics[0]["support"] == 2
        assert metrics[0]["recall"] == 1.0  # 2/2 correct
        assert metrics[1]["recall"] == 0.5  # 1/2 correct
        assert metrics[2]["recall"] == 1.0  # 2/2 correct
    
    def test_get_global_metrics(self):
        """Test global metrics computation."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])  # All correct
        y_logits = np.random.randn(6, 3)
        
        result = EvaluationResult(
            y_true=y_true,
            y_pred=y_pred,
            y_logits=y_logits,
            num_samples=6,
            num_classes=3
        )
        
        global_metrics = result.get_global_metrics()
        
        assert global_metrics["accuracy"] == 1.0
        assert global_metrics["f1_macro"] == pytest.approx(1.0)
        assert global_metrics["num_samples"] == 6


# =============================================================================
# FinalEvaluator Tests
# =============================================================================

class TestFinalEvaluator:
    """Test FinalEvaluator class."""
    
    def test_evaluate_runs(self, model, val_loader):
        """Test that evaluate() runs without errors."""
        evaluator = FinalEvaluator(
            model=model,
            dataloader=val_loader,
            num_classes=5
        )
        
        result = evaluator.evaluate()
        
        assert result is not None
        assert result.num_samples == 20
        assert result.num_classes == 5
        assert len(result.y_true) == 20
        assert len(result.y_pred) == 20
    
    def test_evaluate_model_in_eval_mode(self, model, val_loader):
        """Test that model is in eval mode during evaluation."""
        model.train()  # Start in train mode
        
        evaluator = FinalEvaluator(
            model=model,
            dataloader=val_loader,
            num_classes=5
        )
        
        evaluator.evaluate()
        
        # Model should be in eval mode after evaluation
        assert not model.training
    
    def test_evaluate_no_gradients(self, model, val_loader):
        """Test that no gradients are computed."""
        evaluator = FinalEvaluator(
            model=model,
            dataloader=val_loader,
            num_classes=5
        )
        
        # Enable gradients
        for param in model.parameters():
            param.requires_grad = True
        
        evaluator.evaluate()
        
        # No gradients should be accumulated
        for param in model.parameters():
            assert param.grad is None or param.grad.abs().sum() == 0
    
    def test_save_artifacts(self, model, val_loader, class_names):
        """Test artifact saving."""
        evaluator = FinalEvaluator(
            model=model,
            dataloader=val_loader,
            num_classes=5,
            class_names=class_names
        )
        
        evaluator.evaluate()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = evaluator.save_artifacts(
                output_dir=tmpdir,
                dataset_name="test",
                epoch=10
            )
            
            # Check all files exist
            assert Path(paths["metrics_json"]).exists()
            assert Path(paths["metrics_csv"]).exists()
            assert Path(paths["confusion_matrix_csv"]).exists()
            assert Path(paths["summary_json"]).exists()
            
            # Verify JSON content
            with open(paths["metrics_json"], "r") as f:
                data = json.load(f)
            assert "metrics_by_class" in data
            assert len(data["metrics_by_class"]) == 5
    
    def test_save_artifacts_with_prefix(self, model, val_loader):
        """Test artifact saving with prefix."""
        evaluator = FinalEvaluator(
            model=model,
            dataloader=val_loader,
            num_classes=5
        )
        
        evaluator.evaluate()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = evaluator.save_artifacts(
                output_dir=tmpdir,
                prefix="experiment_001"
            )
            
            # Check prefix in filenames
            for path in paths.values():
                assert "experiment_001" in Path(path).name
    
    def test_print_summary(self, model, val_loader, capsys):
        """Test summary printing."""
        evaluator = FinalEvaluator(
            model=model,
            dataloader=val_loader,
            num_classes=5
        )
        
        evaluator.evaluate()
        evaluator.print_summary()
        
        captured = capsys.readouterr()
        assert "FINAL EVALUATION SUMMARY" in captured.out
        assert "Accuracy" in captured.out


# =============================================================================
# run_final_evaluation Tests
# =============================================================================

class TestRunFinalEvaluation:
    """Test run_final_evaluation convenience function."""
    
    def test_run_final_evaluation(self, model, val_loader, class_names):
        """Test complete evaluation pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, paths = run_final_evaluation(
                model=model,
                val_loader=val_loader,
                num_classes=5,
                output_dir=tmpdir,
                class_names=class_names,
                dataset_name="validation",
                epoch=5
            )
            
            # Check result
            assert result.num_samples == 20
            
            # Check paths
            assert len(paths) >= 4
            
            # Check summary includes epoch
            with open(paths["summary_json"], "r") as f:
                summary = json.load(f)
            assert summary["epoch"] == 5
            assert summary["dataset"] == "validation"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_class(self, val_loader):
        """Test with single class."""
        model = SimpleClassifier(input_dim=10, num_classes=1)
        
        # Modify loader to have single class
        def collate_fn(batch):
            return {
                "hand": torch.randn(len(batch), 8, 10),
                "body": torch.randn(len(batch), 8, 10),
                "face": torch.randn(len(batch), 8, 10),
                "labels": torch.zeros(len(batch), dtype=torch.long),
                "lengths": torch.full((len(batch),), 8, dtype=torch.long),
                "mask": None
            }
        
        loader = DataLoader(range(10), batch_size=2, collate_fn=collate_fn)
        
        evaluator = FinalEvaluator(
            model=model,
            dataloader=loader,
            num_classes=1
        )
        
        result = evaluator.evaluate()
        assert result.num_classes == 1
    
    def test_evaluate_before_save_raises(self, model, val_loader):
        """Test that save_artifacts raises if evaluate not called."""
        evaluator = FinalEvaluator(
            model=model,
            dataloader=val_loader,
            num_classes=5
        )
        
        with pytest.raises(RuntimeError, match="Must call evaluate"):
            evaluator.save_artifacts(output_dir="/tmp")
    
    def test_empty_class_names(self, model, val_loader):
        """Test with empty class names."""
        evaluator = FinalEvaluator(
            model=model,
            dataloader=val_loader,
            num_classes=5,
            class_names=[]
        )
        
        result = evaluator.evaluate()
        metrics = result.get_per_class_metrics()
        
        # Should still work, but class_name will be None
        assert metrics[0]["class_name"] is None


# =============================================================================
# Confusion Matrix Criteria Tests
# =============================================================================

class TestConfusionMatrixCriteria:
    """
    Tests that verify confusion matrix meets specified criteria:
    - y_true: labels reales del conjunto de validación
    - y_pred: argmax(logits) del modelo
    - Calculada al final del entrenamiento
    - Basada únicamente en predicciones del modelo
    - C[i][j] = instancias de clase real i predichas como j
    """
    
    def test_cm_uses_argmax_predictions(self):
        """Verify y_pred comes from argmax(logits)."""
        # Create known logits where argmax is predictable
        y_logits = np.array([
            [2.0, 0.5, 0.1],  # argmax = 0
            [0.1, 2.0, 0.5],  # argmax = 1
            [0.1, 0.5, 2.0],  # argmax = 2
        ])
        y_true = np.array([0, 1, 2])
        y_pred = y_logits.argmax(axis=1)
        
        result = EvaluationResult(
            y_true=y_true,
            y_pred=y_pred,
            y_logits=y_logits,
            num_samples=3,
            num_classes=3
        )
        
        cm = result.get_confusion_matrix()
        
        # With these logits, all predictions match true labels
        # So diagonal should be 1, 1, 1
        assert cm[0, 0] == 1
        assert cm[1, 1] == 1
        assert cm[2, 2] == 1
    
    def test_cm_definition_cij(self):
        """Verify C[i][j] = count of true i predicted as j."""
        y_true = np.array([0, 0, 0, 1, 1, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2])
        # True 0 -> Pred 0: 1
        # True 0 -> Pred 1: 2
        # True 1 -> Pred 1: 1
        # True 1 -> Pred 2: 1
        # True 2 -> Pred 2: 1
        
        result = EvaluationResult(
            y_true=y_true,
            y_pred=y_pred,
            y_logits=np.zeros((6, 3)),
            num_samples=6,
            num_classes=3
        )
        
        cm = result.get_confusion_matrix()
        
        assert cm[0, 0] == 1  # True 0, Pred 0
        assert cm[0, 1] == 2  # True 0, Pred 1
        assert cm[1, 1] == 1  # True 1, Pred 1
        assert cm[1, 2] == 1  # True 1, Pred 2
        assert cm[2, 2] == 1  # True 2, Pred 2
    
    def test_cm_post_forward_only(self, model, val_loader):
        """Verify CM is based on model's forward pass output."""
        evaluator = FinalEvaluator(
            model=model,
            dataloader=val_loader,
            num_classes=5
        )
        
        result = evaluator.evaluate()
        
        # Verify y_pred matches what argmax(logits) would give
        expected_preds = result.y_logits.argmax(axis=1)
        np.testing.assert_array_equal(result.y_pred, expected_preds)
    
    def test_cm_no_external_info(self, model, val_loader):
        """Verify CM doesn't use class frequency or dict.json."""
        evaluator = FinalEvaluator(
            model=model,
            dataloader=val_loader,
            num_classes=5
        )
        
        result = evaluator.evaluate()
        cm = result.get_confusion_matrix()
        
        # CM should only reflect what model predicted
        # Total should equal number of samples
        assert cm.sum() == result.num_samples
        
        # Each row sum = support for that class
        for i in range(5):
            assert cm[i].sum() == (result.y_true == i).sum()
