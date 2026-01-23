"""
Tests for inference pipeline.
"""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import torch.nn as nn

from inference.types import (
    PredictionResult,
    TopKPrediction,
    InferenceConfig,
    ModelInfo
)
from inference.predictor import Predictor, create_predictor
from inference.loader import InferenceLoader


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_classes = 10
        
        def forward(self, hand, body, face, lengths=None):
            batch_size = hand.shape[0]
            # Return logits that favor class 3
            logits = torch.randn(batch_size, 10)
            logits[:, 3] = 5.0  # Make class 3 highest
            return logits
        
        def eval(self):
            return self
    
    return FakeModel()


@pytest.fixture
def class_names():
    """Sample class names."""
    return {
        0: "HOLA",
        1: "GRACIAS",
        2: "POR_FAVOR",
        3: "YO",
        4: "TU",
        5: "EL",
        6: "NOSOTROS",
        7: "USTEDES",
        8: "ELLOS",
        9: "OTHER"
    }


@pytest.fixture
def sample_features():
    """Create sample feature tensors."""
    T = 30  # frames
    return {
        "hand": torch.randn(1, T, 168),   # 21 * 4 * 2
        "body": torch.randn(1, T, 132),   # 33 * 4
        "face": torch.randn(1, T, 1872),  # 468 * 4
        "lengths": torch.tensor([T])
    }


@pytest.fixture
def sample_features_2d():
    """Create sample 2D features (no batch dim)."""
    T = 30
    return {
        "hand": torch.randn(T, 168),
        "body": torch.randn(T, 132),
        "face": torch.randn(T, 1872)
    }


# =============================================================================
# TopKPrediction Tests
# =============================================================================

class TestTopKPrediction:
    """Tests for TopKPrediction dataclass."""
    
    def test_creation(self):
        """Test creating a TopKPrediction."""
        pred = TopKPrediction(
            rank=1,
            class_id=5,
            class_name="YO",
            score=0.85
        )
        
        assert pred.rank == 1
        assert pred.class_id == 5
        assert pred.class_name == "YO"
        assert pred.score == 0.85
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        pred = TopKPrediction(rank=1, class_id=5, class_name="YO", score=0.85)
        d = pred.to_dict()
        
        assert d["rank"] == 1
        assert d["class_id"] == 5
        assert d["class_name"] == "YO"
        assert d["score"] == 0.85


# =============================================================================
# PredictionResult Tests
# =============================================================================

class TestPredictionResult:
    """Tests for PredictionResult dataclass."""
    
    def test_creation(self, class_names):
        """Test creating a PredictionResult."""
        topk = [
            TopKPrediction(1, 3, "YO", 0.62),
            TopKPrediction(2, 4, "TU", 0.14),
            TopKPrediction(3, 5, "EL", 0.09)
        ]
        
        result = PredictionResult(
            top1_class_id=3,
            top1_class_name="YO",
            top1_score=0.62,
            topk=topk,
            is_other=False
        )
        
        assert result.top1_class_id == 3
        assert result.top1_class_name == "YO"
        assert result.top1_score == 0.62
        assert len(result.topk) == 3
        assert result.is_other is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        topk = [TopKPrediction(1, 3, "YO", 0.62)]
        result = PredictionResult(
            top1_class_id=3,
            top1_class_name="YO",
            top1_score=0.62,
            topk=topk,
            is_other=False
        )
        
        d = result.to_dict()
        
        assert d["top1_class_id"] == 3
        assert d["top1_class_name"] == "YO"
        assert "topk" in d
        assert len(d["topk"]) == 1
    
    def test_format_output(self):
        """Test formatted string output."""
        topk = [
            TopKPrediction(1, 3, "YO", 0.62),
            TopKPrediction(2, 4, "TU", 0.14)
        ]
        result = PredictionResult(
            top1_class_id=3,
            top1_class_name="YO",
            top1_score=0.62,
            topk=topk,
            is_other=False
        )
        
        output = result.format_output()
        
        assert "Inference Result" in output
        assert "YO" in output
        assert "0.62" in output
        assert "Top-2 Predictions:" in output
        assert "Is OTHER: False" in output
    
    def test_format_output_with_other(self):
        """Test formatted output when is_other=True."""
        topk = [TopKPrediction(1, 9, "OTHER", 0.55)]
        result = PredictionResult(
            top1_class_id=9,
            top1_class_name="OTHER",
            top1_score=0.55,
            topk=topk,
            is_other=True
        )
        
        output = result.format_output()
        assert "Is OTHER: True" in output


# =============================================================================
# InferenceConfig Tests
# =============================================================================

class TestInferenceConfig:
    """Tests for InferenceConfig."""
    
    def test_defaults(self):
        """Test default configuration."""
        config = InferenceConfig()
        
        assert config.device == "cpu"
        assert config.topk == 5
        assert config.return_logits is False
        assert config.return_all_scores is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = InferenceConfig(
            device="cuda",
            topk=10,
            return_logits=True
        )
        
        assert config.device == "cuda"
        assert config.topk == 10
        assert config.return_logits is True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = InferenceConfig(device="cuda", topk=3)
        d = config.to_dict()
        
        assert d["device"] == "cuda"
        assert d["topk"] == 3


# =============================================================================
# ModelInfo Tests
# =============================================================================

class TestModelInfo:
    """Tests for ModelInfo."""
    
    def test_creation(self):
        """Test creating ModelInfo."""
        info = ModelInfo(
            checkpoint_path="/path/to/best.pt",
            epoch=50,
            num_classes=100,
            other_class_id=99,
            tail_to_other=True,
            metrics={"f1_macro": 0.75}
        )
        
        assert info.checkpoint_path == "/path/to/best.pt"
        assert info.epoch == 50
        assert info.num_classes == 100
        assert info.other_class_id == 99
        assert info.tail_to_other is True
    
    def test_defaults(self):
        """Test default values."""
        info = ModelInfo(
            checkpoint_path="/path/to/model.pt",
            epoch=10,
            num_classes=50
        )
        
        assert info.other_class_id == -1
        assert info.tail_to_other is False
        assert info.metrics == {}


# =============================================================================
# Predictor Tests
# =============================================================================

class TestPredictor:
    """Tests for Predictor class."""
    
    def test_init(self, mock_model, class_names):
        """Test predictor initialization."""
        predictor = Predictor(
            model=mock_model,
            class_names=class_names,
            other_class_id=9,
            device="cpu",
            topk=5
        )
        
        assert predictor.model is mock_model
        assert predictor.other_class_id == 9
        assert predictor.topk == 5
    
    def test_predict(self, mock_model, class_names, sample_features):
        """Test basic prediction."""
        predictor = Predictor(
            model=mock_model,
            class_names=class_names,
            other_class_id=9,
            topk=5
        )
        
        result = predictor.predict(
            sample_features["hand"],
            sample_features["body"],
            sample_features["face"],
            sample_features["lengths"]
        )
        
        assert isinstance(result, PredictionResult)
        assert result.top1_class_id == 3  # Mock returns highest for class 3
        assert result.top1_class_name == "YO"
        assert 0 <= result.top1_score <= 1
        assert len(result.topk) == 5
        assert result.is_other is False
    
    def test_predict_2d_input(self, mock_model, class_names, sample_features_2d):
        """Test prediction with 2D input (no batch dim)."""
        predictor = Predictor(
            model=mock_model,
            class_names=class_names,
            topk=3
        )
        
        result = predictor.predict(
            sample_features_2d["hand"],
            sample_features_2d["body"],
            sample_features_2d["face"]
        )
        
        assert isinstance(result, PredictionResult)
        assert len(result.topk) == 3
    
    def test_predict_from_features(self, mock_model, class_names, sample_features):
        """Test prediction from feature dictionary."""
        predictor = Predictor(
            model=mock_model,
            class_names=class_names,
            topk=5
        )
        
        result = predictor.predict_from_features(sample_features)
        
        assert isinstance(result, PredictionResult)
        assert result.top1_class_id == 3
    
    def test_is_other_detection(self, class_names):
        """Test OTHER class detection."""
        # Create model that predicts class 9 (OTHER)
        class OtherModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_classes = 10
            
            def forward(self, hand, body, face, lengths=None):
                batch_size = hand.shape[0]
                logits = torch.randn(batch_size, 10)
                logits[:, 9] = 10.0  # Make OTHER highest
                return logits
            
            def eval(self):
                return self
        
        model = OtherModel()
        
        predictor = Predictor(
            model=model,
            class_names=class_names,
            other_class_id=9,
            topk=5
        )
        
        features = {
            "hand": torch.randn(1, 30, 168),
            "body": torch.randn(1, 30, 132),
            "face": torch.randn(1, 30, 1872)
        }
        
        result = predictor.predict_from_features(features)
        
        assert result.top1_class_id == 9
        assert result.top1_class_name == "OTHER"
        assert result.is_other is True
    
    def test_topk_ordering(self, mock_model, class_names, sample_features):
        """Test that top-k results are properly ordered."""
        predictor = Predictor(
            model=mock_model,
            class_names=class_names,
            topk=5
        )
        
        result = predictor.predict_from_features(sample_features)
        
        # Check ordering
        for i in range(len(result.topk) - 1):
            assert result.topk[i].score >= result.topk[i + 1].score
            assert result.topk[i].rank == i + 1
    
    def test_topk_exceeds_num_classes(self, mock_model, class_names, sample_features):
        """Test when topk > num_classes."""
        predictor = Predictor(
            model=mock_model,
            class_names=class_names,
            topk=20  # More than 10 classes
        )
        
        result = predictor.predict_from_features(sample_features)
        
        # Should return min(topk, num_classes)
        assert len(result.topk) == 10
    
    def test_unknown_class_name(self, mock_model, sample_features):
        """Test handling of unknown class names."""
        # Sparse class names
        sparse_names = {0: "KNOWN_CLASS"}
        
        predictor = Predictor(
            model=mock_model,
            class_names=sparse_names,
            topk=3
        )
        
        result = predictor.predict_from_features(sample_features)
        
        # Unknown classes should have fallback names
        for pred in result.topk:
            if pred.class_id not in sparse_names:
                assert pred.class_name.startswith("CLASS_")
    
    def test_predict_batch(self, mock_model, class_names):
        """Test batch prediction."""
        predictor = Predictor(
            model=mock_model,
            class_names=class_names,
            topk=3
        )
        
        batch_size = 4
        T = 30
        features = {
            "hand": torch.randn(batch_size, T, 168),
            "body": torch.randn(batch_size, T, 132),
            "face": torch.randn(batch_size, T, 1872)
        }
        
        results = predictor.predict_batch(
            features["hand"],
            features["body"],
            features["face"]
        )
        
        assert len(results) == batch_size
        for result in results:
            assert isinstance(result, PredictionResult)
            assert len(result.topk) == 3
    
    def test_get_prediction_summary(self, mock_model, class_names, sample_features):
        """Test summary string generation."""
        predictor = Predictor(
            model=mock_model,
            class_names=class_names,
            other_class_id=9,
            topk=3
        )
        
        result = predictor.predict_from_features(sample_features)
        summary = predictor.get_prediction_summary(result)
        
        assert "YO" in summary
        assert "OTHER: No" in summary


class TestCreatePredictor:
    """Tests for create_predictor factory function."""
    
    def test_create_predictor(self, mock_model, class_names):
        """Test factory function."""
        predictor = create_predictor(
            model=mock_model,
            class_names=class_names,
            other_class_id=9,
            device="cpu",
            topk=10
        )
        
        assert isinstance(predictor, Predictor)
        assert predictor.topk == 10


# =============================================================================
# InferenceLoader Tests
# =============================================================================

class TestInferenceLoader:
    """Tests for InferenceLoader class."""
    
    def test_init_file_not_found(self):
        """Test error when checkpoint doesn't exist."""
        with pytest.raises(FileNotFoundError):
            InferenceLoader(
                checkpoint_path=Path("/nonexistent/path/model.pt")
            )
    
    def test_class_mapping_path_inference(self, tmp_path):
        """Test automatic class mapping path inference."""
        # Create directory structure
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        checkpoint_path = checkpoints_dir / "best.pt"
        
        # Create dummy checkpoint
        torch.save({"model_state": {}, "epoch": 1, "num_classes": 10}, checkpoint_path)
        
        loader = InferenceLoader(checkpoint_path=checkpoint_path)
        
        # Should infer class_mapping.json in parent directory
        expected_mapping_path = tmp_path / "class_mapping.json"
        assert loader.class_mapping_path == expected_mapping_path
    
    def test_load_class_mapping(self, tmp_path):
        """Test loading class mapping JSON."""
        # Create checkpoint
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        checkpoint_path = checkpoints_dir / "best.pt"
        torch.save({"model_state": {}, "epoch": 1, "num_classes": 10}, checkpoint_path)
        
        # Create class mapping
        mapping = {
            "config": {"strategy": "tail_to_other"},
            "old_to_new": {"0": 0, "1": 1},
            "new_to_old": {"0": [0], "1": [1]},
            "new_class_names": {"0": "HOLA", "1": "OTHER"},
            "statistics": {
                "num_classes_original": 100,
                "num_classes_remapped": 2,
                "other_class_id": 1
            }
        }
        mapping_path = tmp_path / "class_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f)
        
        loader = InferenceLoader(checkpoint_path=checkpoint_path)
        loaded_mapping = loader.load_class_mapping()
        
        assert loaded_mapping["statistics"]["other_class_id"] == 1
        assert loaded_mapping["config"]["strategy"] == "tail_to_other"
    
    def test_get_num_classes_from_metadata(self, tmp_path):
        """Test getting num_classes from checkpoint metadata."""
        checkpoint_path = tmp_path / "model.pt"
        torch.save({"model_state": {}, "epoch": 1, "num_classes": 50}, checkpoint_path)
        
        loader = InferenceLoader(checkpoint_path=checkpoint_path)
        
        assert loader.get_num_classes() == 50
    
    def test_get_num_classes_from_weights(self, tmp_path):
        """Test inferring num_classes from classifier weights."""
        checkpoint_path = tmp_path / "model.pt"
        
        # Save checkpoint with classifier weights but no metadata
        checkpoint = {
            "model_state": {
                "classifier.weight": torch.randn(75, 512),
                "classifier.bias": torch.randn(75)
            },
            "epoch": 1
        }
        torch.save(checkpoint, checkpoint_path)
        
        loader = InferenceLoader(checkpoint_path=checkpoint_path)
        
        assert loader.get_num_classes() == 75
    
    def test_get_class_names(self, tmp_path):
        """Test getting class names from mapping."""
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        checkpoint_path = checkpoints_dir / "best.pt"
        torch.save({"model_state": {}, "num_classes": 3}, checkpoint_path)
        
        mapping = {
            "config": {},
            "new_class_names": {"0": "HOLA", "1": "GRACIAS", "2": "OTHER"},
            "statistics": {"other_class_id": 2}
        }
        with open(tmp_path / "class_mapping.json", "w") as f:
            json.dump(mapping, f)
        
        loader = InferenceLoader(checkpoint_path=checkpoint_path)
        names = loader.get_class_names()
        
        assert names[0] == "HOLA"
        assert names[1] == "GRACIAS"
        assert names[2] == "OTHER"
    
    def test_was_tail_to_other_from_checkpoint(self, tmp_path):
        """Test checking tail_to_other from checkpoint."""
        checkpoint_path = tmp_path / "model.pt"
        torch.save({
            "model_state": {},
            "num_classes": 10,
            "tail_to_other": True
        }, checkpoint_path)
        
        loader = InferenceLoader(checkpoint_path=checkpoint_path)
        
        assert loader.was_tail_to_other() is True
    
    def test_was_tail_to_other_from_mapping(self, tmp_path):
        """Test checking tail_to_other from class mapping."""
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        checkpoint_path = checkpoints_dir / "best.pt"
        torch.save({"model_state": {}, "num_classes": 10}, checkpoint_path)
        
        mapping = {
            "config": {"strategy": "tail_to_other"},
            "statistics": {"other_class_id": 9}
        }
        with open(tmp_path / "class_mapping.json", "w") as f:
            json.dump(mapping, f)
        
        loader = InferenceLoader(checkpoint_path=checkpoint_path)
        
        assert loader.was_tail_to_other() is True
    
    def test_missing_class_mapping_returns_empty(self, tmp_path):
        """Test that missing class mapping returns empty dict."""
        checkpoint_path = tmp_path / "model.pt"
        torch.save({"model_state": {}, "num_classes": 10}, checkpoint_path)
        
        loader = InferenceLoader(checkpoint_path=checkpoint_path)
        mapping = loader.load_class_mapping()
        
        assert mapping == {}


# =============================================================================
# Integration Tests
# =============================================================================

class TestInferenceIntegration:
    """Integration tests for the full inference pipeline."""
    
    def test_end_to_end_with_mock_model(self, mock_model, class_names):
        """Test complete inference flow with mock model."""
        # Create predictor
        predictor = Predictor(
            model=mock_model,
            class_names=class_names,
            other_class_id=9,
            topk=5
        )
        
        # Create features
        features = {
            "hand": torch.randn(1, 30, 168),
            "body": torch.randn(1, 30, 132),
            "face": torch.randn(1, 30, 1872)
        }
        
        # Run inference
        result = predictor.predict_from_features(features)
        
        # Validate result structure
        assert result.top1_class_id >= 0
        assert result.top1_class_name in class_names.values() or result.top1_class_name.startswith("CLASS_")
        assert 0 <= result.top1_score <= 1
        assert len(result.topk) <= 5
        assert isinstance(result.is_other, bool)
        
        # Test output formatting
        output = result.format_output()
        assert "Inference Result" in output
        
        # Test JSON serialization
        d = result.to_dict()
        json_str = json.dumps(d)
        assert "top1_class_id" in json_str
