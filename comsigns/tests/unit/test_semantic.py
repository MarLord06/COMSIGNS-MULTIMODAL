"""
Tests for backend semantic resolution layer.
"""

import json
import tempfile
from pathlib import Path

import pytest

from backend.semantic.types import (
    SemanticPrediction,
    SemanticClassInfo,
    SemanticTopK,
    SemanticMappingStats
)
from backend.semantic.loader import SemanticMappingLoader
from backend.semantic.resolver import SemanticResolver, create_semantic_resolver


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_class_mapping():
    """Sample class_mapping.json content."""
    return {
        "config": {
            "strategy": "tail_to_other",
            "head_threshold": 10,
            "mid_range": [3, 9],
            "other_class_name": "OTHER"
        },
        "new_class_names": {
            "0": "HEAD_14",
            "1": "HEAD_18",
            "2": "MID_20",
            "3": "MID_26",
            "4": "OTHER"
        },
        "statistics": {
            "num_classes_original": 100,
            "num_classes_remapped": 5,
            "head_count": 2,
            "mid_count": 2,
            "tail_count": 96,
            "other_class_id": 4
        }
    }


@pytest.fixture
def sample_dict():
    """Sample dict.json content."""
    return {
        "14": {"gloss": "HOLA", "instances": []},
        "18": {"gloss": "GRACIAS", "instances": []},
        "20": {"gloss": "POR_FAVOR", "instances": []},
        "26": {"gloss": "YO", "instances": []}
    }


@pytest.fixture
def mapping_files(tmp_path, sample_class_mapping, sample_dict):
    """Create temporary mapping files."""
    class_mapping_path = tmp_path / "class_mapping.json"
    dict_path = tmp_path / "dict.json"
    
    with open(class_mapping_path, "w") as f:
        json.dump(sample_class_mapping, f)
    
    with open(dict_path, "w") as f:
        json.dump(sample_dict, f)
    
    return class_mapping_path, dict_path


# =============================================================================
# SemanticClassInfo Tests
# =============================================================================

class TestSemanticClassInfo:
    """Tests for SemanticClassInfo dataclass."""
    
    def test_creation(self):
        """Test creating SemanticClassInfo."""
        info = SemanticClassInfo(
            new_class_id=28,
            old_class_id=319,
            bucket="HEAD",
            gloss="YO",
            is_other=False
        )
        
        assert info.new_class_id == 28
        assert info.old_class_id == 319
        assert info.bucket == "HEAD"
        assert info.gloss == "YO"
        assert info.is_other is False
    
    def test_other_class(self):
        """Test OTHER class creation."""
        info = SemanticClassInfo(
            new_class_id=141,
            old_class_id=None,
            bucket="OTHER",
            gloss="OTHER",
            is_other=True
        )
        
        assert info.is_other is True
        assert info.old_class_id is None
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = SemanticClassInfo(
            new_class_id=28,
            old_class_id=319,
            bucket="HEAD",
            gloss="YO"
        )
        
        d = info.to_dict()
        
        assert d["new_class_id"] == 28
        assert d["old_class_id"] == 319
        assert d["bucket"] == "HEAD"
        assert d["gloss"] == "YO"


# =============================================================================
# SemanticPrediction Tests
# =============================================================================

class TestSemanticPrediction:
    """Tests for SemanticPrediction dataclass."""
    
    def test_creation(self):
        """Test creating SemanticPrediction."""
        pred = SemanticPrediction(
            gloss="YO",
            confidence=0.85,
            bucket="HEAD",
            old_class_id=319,
            new_class_id=28,
            is_other=False
        )
        
        assert pred.gloss == "YO"
        assert pred.confidence == 0.85
        assert pred.bucket == "HEAD"
        assert pred.is_other is False
    
    def test_other_prediction(self):
        """Test OTHER class prediction."""
        pred = SemanticPrediction(
            gloss="OTHER",
            confidence=0.42,
            bucket="OTHER",
            old_class_id=None,
            new_class_id=141,
            is_other=True
        )
        
        assert pred.is_other is True
        assert pred.old_class_id is None
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        pred = SemanticPrediction(
            gloss="YO",
            confidence=0.85,
            bucket="HEAD",
            old_class_id=319,
            new_class_id=28
        )
        
        d = pred.to_dict()
        
        assert d["gloss"] == "YO"
        assert d["confidence"] == 0.85
        assert "is_other" in d


# =============================================================================
# SemanticTopK Tests
# =============================================================================

class TestSemanticTopK:
    """Tests for SemanticTopK dataclass."""
    
    def test_creation(self):
        """Test creating SemanticTopK."""
        preds = [
            SemanticPrediction("YO", 0.6, "HEAD", 319, 28, False),
            SemanticPrediction("HOLA", 0.3, "MID", 14, 0, False),
        ]
        
        topk = SemanticTopK(predictions=preds)
        
        assert len(topk.predictions) == 2
    
    def test_top1_property(self):
        """Test top1 property."""
        preds = [
            SemanticPrediction("YO", 0.6, "HEAD", 319, 28, False),
            SemanticPrediction("HOLA", 0.3, "MID", 14, 0, False),
        ]
        
        topk = SemanticTopK(predictions=preds)
        
        assert topk.top1.gloss == "YO"
        assert topk.top1.confidence == 0.6
    
    def test_empty_topk(self):
        """Test empty predictions."""
        topk = SemanticTopK(predictions=[])
        
        assert topk.top1 is None


# =============================================================================
# SemanticMappingLoader Tests
# =============================================================================

class TestSemanticMappingLoader:
    """Tests for SemanticMappingLoader class."""
    
    def test_load_class_mapping(self, mapping_files):
        """Test loading class_mapping.json."""
        class_mapping_path, dict_path = mapping_files
        
        loader = SemanticMappingLoader(class_mapping_path)
        loader.load()
        
        assert loader.is_loaded
        assert len(loader.new_class_names) == 5
        assert loader.new_class_names[0] == "HEAD_14"
        assert loader.new_class_names[4] == "OTHER"
    
    def test_load_with_dict(self, mapping_files):
        """Test loading with dict.json."""
        class_mapping_path, dict_path = mapping_files
        
        loader = SemanticMappingLoader(class_mapping_path, dict_path)
        loader.load()
        
        assert len(loader.old_to_gloss) == 4
        assert loader.get_gloss(14) == "HOLA"
        assert loader.get_gloss(26) == "YO"
    
    def test_statistics(self, mapping_files):
        """Test statistics loading."""
        class_mapping_path, _ = mapping_files
        
        loader = SemanticMappingLoader(class_mapping_path)
        loader.load()
        
        assert loader.statistics is not None
        assert loader.statistics.num_classes_remapped == 5
        assert loader.statistics.other_class_id == 4
    
    def test_file_not_found(self, tmp_path):
        """Test error when file not found."""
        loader = SemanticMappingLoader(tmp_path / "nonexistent.json")
        
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_missing_dict_warning(self, mapping_files, tmp_path):
        """Test warning when dict.json missing."""
        class_mapping_path, _ = mapping_files
        
        loader = SemanticMappingLoader(
            class_mapping_path,
            tmp_path / "nonexistent_dict.json"
        )
        loader.load()
        
        # Should load successfully with empty old_to_gloss
        assert len(loader.old_to_gloss) == 0


# =============================================================================
# SemanticResolver Tests
# =============================================================================

class TestSemanticResolver:
    """Tests for SemanticResolver class."""
    
    def test_resolve_head_class(self, mapping_files):
        """Test resolving HEAD class."""
        class_mapping_path, dict_path = mapping_files
        
        loader = SemanticMappingLoader(class_mapping_path, dict_path)
        loader.load()
        resolver = SemanticResolver(loader)
        
        pred = resolver.resolve(new_class_id=0, score=0.85)
        
        assert pred.gloss == "HOLA"
        assert pred.bucket == "HEAD"
        assert pred.old_class_id == 14
        assert pred.is_other is False
    
    def test_resolve_mid_class(self, mapping_files):
        """Test resolving MID class."""
        class_mapping_path, dict_path = mapping_files
        
        loader = SemanticMappingLoader(class_mapping_path, dict_path)
        loader.load()
        resolver = SemanticResolver(loader)
        
        pred = resolver.resolve(new_class_id=3, score=0.5)
        
        assert pred.gloss == "YO"
        assert pred.bucket == "MID"
        assert pred.old_class_id == 26
    
    def test_resolve_other_class(self, mapping_files):
        """Test resolving OTHER class."""
        class_mapping_path, dict_path = mapping_files
        
        loader = SemanticMappingLoader(class_mapping_path, dict_path)
        loader.load()
        resolver = SemanticResolver(loader)
        
        pred = resolver.resolve(new_class_id=4, score=0.42)
        
        assert pred.gloss == "OTHER"
        assert pred.bucket == "OTHER"
        assert pred.old_class_id is None
        assert pred.is_other is True
    
    def test_resolve_topk(self, mapping_files):
        """Test resolving top-K predictions."""
        class_mapping_path, dict_path = mapping_files
        
        loader = SemanticMappingLoader(class_mapping_path, dict_path)
        loader.load()
        resolver = SemanticResolver(loader)
        
        topk = resolver.resolve_topk(
            class_ids=[0, 3, 4],
            scores=[0.6, 0.3, 0.1]
        )
        
        assert len(topk.predictions) == 3
        assert topk.top1.gloss == "HOLA"
        assert topk.predictions[2].is_other is True
    
    def test_is_other_class(self, mapping_files):
        """Test is_other_class method."""
        class_mapping_path, dict_path = mapping_files
        
        loader = SemanticMappingLoader(class_mapping_path, dict_path)
        loader.load()
        resolver = SemanticResolver(loader)
        
        assert resolver.is_other_class(4) is True
        assert resolver.is_other_class(0) is False
    
    def test_get_all_glosses(self, mapping_files):
        """Test get_all_glosses method."""
        class_mapping_path, dict_path = mapping_files
        
        loader = SemanticMappingLoader(class_mapping_path, dict_path)
        loader.load()
        resolver = SemanticResolver(loader)
        
        glosses = resolver.get_all_glosses()
        
        assert len(glosses) == 5
        assert glosses[0] == "HOLA"
        assert glosses[4] == "OTHER"
    
    def test_unknown_class(self, mapping_files):
        """Test handling of unknown class ID."""
        class_mapping_path, dict_path = mapping_files
        
        loader = SemanticMappingLoader(class_mapping_path, dict_path)
        loader.load()
        resolver = SemanticResolver(loader)
        
        pred = resolver.resolve(new_class_id=999, score=0.5)
        
        assert "UNKNOWN" in pred.gloss
        assert pred.bucket == "UNKNOWN"
    
    def test_loader_not_loaded_raises(self, mapping_files):
        """Test error when loader not loaded."""
        class_mapping_path, _ = mapping_files
        
        loader = SemanticMappingLoader(class_mapping_path)
        # Don't call load()
        
        with pytest.raises(ValueError, match="must be loaded"):
            SemanticResolver(loader)


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateSemanticResolver:
    """Tests for create_semantic_resolver factory function."""
    
    def test_create_resolver(self, mapping_files):
        """Test factory function."""
        class_mapping_path, dict_path = mapping_files
        
        resolver = create_semantic_resolver(
            class_mapping_path=str(class_mapping_path),
            dict_path=str(dict_path)
        )
        
        assert isinstance(resolver, SemanticResolver)
        
        pred = resolver.resolve(0, 0.85)
        assert pred.gloss == "HOLA"


# =============================================================================
# Integration Tests
# =============================================================================

class TestSemanticIntegration:
    """Integration tests for semantic layer."""
    
    def test_full_flow(self, mapping_files):
        """Test complete semantic resolution flow."""
        class_mapping_path, dict_path = mapping_files
        
        # Load
        loader = SemanticMappingLoader(class_mapping_path, dict_path)
        loader.load()
        
        # Create resolver
        resolver = SemanticResolver(loader)
        
        # Simulate model predictions
        model_class_ids = [0, 3, 4]
        model_scores = [0.6, 0.25, 0.15]
        
        # Resolve
        topk = resolver.resolve_topk(model_class_ids, model_scores)
        
        # Verify
        assert topk.top1.gloss == "HOLA"
        assert topk.top1.confidence == 0.6
        assert topk.top1.bucket == "HEAD"
        
        # Check serialization
        d = topk.to_dict()
        assert "predictions" in d
        assert d["top1"]["gloss"] == "HOLA"
