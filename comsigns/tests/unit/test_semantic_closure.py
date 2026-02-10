"""
Unit tests for semantic_closure module.

Tests filtering utilities that ensure semantic resolution:
new_class_id → class_mapping.json → dict.json → gloss
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from training.semantic_closure import (
    load_dict_mapping,
    build_semantic_whitelist,
    SemanticClosureReport,
    DictIdDataset,
    FilteredGlossDataset,
    LowSupportFilter
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_dict_json():
    """Sample dict.json structure."""
    return {
        "0": {"gloss": "hola", "instances": 15},
        "1": {"gloss": "adios", "instances": 8},
        "2": {"gloss": "gracias", "instances": 5},
        "3": {"gloss": "por favor", "instances": 3},
        "100": {"gloss": "casa", "instances": 20}
    }


@pytest.fixture
def sample_class_mapping():
    """Sample class_mapping.json structure."""
    return {
        "new_class_names": {
            "0": "HEAD_0",
            "1": "HEAD_100",
            "2": "MID_1",
            "3": "MID_2",
            "4": "OTHER"
        },
        "new_to_old": {
            "0": 0,
            "1": 100,
            "2": 1,
            "3": 2,
            "4": -1  # OTHER has no old_id
        },
        "old_to_new": {
            "0": 0,
            "100": 1,
            "1": 2,
            "2": 3
        }
    }


@pytest.fixture
def dict_json_path(sample_dict_json, tmp_path):
    """Create temporary dict.json file."""
    path = tmp_path / "dict.json"
    with open(path, 'w') as f:
        json.dump(sample_dict_json, f)
    return path


@pytest.fixture
def class_mapping_path(sample_class_mapping, tmp_path):
    """Create temporary class_mapping.json file."""
    path = tmp_path / "class_mapping.json"
    with open(path, 'w') as f:
        json.dump(sample_class_mapping, f)
    return path


@pytest.fixture
def mock_base_dataset():
    """Create a mock base dataset."""
    dataset = MagicMock()
    dataset.gloss_to_id = {
        "hola": 0,
        "adios": 1,
        "gracias": 2,
        "desconocido": 99  # Not in dict.json
    }
    dataset.id_to_gloss = {v: k for k, v in dataset.gloss_to_id.items()}
    
    # Sample data
    samples = [
        {"keypoints": [1, 2, 3], "gloss": "hola", "class_id": 0},
        {"keypoints": [4, 5, 6], "gloss": "hola", "class_id": 0},
        {"keypoints": [7, 8, 9], "gloss": "adios", "class_id": 1},
        {"keypoints": [10, 11, 12], "gloss": "gracias", "class_id": 2},
        {"keypoints": [13, 14, 15], "gloss": "desconocido", "class_id": 99},
    ]
    dataset.__len__ = MagicMock(return_value=len(samples))
    dataset.__getitem__ = MagicMock(side_effect=lambda i: samples[i])
    
    return dataset


# =============================================================================
# Tests: load_dict_mapping
# =============================================================================

class TestLoadDictMapping:
    """Tests for load_dict_mapping function."""
    
    def test_loads_valid_dict_json(self, dict_json_path, sample_dict_json):
        """Should load and parse dict.json correctly."""
        mapping = load_dict_mapping(dict_json_path)
        
        assert len(mapping) == len(sample_dict_json)
        assert mapping[0] == "hola"
        assert mapping[100] == "casa"
    
    def test_returns_old_id_to_gloss_mapping(self, dict_json_path):
        """Should return dict with int keys and string values."""
        mapping = load_dict_mapping(dict_json_path)
        
        for key, value in mapping.items():
            assert isinstance(key, int)
            assert isinstance(value, str)
    
    def test_raises_on_missing_file(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_dict_mapping(tmp_path / "nonexistent.json")


# =============================================================================
# Tests: build_semantic_whitelist
# =============================================================================

class TestBuildSemanticWhitelist:
    """Tests for build_semantic_whitelist function."""
    
    def test_builds_whitelist_from_dict_and_mapping(
        self, dict_json_path, class_mapping_path
    ):
        """Should build whitelist of valid new_class_ids."""
        whitelist, report = build_semantic_whitelist(
            dict_json_path, class_mapping_path
        )
        
        # Should include classes that resolve to valid glosses
        assert 0 in whitelist  # HEAD_0 → old_id=0 → "hola"
        assert 1 in whitelist  # HEAD_100 → old_id=100 → "casa"
        assert 2 in whitelist  # MID_1 → old_id=1 → "adios"
        assert 3 in whitelist  # MID_2 → old_id=2 → "gracias"
        
        # OTHER should NOT be in whitelist (no valid gloss)
        assert 4 not in whitelist
    
    def test_returns_semantic_closure_report(
        self, dict_json_path, class_mapping_path
    ):
        """Should return a SemanticClosureReport."""
        whitelist, report = build_semantic_whitelist(
            dict_json_path, class_mapping_path
        )
        
        assert isinstance(report, SemanticClosureReport)
        assert report.total_new_classes > 0
        assert report.valid_classes > 0
    
    def test_report_contains_invalid_classes(
        self, dict_json_path, class_mapping_path
    ):
        """Report should list classes that failed resolution."""
        _, report = build_semantic_whitelist(
            dict_json_path, class_mapping_path
        )
        
        # OTHER class should be in invalid list
        assert len(report.invalid_class_ids) >= 1


# =============================================================================
# Tests: SemanticClosureReport
# =============================================================================

class TestSemanticClosureReport:
    """Tests for SemanticClosureReport dataclass."""
    
    def test_summary_shows_statistics(self):
        """Summary should include key statistics."""
        report = SemanticClosureReport(
            total_new_classes=10,
            valid_classes=8,
            invalid_class_ids=[4, 9],
            valid_glosses=["hola", "adios", "gracias"]
        )
        
        summary = report.get_summary()
        
        assert "10" in summary  # total
        assert "8" in summary   # valid
        assert "2" in summary or "invalid" in summary.lower()


# =============================================================================
# Tests: DictIdDataset
# =============================================================================

class TestDictIdDataset:
    """Tests for DictIdDataset wrapper."""
    
    def test_uses_old_class_id_from_dict(self, mock_base_dataset, sample_dict_json):
        """Should remap class_ids to dict.json IDs."""
        gloss_to_dict_id = {v["gloss"]: int(k) for k, v in sample_dict_json.items()}
        
        wrapped = DictIdDataset(mock_base_dataset, gloss_to_dict_id)
        
        sample = wrapped[0]  # "hola"
        assert sample["class_id"] == 0  # dict.json ID for "hola"
    
    def test_preserves_other_sample_fields(self, mock_base_dataset, sample_dict_json):
        """Should preserve keypoints and other fields."""
        gloss_to_dict_id = {v["gloss"]: int(k) for k, v in sample_dict_json.items()}
        
        wrapped = DictIdDataset(mock_base_dataset, gloss_to_dict_id)
        
        sample = wrapped[0]
        assert "keypoints" in sample
        assert sample["keypoints"] == [1, 2, 3]


# =============================================================================
# Tests: FilteredGlossDataset
# =============================================================================

class TestFilteredGlossDataset:
    """Tests for FilteredGlossDataset wrapper."""
    
    def test_filters_to_valid_glosses_only(self, mock_base_dataset):
        """Should only include samples with valid glosses."""
        valid_glosses = {"hola", "adios"}  # Not including "gracias" or "desconocido"
        
        filtered = FilteredGlossDataset(mock_base_dataset, valid_glosses)
        
        # Original has 5 samples, should have 3 (2 hola + 1 adios)
        assert len(filtered) == 3
    
    def test_excludes_unknown_glosses(self, mock_base_dataset):
        """Should exclude glosses not in valid set."""
        valid_glosses = {"hola", "adios", "gracias"}
        
        filtered = FilteredGlossDataset(mock_base_dataset, valid_glosses)
        
        # Should exclude "desconocido"
        for i in range(len(filtered)):
            sample = filtered[i]
            assert sample["gloss"] != "desconocido"
    
    def test_remaps_class_ids_when_enabled(self, mock_base_dataset):
        """Should remap class_ids to contiguous range when remap_ids=True."""
        valid_glosses = {"hola", "adios"}
        
        filtered = FilteredGlossDataset(
            mock_base_dataset, 
            valid_glosses, 
            remap_ids=True
        )
        
        # Class IDs should be 0, 1 (contiguous)
        class_ids = set()
        for i in range(len(filtered)):
            class_ids.add(filtered[i]["class_id"])
        
        assert class_ids == {0, 1}


# =============================================================================
# Tests: LowSupportFilter
# =============================================================================

class TestLowSupportFilter:
    """Tests for LowSupportFilter class."""
    
    def test_filters_classes_below_threshold(self, mock_base_dataset):
        """Should remove classes with support below threshold."""
        # Class support: hola=2, adios=1, gracias=1, desconocido=1
        
        filter_obj = LowSupportFilter(min_support=2)
        filtered = filter_obj.filter(mock_base_dataset)
        
        # Only "hola" has support >= 2
        assert len(filtered) == 2  # 2 samples of "hola"
    
    def test_preserves_classes_above_threshold(self, mock_base_dataset):
        """Should keep classes with support >= threshold."""
        filter_obj = LowSupportFilter(min_support=1)
        filtered = filter_obj.filter(mock_base_dataset)
        
        # All classes have support >= 1
        assert len(filtered) == len(mock_base_dataset)
    
    def test_returns_removal_report(self, mock_base_dataset):
        """Should report which classes were removed."""
        filter_obj = LowSupportFilter(min_support=2)
        filtered, removed = filter_obj.filter_with_report(mock_base_dataset)
        
        assert "adios" in removed or len(removed) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestSemanticClosureIntegration:
    """Integration tests for the full semantic closure pipeline."""
    
    def test_full_pipeline_filters_correctly(
        self, 
        dict_json_path, 
        class_mapping_path,
        mock_base_dataset
    ):
        """Full pipeline should filter and remap correctly."""
        # Build whitelist
        whitelist, report = build_semantic_whitelist(
            dict_json_path, class_mapping_path
        )
        
        # Get valid glosses from dict
        dict_mapping = load_dict_mapping(dict_json_path)
        valid_glosses = set(dict_mapping.values())
        
        # Filter dataset
        filtered = FilteredGlossDataset(
            mock_base_dataset, 
            valid_glosses,
            remap_ids=True
        )
        
        # All remaining samples should have valid glosses
        for i in range(len(filtered)):
            sample = filtered[i]
            assert sample["gloss"] in valid_glosses
