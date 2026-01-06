"""
Unit tests for AEC dataset module.

Tests cover:
- Path resolver functionality
- Converter functions (pure/stateless)
- Dataset loading and indexing
- Gloss vocabulary mappings
- Shape compatibility with encoder
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch, MagicMock

from comsigns.core.data.datasets.base import BaseDataset, KeypointResolverInterface
from comsigns.core.data.datasets.sample import Sample, EncoderReadySample
from comsigns.core.data.datasets.aec.resolver import AECKeypointResolver
from comsigns.core.data.datasets.aec.converters import (
    aec_frame_to_encoder_arrays,
    aec_keypoints_to_encoder_format,
    validate_aec_frame,
    _xy_to_xyconf
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_aec_frame() -> Dict:
    """Create a sample AEC frame with realistic structure."""
    return {
        'pose': {
            'x': [0.5] * 33,
            'y': [0.5] * 33
        },
        'left_hand': {
            'x': [0.3] * 21,
            'y': [0.3] * 21
        },
        'right_hand': {
            'x': [0.7] * 21,
            'y': [0.7] * 21
        },
        'face': {
            'x': [0.5] * 468,
            'y': [0.5] * 468
        }
    }


@pytest.fixture
def sample_aec_frames(sample_aec_frame) -> List[Dict]:
    """Create a list of sample AEC frames."""
    return [sample_aec_frame for _ in range(5)]


@pytest.fixture
def mock_dataset_root(tmp_path) -> Path:
    """Create a mock dataset directory structure."""
    keypoints_dir = tmp_path / "Keypoints" / "pkl" / "test_video"
    keypoints_dir.mkdir(parents=True)
    return tmp_path


# ============================================================================
# Resolver Tests
# ============================================================================

class TestAECKeypointResolver:
    """Tests for AECKeypointResolver."""
    
    def test_resolve_standard_path(self, mock_dataset_root):
        """Test resolving a standard AEC path."""
        resolver = AECKeypointResolver(mock_dataset_root)
        raw_path = "./Data/AEC/Keypoints/pkl/ira_alegria/yo_1.pkl"
        
        resolved = resolver.resolve(raw_path)
        
        assert resolved == mock_dataset_root / "Keypoints/pkl/ira_alegria/yo_1.pkl"
    
    def test_resolve_with_backslashes(self, mock_dataset_root):
        """Test that backslashes are normalized."""
        resolver = AECKeypointResolver(mock_dataset_root)
        raw_path = ".\\Data\\AEC\\Keypoints\\pkl\\test\\sample.pkl"
        
        resolved = resolver.resolve(raw_path)
        
        assert "Keypoints" in str(resolved)
        assert "\\" not in str(resolved)
    
    def test_exists_returns_false_for_missing(self, mock_dataset_root):
        """Test exists() returns False for missing files."""
        resolver = AECKeypointResolver(mock_dataset_root)
        
        result = resolver.exists("./Data/AEC/Keypoints/pkl/nonexistent.pkl")
        
        assert result is False
    
    def test_custom_prefix(self, mock_dataset_root):
        """Test resolver with custom prefix."""
        resolver = AECKeypointResolver(
            mock_dataset_root, 
            prefix_to_strip="./custom/"
        )
        raw_path = "./custom/some/path.pkl"
        
        resolved = resolver.resolve(raw_path)
        
        assert resolved == mock_dataset_root / "some/path.pkl"


# ============================================================================
# Converter Tests
# ============================================================================

class TestConverters:
    """Tests for stateless converter functions."""
    
    def test_xy_to_xyconf_basic(self):
        """Test basic x,y to x,y,z,conf conversion."""
        x = [0.1, 0.2]
        y = [0.3, 0.4]
        
        result = _xy_to_xyconf(x, y)
        
        assert result.shape == (8,)  # 2 points * 4 values
        assert result.dtype == np.float32
        # First point: x=0.1, y=0.3, z=0.0, conf=1.0
        assert result[0] == pytest.approx(0.1)
        assert result[1] == pytest.approx(0.3)
        assert result[2] == pytest.approx(0.0)  # default z
        assert result[3] == pytest.approx(1.0)  # default conf
    
    def test_aec_frame_to_encoder_arrays_shapes(self, sample_aec_frame):
        """Test that converter produces correct shapes."""
        result = aec_frame_to_encoder_arrays(sample_aec_frame)
        
        assert result['hand'].shape == (168,)  # 2 hands * 21 kp * 4
        assert result['body'].shape == (132,)  # 33 kp * 4
        assert result['face'].shape == (1872,)  # 468 kp * 4
    
    def test_aec_frame_to_encoder_arrays_dtype(self, sample_aec_frame):
        """Test that output is float32."""
        result = aec_frame_to_encoder_arrays(sample_aec_frame)
        
        assert result['hand'].dtype == np.float32
        assert result['body'].dtype == np.float32
        assert result['face'].dtype == np.float32
    
    def test_aec_keypoints_to_encoder_format(self, sample_aec_frames):
        """Test conversion of multiple frames."""
        result = aec_keypoints_to_encoder_format(sample_aec_frames)
        
        T = len(sample_aec_frames)
        assert result['hand'].shape == (T, 168)
        assert result['body'].shape == (T, 132)
        assert result['face'].shape == (T, 1872)
    
    def test_empty_frames_raises_error(self):
        """Test that empty frames list raises ValueError."""
        with pytest.raises(ValueError, match="empty frames"):
            aec_keypoints_to_encoder_format([])
    
    def test_converter_is_pure(self, sample_aec_frame):
        """Test that converter has no side effects (pure function)."""
        original_frame = {
            k: {kk: list(vv) for kk, vv in v.items()}
            for k, v in sample_aec_frame.items()
        }
        
        # Call converter
        aec_frame_to_encoder_arrays(sample_aec_frame)
        
        # Verify frame was not modified
        assert sample_aec_frame == original_frame


class TestValidateAECFrame:
    """Tests for frame validation function."""
    
    def test_valid_frame(self, sample_aec_frame):
        """Test validation of valid frame."""
        assert validate_aec_frame(sample_aec_frame) is True
    
    def test_missing_key(self, sample_aec_frame):
        """Test validation fails with missing key."""
        del sample_aec_frame['face']
        
        assert validate_aec_frame(sample_aec_frame) is False
    
    def test_wrong_size(self, sample_aec_frame):
        """Test validation fails with wrong keypoint count."""
        sample_aec_frame['pose']['x'] = [0.5] * 30  # Should be 33
        
        assert validate_aec_frame(sample_aec_frame) is False
    
    def test_not_dict(self):
        """Test validation fails for non-dict input."""
        assert validate_aec_frame("not a dict") is False
        assert validate_aec_frame(None) is False


# ============================================================================
# Sample Dataclass Tests
# ============================================================================

class TestSampleDataclasses:
    """Tests for Sample and EncoderReadySample dataclasses."""
    
    def test_sample_num_frames(self, sample_aec_frames):
        """Test Sample.num_frames property."""
        sample = Sample(
            gloss="test",
            keypoints=sample_aec_frames,
            frame_start=0,
            frame_end=4,
            unique_name="test_1"
        )
        
        assert sample.num_frames == 5
    
    def test_encoder_ready_sample_to_tensors(self):
        """Test EncoderReadySample.to_tensors() method."""
        hand = np.zeros((5, 168), dtype=np.float32)
        body = np.zeros((5, 132), dtype=np.float32)
        face = np.zeros((5, 1872), dtype=np.float32)
        
        sample = EncoderReadySample(
            gloss="test",
            gloss_id=0,
            hand_keypoints=hand,
            body_keypoints=body,
            face_keypoints=face,
            unique_name="test_1"
        )
        
        tensors = sample.to_tensors()
        
        assert 'hand' in tensors
        assert 'body' in tensors
        assert 'face' in tensors
        np.testing.assert_array_equal(tensors['hand'], hand)


# ============================================================================
# Integration Tests (with real dataset if available)
# ============================================================================

class TestAECDatasetIntegration:
    """Integration tests that run only if the dataset is available."""
    
    @pytest.fixture
    def dataset_path(self):
        """Get the dataset path, skip if not available."""
        path = Path("data/raw/lsp_aec")
        if not path.exists():
            pytest.skip("AEC dataset not available")
        return path
    
    def test_dataset_loads(self, dataset_path):
        """Test that dataset loads successfully."""
        from comsigns.core.data.datasets.aec import AECDataset
        
        dataset = AECDataset(dataset_path)
        
        assert len(dataset) > 0
        assert dataset.num_classes > 0
    
    def test_sample_shapes(self, dataset_path):
        """Test that samples have correct shapes."""
        from comsigns.core.data.datasets.aec import AECDataset
        
        dataset = AECDataset(dataset_path)
        sample = dataset[0]
        
        assert sample.hand_keypoints.shape[1] == 168
        assert sample.body_keypoints.shape[1] == 132
        assert sample.face_keypoints.shape[1] == 1872
    
    def test_gloss_mapping_consistency(self, dataset_path):
        """Test that gloss mappings are consistent."""
        from comsigns.core.data.datasets.aec import AECDataset
        
        dataset = AECDataset(dataset_path)
        
        for gloss, gloss_id in dataset.gloss_to_id.items():
            assert dataset.id_to_gloss[gloss_id] == gloss
