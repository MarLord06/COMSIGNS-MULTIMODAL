"""
Unit tests for video inference pipeline.

Tests keypoint extraction and video preprocessing.
"""

import pytest
import numpy as np
from pathlib import Path


# ============================================================
# Keypoint Extractor Tests
# ============================================================

class TestKeypointExtractorConstants:
    """Test keypoint dimension constants."""
    
    def test_hand_dimensions(self):
        """Hand keypoints should be 21 landmarks * 2 hands * 4 coords."""
        from backend.services.keypoint_extractor import HAND_DIM, HAND_LANDMARKS
        assert HAND_LANDMARKS == 21
        assert HAND_DIM == 21 * 2 * 4  # 168
    
    def test_body_dimensions(self):
        """Body keypoints should be 33 landmarks * 4 coords."""
        from backend.services.keypoint_extractor import BODY_DIM, POSE_LANDMARKS
        assert POSE_LANDMARKS == 33
        assert BODY_DIM == 33 * 4  # 132
    
    def test_face_dimensions(self):
        """Face keypoints should be 468 landmarks * 4 coords."""
        from backend.services.keypoint_extractor import FACE_DIM, FACE_LANDMARKS
        assert FACE_LANDMARKS == 468
        assert FACE_DIM == 468 * 4  # 1872


class TestKeypointExtractor:
    """Test KeypointExtractor class."""
    
    def test_extractor_creation(self):
        """KeypointExtractor should be creatable."""
        from backend.services.keypoint_extractor import KeypointExtractor
        extractor = KeypointExtractor()
        assert extractor is not None
        assert extractor.min_detection_confidence == 0.5
        assert extractor.min_tracking_confidence == 0.5
    
    def test_singleton_getter(self):
        """get_keypoint_extractor should return singleton."""
        from backend.services.keypoint_extractor import get_keypoint_extractor
        
        extractor1 = get_keypoint_extractor()
        extractor2 = get_keypoint_extractor()
        
        assert extractor1 is extractor2
    
    def test_extract_from_black_frame(self):
        """Extraction from black frame should return zeros with correct dims."""
        from backend.services.keypoint_extractor import (
            KeypointExtractor, HAND_DIM, BODY_DIM, FACE_DIM
        )
        
        extractor = KeypointExtractor()
        
        # Black frame (no features to detect)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.extract_from_frame(frame)
        
        assert "hand" in result
        assert "body" in result
        assert "face" in result
        
        # Critical: verify exact dimensions match model expectations
        assert result["hand"].shape == (HAND_DIM,), f"Expected hand dim {HAND_DIM}, got {result['hand'].shape}"
        assert result["body"].shape == (BODY_DIM,), f"Expected body dim {BODY_DIM}, got {result['body'].shape}"
        assert result["face"].shape == (FACE_DIM,), f"Expected face dim {FACE_DIM}, got {result['face'].shape}"
        
        # Verify exact values: 168, 132, 1872
        assert HAND_DIM == 168
        assert BODY_DIM == 132
        assert FACE_DIM == 1872
        
        # Black frame should produce zeros (no detections)
        assert np.sum(result["hand"]) == 0  # No hands detected
    
    def test_extract_face_never_exceeds_bounds(self):
        """Face extraction should never access index >= 468."""
        from backend.services.keypoint_extractor import (
            KeypointExtractor, FACE_LANDMARKS, FACE_DIM
        )
        
        extractor = KeypointExtractor()
        
        # Random noise frame (may detect partial faces)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = extractor.extract_from_frame(frame)
        
        # Must always return correct shape, never IndexError
        assert result["face"].shape == (FACE_DIM,)
        assert FACE_LANDMARKS == 468


# ============================================================
# Video Preprocessor Tests
# ============================================================

class TestVideoPreprocessor:
    """Test VideoPreprocessor class."""
    
    def test_preprocessor_creation(self):
        """VideoPreprocessor should be creatable."""
        from backend.services.video_preprocess import VideoPreprocessor
        
        preprocessor = VideoPreprocessor(max_frames=100, min_frames=3)
        assert preprocessor.max_frames == 100
        assert preprocessor.min_frames == 3
    
    def test_singleton_getter(self):
        """get_video_preprocessor should return instance."""
        from backend.services.video_preprocess import get_video_preprocessor
        
        preprocessor = get_video_preprocessor()
        assert preprocessor is not None
        assert preprocessor.max_frames == 150  # Default
    
    def test_invalid_video_source_type(self):
        """Processing invalid source type should raise ValueError."""
        from backend.services.video_preprocess import VideoPreprocessor
        
        preprocessor = VideoPreprocessor()
        
        with pytest.raises(ValueError, match="Unsupported video source type"):
            preprocessor.process_video(12345)  # Invalid type


# ============================================================
# Video Route Tests  
# ============================================================

class TestVideoRouteConfig:
    """Test video route configuration."""
    
    def test_allowed_extensions(self):
        """Should have standard video extensions."""
        from backend.api.routes.video import ALLOWED_EXTENSIONS
        
        assert ".mp4" in ALLOWED_EXTENSIONS
        assert ".mov" in ALLOWED_EXTENSIONS
        assert ".avi" in ALLOWED_EXTENSIONS
    
    def test_video_constraints(self):
        """Should have reasonable constraints."""
        from backend.api.routes.video import (
            MAX_VIDEO_SIZE_MB, MIN_DURATION_SEC, MAX_DURATION_SEC
        )
        
        assert MAX_VIDEO_SIZE_MB > 0
        assert MIN_DURATION_SEC > 0
        assert MAX_DURATION_SEC > MIN_DURATION_SEC


class TestVideoValidation:
    """Test video validation functions."""
    
    def test_valid_mp4_file(self):
        """Valid .mp4 file should pass validation."""
        from backend.api.routes.video import validate_video_file
        
        # Should not raise
        validate_video_file("test.mp4", 1024 * 1024)  # 1MB
    
    def test_invalid_extension(self):
        """Invalid extension should raise HTTPException."""
        from backend.api.routes.video import validate_video_file
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            validate_video_file("test.txt", 1024)
        
        assert exc_info.value.status_code == 400
        assert "Invalid file type" in exc_info.value.detail
    
    def test_file_too_large(self):
        """Too large file should raise HTTPException."""
        from backend.api.routes.video import validate_video_file, MAX_VIDEO_SIZE_MB
        from fastapi import HTTPException
        
        large_size = (MAX_VIDEO_SIZE_MB + 1) * 1024 * 1024
        
        with pytest.raises(HTTPException) as exc_info:
            validate_video_file("test.mp4", large_size)
        
        assert exc_info.value.status_code == 400
        assert "too large" in exc_info.value.detail


# ============================================================
# Integration Tests (require model)
# ============================================================

@pytest.mark.integration
class TestVideoInferenceIntegration:
    """Integration tests for full video inference pipeline.
    
    These tests require the model to be available.
    Run with: pytest -m integration
    """
    
    @pytest.fixture
    def sample_video_path(self):
        """Path to a test video if available."""
        # This would need to point to an actual test video
        return None
    
    def test_full_pipeline_placeholder(self, sample_video_path):
        """Placeholder for full pipeline test."""
        if sample_video_path is None:
            pytest.skip("No test video available")
