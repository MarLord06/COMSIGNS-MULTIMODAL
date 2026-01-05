"""
Tests unitarios para esquemas Pydantic
"""

import pytest
from comsigns.services.schemas import (
    Keypoint,
    FrameKeypoints,
    FeatureClip,
    ClipMetadata,
    VideoManifest
)


class TestKeypoint:
    """Tests para Keypoint"""

    def test_valid_keypoint(self):
        """Test keypoint válido"""
        kp = Keypoint(x=0.5, y=0.5, confidence=0.9)
        assert kp.x == 0.5
        assert kp.y == 0.5
        assert kp.confidence == 0.9

    def test_invalid_coordinates(self):
        """Test coordenadas fuera de rango"""
        with pytest.raises(ValueError):
            Keypoint(x=1.5, y=0.5, confidence=0.9)

        with pytest.raises(ValueError):
            Keypoint(x=0.5, y=-0.1, confidence=0.9)


class TestFrameKeypoints:
    """Tests para FrameKeypoints"""

    def test_valid_frame(self):
        """Test frame válido"""
        frame = FrameKeypoints(
            t=0.033,
            hand_keypoints=[[0.5, 0.5, 0.1, 0.9]],
            body_keypoints=[[0.5, 0.5, 0.1, 0.8]],
            face_keypoints=[[0.5, 0.5, 0.1, 0.9]]
        )
        assert frame.t == 0.033
        assert len(frame.hand_keypoints) == 1

    def test_invalid_keypoint_format(self):
        """Test formato de keypoint inválido"""
        with pytest.raises(ValueError):
            FrameKeypoints(
                t=0.033,
                hand_keypoints=[[0.5, 0.5]],  # Falta confidence
                body_keypoints=[],
                face_keypoints=[]
            )


class TestFeatureClip:
    """Tests para FeatureClip"""

    def test_valid_clip(self):
        """Test clip válido"""
        frame = FrameKeypoints(
            t=0.0,
            hand_keypoints=[[0.5, 0.5, 0.1, 0.9]],
            body_keypoints=[[0.5, 0.5, 0.1, 0.8]],
            face_keypoints=[[0.5, 0.5, 0.1, 0.9]]
        )

        clip = FeatureClip(
            clip_id="test-123",
            fps=30.0,
            frames=[frame],
            meta=ClipMetadata()
        )

        assert clip.clip_id == "test-123"
        assert clip.fps == 30.0
        assert len(clip.frames) == 1

    def test_invalid_fps(self):
        """Test FPS inválido"""
        frame = FrameKeypoints(
            t=0.0,
            hand_keypoints=[],
            body_keypoints=[],
            face_keypoints=[]
        )

        with pytest.raises(ValueError):
            FeatureClip(
                clip_id="test-123",
                fps=0.0,  # FPS inválido
                frames=[frame]
            )

    def test_frames_ordering(self):
        """Test que los frames deben estar ordenados"""
        frame1 = FrameKeypoints(t=0.1, hand_keypoints=[], body_keypoints=[], face_keypoints=[])
        frame2 = FrameKeypoints(t=0.0, hand_keypoints=[], body_keypoints=[], face_keypoints=[])

        with pytest.raises(ValueError):
            FeatureClip(
                clip_id="test-123",
                fps=30.0,
                frames=[frame1, frame2]  # Desordenados
            )

    def test_to_dict_from_dict(self):
        """Test conversión a/desde diccionario"""
        frame = FrameKeypoints(
            t=0.0,
            hand_keypoints=[[0.5, 0.5, 0.1, 0.9]],
            body_keypoints=[],
            face_keypoints=[]
        )

        clip = FeatureClip(
            clip_id="test-123",
            fps=30.0,
            frames=[frame]
        )

        # Convertir a dict
        clip_dict = clip.to_dict()
        assert isinstance(clip_dict, dict)
        assert clip_dict['clip_id'] == "test-123"

        # Recrear desde dict
        clip_restored = FeatureClip.from_dict(clip_dict)
        assert clip_restored.clip_id == clip.clip_id
        assert clip_restored.fps == clip.fps

