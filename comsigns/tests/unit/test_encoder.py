"""
Tests unitarios para encoder multimodal
"""

import pytest
import torch

from comsigns.services.encoder.model import (
    MultimodalEncoder,
    HandBranch,
    BodyBranch,
    FaceBranch,
    create_encoder
)
from comsigns.services.encoder.utils import keypoints_to_tensor, feature_clip_to_tensors
from comsigns.services.schemas import FeatureClip, FrameKeypoints, ClipMetadata


class TestEncoderBranches:
    """Tests para las ramas del encoder"""

    def test_hand_branch(self):
        """Test rama de manos"""
        branch = HandBranch(input_dim=21 * 4, hidden_dim=128, num_layers=1)
        x = torch.randn(2, 10, 21 * 4)  # (batch, seq_len, input_dim)
        output = branch(x)
        assert output.shape == (2, 10, 128)

    def test_body_branch(self):
        """Test rama del cuerpo"""
        branch = BodyBranch(input_dim=33 * 4, hidden_dim=128, num_layers=1)
        x = torch.randn(2, 10, 33 * 4)
        output = branch(x)
        assert output.shape == (2, 10, 128)

    def test_face_branch(self):
        """Test rama del rostro"""
        branch = FaceBranch(input_dim=468 * 4, hidden_dim=128, num_layers=1)
        x = torch.randn(2, 10, 468 * 4)
        output = branch(x)
        assert output.shape == (2, 10, 128)


class TestMultimodalEncoder:
    """Tests para el encoder multimodal completo"""

    def test_encoder_forward(self):
        """Test forward pass del encoder"""
        encoder = MultimodalEncoder(
            hand_input_dim=21 * 4,
            body_input_dim=33 * 4,
            face_input_dim=468 * 4,
            hidden_dim=128,
            output_dim=256,
            num_layers=1
        )

        batch_size = 2
        seq_len = 10

        hand_kp = torch.randn(batch_size, seq_len, 21 * 4)
        body_kp = torch.randn(batch_size, seq_len, 33 * 4)
        face_kp = torch.randn(batch_size, seq_len, 468 * 4)

        output = encoder(hand_kp, body_kp, face_kp)

        assert output.shape == (batch_size, seq_len, 256)

    def test_encoder_encode_features(self):
        """Test método encode_features"""
        encoder = MultimodalEncoder(
            hand_input_dim=21 * 4,
            body_input_dim=33 * 4,
            face_input_dim=468 * 4,
            hidden_dim=128,
            output_dim=256
        )

        features = {
            'hand': torch.randn(1, 10, 21 * 4),
            'body': torch.randn(1, 10, 33 * 4),
            'face': torch.randn(1, 10, 468 * 4)
        }

        output = encoder.encode_features(features)
        assert output.shape == (1, 10, 256)


class TestEncoderUtils:
    """Tests para utilidades del encoder"""

    def test_keypoints_to_tensor(self):
        """Test conversión de keypoints a tensor"""
        keypoints = [
            [0.5, 0.5, 0.1, 0.9],
            [0.6, 0.6, 0.2, 0.8]
        ]

        tensor = keypoints_to_tensor(keypoints)
        assert tensor.shape == (8,)  # 2 keypoints * 4 valores

    def test_keypoints_to_tensor_empty(self):
        """Test con lista vacía"""
        tensor = keypoints_to_tensor([])
        assert tensor.shape == (0, 4)

    def test_feature_clip_to_tensors(self):
        """Test conversión de FeatureClip a tensores"""
        # Crear FeatureClip de prueba
        frames = [
            FrameKeypoints(
                t=0.0,
                hand_keypoints=[[0.5, 0.5, 0.1, 0.9] * 21],  # 21 keypoints aplanados
                body_keypoints=[[0.5, 0.5, 0.1, 0.8] * 33],  # 33 keypoints
                face_keypoints=[[0.5, 0.5, 0.1, 0.9] * 468]  # 468 keypoints
            ),
            FrameKeypoints(
                t=0.033,
                hand_keypoints=[[0.6, 0.6, 0.2, 0.9] * 21],
                body_keypoints=[[0.6, 0.6, 0.2, 0.8] * 33],
                face_keypoints=[[0.6, 0.6, 0.2, 0.9] * 468]
            )
        ]

        feature_clip = FeatureClip(
            clip_id="test-123",
            fps=30.0,
            frames=frames,
            meta=ClipMetadata()
        )

        tensors = feature_clip_to_tensors(feature_clip)

        assert 'hand' in tensors
        assert 'body' in tensors
        assert 'face' in tensors

        # Verificar formas
        assert tensors['hand'].shape[0] == 2  # seq_len
        assert tensors['body'].shape[0] == 2
        assert tensors['face'].shape[0] == 2

