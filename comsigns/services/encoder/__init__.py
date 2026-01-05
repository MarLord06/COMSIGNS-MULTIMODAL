"""
Módulo de encoder multimodal: tres ramas (manos, cuerpo, rostro)
"""

from .model import MultimodalEncoder, create_encoder
from .utils import feature_clip_to_tensors, keypoints_to_tensor

__all__ = [
    'MultimodalEncoder',
    'create_encoder',
    'feature_clip_to_tensors',
    'keypoints_to_tensor',
]

