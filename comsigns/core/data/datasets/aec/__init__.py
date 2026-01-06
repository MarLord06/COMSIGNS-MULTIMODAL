"""
AEC (Peruvian Sign Language) dataset module.

Provides the AECDataset class for loading and iterating over the
Lengua de Señas Peruana dataset from Asociación de Estudio del Conocimiento.

Example:
    >>> from comsigns.core.data.datasets.aec import AECDataset
    >>> from pathlib import Path
    >>> 
    >>> dataset = AECDataset(Path("data/raw/lsp_aec"))
    >>> sample = dataset[0]
    >>> print(f"Gloss: {sample.gloss}, Shape: {sample.hand_keypoints.shape}")
"""

from .aec_dataset import AECDataset
from .resolver import AECKeypointResolver
from .converters import (
    aec_frame_to_encoder_arrays,
    aec_keypoints_to_encoder_format,
    validate_aec_frame
)

__all__ = [
    'AECDataset',
    'AECKeypointResolver',
    'aec_frame_to_encoder_arrays',
    'aec_keypoints_to_encoder_format',
    'validate_aec_frame',
]
