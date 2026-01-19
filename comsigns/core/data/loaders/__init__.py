"""
Data loaders and collate functions for PyTorch DataLoader integration.

This module provides utilities for batching variable-length sequences
from sign language datasets in a way that's compatible with the
MultimodalEncoder pipeline.
"""

from .collate import (
    encoder_collate_fn,
    create_encoder_collate_fn,
    EncoderBatch,
)

__all__ = [
    "encoder_collate_fn",
    "create_encoder_collate_fn", 
    "EncoderBatch",
]
