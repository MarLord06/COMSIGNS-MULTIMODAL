"""
Core data structures and interfaces for sign language datasets.

This module provides:
- BaseDataset: Abstract base class for dataset implementations
- KeypointResolverInterface: Interface for path resolution
- Sample: Raw sample representation
- EncoderReadySample: Encoder-compatible sample format
"""

from .base import BaseDataset, KeypointResolverInterface
from .sample import Sample, EncoderReadySample

__all__ = [
    'BaseDataset',
    'KeypointResolverInterface',
    'Sample',
    'EncoderReadySample',
]
