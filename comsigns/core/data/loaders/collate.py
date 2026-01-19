"""
Collate functions for batching variable-length keypoint sequences.

This module provides a pure, generic collate_fn that works with any dataset
implementing BaseDataset and returning EncoderReadySample instances.

Design Principles:
- Pure function: No side effects, no state
- Generic: Works with any EncoderReadySample, not coupled to AEC
- Infers dimensions from data (no hardcoding 168, 132, 1872)
- Returns explicit lengths for masking in attention layers

Usage:
    from torch.utils.data import DataLoader
    from comsigns.core.data.loaders import encoder_collate_fn
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=encoder_collate_fn
    )
    
    for batch in dataloader:
        hand = batch["hand"]      # [B, T_max, 168]
        body = batch["body"]      # [B, T_max, 132]
        face = batch["face"]      # [B, T_max, 1872]
        labels = batch["labels"]  # [B]
        lengths = batch["lengths"] # [B]
        mask = batch["mask"]      # [B, T_max] (optional, for attention)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TypedDict, Protocol, runtime_checkable
import numpy as np
import torch


# Type definitions for better IDE support and documentation
class EncoderBatch(TypedDict):
    """
    Type definition for the batch dictionary returned by encoder_collate_fn.
    
    Attributes:
        hand: Padded hand keypoints tensor [batch, T_max, hand_dim]
        body: Padded body keypoints tensor [batch, T_max, body_dim]
        face: Padded face keypoints tensor [batch, T_max, face_dim]
        labels: Class labels tensor [batch]
        lengths: Original sequence lengths before padding [batch]
        mask: Boolean attention mask [batch, T_max], True for valid positions
    """
    hand: torch.Tensor
    body: torch.Tensor
    face: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor
    mask: torch.Tensor


@runtime_checkable
class EncoderReadySampleProtocol(Protocol):
    """
    Protocol defining the minimum interface required for samples.
    
    This allows the collate function to work with any sample type
    that provides these attributes, not just EncoderReadySample.
    """
    gloss_id: int
    hand_keypoints: np.ndarray
    body_keypoints: np.ndarray
    face_keypoints: np.ndarray
    
    @property
    def num_frames(self) -> int: ...


def _pad_sequence(
    arrays: List[np.ndarray],
    max_len: int,
    pad_value: float = 0.0,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Pad a list of variable-length arrays to a common length.
    
    This is a pure helper function with no side effects.
    
    Args:
        arrays: List of numpy arrays with shape [T_i, D] where T_i varies
        max_len: Target length T_max to pad to
        pad_value: Value to use for padding (default 0.0)
        dtype: Numpy dtype for the output array (default float32)
    
    Returns:
        Numpy array of shape [len(arrays), max_len, D]
    
    Example:
        >>> arr1 = np.ones((3, 10))  # T=3, D=10
        >>> arr2 = np.ones((5, 10))  # T=5, D=10
        >>> padded = _pad_sequence([arr1, arr2], max_len=5)
        >>> padded.shape
        (2, 5, 10)
    """
    if not arrays:
        raise ValueError("Cannot pad empty list of arrays")
    
    # Infer feature dimension from first array
    feature_dim = arrays[0].shape[1] if arrays[0].ndim > 1 else arrays[0].shape[0]
    batch_size = len(arrays)
    
    # Pre-allocate output array filled with pad_value
    padded = np.full(
        (batch_size, max_len, feature_dim),
        fill_value=pad_value,
        dtype=dtype
    )
    
    # Copy actual values
    for i, arr in enumerate(arrays):
        seq_len = arr.shape[0]
        padded[i, :seq_len, :] = arr
    
    return padded


def _create_attention_mask(lengths: List[int], max_len: int) -> np.ndarray:
    """
    Create boolean attention mask from sequence lengths.
    
    The mask is True for valid (non-padded) positions and False for padding.
    This follows the PyTorch convention where True means "attend to this position".
    
    Args:
        lengths: List of original sequence lengths
        max_len: Maximum sequence length (T_max)
    
    Returns:
        Boolean numpy array of shape [batch_size, max_len]
        True for valid positions, False for padding
    
    Example:
        >>> mask = _create_attention_mask([3, 5, 2], max_len=5)
        >>> mask
        array([[ True,  True,  True, False, False],
               [ True,  True,  True,  True,  True],
               [ True,  True, False, False, False]])
    """
    batch_size = len(lengths)
    mask = np.zeros((batch_size, max_len), dtype=bool)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask


def encoder_collate_fn(
    batch: List[Any],
    pad_value: float = 0.0,
    include_mask: bool = True
) -> EncoderBatch:
    """
    Collate function for batching EncoderReadySample instances.
    
    This is the main collate function to use with torch.utils.data.DataLoader.
    It handles variable-length temporal sequences by padding to the maximum
    length in the batch.
    
    PURE FUNCTION: No side effects, no external dependencies, easily testable.
    
    Args:
        batch: List of samples, each must have:
               - gloss_id: int
               - hand_keypoints: np.ndarray [T, hand_dim]
               - body_keypoints: np.ndarray [T, body_dim]
               - face_keypoints: np.ndarray [T, face_dim]
               - num_frames property: int
        pad_value: Value to use for padding (default 0.0)
        include_mask: Whether to include attention mask in output (default True)
    
    Returns:
        EncoderBatch dictionary with:
        - hand: Tensor [batch_size, T_max, hand_dim]
        - body: Tensor [batch_size, T_max, body_dim]
        - face: Tensor [batch_size, T_max, face_dim]
        - labels: Tensor [batch_size] (int64 for CrossEntropyLoss)
        - lengths: Tensor [batch_size] (int64)
        - mask: Tensor [batch_size, T_max] (bool, only if include_mask=True)
    
    Raises:
        ValueError: If batch is empty
        AttributeError: If samples don't have required attributes
    
    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=encoder_collate_fn)
        >>> batch = next(iter(loader))
        >>> batch["hand"].shape  # [32, T_max, 168]
    
    Performance Notes:
        - Pre-allocates output arrays to minimize memory allocations
        - Uses numpy operations for efficiency before converting to torch
        - Converts to torch tensors only once at the end
    """
    if not batch:
        raise ValueError("Cannot collate empty batch")
    
    # Extract data from samples
    hand_arrays: List[np.ndarray] = []
    body_arrays: List[np.ndarray] = []
    face_arrays: List[np.ndarray] = []
    labels: List[int] = []
    lengths: List[int] = []
    
    for sample in batch:
        # Validate sample has required attributes
        if not hasattr(sample, 'hand_keypoints'):
            raise AttributeError(
                f"Sample missing 'hand_keypoints'. Got type: {type(sample).__name__}"
            )
        
        hand_arrays.append(sample.hand_keypoints)
        body_arrays.append(sample.body_keypoints)
        face_arrays.append(sample.face_keypoints)
        labels.append(sample.gloss_id)
        lengths.append(sample.num_frames)
    
    # Compute max length for this batch
    max_len = max(lengths)
    
    # Pad all sequences
    hand_padded = _pad_sequence(hand_arrays, max_len, pad_value)
    body_padded = _pad_sequence(body_arrays, max_len, pad_value)
    face_padded = _pad_sequence(face_arrays, max_len, pad_value)
    
    # Create attention mask
    mask_array = _create_attention_mask(lengths, max_len)
    
    # Convert to PyTorch tensors
    result: EncoderBatch = {
        "hand": torch.from_numpy(hand_padded).float(),
        "body": torch.from_numpy(body_padded).float(),
        "face": torch.from_numpy(face_padded).float(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "mask": torch.from_numpy(mask_array) if include_mask else torch.empty(0),
    }
    
    return result


def create_encoder_collate_fn(
    pad_value: float = 0.0,
    include_mask: bool = True
):
    """
    Factory function to create a configured collate_fn.
    
    Use this when you need to customize padding behavior.
    
    Args:
        pad_value: Value to use for padding keypoints
        include_mask: Whether to include attention mask in batch output
    
    Returns:
        Configured collate function compatible with DataLoader
    
    Example:
        >>> collate_fn = create_encoder_collate_fn(pad_value=-1.0)
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    def collate_fn(batch: List[Any]) -> EncoderBatch:
        return encoder_collate_fn(batch, pad_value=pad_value, include_mask=include_mask)
    
    return collate_fn


# Convenience aliases for common use cases
default_collate = encoder_collate_fn
