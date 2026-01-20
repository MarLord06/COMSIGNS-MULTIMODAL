"""
Dataset splitting utilities for train/validation/test splits.

Provides pure functions for splitting datasets without modifying
the original dataset class. Follows Clean Architecture principles.

Example:
    from comsigns.core.data.splits import create_train_val_split
    
    train_set, val_set = create_train_val_split(dataset, val_ratio=0.2)
"""

import torch
from torch.utils.data import Dataset, Subset, random_split
from typing import Tuple, Optional, List
import math


def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.2,
    seed: Optional[int] = 42
) -> Tuple[Subset, Subset]:
    """
    Split a dataset into training and validation subsets.
    
    Uses torch.utils.data.random_split for random splitting.
    The split is reproducible when seed is provided.
    
    Args:
        dataset: The full dataset to split
        val_ratio: Fraction of data for validation (0 < val_ratio < 1)
        seed: Random seed for reproducibility. None for random split.
    
    Returns:
        Tuple of (train_subset, val_subset)
    
    Raises:
        ValueError: If val_ratio is not in (0, 1)
    
    Example:
        >>> train_set, val_set = create_train_val_split(dataset, val_ratio=0.2)
        >>> len(train_set)  # 80% of dataset
        >>> len(val_set)    # 20% of dataset
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    
    total_size = len(dataset)
    val_size = int(math.floor(total_size * val_ratio))
    train_size = total_size - val_size
    
    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"Dataset too small for split. Total: {total_size}, "
            f"would get train: {train_size}, val: {val_size}"
        )
    
    # Create generator for reproducibility
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    
    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )
    
    return train_subset, val_subset


def create_train_val_test_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    Split a dataset into training, validation, and test subsets.
    
    Args:
        dataset: The full dataset to split
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for test (default 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    
    Example:
        >>> train, val, test = create_train_val_test_split(dataset)
        >>> # 80% train, 10% val, 10% test
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
    if val_ratio + test_ratio >= 1:
        raise ValueError(
            f"val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio}"
        )
    
    total_size = len(dataset)
    val_size = int(math.floor(total_size * val_ratio))
    test_size = int(math.floor(total_size * test_ratio))
    train_size = total_size - val_size - test_size
    
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    
    train_subset, val_subset, test_subset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    return train_subset, val_subset, test_subset


def get_split_indices(
    total_size: int,
    val_ratio: float = 0.2,
    seed: Optional[int] = 42
) -> Tuple[List[int], List[int]]:
    """
    Get indices for train/val split without creating subsets.
    
    Useful when you need direct access to indices for debugging
    or custom sampling.
    
    Args:
        total_size: Total number of samples
        val_ratio: Fraction for validation
        seed: Random seed
    
    Returns:
        Tuple of (train_indices, val_indices)
    """
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()
    
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    val_size = int(math.floor(total_size * val_ratio))
    train_size = total_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    return train_indices, val_indices
