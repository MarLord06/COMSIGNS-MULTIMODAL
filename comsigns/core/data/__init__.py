"""
Core data utilities for ComSigns.

Provides dataset loading, splitting, and batching utilities.
"""

from .splits import (
    create_train_val_split,
    create_train_val_test_split,
    get_split_indices,
)

__all__ = [
    "create_train_val_split",
    "create_train_val_test_split",
    "get_split_indices",
]
