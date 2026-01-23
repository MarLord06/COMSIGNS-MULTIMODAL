"""
Class Remapping Module for TAIL → OTHER Experiment.

This module provides tools to collapse long-tail classes into a single
"OTHER" class for controlled experimentation with class imbalance.

Key Components:
- RemapConfig: Configuration for remapping strategy
- ClassRemapper: Maps original class IDs to new collapsed IDs
- RemappedDataset: Dataset wrapper that applies remapping on-the-fly

Usage:
    >>> config = RemapConfig(strategy="tail_to_other")
    >>> remapper = ClassRemapper(config)
    >>> remapper.fit(train_support)
    >>> wrapped_dataset = RemappedDataset(original_dataset, remapper)
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class Bucket(Enum):
    """Support bucket classification."""
    HEAD = "HEAD"   # ≥ threshold samples
    MID = "MID"     # mid_range samples
    TAIL = "TAIL"   # ≤ tail_max samples


@dataclass
class RemapConfig:
    """Configuration for class remapping.
    
    Attributes:
        strategy: Remapping strategy ("tail_to_other" or "tail_exclude")
        head_threshold: Minimum samples for HEAD bucket (default: 10)
        mid_range: (min, max) samples for MID bucket (default: (3, 9))
        other_class_name: Name for the collapsed OTHER class
    """
    strategy: Literal["tail_to_other", "tail_exclude"] = "tail_to_other"
    head_threshold: int = 10
    mid_range: Tuple[int, int] = (3, 9)
    other_class_name: str = "OTHER"
    
    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy,
            "head_threshold": self.head_threshold,
            "mid_range": list(self.mid_range),
            "other_class_name": self.other_class_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RemapConfig":
        return cls(
            strategy=data["strategy"],
            head_threshold=data["head_threshold"],
            mid_range=tuple(data["mid_range"]),
            other_class_name=data["other_class_name"]
        )


class ClassRemapper:
    """Maps original class IDs to new collapsed IDs.
    
    This class handles the remapping of class indices when collapsing
    TAIL classes into a single OTHER class.
    
    The remapping is deterministic:
    - HEAD classes get indices 0, 1, 2, ... (sorted by original ID)
    - MID classes get indices after HEAD (sorted by original ID)
    - TAIL classes all map to the last index (OTHER)
    
    Example:
        >>> config = RemapConfig(strategy="tail_to_other")
        >>> remapper = ClassRemapper(config)
        >>> remapper.fit(train_support)  # {class_id: sample_count}
        >>> new_id = remapper.transform(old_id)
        >>> old_ids = remapper.inverse_transform(new_id)
    """
    
    def __init__(self, config: Optional[RemapConfig] = None):
        """Initialize remapper with configuration.
        
        Args:
            config: Remapping configuration. If None, uses defaults.
        """
        self.config = config or RemapConfig()
        
        # Mappings (populated by fit())
        self.old_to_new: Dict[int, int] = {}
        self.new_to_old: Dict[int, List[int]] = {}
        self.new_class_names: Dict[int, str] = {}
        
        # Bucket assignments
        self.bucket_to_old_ids: Dict[Bucket, List[int]] = {
            Bucket.HEAD: [],
            Bucket.MID: [],
            Bucket.TAIL: []
        }
        self.old_id_to_bucket: Dict[int, Bucket] = {}
        
        # Class info
        self.class_names: Dict[int, str] = {}  # Original class names
        self._fitted = False
        
        # Statistics
        self.num_classes_original: int = 0
        self.num_classes_remapped: int = 0
        self.other_class_id: int = -1
        self.classes_collapsed: int = 0
        self.samples_in_other: int = 0
    
    def _classify_bucket(self, support: int) -> Bucket:
        """Classify a class into a bucket based on support count."""
        if support >= self.config.head_threshold:
            return Bucket.HEAD
        elif self.config.mid_range[0] <= support <= self.config.mid_range[1]:
            return Bucket.MID
        else:
            return Bucket.TAIL
    
    def fit(
        self,
        class_support: Dict[int, int],
        class_names: Optional[Dict[int, str]] = None
    ) -> "ClassRemapper":
        """Build mapping from class support counts.
        
        Args:
            class_support: Dict mapping class_id to sample count in training set
            class_names: Optional dict mapping class_id to class name
        
        Returns:
            self for method chaining
        """
        self.class_names = class_names or {}
        self.num_classes_original = len(class_support)
        
        # Reset bucket assignments
        self.bucket_to_old_ids = {b: [] for b in Bucket}
        self.old_id_to_bucket = {}
        
        # Classify each class into bucket
        for old_id, support in class_support.items():
            bucket = self._classify_bucket(support)
            self.bucket_to_old_ids[bucket].append(old_id)
            self.old_id_to_bucket[old_id] = bucket
        
        # Sort for deterministic ordering
        for bucket in Bucket:
            self.bucket_to_old_ids[bucket].sort()
        
        # Build mapping based on strategy
        if self.config.strategy == "tail_to_other":
            self._build_tail_to_other_mapping(class_support)
        elif self.config.strategy == "tail_exclude":
            self._build_tail_exclude_mapping(class_support)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        self._fitted = True
        
        logger.info(
            f"ClassRemapper fitted: {self.num_classes_original} → "
            f"{self.num_classes_remapped} classes "
            f"(HEAD={len(self.bucket_to_old_ids[Bucket.HEAD])}, "
            f"MID={len(self.bucket_to_old_ids[Bucket.MID])}, "
            f"TAIL={len(self.bucket_to_old_ids[Bucket.TAIL])} → OTHER)"
        )
        
        return self
    
    def _build_tail_to_other_mapping(self, class_support: Dict[int, int]) -> None:
        """Build mapping that collapses TAIL into OTHER."""
        self.old_to_new = {}
        self.new_to_old = {}
        self.new_class_names = {}
        
        new_id = 0
        
        # HEAD classes: keep separate, new sequential IDs
        for old_id in self.bucket_to_old_ids[Bucket.HEAD]:
            self.old_to_new[old_id] = new_id
            self.new_to_old[new_id] = [old_id]
            self.new_class_names[new_id] = self.class_names.get(old_id, f"HEAD_{old_id}")
            new_id += 1
        
        # MID classes: keep separate, new sequential IDs
        for old_id in self.bucket_to_old_ids[Bucket.MID]:
            self.old_to_new[old_id] = new_id
            self.new_to_old[new_id] = [old_id]
            self.new_class_names[new_id] = self.class_names.get(old_id, f"MID_{old_id}")
            new_id += 1
        
        # TAIL classes: all map to OTHER (last index)
        self.other_class_id = new_id
        self.new_to_old[self.other_class_id] = []
        self.new_class_names[self.other_class_id] = self.config.other_class_name
        
        for old_id in self.bucket_to_old_ids[Bucket.TAIL]:
            self.old_to_new[old_id] = self.other_class_id
            self.new_to_old[self.other_class_id].append(old_id)
        
        # Statistics
        self.num_classes_remapped = new_id + 1  # +1 for OTHER
        self.classes_collapsed = len(self.bucket_to_old_ids[Bucket.TAIL])
        self.samples_in_other = sum(
            class_support.get(old_id, 0) 
            for old_id in self.bucket_to_old_ids[Bucket.TAIL]
        )
    
    def _build_tail_exclude_mapping(self, class_support: Dict[int, int]) -> None:
        """Build mapping that excludes TAIL entirely."""
        self.old_to_new = {}
        self.new_to_old = {}
        self.new_class_names = {}
        
        new_id = 0
        
        # HEAD classes
        for old_id in self.bucket_to_old_ids[Bucket.HEAD]:
            self.old_to_new[old_id] = new_id
            self.new_to_old[new_id] = [old_id]
            self.new_class_names[new_id] = self.class_names.get(old_id, f"HEAD_{old_id}")
            new_id += 1
        
        # MID classes
        for old_id in self.bucket_to_old_ids[Bucket.MID]:
            self.old_to_new[old_id] = new_id
            self.new_to_old[new_id] = [old_id]
            self.new_class_names[new_id] = self.class_names.get(old_id, f"MID_{old_id}")
            new_id += 1
        
        # TAIL classes: map to -1 (excluded)
        for old_id in self.bucket_to_old_ids[Bucket.TAIL]:
            self.old_to_new[old_id] = -1
        
        self.num_classes_remapped = new_id
        self.other_class_id = -1
        self.classes_collapsed = len(self.bucket_to_old_ids[Bucket.TAIL])
        self.samples_in_other = 0
    
    def transform(self, old_class_id: int) -> int:
        """Map old class ID to new class ID.
        
        Args:
            old_class_id: Original class ID
        
        Returns:
            New class ID (or -1 if excluded in tail_exclude strategy)
        
        Raises:
            ValueError: If class ID not found and remapper is fitted
        """
        if not self._fitted:
            raise RuntimeError("ClassRemapper not fitted. Call fit() first.")
        
        if old_class_id not in self.old_to_new:
            raise ValueError(f"Unknown class ID: {old_class_id}")
        
        return self.old_to_new[old_class_id]
    
    def inverse_transform(self, new_class_id: int) -> List[int]:
        """Get original class IDs for a new class ID.
        
        Args:
            new_class_id: Remapped class ID
        
        Returns:
            List of original class IDs that map to this new ID
        """
        if not self._fitted:
            raise RuntimeError("ClassRemapper not fitted. Call fit() first.")
        
        if new_class_id not in self.new_to_old:
            raise ValueError(f"Unknown new class ID: {new_class_id}")
        
        return self.new_to_old[new_class_id]
    
    def get_bucket(self, old_class_id: int) -> Bucket:
        """Get bucket for an original class ID."""
        if old_class_id not in self.old_id_to_bucket:
            raise ValueError(f"Unknown class ID: {old_class_id}")
        return self.old_id_to_bucket[old_class_id]
    
    def get_new_class_bucket(self, new_class_id: int) -> Bucket:
        """Get bucket for a new (remapped) class ID."""
        if new_class_id == self.other_class_id:
            return Bucket.TAIL  # OTHER represents collapsed TAIL
        
        old_ids = self.inverse_transform(new_class_id)
        if old_ids:
            return self.get_bucket(old_ids[0])
        raise ValueError(f"Unknown new class ID: {new_class_id}")
    
    def is_other(self, new_class_id: int) -> bool:
        """Check if a new class ID is the OTHER class."""
        return new_class_id == self.other_class_id
    
    def save(self, path: Path) -> None:
        """Persist mapping to JSON.
        
        Args:
            path: Output path for JSON file
        """
        if not self._fitted:
            raise RuntimeError("ClassRemapper not fitted. Call fit() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": self.config.to_dict(),
            "old_to_new": {str(k): v for k, v in self.old_to_new.items()},
            "new_to_old": {str(k): v for k, v in self.new_to_old.items()},
            "new_class_names": {str(k): v for k, v in self.new_class_names.items()},
            "bucket_to_old_ids": {
                b.value: ids for b, ids in self.bucket_to_old_ids.items()
            },
            "statistics": {
                "num_classes_original": self.num_classes_original,
                "num_classes_remapped": self.num_classes_remapped,
                "other_class_id": self.other_class_id,
                "classes_collapsed": self.classes_collapsed,
                "samples_in_other": self.samples_in_other,
                "head_count": len(self.bucket_to_old_ids[Bucket.HEAD]),
                "mid_count": len(self.bucket_to_old_ids[Bucket.MID]),
                "tail_count": len(self.bucket_to_old_ids[Bucket.TAIL])
            }
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"ClassRemapper saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "ClassRemapper":
        """Load mapping from JSON.
        
        Args:
            path: Path to JSON file
        
        Returns:
            Loaded ClassRemapper instance
        """
        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        config = RemapConfig.from_dict(data["config"])
        remapper = cls(config)
        
        remapper.old_to_new = {int(k): v for k, v in data["old_to_new"].items()}
        remapper.new_to_old = {int(k): v for k, v in data["new_to_old"].items()}
        remapper.new_class_names = {int(k): v for k, v in data["new_class_names"].items()}
        remapper.bucket_to_old_ids = {
            Bucket(k): v for k, v in data["bucket_to_old_ids"].items()
        }
        remapper.old_id_to_bucket = {}
        for bucket, ids in remapper.bucket_to_old_ids.items():
            for old_id in ids:
                remapper.old_id_to_bucket[old_id] = bucket
        
        stats = data["statistics"]
        remapper.num_classes_original = stats["num_classes_original"]
        remapper.num_classes_remapped = stats["num_classes_remapped"]
        remapper.other_class_id = stats["other_class_id"]
        remapper.classes_collapsed = stats["classes_collapsed"]
        remapper.samples_in_other = stats["samples_in_other"]
        remapper._fitted = True
        
        logger.info(f"ClassRemapper loaded from {path}")
        return remapper
    
    def get_config_summary(self) -> Dict:
        """Get configuration summary for experiment logging."""
        return {
            "strategy": self.config.strategy,
            "head_threshold": self.config.head_threshold,
            "mid_range": list(self.config.mid_range),
            "other_class_name": self.config.other_class_name,
            "num_classes_original": self.num_classes_original,
            "num_classes_remapped": self.num_classes_remapped,
            "other_class_id": self.other_class_id,
            "classes_collapsed": self.classes_collapsed,
            "samples_in_other": self.samples_in_other,
            "head_classes": len(self.bucket_to_old_ids[Bucket.HEAD]),
            "mid_classes": len(self.bucket_to_old_ids[Bucket.MID]),
            "tail_classes": len(self.bucket_to_old_ids[Bucket.TAIL])
        }


class RemappedDataset(Dataset):
    """Dataset wrapper that applies class remapping on-the-fly.
    
    This wrapper transparently remaps gloss_id values when samples
    are accessed, without modifying the underlying dataset.
    
    Compatibility:
    - ✅ Existing collate functions work without changes
    - ✅ Trainer receives correct num_classes
    - ✅ Original dataset is not modified
    
    Example:
        >>> remapper = ClassRemapper(config)
        >>> remapper.fit(train_support)
        >>> wrapped = RemappedDataset(original_dataset, remapper)
        >>> sample = wrapped[0]  # gloss_id is already remapped
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        remapper: ClassRemapper,
        filter_excluded: bool = True
    ):
        """Initialize wrapped dataset.
        
        Args:
            base_dataset: Original dataset to wrap
            remapper: Fitted ClassRemapper instance
            filter_excluded: If True, filter out samples with excluded classes
                           (only relevant for tail_exclude strategy)
        """
        self.base_dataset = base_dataset
        self.remapper = remapper
        self.filter_excluded = filter_excluded
        
        # Build index mapping if filtering
        self._index_map: Optional[List[int]] = None
        if filter_excluded and remapper.config.strategy == "tail_exclude":
            self._build_index_map()
    
    def _build_index_map(self) -> None:
        """Build mapping from wrapped indices to base dataset indices."""
        self._index_map = []
        for i in range(len(self.base_dataset)):
            sample = self.base_dataset[i]
            old_id = sample.gloss_id
            new_id = self.remapper.transform(old_id)
            if new_id != -1:  # Not excluded
                self._index_map.append(i)
        
        logger.info(
            f"RemappedDataset: filtered {len(self.base_dataset)} → "
            f"{len(self._index_map)} samples"
        )
    
    def __getitem__(self, idx: int) -> Any:
        """Get sample with remapped gloss_id.
        
        Args:
            idx: Sample index
        
        Returns:
            Sample with gloss_id remapped to new class ID
        """
        # Map index if filtering
        if self._index_map is not None:
            base_idx = self._index_map[idx]
        else:
            base_idx = idx
        
        # Get original sample
        sample = self.base_dataset[base_idx]
        
        # Remap gloss_id
        # Handle both object and dict-like samples
        if hasattr(sample, 'gloss_id'):
            old_id = sample.gloss_id
            new_id = self.remapper.transform(old_id)
            sample.gloss_id = new_id
        elif isinstance(sample, dict) and 'gloss_id' in sample:
            old_id = sample['gloss_id']
            new_id = self.remapper.transform(old_id)
            sample['gloss_id'] = new_id
        
        return sample
    
    def __len__(self) -> int:
        """Get number of samples (after filtering if applicable)."""
        if self._index_map is not None:
            return len(self._index_map)
        return len(self.base_dataset)
    
    @property
    def num_classes(self) -> int:
        """Get number of classes after remapping."""
        return self.remapper.num_classes_remapped
    
    @property
    def gloss_to_id(self) -> Dict[str, int]:
        """Get new gloss to ID mapping."""
        return {v: k for k, v in self.remapper.new_class_names.items()}
    
    @property
    def id_to_gloss(self) -> Dict[int, str]:
        """Get new ID to gloss mapping."""
        return self.remapper.new_class_names.copy()


def compute_class_support(dataset: Dataset) -> Dict[int, int]:
    """Compute sample count per class from a dataset.
    
    Args:
        dataset: Dataset to analyze
    
    Returns:
        Dict mapping class_id to sample count
    """
    support = Counter()
    
    for i in range(len(dataset)):
        sample = dataset[i]
        if hasattr(sample, 'gloss_id'):
            support[sample.gloss_id] += 1
        elif isinstance(sample, dict) and 'gloss_id' in sample:
            support[sample['gloss_id']] += 1
    
    return dict(support)


def create_remapped_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: Optional[RemapConfig] = None,
    class_names: Optional[Dict[int, str]] = None
) -> Tuple[RemappedDataset, RemappedDataset, ClassRemapper]:
    """Create remapped train and validation datasets.
    
    Convenience function that:
    1. Computes training support
    2. Creates and fits ClassRemapper
    3. Wraps both datasets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Remapping configuration (uses defaults if None)
        class_names: Optional class name mapping
    
    Returns:
        Tuple of (wrapped_train, wrapped_val, remapper)
    
    Example:
        >>> train_wrapped, val_wrapped, remapper = create_remapped_datasets(
        ...     train_ds, val_ds, config=RemapConfig(strategy="tail_to_other")
        ... )
    """
    config = config or RemapConfig()
    
    # Compute support from training set
    train_support = compute_class_support(train_dataset)
    
    # Create and fit remapper
    remapper = ClassRemapper(config)
    remapper.fit(train_support, class_names)
    
    # Wrap datasets
    train_wrapped = RemappedDataset(train_dataset, remapper)
    val_wrapped = RemappedDataset(val_dataset, remapper)
    
    return train_wrapped, val_wrapped, remapper
