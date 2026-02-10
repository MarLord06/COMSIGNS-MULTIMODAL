"""Dataset rebalancing utilities."""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .augmentation import KeypointAugmenter, AugmentConfig

logger = logging.getLogger(__name__)


@dataclass
class RebalanceConfig:
    target_strategy: str = "median"  # "median" or "max"
    other_max_multiplier: float = 2.0
    other_max_cap: int = 0  # 0 = no cap
    min_support: int = 1


class RebalancedDataset:
    """Create a rebalanced dataset with optional augmentation.

    - Upsamples non-OTHER classes to target count using augmentation
    - Downsamples OTHER to a capped count
    """

    def __init__(
        self,
        base_dataset,
        other_class_id: Optional[int],
        class_counts: Dict[int, int],
        augmenter: Optional[KeypointAugmenter] = None,
        config: Optional[RebalanceConfig] = None
    ):
        self.base_dataset = base_dataset
        self.other_class_id = other_class_id
        self.class_counts = class_counts
        self.augmenter = augmenter
        self.config = config or RebalanceConfig()

        # Build indices per class
        self.indices_by_class: Dict[int, List[int]] = {}
        for i in range(len(base_dataset)):
            sample = base_dataset[i]
            class_id = sample.gloss_id
            self.indices_by_class.setdefault(class_id, []).append(i)

        # Determine target count for non-OTHER classes
        non_other_counts = [
            count for cid, count in class_counts.items()
            if cid != self.other_class_id
        ]
        if not non_other_counts:
            self.target_count = 0
        elif self.config.target_strategy == "max":
            self.target_count = max(non_other_counts)
        else:
            self.target_count = int(np.median(non_other_counts))

        # Build index map with augmentation flags
        self.index_map: List[Tuple[int, bool]] = []

        for class_id, indices in self.indices_by_class.items():
            count = len(indices)

            if class_id == self.other_class_id:
                # Downsample OTHER
                max_allowed = int(self.config.other_max_multiplier * self.target_count)
                if self.config.other_max_cap > 0:
                    max_allowed = min(max_allowed, self.config.other_max_cap)
                if max_allowed <= 0:
                    max_allowed = count
                selected = indices[:max_allowed]
                self.index_map.extend([(idx, False) for idx in selected])
                continue

            # Keep originals
            self.index_map.extend([(idx, False) for idx in indices])

            # Upsample with augmentation if needed
            if self.target_count > count:
                to_add = self.target_count - count
                # Sample with replacement
                extra = np.random.choice(indices, size=to_add, replace=True)
                self.index_map.extend([(idx, True) for idx in extra])

        logger.info(
            f"RebalancedDataset: base={len(base_dataset)} → rebalanced={len(self.index_map)} "
            f"(target_count={self.target_count}, other_id={self.other_class_id})"
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        base_idx, do_augment = self.index_map[idx]
        sample = self.base_dataset[base_idx]
        if do_augment and self.augmenter is not None:
            sample = self.augmenter.apply(sample)
        return sample

    @property
    def gloss_to_id(self) -> Dict[str, int]:
        return getattr(self.base_dataset, "gloss_to_id", {})

    @property
    def id_to_gloss(self) -> Dict[int, str]:
        return getattr(self.base_dataset, "id_to_gloss", {})
