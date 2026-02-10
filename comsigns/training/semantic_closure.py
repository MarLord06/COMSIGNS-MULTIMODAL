"""
Semantic closure utilities for ComSigns training.

Ensures training classes have valid semantic mapping:
new_class_id -> old_id -> gloss (dict.json)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SemanticClosureReport:
    """Summary of semantic closure filtering."""
    total_classes_before: int
    total_classes_after: int
    removed_old_ids: List[int]
    removed_counts: Dict[int, int]
    missing_gloss_old_ids: List[int]
    valid_old_ids: List[int]


def load_dict_mapping(dict_path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Load dict.json and return (old_id->gloss, gloss->old_id)."""
    dict_path = Path(dict_path)
    if not dict_path.exists():
        raise FileNotFoundError(f"dict.json not found: {dict_path}")

    with open(dict_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    old_to_gloss: Dict[int, str] = {}
    gloss_to_old: Dict[str, int] = {}

    for old_id_str, entry in data.items():
        old_id = int(old_id_str)
        gloss = entry.get("gloss", "").strip()
        if not gloss:
            continue
        old_to_gloss[old_id] = gloss
        if gloss in gloss_to_old and gloss_to_old[gloss] != old_id:
            logger.warning(
                f"Duplicate gloss '{gloss}' with different IDs: "
                f"{gloss_to_old[gloss]} and {old_id}. Keeping first."
            )
            continue
        gloss_to_old[gloss] = old_id

    return old_to_gloss, gloss_to_old


def load_class_mapping_old_ids(class_mapping_path: Path) -> Set[int]:
    """Load class_mapping.json and extract old IDs from HEAD_/MID_ names."""
    class_mapping_path = Path(class_mapping_path)
    if not class_mapping_path.exists():
        raise FileNotFoundError(f"class_mapping.json not found: {class_mapping_path}")

    with open(class_mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_names = data.get("new_class_names", {})
    valid_old_ids: Set[int] = set()

    for _, class_name in raw_names.items():
        if not isinstance(class_name, str):
            continue
        if class_name == "OTHER":
            continue
        if class_name.startswith("HEAD_") or class_name.startswith("MID_"):
            try:
                old_id = int(class_name.split("_", 1)[1])
                valid_old_ids.add(old_id)
            except (ValueError, IndexError):
                logger.warning(f"Invalid class name format: {class_name}")

    return valid_old_ids


def build_semantic_whitelist(
    dict_path: Path,
    class_mapping_path: Optional[Path] = None
) -> Tuple[Set[int], List[int]]:
    """Build whitelist of valid old IDs with gloss mapping.

    If class_mapping_path is provided, only IDs present in HEAD_/MID_ are allowed.
    Otherwise, all IDs in dict.json are considered valid.

    Returns:
        (valid_old_ids, missing_gloss_old_ids)
    """
    old_to_gloss, _ = load_dict_mapping(dict_path)

    if class_mapping_path:
        allowed_old_ids = load_class_mapping_old_ids(class_mapping_path)
    else:
        allowed_old_ids = set(old_to_gloss.keys())

    missing_gloss = [old_id for old_id in allowed_old_ids if old_id not in old_to_gloss]
    valid_old_ids = set(old_to_gloss.keys()) & allowed_old_ids

    return valid_old_ids, sorted(missing_gloss)


class DictIdDataset:
    """Dataset wrapper that uses dict.json old_id as gloss_id.

    Ensures labels are aligned with dict.json keys for semantic compatibility.
    """

    def __init__(self, base_dataset, dict_path: Path):
        self.base_dataset = base_dataset
        self.dict_path = Path(dict_path)
        self.old_to_gloss, self.gloss_to_old = load_dict_mapping(self.dict_path)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        gloss = sample.gloss
        if gloss not in self.gloss_to_old:
            raise ValueError(f"Gloss '{gloss}' not found in dict.json mapping")
        old_id = self.gloss_to_old[gloss]
        if hasattr(sample, "gloss_id"):
            sample.gloss_id = old_id
        elif isinstance(sample, dict) and "gloss_id" in sample:
            sample["gloss_id"] = old_id
        return sample

    @property
    def gloss_to_id(self) -> Dict[str, int]:
        return self.gloss_to_old

    @property
    def id_to_gloss(self) -> Dict[int, str]:
        return self.old_to_gloss


class FilteredGlossDataset:
    """Filter dataset to only allowed old_ids (optional remap)."""

    def __init__(self, base_dataset, allowed_old_ids: Set[int], remap_ids: bool = True):
        self.base_dataset = base_dataset
        self.allowed_old_ids = set(allowed_old_ids)
        self.remap_ids = remap_ids

        # Build contiguous mapping if requested
        if self.remap_ids:
            self._old_to_new = {old_id: i for i, old_id in enumerate(sorted(self.allowed_old_ids))}
            self._new_to_old = {v: k for k, v in self._old_to_new.items()}
        else:
            self._old_to_new = {old_id: old_id for old_id in self.allowed_old_ids}
            self._new_to_old = {old_id: old_id for old_id in self.allowed_old_ids}

        # Build index map using gloss labels without loading keypoints if possible
        self._index_map: List[int] = []
        if hasattr(base_dataset, "get_instances_by_gloss") and hasattr(base_dataset, "id_to_gloss"):
            for old_id in sorted(self.allowed_old_ids):
                gloss = base_dataset.id_to_gloss.get(old_id)
                if gloss is None:
                    continue
                indices = base_dataset.get_instances_by_gloss(gloss)
                self._index_map.extend(indices)
        else:
            # Fallback: iterate through dataset (may load keypoints)
            for i in range(len(base_dataset)):
                sample = base_dataset[i]
                old_id = sample.gloss_id
                if old_id in self.allowed_old_ids:
                    self._index_map.append(i)

        logger.info(
            f"FilteredGlossDataset: {len(base_dataset)} → {len(self._index_map)} samples, "
            f"classes={len(self._old_to_new)} (remap={self.remap_ids})"
        )

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int):
        base_idx = self._index_map[idx]
        sample = self.base_dataset[base_idx]
        old_id = sample.gloss_id
        if old_id not in self._old_to_new:
            raise ValueError(f"Old class ID {old_id} not in whitelist")
        new_id = self._old_to_new[old_id]
        if hasattr(sample, "gloss_id"):
            sample.gloss_id = new_id
        elif isinstance(sample, dict) and "gloss_id" in sample:
            sample["gloss_id"] = new_id
        return sample

    @property
    def gloss_to_id(self) -> Dict[str, int]:
        # Use base dataset glosses for allowed IDs
        mapping = {}
        if hasattr(self.base_dataset, "id_to_gloss"):
            for new_id, old_id in self._new_to_old.items():
                gloss = self.base_dataset.id_to_gloss.get(old_id)
                if gloss:
                    mapping[gloss] = new_id
        return mapping

    @property
    def id_to_gloss(self) -> Dict[int, str]:
        mapping = {}
        if hasattr(self.base_dataset, "id_to_gloss"):
            for new_id, old_id in self._new_to_old.items():
                gloss = self.base_dataset.id_to_gloss.get(old_id)
                if gloss:
                    mapping[new_id] = gloss
        return mapping


class LowSupportFilter:
    """Utility to remove classes with insufficient samples."""

    @staticmethod
    def filter_by_min_support(
        dataset,
        min_support: int
    ) -> Tuple[Set[int], Dict[int, int]]:
        """Return allowed class IDs and per-class support counts."""
        counts: Dict[int, int] = {}

        # Try to use optimized gloss counts if available
        if hasattr(dataset, "get_sample_counts_per_gloss") and hasattr(dataset, "gloss_to_id"):
            gloss_counts = dataset.get_sample_counts_per_gloss()
            for gloss, count in gloss_counts.items():
                class_id = dataset.gloss_to_id.get(gloss)
                if class_id is not None:
                    counts[class_id] = count
        else:
            for i in range(len(dataset)):
                sample = dataset[i]
                class_id = sample.gloss_id
                counts[class_id] = counts.get(class_id, 0) + 1

        allowed = {cls_id for cls_id, count in counts.items() if count >= min_support}
        return allowed, counts
