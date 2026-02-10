"""
Gloss cleaner for ComSigns training.

Removes invalid/noisy glosses before any training:
- ??? prefixed glosses
- Deletreos artificiales (A-L-E-X_46)
- Prefijos/sufijos no semánticos (-adulto_2196)
- Glosas que no existen en dict.json
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class GlossCleaningReport:
    """Report of gloss cleaning results."""
    total_glosses_before: int
    total_glosses_after: int
    total_samples_before: int
    total_samples_after: int
    removed_question_marks: List[str] = field(default_factory=list)
    removed_deletreos: List[str] = field(default_factory=list)
    removed_invalid_prefix: List[str] = field(default_factory=list)
    removed_not_in_dict: List[str] = field(default_factory=list)
    removed_other: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "total_glosses_before": self.total_glosses_before,
            "total_glosses_after": self.total_glosses_after,
            "total_samples_before": self.total_samples_before,
            "total_samples_after": self.total_samples_after,
            "removed_due_to_question_marks": self.removed_question_marks,
            "removed_due_to_deletreos": self.removed_deletreos,
            "removed_due_to_invalid_prefix": self.removed_invalid_prefix,
            "removed_due_to_missing_dict": self.removed_not_in_dict,
            "removed_due_to_other": self.removed_other,
            "final_class_count": self.total_glosses_after
        }
    
    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    def get_summary(self) -> str:
        lines = [
            "=" * 60,
            "GLOSS CLEANING REPORT",
            "=" * 60,
            f"Glosses: {self.total_glosses_before} → {self.total_glosses_after}",
            f"Samples: {self.total_samples_before} → {self.total_samples_after}",
            f"Removed (???): {len(self.removed_question_marks)}",
            f"Removed (deletreos): {len(self.removed_deletreos)}",
            f"Removed (invalid prefix): {len(self.removed_invalid_prefix)}",
            f"Removed (not in dict): {len(self.removed_not_in_dict)}",
            f"Removed (other): {len(self.removed_other)}",
            "=" * 60,
        ]
        return "\n".join(lines)


class GlossCleaner:
    """
    Cleans glosses by removing invalid/noisy entries.
    
    Patterns removed:
    1. ??? prefixed (e.g., ???_111, ??_2280)
    2. Deletreos artificiales (e.g., A-L-E-X_46, contains multiple hyphens between letters)
    3. Invalid prefix/suffix (e.g., -adulto_2196, -ustedes_1353)
    4. Glosses not in dict.json
    """
    
    # Patterns for invalid glosses
    QUESTION_MARK_PATTERN = re.compile(r'^\?+')  # Starts with one or more ?
    DELETREO_PATTERN = re.compile(r'^[A-Z](-[A-Z])+_\d+$')  # A-L-E-X_46 pattern
    INVALID_PREFIX_PATTERN = re.compile(r'^-')  # Starts with hyphen
    
    # Explicit list of known invalid glosses (from user specification)
    EXPLICIT_INVALID = {
        "-adulto_2196",
        "-ustedes_1353",
        "???_111",
        "???_1112",
        "???_137",
        "???_1557",
        "???_1571",
        "???_1618",
        "???_1688",
        "???_1862",
        "???_1880",
        "???_2021",
        "???_2031",
        "???_2036",
        "???_234",
        "???_38",
        "???_54",
        "???_580",
        "???_615",
        "???_648",
        "???_869",
        "??_2280",
        "A-L-E-X_46",
    }
    
    def __init__(self, dict_path: Optional[Path] = None):
        """
        Initialize the cleaner.
        
        Args:
            dict_path: Path to dict.json for semantic validation
        """
        self.dict_path = Path(dict_path) if dict_path else None
        self.valid_glosses: Optional[Set[str]] = None
        
        if self.dict_path and self.dict_path.exists():
            self._load_dict()
    
    def _load_dict(self):
        """Load valid glosses from dict.json."""
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.valid_glosses = set()
        for entry in data.values():
            gloss = entry.get("gloss", "").strip()
            if gloss:
                self.valid_glosses.add(gloss)
        
        logger.info(f"Loaded {len(self.valid_glosses)} valid glosses from dict.json")
    
    def is_question_mark_gloss(self, gloss: str) -> bool:
        """Check if gloss starts with question marks."""
        return bool(self.QUESTION_MARK_PATTERN.match(gloss))
    
    def is_deletreo(self, gloss: str) -> bool:
        """Check if gloss is an artificial spelling (A-L-E-X_46)."""
        return bool(self.DELETREO_PATTERN.match(gloss))
    
    def is_invalid_prefix(self, gloss: str) -> bool:
        """Check if gloss has invalid prefix (starts with -)."""
        return bool(self.INVALID_PREFIX_PATTERN.match(gloss))
    
    def is_in_dict(self, gloss: str) -> bool:
        """Check if gloss exists in dict.json."""
        if self.valid_glosses is None:
            return True  # No dict loaded, assume valid
        
        # Extract base gloss (remove _ID suffix if present)
        base_gloss = gloss.rsplit('_', 1)[0] if '_' in gloss else gloss
        return base_gloss in self.valid_glosses or gloss in self.valid_glosses
    
    def classify_invalid_reason(self, gloss: str) -> Optional[str]:
        """
        Classify why a gloss is invalid.
        
        Returns:
            None if valid, otherwise reason string
        """
        # Check explicit list first
        if gloss in self.EXPLICIT_INVALID:
            if self.is_question_mark_gloss(gloss):
                return "question_marks"
            elif self.is_deletreo(gloss):
                return "deletreo"
            elif self.is_invalid_prefix(gloss):
                return "invalid_prefix"
            return "explicit_invalid"
        
        # Check patterns
        if self.is_question_mark_gloss(gloss):
            return "question_marks"
        if self.is_deletreo(gloss):
            return "deletreo"
        if self.is_invalid_prefix(gloss):
            return "invalid_prefix"
        if not self.is_in_dict(gloss):
            return "not_in_dict"
        
        return None  # Valid
    
    def get_valid_glosses(self, all_glosses: Set[str]) -> Tuple[Set[str], GlossCleaningReport]:
        """
        Filter glosses and return only valid ones.
        
        Args:
            all_glosses: Set of all gloss names in dataset
            
        Returns:
            (valid_glosses, cleaning_report)
        """
        valid = set()
        report = GlossCleaningReport(
            total_glosses_before=len(all_glosses),
            total_glosses_after=0,
            total_samples_before=0,
            total_samples_after=0
        )
        
        for gloss in all_glosses:
            reason = self.classify_invalid_reason(gloss)
            
            if reason is None:
                valid.add(gloss)
            elif reason == "question_marks":
                report.removed_question_marks.append(gloss)
            elif reason == "deletreo":
                report.removed_deletreos.append(gloss)
            elif reason == "invalid_prefix":
                report.removed_invalid_prefix.append(gloss)
            elif reason == "not_in_dict":
                report.removed_not_in_dict.append(gloss)
            else:
                report.removed_other.append(gloss)
        
        report.total_glosses_after = len(valid)
        return valid, report


class CleanedGlossDataset:
    """Dataset wrapper that filters out invalid glosses."""
    
    def __init__(
        self, 
        base_dataset, 
        cleaner: GlossCleaner,
        remap_ids: bool = True
    ):
        """
        Initialize cleaned dataset.
        
        Args:
            base_dataset: Original dataset with gloss_to_id, id_to_gloss
            cleaner: GlossCleaner instance
            remap_ids: Whether to remap class IDs to contiguous range
        """
        self.base_dataset = base_dataset
        self.cleaner = cleaner
        self.remap_ids = remap_ids
        
        # Get all glosses from dataset
        if hasattr(base_dataset, 'gloss_to_id'):
            all_glosses = set(base_dataset.gloss_to_id.keys())
        else:
            all_glosses = set()
            for i in range(len(base_dataset)):
                sample = base_dataset[i]
                gloss = getattr(sample, 'gloss', None) or sample.get('gloss')
                if gloss:
                    all_glosses.add(gloss)
        
        # Get valid glosses
        self._valid_glosses, self._report = cleaner.get_valid_glosses(all_glosses)
        
        # Build ID mappings
        if remap_ids:
            sorted_glosses = sorted(self._valid_glosses)
            self._gloss_to_new_id = {g: i for i, g in enumerate(sorted_glosses)}
            self._new_id_to_gloss = {i: g for g, i in self._gloss_to_new_id.items()}
        else:
            # Keep original IDs
            self._gloss_to_new_id = {
                g: base_dataset.gloss_to_id[g] 
                for g in self._valid_glosses 
                if g in base_dataset.gloss_to_id
            }
            self._new_id_to_gloss = {v: k for k, v in self._gloss_to_new_id.items()}
        
        # Build index map (which samples to include)
        self._index_map: List[int] = []
        samples_per_gloss: Dict[str, int] = {}
        
        for i in range(len(base_dataset)):
            sample = base_dataset[i]
            gloss = getattr(sample, 'gloss', None) or sample.get('gloss')
            
            if gloss in self._valid_glosses:
                self._index_map.append(i)
                samples_per_gloss[gloss] = samples_per_gloss.get(gloss, 0) + 1
        
        # Update report with sample counts
        self._report.total_samples_before = len(base_dataset)
        self._report.total_samples_after = len(self._index_map)
        
        logger.info(
            f"CleanedGlossDataset: {len(base_dataset)} → {len(self._index_map)} samples, "
            f"glosses: {self._report.total_glosses_before} → {self._report.total_glosses_after}"
        )
    
    def __len__(self) -> int:
        return len(self._index_map)
    
    def __getitem__(self, idx: int):
        base_idx = self._index_map[idx]
        sample = self.base_dataset[base_idx]
        
        gloss = getattr(sample, 'gloss', None) or sample.get('gloss')
        new_id = self._gloss_to_new_id[gloss]
        
        # Update class ID in sample
        if hasattr(sample, 'gloss_id'):
            sample.gloss_id = new_id
        elif isinstance(sample, dict) and 'gloss_id' in sample:
            sample['gloss_id'] = new_id
        
        return sample
    
    @property
    def gloss_to_id(self) -> Dict[str, int]:
        return self._gloss_to_new_id
    
    @property
    def id_to_gloss(self) -> Dict[int, str]:
        return self._new_id_to_gloss
    
    @property
    def cleaning_report(self) -> GlossCleaningReport:
        return self._report
    
    @property
    def num_classes(self) -> int:
        return len(self._gloss_to_new_id)
