"""
Semantic mapping loader.

Loads class_mapping.json and dict.json to enable
semantic resolution of model predictions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

from .types import SemanticMappingStats

logger = logging.getLogger(__name__)


class SemanticMappingLoader:
    """Loads and provides access to semantic mapping artifacts.
    
    Responsible for loading:
    - class_mapping.json: Model class mapping (new_class_names, statistics)
    - dict.json: Original dataset dictionary (old_class_id → gloss)
    
    Example:
        >>> loader = SemanticMappingLoader(
        ...     class_mapping_path=Path("artifacts/class_mapping.json"),
        ...     dict_path=Path("artifacts/dict.json")
        ... )
        >>> loader.load()
        >>> 
        >>> # Access new class names
        >>> print(loader.new_class_names[28])  # "HEAD_319"
        >>> 
        >>> # Access gloss from old class id
        >>> print(loader.get_gloss(319))  # "yo"
    
    Attributes:
        class_mapping_path: Path to class_mapping.json
        dict_path: Path to dict.json
        new_class_names: Mapping new_class_id → "BUCKET_old_id"
        statistics: Mapping statistics
    """
    
    def __init__(
        self,
        class_mapping_path: Path,
        dict_path: Optional[Path] = None
    ):
        """Initialize the loader.
        
        Args:
            class_mapping_path: Path to class_mapping.json from training
            dict_path: Path to dict.json from dataset.
                      If None, gloss resolution will be unavailable.
        """
        self.class_mapping_path = Path(class_mapping_path)
        self.dict_path = Path(dict_path) if dict_path else None
        
        # Loaded data
        self._class_mapping: Optional[Dict[str, Any]] = None
        self._dict_data: Optional[Dict[str, Any]] = None
        self._loaded = False
        
        # Processed mappings
        self.new_class_names: Dict[int, str] = {}
        self.old_to_gloss: Dict[int, str] = {}
        self.statistics: Optional[SemanticMappingStats] = None
    
    def load(self) -> "SemanticMappingLoader":
        """Load all mapping files.
        
        Returns:
            self for method chaining
        
        Raises:
            FileNotFoundError: If class_mapping.json not found
        """
        self._load_class_mapping()
        
        if self.dict_path is not None:
            self._load_dict()
        
        self._loaded = True
        logger.info(
            f"SemanticMappingLoader loaded: "
            f"{len(self.new_class_names)} classes, "
            f"{len(self.old_to_gloss)} glosses"
        )
        
        return self
    
    def _load_class_mapping(self) -> None:
        """Load class_mapping.json."""
        if not self.class_mapping_path.exists():
            raise FileNotFoundError(
                f"class_mapping.json not found: {self.class_mapping_path}"
            )
        
        logger.debug(f"Loading class mapping from {self.class_mapping_path}")
        
        with open(self.class_mapping_path, "r", encoding="utf-8") as f:
            self._class_mapping = json.load(f)
        
        # Extract new_class_names (convert string keys to int)
        raw_names = self._class_mapping.get("new_class_names", {})
        self.new_class_names = {int(k): v for k, v in raw_names.items()}
        
        # Extract statistics
        stats = self._class_mapping.get("statistics", {})
        if stats:
            self.statistics = SemanticMappingStats(
                num_classes_original=stats.get("num_classes_original", 0),
                num_classes_remapped=stats.get("num_classes_remapped", 0),
                head_count=stats.get("head_count", 0),
                mid_count=stats.get("mid_count", 0),
                tail_count=stats.get("tail_count", 0),
                other_class_id=stats.get("other_class_id", -1)
            )
        
        logger.debug(f"Loaded {len(self.new_class_names)} new class names")
    
    def _load_dict(self) -> None:
        """Load dict.json from dataset."""
        if not self.dict_path.exists():
            logger.warning(f"dict.json not found: {self.dict_path}")
            return
        
        logger.debug(f"Loading dict from {self.dict_path}")
        
        with open(self.dict_path, "r", encoding="utf-8") as f:
            self._dict_data = json.load(f)
        
        # Build old_class_id → gloss mapping
        for old_id_str, entry in self._dict_data.items():
            old_id = int(old_id_str)
            gloss = entry.get("gloss", "")
            if gloss:
                self.old_to_gloss[old_id] = gloss
        
        logger.debug(f"Loaded {len(self.old_to_gloss)} glosses from dict")
    
    def get_gloss(self, old_class_id: int) -> Optional[str]:
        """Get gloss for an old class ID.
        
        Args:
            old_class_id: Original dataset class ID
        
        Returns:
            Gloss string or None if not found
        """
        return self.old_to_gloss.get(old_class_id)
    
    def get_new_class_name(self, new_class_id: int) -> Optional[str]:
        """Get the bucket_old_id string for a new class ID.
        
        Args:
            new_class_id: Model output class ID
        
        Returns:
            String like "HEAD_319" or "OTHER", or None if not found
        """
        return self.new_class_names.get(new_class_id)
    
    def get_other_class_id(self) -> int:
        """Get the new_class_id for OTHER.
        
        Returns:
            OTHER class ID, or -1 if statistics not loaded
        """
        if self.statistics:
            return self.statistics.other_class_id
        return -1
    
    @property
    def is_loaded(self) -> bool:
        """Check if mapping files have been loaded."""
        return self._loaded
    
    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"SemanticMappingLoader({status}, "
            f"classes={len(self.new_class_names)}, "
            f"glosses={len(self.old_to_gloss)})"
        )
