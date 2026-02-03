"""
Semantic resolver for model predictions.

Transforms model output (new_class_id, score) into
human-readable predictions with gloss and metadata.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from .loader import SemanticMappingLoader
from .types import SemanticPrediction, SemanticClassInfo, SemanticTopK

logger = logging.getLogger(__name__)


# Pattern to parse "BUCKET_old_id" format
CLASS_NAME_PATTERN = re.compile(r"^(HEAD|MID)_(\d+)$")


class SemanticResolver:
    """Resolves model predictions to human-readable semantics.
    
    Given a new_class_id from model output, resolves:
    - bucket: HEAD, MID, or OTHER
    - old_class_id: Original dataset class ID
    - gloss: Human-readable sign name
    
    Example:
        >>> loader = SemanticMappingLoader(class_mapping_path, dict_path)
        >>> loader.load()
        >>> resolver = SemanticResolver(loader)
        >>> 
        >>> # Single prediction
        >>> pred = resolver.resolve(new_class_id=28, score=0.85)
        >>> print(pred.gloss)  # "yo"
        >>> print(pred.bucket)  # "HEAD"
        >>> 
        >>> # Top-K predictions
        >>> topk = resolver.resolve_topk(
        ...     class_ids=[28, 45, 141],
        ...     scores=[0.6, 0.25, 0.1]
        ... )
    
    Attributes:
        loader: SemanticMappingLoader with loaded mappings
        class_info_cache: Cached SemanticClassInfo by new_class_id
    """
    
    def __init__(self, loader: SemanticMappingLoader):
        """Initialize the resolver.
        
        Args:
            loader: Loaded SemanticMappingLoader instance
        
        Raises:
            ValueError: If loader hasn't been loaded
        """
        if not loader.is_loaded:
            raise ValueError("SemanticMappingLoader must be loaded first")
        
        self.loader = loader
        self._class_info_cache: Dict[int, SemanticClassInfo] = {}
        
        # Build cache for all known classes
        self._build_class_info_cache()
    
    def _build_class_info_cache(self) -> None:
        """Pre-compute SemanticClassInfo for all classes.
        
        Also validates the mapping and logs any resolution issues.
        """
        unresolved_glosses = []
        
        for new_class_id, class_name in self.loader.new_class_names.items():
            info = self._parse_class_name(new_class_id, class_name)
            self._class_info_cache[new_class_id] = info
            
            # Track unresolved glosses (non-OTHER, where gloss == class_name)
            if not info.is_other and info.gloss == class_name and info.old_class_id is not None:
                unresolved_glosses.append((new_class_id, class_name, info.old_class_id))
        
        logger.info(f"SemanticResolver: Built cache with {len(self._class_info_cache)} classes")
        
        if unresolved_glosses:
            logger.warning(
                f"SemanticResolver: {len(unresolved_glosses)} glosses could not be resolved from dict.json:"
            )
            for new_id, name, old_id in unresolved_glosses[:5]:  # Show first 5
                logger.warning(f"  new_id={new_id} -> {name} (old_id={old_id} not in dict)")
            if len(unresolved_glosses) > 5:
                logger.warning(f"  ... and {len(unresolved_glosses) - 5} more")
    
    def _parse_class_name(self, new_class_id: int, class_name: str) -> SemanticClassInfo:
        """Parse a class name string into SemanticClassInfo.
        
        Args:
            new_class_id: Model output class ID
            class_name: String like "HEAD_319" or "OTHER"
        
        Returns:
            SemanticClassInfo with resolved bucket, old_class_id, and gloss
        """
        # Handle OTHER class
        if class_name == "OTHER":
            return SemanticClassInfo(
                new_class_id=new_class_id,
                old_class_id=None,
                bucket="OTHER",
                gloss="OTHER",
                is_other=True
            )
        
        # Parse "BUCKET_old_id" format
        match = CLASS_NAME_PATTERN.match(class_name)
        
        if not match:
            # Fallback for unexpected format
            logger.warning(f"Unexpected class name format: {class_name}")
            return SemanticClassInfo(
                new_class_id=new_class_id,
                old_class_id=None,
                bucket="UNKNOWN",
                gloss=class_name,
                is_other=False
            )
        
        bucket = match.group(1)  # "HEAD" or "MID"
        old_class_id = int(match.group(2))
        
        # Resolve gloss from old_class_id
        gloss = self.loader.get_gloss(old_class_id)
        if gloss is None:
            # Fallback to class name if gloss not found
            gloss = class_name
            logger.debug(f"Gloss not found for old_class_id={old_class_id}")
        
        return SemanticClassInfo(
            new_class_id=new_class_id,
            old_class_id=old_class_id,
            bucket=bucket,
            gloss=gloss,
            is_other=False
        )
    
    def get_class_info(self, new_class_id: int) -> Optional[SemanticClassInfo]:
        """Get cached class info for a new_class_id.
        
        Args:
            new_class_id: Model output class ID
        
        Returns:
            SemanticClassInfo or None if not found
        """
        return self._class_info_cache.get(new_class_id)
    
    def resolve(self, new_class_id: int, score: float) -> SemanticPrediction:
        """Resolve a single prediction.
        
        Args:
            new_class_id: Model output class ID
            score: Model confidence score (0.0 to 1.0)
        
        Returns:
            SemanticPrediction with resolved gloss and metadata
        """
        info = self.get_class_info(new_class_id)
        
        if info is None:
            # Handle unknown class
            logger.warning(f"Unknown new_class_id: {new_class_id}")
            return SemanticPrediction(
                gloss=f"UNKNOWN_{new_class_id}",
                confidence=score,
                bucket="UNKNOWN",
                old_class_id=None,
                new_class_id=new_class_id,
                is_other=False
            )
        
        return SemanticPrediction(
            gloss=info.gloss,
            confidence=score,
            bucket=info.bucket,
            old_class_id=info.old_class_id,
            new_class_id=new_class_id,
            is_other=info.is_other
        )
    
    def resolve_topk(
        self,
        class_ids: List[int],
        scores: List[float]
    ) -> SemanticTopK:
        """Resolve top-K predictions.
        
        Args:
            class_ids: List of new_class_id values (in order)
            scores: Corresponding confidence scores
        
        Returns:
            SemanticTopK with resolved predictions
        """
        predictions = []
        
        for class_id, score in zip(class_ids, scores):
            pred = self.resolve(class_id, score)
            predictions.append(pred)
        
        return SemanticTopK(predictions=predictions)
    
    def is_other_class(self, new_class_id: int) -> bool:
        """Check if a new_class_id is the OTHER class.
        
        Args:
            new_class_id: Model output class ID
        
        Returns:
            True if this is the OTHER class
        """
        info = self.get_class_info(new_class_id)
        return info.is_other if info else False
    
    def get_all_glosses(self) -> Dict[int, str]:
        """Get all gloss mappings.
        
        Returns:
            Dict mapping new_class_id → gloss
        """
        return {
            info.new_class_id: info.gloss
            for info in self._class_info_cache.values()
        }
    
    def __repr__(self) -> str:
        return f"SemanticResolver(classes={len(self._class_info_cache)})"


def create_semantic_resolver(
    class_mapping_path: str,
    dict_path: Optional[str] = None
) -> SemanticResolver:
    """Factory function to create a SemanticResolver.
    
    Args:
        class_mapping_path: Path to class_mapping.json
        dict_path: Optional path to dict.json
    
    Returns:
        Configured SemanticResolver
    """
    from pathlib import Path
    
    loader = SemanticMappingLoader(
        class_mapping_path=Path(class_mapping_path),
        dict_path=Path(dict_path) if dict_path else None
    )
    loader.load()
    
    return SemanticResolver(loader)
