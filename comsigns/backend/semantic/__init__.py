"""
Semantic resolution layer for ComSigns backend.

Transforms model predictions (new_class_id, score) into
human-readable results with glosses and metadata.

Usage:
    >>> from backend.semantic import SemanticResolver, SemanticMappingLoader
    >>> 
    >>> # Load mappings
    >>> loader = SemanticMappingLoader(
    ...     class_mapping_path=Path("artifacts/class_mapping.json"),
    ...     dict_path=Path("artifacts/dict.json")
    ... )
    >>> loader.load()
    >>> 
    >>> # Create resolver
    >>> resolver = SemanticResolver(loader)
    >>> 
    >>> # Resolve predictions
    >>> pred = resolver.resolve(new_class_id=28, score=0.85)
    >>> print(pred.gloss)     # "yo"
    >>> print(pred.bucket)    # "HEAD"
    >>> print(pred.is_other)  # False

Or use the factory function:
    >>> from backend.semantic import create_semantic_resolver
    >>> 
    >>> resolver = create_semantic_resolver(
    ...     class_mapping_path="artifacts/class_mapping.json",
    ...     dict_path="artifacts/dict.json"
    ... )
"""

from .types import (
    SemanticPrediction,
    SemanticClassInfo,
    SemanticTopK,
    SemanticMappingStats
)

from .loader import SemanticMappingLoader

from .resolver import (
    SemanticResolver,
    create_semantic_resolver
)


__all__ = [
    # Types
    "SemanticPrediction",
    "SemanticClassInfo",
    "SemanticTopK",
    "SemanticMappingStats",
    # Loader
    "SemanticMappingLoader",
    # Resolver
    "SemanticResolver",
    "create_semantic_resolver",
]
