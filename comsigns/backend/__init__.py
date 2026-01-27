"""
ComSigns Backend package.

Provides semantic resolution and prediction services.
"""

from .semantic import (
    SemanticPrediction,
    SemanticClassInfo,
    SemanticTopK,
    SemanticMappingStats,
    SemanticMappingLoader,
    SemanticResolver,
    create_semantic_resolver
)


__all__ = [
    # Semantic types
    "SemanticPrediction",
    "SemanticClassInfo", 
    "SemanticTopK",
    "SemanticMappingStats",
    # Semantic loader
    "SemanticMappingLoader",
    # Semantic resolver
    "SemanticResolver",
    "create_semantic_resolver",
]
