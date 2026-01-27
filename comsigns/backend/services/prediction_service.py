"""
Prediction service combining inference and semantic resolution.

Provides a unified API for running model inference and
returning human-readable predictions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..semantic import (
    SemanticResolver,
    SemanticMappingLoader,
    SemanticPrediction,
    SemanticTopK
)

logger = logging.getLogger(__name__)


class PredictionService:
    """Unified service for prediction with semantic resolution.
    
    Combines:
    - Model inference (via Predictor)
    - Semantic resolution (via SemanticResolver)
    
    to provide end-to-end predictions with human-readable output.
    
    Example:
        >>> service = PredictionService(
        ...     class_mapping_path="artifacts/class_mapping.json",
        ...     dict_path="artifacts/dict.json"
        ... )
        >>> 
        >>> # With raw model outputs
        >>> result = service.resolve_prediction(new_class_id=28, score=0.85)
        >>> print(result.gloss)  # "yo"
        >>> 
        >>> # With top-k outputs
        >>> results = service.resolve_topk(
        ...     class_ids=[28, 45, 141],
        ...     scores=[0.6, 0.25, 0.1]
        ... )
    
    Attributes:
        resolver: SemanticResolver instance
        model_loaded: Whether inference model is loaded
    """
    
    def __init__(
        self,
        class_mapping_path: Path,
        dict_path: Optional[Path] = None,
        lazy_load: bool = False
    ):
        """Initialize the prediction service.
        
        Args:
            class_mapping_path: Path to class_mapping.json
            dict_path: Path to dict.json (for gloss resolution)
            lazy_load: If True, don't load mappings until first use
        """
        self.class_mapping_path = Path(class_mapping_path)
        self.dict_path = Path(dict_path) if dict_path else None
        
        self._loader: Optional[SemanticMappingLoader] = None
        self._resolver: Optional[SemanticResolver] = None
        
        if not lazy_load:
            self._ensure_loaded()
    
    def _ensure_loaded(self) -> None:
        """Ensure semantic mappings are loaded."""
        if self._resolver is not None:
            return
        
        logger.info("Loading semantic mappings...")
        
        self._loader = SemanticMappingLoader(
            class_mapping_path=self.class_mapping_path,
            dict_path=self.dict_path
        )
        self._loader.load()
        
        self._resolver = SemanticResolver(self._loader)
        
        logger.info(f"PredictionService ready: {self._resolver}")
    
    @property
    def resolver(self) -> SemanticResolver:
        """Get the semantic resolver (loads if needed)."""
        self._ensure_loaded()
        return self._resolver
    
    def resolve_prediction(
        self,
        new_class_id: int,
        score: float
    ) -> SemanticPrediction:
        """Resolve a single prediction to semantic output.
        
        Args:
            new_class_id: Model output class ID
            score: Model confidence score
        
        Returns:
            SemanticPrediction with gloss and metadata
        """
        return self.resolver.resolve(new_class_id, score)
    
    def resolve_topk(
        self,
        class_ids: List[int],
        scores: List[float]
    ) -> SemanticTopK:
        """Resolve top-K predictions to semantic output.
        
        Args:
            class_ids: List of model output class IDs (in order)
            scores: Corresponding confidence scores
        
        Returns:
            SemanticTopK with resolved predictions
        """
        return self.resolver.resolve_topk(class_ids, scores)
    
    def resolve_from_inference_result(
        self,
        inference_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve predictions from inference pipeline result.
        
        Expects result with keys:
        - top1_class_id: int
        - top1_score: float
        - topk: List[dict] with class_id, score
        
        Args:
            inference_result: Dict from inference pipeline
        
        Returns:
            Dict with semantic predictions added
        """
        self._ensure_loaded()
        
        # Resolve top-1
        top1_pred = self.resolve_prediction(
            new_class_id=inference_result["top1_class_id"],
            score=inference_result["top1_score"]
        )
        
        # Resolve top-k
        topk_preds = []
        for item in inference_result.get("topk", []):
            pred = self.resolve_prediction(
                new_class_id=item["class_id"],
                score=item["score"]
            )
            topk_preds.append(pred.to_dict())
        
        return {
            "top1": top1_pred.to_dict(),
            "topk": topk_preds,
            "is_other": top1_pred.is_other
        }
    
    def get_all_glosses(self) -> Dict[int, str]:
        """Get all available gloss mappings.
        
        Returns:
            Dict mapping new_class_id → gloss
        """
        return self.resolver.get_all_glosses()
    
    def is_other_class(self, new_class_id: int) -> bool:
        """Check if a class ID is the OTHER class.
        
        Args:
            new_class_id: Model output class ID
        
        Returns:
            True if this is the OTHER class
        """
        return self.resolver.is_other_class(new_class_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mapping statistics.
        
        Returns:
            Dict with statistics about the mapping
        """
        self._ensure_loaded()
        
        if self._loader.statistics:
            return self._loader.statistics.to_dict()
        return {}
    
    def __repr__(self) -> str:
        loaded = self._resolver is not None
        return f"PredictionService(loaded={loaded})"


def create_prediction_service(
    class_mapping_path: str,
    dict_path: Optional[str] = None
) -> PredictionService:
    """Factory function to create a PredictionService.
    
    Args:
        class_mapping_path: Path to class_mapping.json
        dict_path: Optional path to dict.json
    
    Returns:
        Configured PredictionService
    """
    return PredictionService(
        class_mapping_path=Path(class_mapping_path),
        dict_path=Path(dict_path) if dict_path else None
    )
