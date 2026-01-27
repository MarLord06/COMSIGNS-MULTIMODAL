"""
Inference service combining model inference and semantic resolution.

Provides a unified API for running inference on samples and
returning human-readable predictions with semantic information.
"""

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, BinaryIO
from dataclasses import dataclass, asdict

import torch

logger = logging.getLogger(__name__)


@dataclass
class InferenceResponse:
    """Response structure for inference API.
    
    Attributes:
        top1: Top-1 prediction with semantic info
        topk: List of top-k predictions
        meta: Model and inference metadata
    """
    top1: Dict[str, Any]
    topk: List[Dict[str, Any]]
    meta: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


class InferenceService:
    """Service for running inference with semantic resolution.
    
    Combines:
    - Model loading and inference (via inference.Predictor)
    - Semantic resolution (via SemanticResolver)
    
    to provide end-to-end predictions with human-readable output.
    
    Example:
        >>> service = InferenceService(
        ...     checkpoint_path="experiments/run/checkpoints/best.pt",
        ...     class_mapping_path="experiments/run/class_mapping.json",
        ...     dict_path="data/raw/lsp_aec/dict.json"
        ... )
        >>> 
        >>> # Infer from file
        >>> result = service.infer_from_file(pkl_path, topk=5)
        >>> print(result.top1["gloss"])  # "yo"
    
    Attributes:
        predictor: Inference predictor
        resolver: Semantic resolver
        model_loaded: Whether model is loaded
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        class_mapping_path: Path,
        dict_path: Optional[Path] = None,
        device: str = "cpu",
        lazy_load: bool = True
    ):
        """Initialize the inference service.
        
        Args:
            checkpoint_path: Path to model checkpoint (best.pt)
            class_mapping_path: Path to class_mapping.json
            dict_path: Path to dict.json for gloss resolution
            device: Device for inference ("cpu" or "cuda")
            lazy_load: If True, load model on first inference
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.class_mapping_path = Path(class_mapping_path)
        self.dict_path = Path(dict_path) if dict_path else None
        self.device = device
        
        self._predictor = None
        self._resolver = None
        self._loader = None
        
        if not lazy_load:
            self._ensure_loaded()
    
    def _ensure_loaded(self) -> None:
        """Ensure model and resolver are loaded."""
        if self._predictor is not None:
            return
        
        logger.info("Loading inference model...")
        
        # Import here to avoid circular imports
        from inference.loader import InferenceLoader
        from inference.predictor import Predictor
        from backend.semantic import SemanticMappingLoader, SemanticResolver
        
        # Load model
        self._loader = InferenceLoader(
            checkpoint_path=self.checkpoint_path,
            class_mapping_path=self.class_mapping_path,
            device=self.device
        )
        
        model = self._loader.load_model()
        class_names = self._loader.get_class_names()
        other_class_id = self._loader.get_other_class_id()
        
        self._predictor = Predictor(
            model=model,
            class_names=class_names,
            other_class_id=other_class_id,
            device=self.device
        )
        
        # Load semantic resolver
        semantic_loader = SemanticMappingLoader(
            class_mapping_path=self.class_mapping_path,
            dict_path=self.dict_path
        )
        semantic_loader.load()
        self._resolver = SemanticResolver(semantic_loader)
        
        logger.info(
            f"InferenceService ready: {self._loader.model_info.num_classes} classes, "
            f"device={self.device}"
        )
    
    @property
    def predictor(self):
        """Get predictor (loads if needed)."""
        self._ensure_loaded()
        return self._predictor
    
    @property
    def resolver(self):
        """Get resolver (loads if needed)."""
        self._ensure_loaded()
        return self._resolver
    
    @property
    def num_classes(self) -> int:
        """Get number of output classes."""
        self._ensure_loaded()
        return self._loader.model_info.num_classes
    
    def load_sample(self, file_path: Path) -> Dict[str, torch.Tensor]:
        """Load sample from .pkl file.
        
        Args:
            file_path: Path to .pkl file with features
        
        Returns:
            Dictionary with tensors: hand, body, face, lengths
        
        Raises:
            ValueError: If file format is invalid
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        # Validate required keys
        required_keys = ["hand", "body", "face"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Sample missing required key: {key}")
        
        # Convert to tensors if needed
        features = {}
        for key in ["hand", "body", "face"]:
            val = data[key]
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32)
            features[key] = val
        
        # Add lengths if present
        if "lengths" in data:
            lengths = data["lengths"]
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths, dtype=torch.long)
            features["lengths"] = lengths
        
        return features
    
    def load_sample_from_bytes(self, file_bytes: bytes) -> Dict[str, torch.Tensor]:
        """Load sample from bytes.
        
        Args:
            file_bytes: Raw bytes of .pkl file
        
        Returns:
            Dictionary with tensors
        """
        # Write to temp file and load
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            return self.load_sample(Path(tmp.name))
    
    def infer(
        self,
        features: Dict[str, torch.Tensor],
        topk: int = 5
    ) -> InferenceResponse:
        """Run inference on features and return semantic result.
        
        Args:
            features: Dict with hand, body, face tensors
            topk: Number of top predictions to return
        
        Returns:
            InferenceResponse with semantic predictions
        """
        # Update predictor topk
        self.predictor.topk = topk
        
        # Run model inference
        result = self.predictor.predict_from_features(features)
        
        # Resolve semantics for top-1
        top1_pred = self.resolver.resolve(
            result.top1_class_id, 
            result.top1_score
        )
        
        # Resolve semantics for top-k
        class_ids = [p.class_id for p in result.topk]
        scores = [p.score for p in result.topk]
        topk_semantic = self.resolver.resolve_topk(class_ids, scores)
        
        # Build response
        top1_dict = {
            "gloss": top1_pred.gloss,
            "confidence": round(top1_pred.confidence, 4),
            "bucket": top1_pred.bucket,
            "is_other": top1_pred.is_other,
            "new_class_id": top1_pred.new_class_id,
            "old_class_id": top1_pred.old_class_id
        }
        
        topk_list = []
        for rank, pred in enumerate(topk_semantic.predictions, start=1):
            topk_list.append({
                "rank": rank,
                "gloss": pred.gloss,
                "confidence": round(pred.confidence, 4),
                "bucket": pred.bucket,
                "is_other": pred.is_other
            })
        
        meta = {
            "model": self.checkpoint_path.name,
            "num_classes": self.num_classes,
            "device": self.device,
            "topk_requested": topk
        }
        
        return InferenceResponse(
            top1=top1_dict,
            topk=topk_list,
            meta=meta
        )
    
    def infer_from_file(
        self,
        file_path: Path,
        topk: int = 5
    ) -> InferenceResponse:
        """Run inference on .pkl file.
        
        Args:
            file_path: Path to .pkl sample
            topk: Number of top predictions
        
        Returns:
            InferenceResponse
        """
        features = self.load_sample(file_path)
        return self.infer(features, topk=topk)
    
    def infer_from_bytes(
        self,
        file_bytes: bytes,
        topk: int = 5
    ) -> InferenceResponse:
        """Run inference on .pkl bytes.
        
        Args:
            file_bytes: Raw bytes of .pkl file
            topk: Number of top predictions
        
        Returns:
            InferenceResponse
        """
        features = self.load_sample_from_bytes(file_bytes)
        return self.infer(features, topk=topk)


# Singleton instance for API use
_service_instance: Optional[InferenceService] = None


def get_inference_service(
    checkpoint_path: Optional[Path] = None,
    class_mapping_path: Optional[Path] = None,
    dict_path: Optional[Path] = None,
    device: str = "cpu"
) -> InferenceService:
    """Get or create the inference service singleton.
    
    Args:
        checkpoint_path: Path to checkpoint (required on first call)
        class_mapping_path: Path to class mapping (required on first call)
        dict_path: Optional path to dict.json
        device: Device for inference
    
    Returns:
        InferenceService instance
    """
    global _service_instance
    
    if _service_instance is None:
        if checkpoint_path is None or class_mapping_path is None:
            raise ValueError(
                "checkpoint_path and class_mapping_path required on first call"
            )
        
        _service_instance = InferenceService(
            checkpoint_path=checkpoint_path,
            class_mapping_path=class_mapping_path,
            dict_path=dict_path,
            device=device
        )
    
    return _service_instance
