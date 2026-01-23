"""
Model loader for inference.

Handles loading checkpoints, class mappings, and model reconstruction.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn

from .types import ModelInfo

logger = logging.getLogger(__name__)


class InferenceLoader:
    """Loads trained model and class mapping for inference.
    
    Handles:
    - Loading checkpoint (best.pt or epoch_XXX.pt)
    - Loading class mapping (class_mapping.json)
    - Reconstructing model architecture
    - Setting model to eval mode with no_grad context
    
    Example:
        >>> loader = InferenceLoader(
        ...     checkpoint_path=Path("experiments/run_001/checkpoints/best.pt"),
        ...     class_mapping_path=Path("experiments/run_001/class_mapping.json"),
        ...     device="cpu"
        ... )
        >>> model = loader.load_model()
        >>> class_mapping = loader.load_class_mapping()
        >>> 
        >>> # Use for inference
        >>> with torch.no_grad():
        ...     logits = model(hand, body, face, lengths)
    
    Attributes:
        checkpoint_path: Path to model checkpoint
        class_mapping_path: Path to class mapping JSON
        device: Device to load model on
        model_info: Metadata about loaded model
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        class_mapping_path: Optional[Path] = None,
        device: str = "cpu"
    ):
        """Initialize the inference loader.
        
        Args:
            checkpoint_path: Path to checkpoint file (.pt)
            class_mapping_path: Path to class_mapping.json. If None,
                               inferred from checkpoint location.
            device: Device to load model on ("cpu" or "cuda")
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )
        
        # Infer class mapping path if not provided
        if class_mapping_path is None:
            # Assume structure: experiments/run_XXX/checkpoints/best.pt
            # class_mapping at: experiments/run_XXX/class_mapping.json
            experiment_dir = self.checkpoint_path.parent.parent
            self.class_mapping_path = experiment_dir / "class_mapping.json"
        else:
            self.class_mapping_path = Path(class_mapping_path)
        
        # State
        self._checkpoint: Optional[Dict] = None
        self._class_mapping: Optional[Dict] = None
        self._model: Optional[nn.Module] = None
        self.model_info: Optional[ModelInfo] = None
    
    def _load_checkpoint(self) -> Dict:
        """Load and cache checkpoint."""
        if self._checkpoint is None:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            self._checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False
            )
            logger.info(f"  Epoch: {self._checkpoint.get('epoch', 'unknown')}")
            if 'metrics' in self._checkpoint:
                metrics = self._checkpoint['metrics']
                logger.info(f"  Metrics: {metrics}")
        return self._checkpoint
    
    def load_class_mapping(self) -> Dict[str, Any]:
        """Load and return class mapping.
        
        Returns:
            Dictionary with class mapping information including:
            - old_to_new: Original ID → New ID mapping
            - new_to_old: New ID → List of original IDs
            - new_class_names: New ID → Class name
            - statistics: Summary statistics
        
        Raises:
            FileNotFoundError: If class_mapping.json doesn't exist
        """
        if self._class_mapping is not None:
            return self._class_mapping
        
        if not self.class_mapping_path.exists():
            logger.warning(
                f"Class mapping not found at {self.class_mapping_path}. "
                "Assuming no TAIL→OTHER remapping."
            )
            return {}
        
        logger.info(f"Loading class mapping from {self.class_mapping_path}")
        with open(self.class_mapping_path, "r", encoding="utf-8") as f:
            self._class_mapping = json.load(f)
        
        stats = self._class_mapping.get("statistics", {})
        logger.info(
            f"  Classes: {stats.get('num_classes_original', '?')} → "
            f"{stats.get('num_classes_remapped', '?')}"
        )
        logger.info(f"  OTHER class ID: {stats.get('other_class_id', 'N/A')}")
        
        return self._class_mapping
    
    def get_num_classes(self) -> int:
        """Get number of output classes from checkpoint.
        
        Returns:
            Number of classes the model was trained on
        """
        checkpoint = self._load_checkpoint()
        
        # Try to get from checkpoint metadata
        if "num_classes" in checkpoint:
            return checkpoint["num_classes"]
        
        # Infer from classifier layer shape
        model_state = checkpoint.get("model_state", checkpoint)
        for key in model_state:
            if "classifier.weight" in key:
                return model_state[key].shape[0]
        
        raise ValueError("Could not determine num_classes from checkpoint")
    
    def get_other_class_id(self) -> int:
        """Get the OTHER class ID if TAIL→OTHER was used.
        
        Returns:
            OTHER class ID, or -1 if not applicable
        """
        class_mapping = self.load_class_mapping()
        if not class_mapping:
            return -1
        return class_mapping.get("statistics", {}).get("other_class_id", -1)
    
    def get_class_names(self) -> Dict[int, str]:
        """Get mapping from class ID to name.
        
        Returns:
            Dictionary mapping class ID to human-readable name
        """
        class_mapping = self.load_class_mapping()
        if not class_mapping:
            # No mapping - return empty dict
            return {}
        
        # new_class_names has string keys in JSON
        new_names = class_mapping.get("new_class_names", {})
        return {int(k): v for k, v in new_names.items()}
    
    def was_tail_to_other(self) -> bool:
        """Check if TAIL→OTHER remapping was used.
        
        Returns:
            True if model was trained with TAIL→OTHER
        """
        checkpoint = self._load_checkpoint()
        if "tail_to_other" in checkpoint:
            return checkpoint["tail_to_other"]
        
        # Check class mapping
        class_mapping = self.load_class_mapping()
        if class_mapping:
            config = class_mapping.get("config", {})
            return config.get("strategy") == "tail_to_other"
        
        return False
    
    def load_model(self) -> nn.Module:
        """Load and return the model ready for inference.
        
        Creates the model architecture, loads weights from checkpoint,
        sets to eval mode and moves to specified device.
        
        Returns:
            PyTorch model in eval mode
        
        Raises:
            ValueError: If model architecture cannot be determined
        """
        if self._model is not None:
            return self._model
        
        checkpoint = self._load_checkpoint()
        num_classes = self.get_num_classes()
        
        # Import model classes
        from services.encoder.model import MultimodalEncoder
        from training.classifier import SignLanguageClassifier
        
        # Create encoder with default config
        encoder = MultimodalEncoder()
        
        # Create classifier
        model = SignLanguageClassifier(
            encoder=encoder,
            num_classes=num_classes
        )
        
        # Load state dict
        model_state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(model_state)
        
        # Move to device and set eval mode
        model = model.to(self.device)
        model.eval()
        
        self._model = model
        
        # Build model info
        self.model_info = ModelInfo(
            checkpoint_path=str(self.checkpoint_path),
            epoch=checkpoint.get("epoch", -1),
            num_classes=num_classes,
            other_class_id=self.get_other_class_id(),
            tail_to_other=self.was_tail_to_other(),
            metrics=checkpoint.get("metrics", {})
        )
        
        logger.info(f"Model loaded: {num_classes} classes, device={self.device}")
        
        return self._model
    
    def load_all(self) -> Tuple[nn.Module, Dict[int, str], int]:
        """Load model, class names, and OTHER class ID.
        
        Convenience method to load everything needed for inference.
        
        Returns:
            Tuple of (model, class_names dict, other_class_id)
        """
        model = self.load_model()
        class_names = self.get_class_names()
        other_class_id = self.get_other_class_id()
        
        return model, class_names, other_class_id


def load_checkpoint_for_inference(
    checkpoint_path: Path,
    class_mapping_path: Optional[Path] = None,
    device: str = "cpu"
) -> Tuple[nn.Module, Dict[int, str], int, ModelInfo]:
    """Convenience function to load everything for inference.
    
    Args:
        checkpoint_path: Path to checkpoint file
        class_mapping_path: Optional path to class mapping JSON
        device: Device to load model on
    
    Returns:
        Tuple of (model, class_names, other_class_id, model_info)
    
    Example:
        >>> model, names, other_id, info = load_checkpoint_for_inference(
        ...     Path("experiments/run_001/checkpoints/best.pt")
        ... )
        >>> print(f"Loaded epoch {info.epoch} with {info.num_classes} classes")
    """
    loader = InferenceLoader(
        checkpoint_path=checkpoint_path,
        class_mapping_path=class_mapping_path,
        device=device
    )
    
    model, class_names, other_class_id = loader.load_all()
    
    return model, class_names, other_class_id, loader.model_info
