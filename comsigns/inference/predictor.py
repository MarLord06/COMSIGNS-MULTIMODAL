"""
Predictor for inference.

Handles forward pass and postprocessing of model outputs.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import PredictionResult, TopKPrediction, InferenceConfig

logger = logging.getLogger(__name__)


class Predictor:
    """Runs inference and postprocesses results.
    
    Takes a loaded model and class mapping, runs forward pass,
    and returns structured prediction results.
    
    Example:
        >>> predictor = Predictor(
        ...     model=model,
        ...     class_names=class_names,
        ...     other_class_id=other_class_id,
        ...     device="cpu",
        ...     topk=5
        ... )
        >>> 
        >>> # From tensors
        >>> result = predictor.predict(hand, body, face, lengths)
        >>> print(result.format_output())
        >>> 
        >>> # From feature dict
        >>> result = predictor.predict_from_features(features)
    
    Attributes:
        model: PyTorch model in eval mode
        class_names: Mapping from class ID to name
        other_class_id: ID of OTHER class (-1 if not applicable)
        device: Device for inference
        topk: Number of top predictions to return
    """
    
    def __init__(
        self,
        model: nn.Module,
        class_names: Dict[int, str],
        other_class_id: int = -1,
        device: str = "cpu",
        topk: int = 5,
        config: Optional[InferenceConfig] = None
    ):
        """Initialize the predictor.
        
        Args:
            model: Trained PyTorch model (should be in eval mode)
            class_names: Dict mapping class ID to human-readable name
            other_class_id: ID of the OTHER class, -1 if not using TAIL→OTHER
            device: Device for inference ("cpu" or "cuda")
            topk: Number of top predictions to return
            config: Optional inference configuration
        """
        self.model = model
        self.class_names = class_names
        self.other_class_id = other_class_id
        self.device = device
        self.topk = topk
        self.config = config or InferenceConfig(device=device, topk=topk)
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Get number of classes from model
        self.num_classes = getattr(model, 'num_classes', len(class_names))
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name for ID, with fallback."""
        return self.class_names.get(class_id, f"CLASS_{class_id}")
    
    def _is_other(self, class_id: int) -> bool:
        """Check if class ID is the OTHER class."""
        return self.other_class_id >= 0 and class_id == self.other_class_id
    
    @torch.no_grad()
    def predict(
        self,
        hand: torch.Tensor,
        body: torch.Tensor,
        face: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> PredictionResult:
        """Run inference on input tensors.
        
        Args:
            hand: Hand keypoints tensor [B, T, hand_dim] or [1, T, hand_dim]
            body: Body keypoints tensor [B, T, body_dim] or [1, T, body_dim]
            face: Face keypoints tensor [B, T, face_dim] or [1, T, face_dim]
            lengths: Optional sequence lengths [B]
        
        Returns:
            PredictionResult with top-1 and top-k predictions
        
        Note:
            Currently supports batch size 1. For batched inference,
            call predict() multiple times or use predict_batch().
        """
        # Move to device
        hand = hand.to(self.device)
        body = body.to(self.device)
        face = face.to(self.device)
        if lengths is not None:
            lengths = lengths.to(self.device)
        
        # Ensure batch dimension
        if hand.dim() == 2:
            hand = hand.unsqueeze(0)
            body = body.unsqueeze(0)
            face = face.unsqueeze(0)
            if lengths is not None:
                lengths = lengths.unsqueeze(0)
        
        # Forward pass
        logits = self.model(hand, body, face, lengths)  # [B, num_classes]
        
        # Get first item if batch
        logits = logits[0]  # [num_classes]
        
        # Softmax for probabilities
        scores = F.softmax(logits, dim=0)  # [num_classes]
        
        # Get top-k
        k = min(self.topk, self.num_classes)
        topk_scores, topk_indices = torch.topk(scores, k)
        
        # Build top-k predictions
        topk_predictions = []
        for rank, (score, class_id) in enumerate(
            zip(topk_scores.tolist(), topk_indices.tolist()), 
            start=1
        ):
            topk_predictions.append(TopKPrediction(
                rank=rank,
                class_id=class_id,
                class_name=self._get_class_name(class_id),
                score=score
            ))
        
        # Build result
        top1_id = topk_indices[0].item()
        result = PredictionResult(
            top1_class_id=top1_id,
            top1_class_name=self._get_class_name(top1_id),
            top1_score=topk_scores[0].item(),
            topk=topk_predictions,
            is_other=self._is_other(top1_id)
        )
        
        # Add optional outputs
        if self.config.return_logits:
            result.raw_logits = logits.tolist()
        if self.config.return_all_scores:
            result.all_scores = scores.tolist()
        
        return result
    
    @torch.no_grad()
    def predict_from_features(
        self,
        features: Dict[str, torch.Tensor]
    ) -> PredictionResult:
        """Run inference from feature dictionary.
        
        Args:
            features: Dictionary with keys "hand", "body", "face",
                     and optionally "lengths"
        
        Returns:
            PredictionResult
        """
        hand = features["hand"]
        body = features["body"]
        face = features["face"]
        lengths = features.get("lengths")
        
        return self.predict(hand, body, face, lengths)
    
    @torch.no_grad()
    def predict_batch(
        self,
        hand: torch.Tensor,
        body: torch.Tensor,
        face: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> List[PredictionResult]:
        """Run inference on a batch of inputs.
        
        Args:
            hand: Hand keypoints tensor [B, T, hand_dim]
            body: Body keypoints tensor [B, T, body_dim]
            face: Face keypoints tensor [B, T, face_dim]
            lengths: Optional sequence lengths [B]
        
        Returns:
            List of PredictionResult, one per batch item
        """
        # Move to device
        hand = hand.to(self.device)
        body = body.to(self.device)
        face = face.to(self.device)
        if lengths is not None:
            lengths = lengths.to(self.device)
        
        # Forward pass
        logits = self.model(hand, body, face, lengths)  # [B, num_classes]
        
        # Softmax
        scores = F.softmax(logits, dim=1)  # [B, num_classes]
        
        # Get top-k
        k = min(self.topk, self.num_classes)
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)  # [B, k]
        
        # Build results for each batch item
        results = []
        batch_size = logits.shape[0]
        
        for b in range(batch_size):
            topk_predictions = []
            for rank, (score, class_id) in enumerate(
                zip(topk_scores[b].tolist(), topk_indices[b].tolist()),
                start=1
            ):
                topk_predictions.append(TopKPrediction(
                    rank=rank,
                    class_id=class_id,
                    class_name=self._get_class_name(class_id),
                    score=score
                ))
            
            top1_id = topk_indices[b, 0].item()
            result = PredictionResult(
                top1_class_id=top1_id,
                top1_class_name=self._get_class_name(top1_id),
                top1_score=topk_scores[b, 0].item(),
                topk=topk_predictions,
                is_other=self._is_other(top1_id)
            )
            
            if self.config.return_logits:
                result.raw_logits = logits[b].tolist()
            if self.config.return_all_scores:
                result.all_scores = scores[b].tolist()
            
            results.append(result)
        
        return results
    
    def get_prediction_summary(self, result: PredictionResult) -> str:
        """Get a one-line summary of prediction.
        
        Args:
            result: PredictionResult to summarize
        
        Returns:
            Summary string like "YO (0.62) [OTHER: No]"
        """
        other_str = "Yes" if result.is_other else "No"
        return f"{result.top1_class_name} ({result.top1_score:.2f}) [OTHER: {other_str}]"


def create_predictor(
    model: nn.Module,
    class_names: Dict[int, str],
    other_class_id: int = -1,
    device: str = "cpu",
    topk: int = 5
) -> Predictor:
    """Factory function to create a Predictor.
    
    Args:
        model: Trained model
        class_names: Class ID to name mapping
        other_class_id: OTHER class ID (-1 if not used)
        device: Inference device
        topk: Number of top predictions
    
    Returns:
        Configured Predictor instance
    """
    return Predictor(
        model=model,
        class_names=class_names,
        other_class_id=other_class_id,
        device=device,
        topk=topk
    )
