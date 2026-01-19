"""
Sign Language Classifier model.

Wraps the MultimodalEncoder with a classification head for
gloss recognition training.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


class SignLanguageClassifier(nn.Module):
    """
    Classification model that wraps MultimodalEncoder.
    
    Takes multimodal keypoint sequences and produces class logits.
    Handles temporal pooling to convert sequence embeddings [B, T, D]
    to fixed-size representations [B, D] for classification.
    
    Attributes:
        encoder: MultimodalEncoder instance
        classifier: Linear classification head
        pooling: Temporal pooling strategy ("mean", "max", or "last")
    
    Example:
        >>> encoder = MultimodalEncoder()
        >>> model = SignLanguageClassifier(encoder, num_classes=100)
        >>> logits = model(hand, body, face, lengths)
        >>> logits.shape
        torch.Size([batch_size, 100])
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        pooling: Literal["mean", "max", "last"] = "mean",
        dropout: float = 0.1
    ):
        """
        Initialize the classifier.
        
        Args:
            encoder: MultimodalEncoder or compatible module that produces
                     embeddings of shape [B, T, encoder_output_dim]
            num_classes: Number of gloss classes for classification
            pooling: Temporal pooling strategy:
                     - "mean": Average over time (respects padding via lengths)
                     - "max": Max over time
                     - "last": Use last valid timestep
            dropout: Dropout before classification head
        """
        super().__init__()
        
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        
        self.encoder = encoder
        self.pooling = pooling
        self.num_classes = num_classes
        
        # Get encoder output dimension
        encoder_dim = getattr(encoder, 'output_dim', 512)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder_dim, num_classes)
    
    def forward(
        self,
        hand: torch.Tensor,
        body: torch.Tensor,
        face: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: keypoints → embeddings → pooled → logits.
        
        Args:
            hand: Hand keypoints [B, T, hand_dim]
            body: Body keypoints [B, T, body_dim]
            face: Face keypoints [B, T, face_dim]
            lengths: Original sequence lengths [B] (for masked pooling)
            mask: Boolean mask [B, T] where True = valid position
        
        Returns:
            Logits tensor of shape [B, num_classes]
        """
        # Encode: [B, T, D]
        embeddings = self.encoder(hand, body, face)
        
        # Pool: [B, T, D] → [B, D]
        pooled = self._temporal_pool(embeddings, lengths, mask)
        
        # Classify: [B, D] → [B, num_classes]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits
    
    def _temporal_pool(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Pool temporal dimension to get fixed-size representation.
        
        Args:
            embeddings: [B, T, D]
            lengths: [B] original lengths (before padding)
            mask: [B, T] boolean mask
        
        Returns:
            Pooled embeddings [B, D]
        """
        B, T, D = embeddings.shape
        
        if self.pooling == "mean":
            return self._masked_mean_pool(embeddings, lengths, mask)
        
        elif self.pooling == "max":
            return self._masked_max_pool(embeddings, mask)
        
        elif self.pooling == "last":
            return self._last_valid_pool(embeddings, lengths)
        
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def _masked_mean_pool(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean pooling that respects sequence lengths."""
        B, T, D = embeddings.shape
        
        if lengths is not None:
            # Create mask from lengths: [B, T]
            mask = torch.arange(T, device=embeddings.device).expand(B, T) < lengths.unsqueeze(1)
        
        if mask is not None:
            # Expand mask for broadcasting: [B, T, 1]
            mask_expanded = mask.unsqueeze(-1).float()
            
            # Sum only valid positions
            summed = (embeddings * mask_expanded).sum(dim=1)  # [B, D]
            
            # Divide by actual lengths
            if lengths is not None:
                counts = lengths.unsqueeze(-1).float()  # [B, 1]
            else:
                counts = mask_expanded.sum(dim=1)  # [B, 1]
            
            # Avoid division by zero
            counts = counts.clamp(min=1.0)
            
            return summed / counts
        else:
            # No mask: simple mean
            return embeddings.mean(dim=1)
    
    def _masked_max_pool(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Max pooling that ignores padding."""
        if mask is not None:
            # Set padding positions to very negative value
            mask_expanded = mask.unsqueeze(-1)
            embeddings = embeddings.masked_fill(~mask_expanded, float('-inf'))
        
        return embeddings.max(dim=1).values
    
    def _last_valid_pool(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Get embedding at last valid timestep."""
        B, T, D = embeddings.shape
        
        if lengths is not None:
            # Clamp lengths to valid range [1, T]
            indices = (lengths - 1).clamp(0, T - 1)  # [B]
            
            # Gather last valid embeddings
            indices_expanded = indices.view(B, 1, 1).expand(B, 1, D)
            pooled = embeddings.gather(1, indices_expanded).squeeze(1)  # [B, D]
            return pooled
        else:
            # No lengths: use actual last position
            return embeddings[:, -1, :]
    
    def get_embeddings(
        self,
        hand: torch.Tensor,
        body: torch.Tensor,
        face: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get pooled embeddings without classification.
        
        Useful for feature extraction or visualization.
        
        Returns:
            Pooled embeddings [B, encoder_output_dim]
        """
        embeddings = self.encoder(hand, body, face)
        pooled = self._temporal_pool(embeddings, lengths, mask)
        return pooled
