"""
Agregador de embeddings por segmento
"""

import torch
from typing import List, Literal

from .config import SegmentationConfig
from .segmenter import Segment


class SegmentAggregator:
    """
    Agrega embeddings de frames dentro de cada segmento.
    
    Produce un embedding representativo por segmento usando mean o max pooling.
    
    Attributes:
        method: Método de agregación ("mean" o "max")
    """
    
    def __init__(self, method: Literal["mean", "max"] = "mean"):
        """
        Inicializa el agregador.
        
        Args:
            method: Método de agregación. "mean" promedia embeddings,
                   "max" toma el máximo por dimensión.
        """
        if method not in ("mean", "max"):
            raise ValueError(f"method debe ser 'mean' o 'max', got '{method}'")
        self.method = method
    
    @classmethod
    def from_config(cls, config: SegmentationConfig) -> "SegmentAggregator":
        """Crea agregador desde configuración."""
        return cls(method=config.aggregation_method)
    
    def aggregate_segment(
        self, 
        embeddings: torch.Tensor, 
        segment: Segment
    ) -> torch.Tensor:
        """
        Agrega embeddings de un solo segmento.
        
        Args:
            embeddings: Tensor [T, D] con embeddings temporales
            segment: Segmento a agregar
            
        Returns:
            Tensor [D] con embedding agregado del segmento
        """
        segment_embeddings = embeddings[segment.start:segment.end + 1]
        
        if self.method == "mean":
            return segment_embeddings.mean(dim=0)
        else:  # max
            return segment_embeddings.max(dim=0).values
    
    def aggregate(
        self, 
        embeddings: torch.Tensor, 
        segments: List[Segment]
    ) -> torch.Tensor:
        """
        Agrega embeddings para todos los segmentos.
        
        Args:
            embeddings: Tensor [T, D] con embeddings temporales
            segments: Lista de segmentos detectados
            
        Returns:
            Tensor [S, D] donde S = número de segmentos
        """
        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensor [T, D], got shape {embeddings.shape}")
        
        if len(segments) == 0:
            D = embeddings.shape[1] if embeddings.shape[0] > 0 else 0
            return torch.empty(0, D, dtype=embeddings.dtype, device=embeddings.device)
        
        aggregated = []
        for segment in segments:
            agg = self.aggregate_segment(embeddings, segment)
            aggregated.append(agg)
        
        return torch.stack(aggregated, dim=0)


def segment_and_aggregate(
    embeddings: torch.Tensor,
    config: SegmentationConfig = None
) -> tuple[torch.Tensor, List[Segment]]:
    """
    Función de conveniencia para segmentar y agregar en un solo paso.
    
    Args:
        embeddings: Tensor [T, D] con embeddings temporales
        config: Configuración opcional
        
    Returns:
        Tupla de (embeddings agregados [S, D], lista de Segments)
    """
    from .segmenter import TemporalSegmenter
    
    config = config or SegmentationConfig()
    
    segmenter = TemporalSegmenter(config)
    segments = segmenter.segment(embeddings)
    
    aggregator = SegmentAggregator.from_config(config)
    aggregated = aggregator.aggregate(embeddings, segments)
    
    return aggregated, segments
