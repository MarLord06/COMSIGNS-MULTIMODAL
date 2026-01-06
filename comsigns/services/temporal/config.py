"""
Configuración para segmentación temporal
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SegmentationConfig:
    """
    Configuración del segmentador temporal.
    
    Attributes:
        similarity_threshold: Umbral base de similitud (usado si adaptive=False)
        adaptive: Si True, calcula umbral dinámico por percentil
        adaptive_percentile: Percentil para umbral adaptativo (10-20 recomendado)
        min_segment_length: Mínimo frames por segmento (aplicado POST-detección)
        smoothing_window: Ventana para suavizar similitudes (1 = sin suavizado)
        aggregation_method: Método de agregación de embeddings por segmento
    """
    similarity_threshold: float = 0.75
    adaptive: bool = True
    adaptive_percentile: int = 15
    min_segment_length: int = 3
    smoothing_window: int = 1
    aggregation_method: Literal["mean", "max"] = "mean"
    
    def __post_init__(self):
        if not 0 < self.similarity_threshold < 1:
            raise ValueError("similarity_threshold debe estar entre 0 y 1")
        if not 1 <= self.adaptive_percentile <= 50:
            raise ValueError("adaptive_percentile debe estar entre 1 y 50")
        if self.min_segment_length < 1:
            raise ValueError("min_segment_length debe ser >= 1")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window debe ser >= 1")
