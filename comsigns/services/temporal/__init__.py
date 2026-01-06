"""
Módulo de segmentación temporal semántica (post-encoder)

Detecta cambios de seña en embeddings [T, 512] usando similitud coseno.
Agrupa frames por segmentos y devuelve embeddings agregados.
"""

from .config import SegmentationConfig
from .segmenter import TemporalSegmenter, Segment
from .aggregator import SegmentAggregator, segment_and_aggregate
from .visualization import plot_similarity_timeline, print_segments_summary

__all__ = [
    'SegmentationConfig',
    'TemporalSegmenter',
    'Segment',
    'SegmentAggregator',
    'segment_and_aggregate',
    'plot_similarity_timeline',
    'print_segments_summary',
]

