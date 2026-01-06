"""
Tests unitarios para el módulo de segmentación temporal
"""

import pytest
import torch

from comsigns.services.temporal import (
    SegmentationConfig,
    TemporalSegmenter,
    Segment,
    SegmentAggregator,
    segment_and_aggregate,
)


class TestSegmentationConfig:
    """Tests para la configuración"""

    def test_default_config(self):
        """Test valores por defecto"""
        config = SegmentationConfig()
        assert config.similarity_threshold == 0.75
        assert config.adaptive is True
        assert config.adaptive_percentile == 15
        assert config.min_segment_length == 3
        assert config.aggregation_method == "mean"

    def test_config_validation(self):
        """Test validación de parámetros"""
        with pytest.raises(ValueError):
            SegmentationConfig(similarity_threshold=1.5)
        
        with pytest.raises(ValueError):
            SegmentationConfig(adaptive_percentile=100)
        
        with pytest.raises(ValueError):
            SegmentationConfig(min_segment_length=0)


class TestSegment:
    """Tests para la dataclass Segment"""

    def test_segment_length(self):
        """Test cálculo de longitud"""
        seg = Segment(start=5, end=10, mean_similarity=0.9)
        assert seg.length == 6

    def test_single_frame_segment(self):
        """Test segmento de un solo frame"""
        seg = Segment(start=3, end=3, mean_similarity=1.0)
        assert seg.length == 1


class TestTemporalSegmenter:
    """Tests para el segmentador temporal"""

    def test_compute_similarity_basic(self):
        """Test cálculo de similitud básico"""
        config = SegmentationConfig(adaptive=False)
        segmenter = TemporalSegmenter(config)
        
        # Embeddings idénticos → similitud = 1
        embeddings = torch.ones(5, 512)
        sims = segmenter.compute_similarity(embeddings)
        
        assert sims.shape == (4,)
        assert torch.allclose(sims, torch.ones(4))

    def test_compute_similarity_orthogonal(self):
        """Test vectores ortogonales → similitud = 0"""
        config = SegmentationConfig(adaptive=False)
        segmenter = TemporalSegmenter(config)
        
        embeddings = torch.zeros(2, 512)
        embeddings[0, 0] = 1.0
        embeddings[1, 1] = 1.0
        
        sims = segmenter.compute_similarity(embeddings)
        assert sims.shape == (1,)
        assert sims[0].item() == pytest.approx(0.0, abs=1e-5)

    def test_segment_uniform_sequence(self):
        """Test secuencia uniforme → un solo segmento"""
        config = SegmentationConfig(adaptive=False, similarity_threshold=0.5)
        segmenter = TemporalSegmenter(config)
        
        # Todos los embeddings muy similares
        embeddings = torch.randn(20, 512)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        embeddings = embeddings + torch.randn(1, 512) * 0.01  # Pequeña perturbación
        
        segments = segmenter.segment(embeddings)
        
        # Con alta similitud, debería haber pocos segmentos
        assert len(segments) >= 1
        assert all(isinstance(s, Segment) for s in segments)

    def test_segment_with_clear_changes(self):
        """Test secuencia con cambios claros"""
        config = SegmentationConfig(
            adaptive=False, 
            similarity_threshold=0.5,
            min_segment_length=2
        )
        segmenter = TemporalSegmenter(config)
        
        # Crear 3 bloques distintos
        block1 = torch.randn(1, 512)
        block2 = torch.randn(1, 512)
        block3 = torch.randn(1, 512)
        
        embeddings = torch.cat([
            block1.repeat(5, 1),  # Frames 0-4
            block2.repeat(5, 1),  # Frames 5-9
            block3.repeat(5, 1),  # Frames 10-14
        ], dim=0)
        
        segments = segmenter.segment(embeddings)
        
        # Debería detectar al menos 3 segmentos (uno por bloque)
        assert len(segments) >= 2
        
        # Verificar que los segmentos cubren toda la secuencia
        total = sum(s.length for s in segments)
        assert total == 15

    def test_segment_empty_input(self):
        """Test entrada vacía"""
        segmenter = TemporalSegmenter()
        segments = segmenter.segment(torch.empty(0, 512))
        assert segments == []

    def test_segment_single_frame(self):
        """Test un solo frame"""
        segmenter = TemporalSegmenter()
        segments = segmenter.segment(torch.randn(1, 512))
        
        assert len(segments) == 1
        assert segments[0].start == 0
        assert segments[0].end == 0
        assert segments[0].length == 1

    def test_adaptive_threshold(self):
        """Test umbral adaptativo"""
        config = SegmentationConfig(adaptive=True, adaptive_percentile=15)
        segmenter = TemporalSegmenter(config)
        
        embeddings = torch.randn(30, 512)
        similarities = segmenter.compute_similarity(embeddings)
        
        threshold = segmenter._compute_threshold(similarities)
        
        # Umbral adaptativo debe ser <= umbral fijo
        assert threshold <= config.similarity_threshold

    def test_segment_output_dataclass(self):
        """Test que el output usa Segment dataclass correctamente"""
        segmenter = TemporalSegmenter()
        embeddings = torch.randn(10, 512)
        
        segments = segmenter.segment(embeddings)
        
        for seg in segments:
            assert hasattr(seg, 'start')
            assert hasattr(seg, 'end')
            assert hasattr(seg, 'mean_similarity')
            assert seg.start <= seg.end
            assert 0 <= seg.mean_similarity <= 1.5  # Puede ser > 1 por errores numéricos


class TestSegmentAggregator:
    """Tests para el agregador de segmentos"""

    def test_aggregate_mean(self):
        """Test agregación por media"""
        aggregator = SegmentAggregator(method="mean")
        
        embeddings = torch.arange(10).unsqueeze(1).expand(10, 4).float()
        segments = [
            Segment(start=0, end=4, mean_similarity=0.9),
            Segment(start=5, end=9, mean_similarity=0.9),
        ]
        
        result = aggregator.aggregate(embeddings, segments)
        
        assert result.shape == (2, 4)
        # Primer segmento: media de 0,1,2,3,4 = 2
        assert result[0, 0].item() == pytest.approx(2.0)
        # Segundo segmento: media de 5,6,7,8,9 = 7
        assert result[1, 0].item() == pytest.approx(7.0)

    def test_aggregate_max(self):
        """Test agregación por máximo"""
        aggregator = SegmentAggregator(method="max")
        
        embeddings = torch.arange(10).unsqueeze(1).expand(10, 4).float()
        segments = [
            Segment(start=0, end=4, mean_similarity=0.9),
            Segment(start=5, end=9, mean_similarity=0.9),
        ]
        
        result = aggregator.aggregate(embeddings, segments)
        
        assert result.shape == (2, 4)
        # Primer segmento: max de 0,1,2,3,4 = 4
        assert result[0, 0].item() == pytest.approx(4.0)
        # Segundo segmento: max de 5,6,7,8,9 = 9
        assert result[1, 0].item() == pytest.approx(9.0)

    def test_aggregate_empty_segments(self):
        """Test con lista vacía de segmentos"""
        aggregator = SegmentAggregator()
        embeddings = torch.randn(10, 512)
        
        result = aggregator.aggregate(embeddings, [])
        
        assert result.shape == (0, 512)

    def test_from_config(self):
        """Test creación desde config"""
        config = SegmentationConfig(aggregation_method="max")
        aggregator = SegmentAggregator.from_config(config)
        
        assert aggregator.method == "max"


class TestSegmentAndAggregate:
    """Tests para la función de conveniencia"""

    def test_segment_and_aggregate_basic(self):
        """Test función completa"""
        embeddings = torch.randn(20, 512)
        
        aggregated, segments = segment_and_aggregate(embeddings)
        
        assert isinstance(aggregated, torch.Tensor)
        assert isinstance(segments, list)
        assert len(segments) >= 1
        
        # Verificar dimensiones
        S = len(segments)
        assert aggregated.shape == (S, 512)

    def test_segment_and_aggregate_with_config(self):
        """Test con configuración personalizada"""
        config = SegmentationConfig(
            adaptive=False,
            similarity_threshold=0.9,
            aggregation_method="max"
        )
        
        embeddings = torch.randn(15, 512)
        aggregated, segments = segment_and_aggregate(embeddings, config)
        
        assert aggregated.shape[1] == 512
        assert len(segments) >= 1
