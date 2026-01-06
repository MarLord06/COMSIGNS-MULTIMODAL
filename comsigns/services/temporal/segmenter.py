"""
Segmentador temporal usando similitud coseno adaptativa
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional

from .config import SegmentationConfig


@dataclass
class Segment:
    """
    Representa un segmento temporal detectado.
    
    Attributes:
        start: Índice del primer frame (inclusive)
        end: Índice del último frame (inclusive)
        mean_similarity: Similitud coseno promedio dentro del segmento
    """
    start: int
    end: int
    mean_similarity: float
    
    @property
    def length(self) -> int:
        """Número de frames en el segmento"""
        return self.end - self.start + 1


class TemporalSegmenter:
    """
    Detecta límites de segmentos semánticos en embeddings temporales.
    
    Usa similitud coseno entre frames consecutivos para detectar cambios.
    Soporta umbral fijo o adaptativo basado en percentil.
    
    Attributes:
        config: Configuración del segmentador
    """
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        """
        Inicializa el segmentador.
        
        Args:
            config: Configuración. Si None, usa valores por defecto.
        """
        self.config = config or SegmentationConfig()
    
    def compute_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calcula similitud coseno entre frames consecutivos.
        
        Args:
            embeddings: Tensor de shape [T, D] donde T=frames, D=dimensión
            
        Returns:
            Tensor de shape [T-1] con similitudes entre frames consecutivos
        """
        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensor [T, D], got shape {embeddings.shape}")
        
        T = embeddings.shape[0]
        if T < 2:
            return torch.tensor([], dtype=embeddings.dtype, device=embeddings.device)
        
        # Normalizar embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Similitud coseno entre frames consecutivos
        similarities = (embeddings_norm[:-1] * embeddings_norm[1:]).sum(dim=1)
        
        # Aplicar suavizado si window > 1
        if self.config.smoothing_window > 1:
            similarities = self._smooth(similarities)
        
        return similarities
    
    def _smooth(self, similarities: torch.Tensor) -> torch.Tensor:
        """Aplica suavizado por media móvil"""
        window = self.config.smoothing_window
        if len(similarities) < window:
            return similarities
        
        # Padding para mantener longitud
        padded = F.pad(similarities.unsqueeze(0).unsqueeze(0), 
                       (window // 2, window - 1 - window // 2), mode='replicate')
        kernel = torch.ones(1, 1, window, device=similarities.device) / window
        smoothed = F.conv1d(padded, kernel).squeeze()
        
        return smoothed
    
    def _compute_threshold(self, similarities: torch.Tensor) -> float:
        """
        Calcula umbral de detección.
        
        Si adaptive=True, usa percentil. Sino, usa threshold fijo.
        """
        if not self.config.adaptive or len(similarities) == 0:
            return self.config.similarity_threshold
        
        # Umbral adaptativo: percentil bajo de similitudes
        percentile_value = torch.quantile(
            similarities, 
            self.config.adaptive_percentile / 100.0
        ).item()
        
        # Usar el mínimo entre umbral fijo y adaptativo para ser conservador
        return min(self.config.similarity_threshold, percentile_value)
    
    def _detect_boundaries(
        self, 
        similarities: torch.Tensor, 
        threshold: float
    ) -> List[int]:
        """
        Detecta índices donde hay cambio de segmento.
        
        Returns:
            Lista de índices de frame donde inicia un nuevo segmento
        """
        boundaries = [0]  # Siempre empieza un segmento en frame 0
        
        for i, sim in enumerate(similarities):
            if sim.item() < threshold:
                # Cambio detectado después del frame i, nuevo segmento en i+1
                boundaries.append(i + 1)
        
        return boundaries
    
    def _merge_short_segments(
        self, 
        boundaries: List[int], 
        similarities: torch.Tensor,
        total_frames: int
    ) -> List[int]:
        """
        Fusiona segmentos más cortos que min_segment_length.
        
        Fusiona hacia el vecino con mayor similitud en el límite.
        """
        min_len = self.config.min_segment_length
        
        if len(boundaries) <= 1:
            return boundaries
        
        # Calcular longitud de cada segmento
        def get_segment_length(boundaries: List[int], idx: int) -> int:
            if idx >= len(boundaries) - 1:
                return total_frames - boundaries[idx]
            return boundaries[idx + 1] - boundaries[idx]
        
        # Iterar hasta que no haya segmentos cortos
        merged = list(boundaries)
        changed = True
        
        while changed and len(merged) > 1:
            changed = False
            new_merged = [merged[0]]
            i = 1
            
            while i < len(merged):
                prev_len = get_segment_length(new_merged, len(new_merged) - 1)
                curr_start = merged[i]
                
                # Calcular longitud del segmento actual
                if i == len(merged) - 1:
                    curr_len = total_frames - curr_start
                else:
                    curr_len = merged[i + 1] - curr_start
                
                # Si el segmento previo es muy corto, fusionarlo con el actual
                if prev_len < min_len and len(new_merged) > 1:
                    # Remover el límite previo (fusionar hacia adelante)
                    new_merged.pop()
                    changed = True
                # Si el segmento actual es muy corto, no agregar su límite
                elif curr_len < min_len and i < len(merged) - 1:
                    # Skip this boundary (fusionar con siguiente)
                    changed = True
                else:
                    new_merged.append(curr_start)
                
                i += 1
            
            # Verificar último segmento
            if len(new_merged) > 1:
                last_len = total_frames - new_merged[-1]
                if last_len < min_len:
                    new_merged.pop()
                    changed = True
            
            merged = new_merged
        
        return merged
    
    def segment(self, embeddings: torch.Tensor) -> List[Segment]:
        """
        Segmenta una secuencia de embeddings.
        
        Args:
            embeddings: Tensor de shape [T, D] con embeddings temporales
            
        Returns:
            Lista de Segment con límites y estadísticas de cada segmento
        """
        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensor [T, D], got shape {embeddings.shape}")
        
        T = embeddings.shape[0]
        
        if T == 0:
            return []
        
        if T == 1:
            return [Segment(start=0, end=0, mean_similarity=1.0)]
        
        # Calcular similitudes
        similarities = self.compute_similarity(embeddings)
        
        # Determinar umbral
        threshold = self._compute_threshold(similarities)
        
        # Detectar límites iniciales
        boundaries = self._detect_boundaries(similarities, threshold)
        
        # Fusionar segmentos cortos
        boundaries = self._merge_short_segments(boundaries, similarities, T)
        
        # Construir lista de Segments
        segments = []
        for i, start in enumerate(boundaries):
            if i == len(boundaries) - 1:
                end = T - 1
            else:
                end = boundaries[i + 1] - 1
            
            # Calcular similitud media del segmento
            if start == end:
                mean_sim = 1.0
            else:
                seg_sims = similarities[start:end]
                mean_sim = seg_sims.mean().item() if len(seg_sims) > 0 else 1.0
            
            segments.append(Segment(
                start=start,
                end=end,
                mean_similarity=mean_sim
            ))
        
        return segments
