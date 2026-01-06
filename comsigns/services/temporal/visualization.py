"""
Visualización opcional para debugging de segmentación temporal
"""

import torch
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import matplotlib.figure

from .segmenter import Segment


def plot_similarity_timeline(
    similarities: torch.Tensor,
    segments: List[Segment],
    threshold: Optional[float] = None,
    title: str = "Temporal Similarity Timeline",
    output_path: Optional[str] = None
) -> Optional["matplotlib.figure.Figure"]:
    """
    Visualiza la línea de tiempo de similitudes con segmentos detectados.
    
    Args:
        similarities: Tensor [T-1] con similitudes entre frames consecutivos
        segments: Lista de segmentos detectados
        threshold: Umbral usado para detección (línea horizontal)
        title: Título del gráfico
        output_path: Si se proporciona, guarda el gráfico en esta ruta
        
    Returns:
        Figure de matplotlib si está disponible, None si no se puede importar
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no está instalado. Instálalo con: pip install matplotlib")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Convertir a numpy
    if isinstance(similarities, torch.Tensor):
        similarities = similarities.detach().cpu().numpy()
    
    # Graficar similitudes
    x = range(len(similarities))
    ax.plot(x, similarities, 'b-', linewidth=1, label='Cosine Similarity')
    ax.fill_between(x, similarities, alpha=0.3)
    
    # Línea de umbral
    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--', 
                   linewidth=1.5, label=f'Threshold ({threshold:.3f})')
    
    # Marcar límites de segmentos
    colors = plt.cm.Set3.colors
    for i, seg in enumerate(segments):
        color = colors[i % len(colors)]
        
        # Región del segmento
        ax.axvspan(seg.start - 0.5, seg.end + 0.5, 
                   alpha=0.2, color=color, label=f'Seg {i}' if i < 5 else None)
        
        # Línea vertical en el inicio (excepto el primero)
        if seg.start > 0:
            ax.axvline(x=seg.start - 0.5, color='gray', 
                       linestyle=':', linewidth=1)
    
    # Configuración
    ax.set_xlabel('Frame')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado en: {output_path}")
    
    return fig


def print_segments_summary(segments: List[Segment]) -> None:
    """
    Imprime un resumen de los segmentos detectados.
    
    Args:
        segments: Lista de segmentos
    """
    print(f"\n{'='*50}")
    print(f"Segmentos detectados: {len(segments)}")
    print(f"{'='*50}")
    
    total_frames = sum(s.length for s in segments)
    
    for i, seg in enumerate(segments):
        pct = (seg.length / total_frames * 100) if total_frames > 0 else 0
        print(f"  [{i}] Frames {seg.start:3d}-{seg.end:3d} "
              f"(len={seg.length:3d}, {pct:5.1f}%) "
              f"| sim={seg.mean_similarity:.3f}")
    
    print(f"{'='*50}")
    print(f"Total frames: {total_frames}")
