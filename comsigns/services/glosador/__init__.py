"""
Módulo Glosador para COMSIGNS
Convierte embeddings temporales en secuencias de glosas (glosses)

Este módulo proporciona una interfaz para el modelo de glosado.
Actualmente usa un placeholder que puede ser reemplazado con el modelo real.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GlosadorPlaceholder(nn.Module):
    """
    Placeholder para el modelo de glosado.
    
    Este modelo será reemplazado por el modelo real (ej: SLTUNET-like con CTC/Seq2Seq).
    La interfaz se mantiene para facilitar la integración.
    """
    
    def __init__(self, embedding_dim: int = 512, vocab_size: int = 1000):
        """
        Args:
            embedding_dim: Dimensión de los embeddings de entrada
            vocab_size: Tamaño del vocabulario de glosas
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Placeholder: red simple para demostración
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
        # Vocabulario de ejemplo (reemplazar con vocabulario real)
        self.idx_to_gloss = {
            0: "HOLA",
            1: "GRACIAS",
            2: "POR_FAVOR",
            3: "ADIOS",
            4: "SI",
            5: "NO",
            6: "AYUDA",
            7: "BIEN",
            8: "MAL",
            9: "COMO_ESTAS",
            # Agregar más glosas según el vocabulario real
        }
        
        logger.info(f"GlosadorPlaceholder inicializado (embedding_dim={embedding_dim}, vocab_size={vocab_size})")
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Procesa embeddings y retorna logits de glosas.
        
        Args:
            embeddings: Tensor de forma (batch_size, seq_len, embedding_dim)
        
        Returns:
            logits: Tensor de forma (batch_size, seq_len, vocab_size)
        """
        # Placeholder: simplemente pasa por una capa lineal
        logits = self.fc(embeddings)
        return logits
    
    def decode(self, embeddings: torch.Tensor, return_confidence: bool = True) -> List[Tuple[str, float]]:
        """
        Decodifica embeddings a glosas con confianza.
        
        Args:
            embeddings: Tensor de forma (batch_size, seq_len, embedding_dim)
            return_confidence: Si retornar confianza junto con la glosa
        
        Returns:
            Lista de tuplas (glosa, confianza) para cada frame
        """
        with torch.no_grad():
            logits = self.forward(embeddings)
            probs = torch.softmax(logits, dim=-1)
            
            # Obtener la glosa más probable y su confianza
            confidences, indices = torch.max(probs, dim=-1)
            
            results = []
            for batch_idx in range(embeddings.size(0)):
                batch_results = []
                for seq_idx in range(embeddings.size(1)):
                    idx = indices[batch_idx, seq_idx].item()
                    conf = confidences[batch_idx, seq_idx].item()
                    gloss = self.idx_to_gloss.get(idx % len(self.idx_to_gloss), "DESCONOCIDO")
                    batch_results.append((gloss, conf))
                results.append(batch_results)
            
            return results[0] if len(results) == 1 else results
    
    def decode_sequence(self, embeddings: torch.Tensor) -> Tuple[str, float]:
        """
        Decodifica una secuencia completa de embeddings a una sola glosa.
        Útil para procesar múltiples frames y obtener la glosa más probable.
        
        Args:
            embeddings: Tensor de forma (batch_size, seq_len, embedding_dim)
        
        Returns:
            Tupla (glosa, confianza_promedio)
        """
        results = self.decode(embeddings, return_confidence=True)
        
        if not results:
            return "DESCONOCIDO", 0.0
        
        # Estrategia simple: tomar la glosa más frecuente
        from collections import Counter
        glosses = [gloss for gloss, _ in results]
        confidences = [conf for _, conf in results]
        
        if not glosses:
            return "DESCONOCIDO", 0.0
        
        # Glosa más común
        most_common_gloss = Counter(glosses).most_common(1)[0][0]
        
        # Confianza promedio para esa glosa
        avg_confidence = sum(conf for gloss, conf in results if gloss == most_common_gloss) / len([g for g in glosses if g == most_common_gloss])
        
        return most_common_gloss, avg_confidence


def create_glosador(model_path: Optional[str] = None, device: str = "cpu") -> GlosadorPlaceholder:
    """
    Crea y carga el modelo de glosado.
    
    Args:
        model_path: Ruta al modelo entrenado (opcional para placeholder)
        device: Dispositivo para el modelo ('cpu' o 'cuda')
    
    Returns:
        Modelo de glosado listo para inferencia
    """
    model = GlosadorPlaceholder()
    model.to(device)
    model.eval()
    
    if model_path:
        try:
            logger.info(f"Cargando modelo de glosado desde {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("Modelo de glosado cargado exitosamente")
        except Exception as e:
            logger.warning(f"No se pudo cargar el modelo: {e}. Usando placeholder.")
    else:
        logger.info("Usando GlosadorPlaceholder (sin modelo entrenado)")
    
    return model


# Ejemplo de uso
if __name__ == "__main__":
    # Crear glosador
    glosador = create_glosador()
    
    # Simular embeddings (batch_size=1, seq_len=30, embedding_dim=512)
    embeddings = torch.randn(1, 30, 512)
    
    # Decodificar secuencia
    gloss, confidence = glosador.decode_sequence(embeddings)
    print(f"Glosa detectada: {gloss} (confianza: {confidence:.2f})")
    
    # Decodificar frame por frame
    results = glosador.decode(embeddings)
    print(f"\nPrimeros 5 frames:")
    for i, (gloss, conf) in enumerate(results[:5]):
        print(f"  Frame {i}: {gloss} ({conf:.2f})")
