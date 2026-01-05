"""
Ejemplo básico de uso del sistema COMSIGNS
"""

import logging
from pathlib import Path

from comsigns.services.ingestion import capture_from_file, validate_video
from comsigns.services.preprocessing import process_video_clip
from comsigns.services.encoder import create_encoder, feature_clip_to_tensors
import torch

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Ejemplo de pipeline completo"""
    
    # 1. Validar video
    video_path = "data/raw/videos/example.mp4"
    logger.info(f"Validando video: {video_path}")
    
    is_valid, error = validate_video(video_path)
    if not is_valid:
        logger.error(f"Video inválido: {error}")
        return
    
    # 2. Procesar video (si no existe, usar capture_from_file primero)
    logger.info("Procesando video y extrayendo keypoints...")
    feature_clip = process_video_clip(
        video_path=video_path,
        fps=30.0,
        normalize=True,
        format="json"
    )
    
    logger.info(f"Keypoints extraídos: {len(feature_clip.frames)} frames")
    
    # 3. Convertir a tensores
    logger.info("Convirtiendo a tensores...")
    tensors = feature_clip_to_tensors(feature_clip)
    
    # 4. Crear encoder
    logger.info("Creando encoder multimodal...")
    encoder = create_encoder()
    encoder.eval()
    
    # 5. Codificar
    logger.info("Codificando features...")
    with torch.no_grad():
        # Agregar dimensión de batch
        hand_t = tensors['hand'].unsqueeze(0)
        body_t = tensors['body'].unsqueeze(0)
        face_t = tensors['face'].unsqueeze(0)
        
        embeddings = encoder(hand_t, body_t, face_t)
    
    logger.info(f"Embeddings generados: {embeddings.shape}")
    logger.info("Pipeline completado exitosamente!")


if __name__ == "__main__":
    main()

