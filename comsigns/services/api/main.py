"""
API FastAPI principal para inferencia de video
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import tempfile
from pathlib import Path
from typing import Optional
import uvicorn

import sys
from pathlib import Path

# Agregar el directorio padre al path para importaciones
# Esto permite que el módulo funcione sin instalación
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from comsigns.services.config import config
    from comsigns.services.schemas import FeatureClip
    from comsigns.services.ingestion import capture_from_file, validate_video
    from comsigns.services.preprocessing import process_video_clip
    from comsigns.services.encoder import create_encoder, feature_clip_to_tensors
except ImportError:
    # Fallback a importaciones relativas si no funciona la absoluta
    from ..config import config
    from ..schemas import FeatureClip
    from ..ingestion import capture_from_file, validate_video
    from ..preprocessing import process_video_clip
    from ..encoder import create_encoder, feature_clip_to_tensors

import torch

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="COMSIGNS API",
    description="API para inferencia de Lengua de Señas",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar configuración
cfg = config.load()

# Inicializar encoder (se carga bajo demanda)
_encoder: Optional[torch.nn.Module] = None


def get_encoder() -> torch.nn.Module:
    """Obtiene o crea el encoder (lazy loading)"""
    global _encoder
    if _encoder is None:
        logger.info("Cargando encoder multimodal...")
        _encoder = create_encoder()
        _encoder.eval()  # Modo evaluación
        logger.info("Encoder cargado")
    return _encoder


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "COMSIGNS API",
        "version": "0.1.0",
        "endpoints": {
            "/infer/video": "POST - Inferencia de video",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.post("/infer/video")
async def infer_video(
    file: UploadFile = File(...),
    fps: Optional[float] = None,
    normalize: bool = True
):
    """
    Endpoint para inferencia de video

    Args:
        file: Archivo de video
        fps: FPS objetivo para procesamiento (None = usar FPS del video)
        normalize: Si normalizar keypoints

    Returns:
        JSON con resultados de inferencia
    """
    try:
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser un video"
            )

        logger.info(f"Procesando video: {file.filename}")

        # Guardar archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Validar video
            is_valid, error_msg = validate_video(tmp_path)
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Video inválido: {error_msg}"
                )

            # Procesar video: extraer keypoints
            logger.info("Extrayendo keypoints...")
            feature_clip = process_video_clip(
                tmp_path,
                fps=fps,
                normalize=normalize,
                format="json"
            )

            # Validar que se extrajeron keypoints
            if not feature_clip.frames:
                raise HTTPException(
                    status_code=500,
                    detail="No se pudieron extraer keypoints del video"
                )

            # Convertir a tensores
            logger.info("Convirtiendo a tensores...")
            tensors = feature_clip_to_tensors(feature_clip)

            # Codificar con el encoder
            logger.info("Codificando con encoder multimodal...")
            encoder = get_encoder()

            with torch.no_grad():
                # Agregar dimensión de batch
                hand_t = tensors['hand'].unsqueeze(0)
                body_t = tensors['body'].unsqueeze(0)
                face_t = tensors['face'].unsqueeze(0)

                # Inferencia
                embeddings = encoder(hand_t, body_t, face_t)

                # Convertir a lista para JSON
                embeddings_list = embeddings.squeeze(0).cpu().numpy().tolist()

            # Preparar respuesta
            response = {
                "clip_id": feature_clip.clip_id,
                "status": "success",
                "fps": feature_clip.fps,
                "num_frames": len(feature_clip.frames),
                "embedding_shape": list(embeddings.shape),
                "embeddings": embeddings_list,
                "metadata": {
                    "duration": feature_clip.meta.duration,
                    "resolution": feature_clip.meta.resolution,
                    "source": feature_clip.meta.source
                }
            }

            logger.info(f"Inferencia completada: {feature_clip.clip_id}")
            return JSONResponse(content=response)

        finally:
            # Limpiar archivo temporal
            Path(tmp_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en inferencia: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando video: {str(e)}"
        )


if __name__ == "__main__":
    api_config = cfg.get('api', {})
    # Usar run_api.py en su lugar, o ejecutar directamente con uvicorn desde el directorio raíz
    uvicorn.run(
        app,  # Usar la instancia de app directamente
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=True
    )

