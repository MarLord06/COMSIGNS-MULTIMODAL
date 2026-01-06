"""
API FastAPI principal para inferencia de video
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict
import uvicorn
import asyncio
import base64
import uuid
import time
import io
from collections import deque
from PIL import Image
import numpy as np

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
    from comsigns.services.glosador import create_glosador
    from comsigns.services.translator import create_translator, TextAccumulator
except ImportError:
    # Fallback a importaciones relativas si no funciona la absoluta
    from ..config import config
    from ..schemas import FeatureClip
    from ..ingestion import capture_from_file, validate_video
    from ..preprocessing import process_video_clip
    from ..encoder import create_encoder, feature_clip_to_tensors
    from ..glosador import create_glosador
    from ..translator import create_translator, TextAccumulator

import torch
import cv2

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

# Inicializar modelos (se cargan bajo demanda)
_encoder: Optional[torch.nn.Module] = None
_glosador: Optional[torch.nn.Module] = None
_translator: Optional[torch.nn.Module] = None

# Sesiones activas de WebSocket
active_sessions: Dict[str, 'SessionState'] = {}


class SessionState:
    """Estado de sesión para conexión WebSocket"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.frame_buffer = deque(maxlen=90)  # 3 segundos a 30fps
        self.text_accumulator = TextAccumulator(max_history=100)
        self.last_activity = time.time()
        self.frame_count = 0
        self.last_gloss = None
        self.last_confidence = 0.0
        
    def add_frame(self, frame_data: dict):
        """Agrega un frame al buffer"""
        self.frame_buffer.append(frame_data)
        self.frame_count += 1
        self.last_activity = time.time()
    
    def add_translation(self, gloss: str, translation: str, confidence: float):
        """Agrega una traducción al acumulador"""
        self.text_accumulator.add(gloss, translation)
        self.last_gloss = gloss
        self.last_confidence = confidence
        
    def reset(self):
        """Resetea el estado de la sesión"""
        self.frame_buffer.clear()
        self.text_accumulator.reset()
        self.frame_count = 0
        self.last_gloss = None
        self.last_confidence = 0.0


def get_encoder() -> torch.nn.Module:
    """Obtiene o crea el encoder (lazy loading)"""
    global _encoder
    if _encoder is None:
        logger.info("Cargando encoder multimodal...")
        _encoder = create_encoder()
        _encoder.eval()  # Modo evaluación
        logger.info("Encoder cargado")
    return _encoder


def get_glosador() -> torch.nn.Module:
    """Obtiene o crea el glosador (lazy loading)"""
    global _glosador
    if _glosador is None:
        logger.info("Cargando glosador...")
        _glosador = create_glosador()
        _glosador.eval()  # Modo evaluación
        logger.info("Glosador cargado")
    return _glosador


def get_translator() -> torch.nn.Module:
    """Obtiene o crea el traductor (lazy loading)"""
    global _translator
    if _translator is None:
        logger.info("Cargando traductor...")
        _translator = create_translator()
        _translator.eval()  # Modo evaluación
        logger.info("Traductor cargado")
    return _translator


def decode_base64_frame(base64_str: str) -> np.ndarray:
    """Decodifica un frame en base64 a numpy array"""
    try:
        # Remover prefijo data:image si existe
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Decodificar base64
        img_bytes = base64.b64decode(base64_str)
        
        # Convertir a imagen PIL
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convertir a numpy array (RGB)
        frame = np.array(img)
        
        # Convertir RGB a BGR para OpenCV si es necesario
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame
    except Exception as e:
        logger.error(f"Error decodificando frame base64: {e}")
        raise ValueError(f"Error decodificando frame: {str(e)}")


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "COMSIGNS API",
        "version": "0.1.0",
        "endpoints": {
            "/infer/video": "POST - Inferencia de video (archivo completo)",
            "/ws/infer": "WebSocket - Inferencia en tiempo real (frames continuos)",
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


@app.websocket("/ws/infer")
async def websocket_infer(websocket: WebSocket):
    """
    Endpoint WebSocket para inferencia en tiempo real
    
    Recibe frames continuos desde el cliente y retorna predicciones instantáneas
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = SessionState(session_id)
    active_sessions[session_id] = session
    
    logger.info(f"Nueva conexión WebSocket: {session_id}")
    
    try:
        # Enviar confirmación de conexión
        await websocket.send_json({
            "type": "status",
            "status": "connected",
            "session_id": session_id,
            "message": "Conexión establecida correctamente"
        })
        
        while True:
            # Recibir mensaje del cliente
            message = await websocket.receive_json()
            message_type = message.get("type")
            
            if message_type == "frame":
                # Procesar frame recibido
                try:
                    start_time = time.time()
                    
                    frame_data = message.get("data", {})
                    frame_base64 = frame_data.get("frame")
                    timestamp = frame_data.get("timestamp")
                    sequence = frame_data.get("sequence", 0)
                    
                    if not frame_base64:
                        await websocket.send_json({
                            "type": "error",
                            "error": "Frame vacío",
                            "code": "EMPTY_FRAME"
                        })
                        continue
                    
                    # Decodificar frame
                    frame = decode_base64_frame(frame_base64)
                    
                    # Guardar frame temporal para procesamiento
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        cv2.imwrite(tmp_file.name, frame)
                        tmp_path = tmp_file.name
                    
                    try:
                        # Procesar frame con MediaPipe (usando process_video_clip con 1 frame)
                        # Nota: Esta es una simplificación, idealmente deberíamos tener
                        # una función específica para procesar frames individuales
                        feature_clip = process_video_clip(
                            tmp_path,
                            fps=30,
                            normalize=True,
                            format="json"
                        )
                        
                        if not feature_clip.frames:
                            await websocket.send_json({
                                "type": "error",
                                "error": "No se pudieron extraer keypoints",
                                "code": "KEYPOINT_EXTRACTION_FAILED"
                            })
                            continue
                        
                        # Agregar frame al buffer de sesión
                        session.add_frame({
                            "timestamp": timestamp,
                            "sequence": sequence,
                            "keypoints": feature_clip.frames[0]
                        })
                        
                        # Convertir a tensores
                        tensors = feature_clip_to_tensors(feature_clip)
                        
                        # Codificar con el encoder
                        encoder = get_encoder()
                        
                        with torch.no_grad():
                            # Agregar dimensión de batch
                            hand_t = tensors['hand'].unsqueeze(0)
                            body_t = tensors['body'].unsqueeze(0)
                            face_t = tensors['face'].unsqueeze(0)
                            
                            # Inferencia con encoder
                            embeddings = encoder(hand_t, body_t, face_t)
                        
                        # Glosador: convertir embeddings a glosa
                        glosador = get_glosador()
                        gloss, confidence = glosador.decode_sequence(embeddings)
                        
                        # Traductor: convertir glosa a texto español
                        translator = get_translator()
                        translation = translator.translate_with_context(
                            gloss,
                            session.text_accumulator.get_recent_glosses(n=5)
                        )
                        
                        # Agregar traducción al acumulador de sesión
                        session.add_translation(gloss, translation, confidence)
                        
                        # Calcular tiempo de procesamiento
                        processing_time = (time.time() - start_time) * 1000
                        
                        # Enviar predicción al cliente
                        await websocket.send_json({
                            "type": "prediction",
                            "data": {
                                "sequence": sequence,
                                "gloss": gloss,
                                "confidence": float(confidence),
                                "text": translation,
                                "accumulated_text": session.text_accumulator.get_accumulated_text(),
                                "processing_time_ms": round(processing_time, 2),
                                "frames_in_buffer": len(session.frame_buffer)
                            }
                        })
                        
                    finally:
                        # Limpiar archivo temporal
                        Path(tmp_path).unlink(missing_ok=True)
                    
                except ValueError as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                        "code": "FRAME_DECODE_ERROR"
                    })
                except Exception as e:
                    logger.error(f"Error procesando frame: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Error procesando frame: {str(e)}",
                        "code": "PROCESSING_ERROR"
                    })
            
            elif message_type == "control":
                # Manejar comandos de control
                action = message.get("action")
                
                if action == "reset":
                    session.reset()
                    await websocket.send_json({
                        "type": "status",
                        "status": "reset",
                        "message": "Sesión reiniciada"
                    })
                elif action == "stop":
                    await websocket.send_json({
                        "type": "status",
                        "status": "stopped",
                        "message": "Procesamiento detenido"
                    })
                    break
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Tipo de mensaje desconocido: {message_type}",
                    "code": "UNKNOWN_MESSAGE_TYPE"
                })
    
    except WebSocketDisconnect:
        logger.info(f"Cliente desconectado: {session_id}")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}", exc_info=True)
    finally:
        # Limpiar sesión
        if session_id in active_sessions:
            del active_sessions[session_id]
        logger.info(f"Sesión cerrada: {session_id}")


if __name__ == "__main__":
    api_config = cfg.get('api', {})
    # Usar run_api.py en su lugar, o ejecutar directamente con uvicorn desde el directorio raíz
    uvicorn.run(
        app,  # Usar la instancia de app directamente
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=True
    )

