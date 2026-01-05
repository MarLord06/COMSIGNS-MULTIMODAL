"""
Utilidades para el módulo de ingestion
"""

import cv2
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from ..config import config

logger = logging.getLogger(__name__)


def validate_video(video_path: str) -> Tuple[bool, Optional[str]]:
    """
    Valida que un archivo de video sea válido y pueda ser procesado

    Args:
        video_path: Ruta al archivo de video

    Returns:
        Tupla (es_válido, mensaje_error)
    """
    video_path = Path(video_path)

    if not video_path.exists():
        return False, f"Archivo no encontrado: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False, f"No se pudo abrir el video: {video_path}"

    # Verificar que tenga frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return False, "El video no contiene frames"

    # Verificar FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return False, "FPS inválido o no disponible"

    # Verificar resolución
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return False, "Resolución inválida"

    cap.release()
    return True, None


def extract_frames(
    video_path: str,
    output_dir: Optional[str] = None,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None
) -> List[str]:
    """
    Extrae frames de un video y los guarda como imágenes

    Args:
        video_path: Ruta al archivo de video
        output_dir: Directorio donde guardar los frames
        fps: FPS objetivo para extracción (None = todos los frames)
        max_frames: Número máximo de frames a extraer (None = todos)

    Returns:
        Lista de rutas a los frames extraídos
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")

    cfg = config.load()
    if output_dir is None:
        output_dir = Path(cfg.get('preprocessing.output_dir', 'data/processed/frames'))
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = fps or video_fps
    frame_interval = max(1, int(video_fps / target_fps)) if target_fps < video_fps else 1

    frame_paths = []
    frame_count = 0
    saved_count = 0

    logger.info(f"Extrayendo frames de {video_path} (FPS objetivo: {target_fps})")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extraer frame según intervalo
            if frame_count % frame_interval == 0:
                frame_filename = f"{video_path.stem}_frame_{saved_count:06d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
                saved_count += 1

                if max_frames and saved_count >= max_frames:
                    break

            frame_count += 1

    finally:
        cap.release()

    logger.info(f"Extraídos {len(frame_paths)} frames en {output_dir}")
    return frame_paths


def get_video_info(video_path: str) -> dict:
    """
    Obtiene información sobre un video

    Args:
        video_path: Ruta al archivo de video

    Returns:
        Diccionario con información del video
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }

    cap.release()
    return info

