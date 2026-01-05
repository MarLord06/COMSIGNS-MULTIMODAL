"""
Procesamiento de clips: normalización y guardado de features
"""

import json
import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd

from ..config import config
from ..schemas import FeatureClip, FrameKeypoints
from .extract_keypoints import KeypointExtractor

logger = logging.getLogger(__name__)


def normalize_keypoints(
    keypoints: List[List[float]],
    method: str = "relative"
) -> List[List[float]]:
    """
    Normaliza keypoints según el método especificado

    Args:
        keypoints: Lista de keypoints [[x, y, z, confidence], ...]
        method: Método de normalización ('relative' o 'absolute')

    Returns:
        Lista de keypoints normalizados
    """
    if not keypoints:
        return []

    if method == "relative":
        # Los keypoints de MediaPipe ya vienen normalizados [0, 1]
        # Solo aseguramos que estén en el rango correcto
        normalized = []
        for kp in keypoints:
            normalized.append([
                max(0.0, min(1.0, kp[0])),  # x
                max(0.0, min(1.0, kp[1])),  # y
                kp[2] if len(kp) > 2 else 0.0,  # z
                max(0.0, min(1.0, kp[3] if len(kp) > 3 else 1.0))  # confidence
            ])
        return normalized

    elif method == "absolute":
        # Normalización centrada en el centroide
        if len(keypoints) == 0:
            return []

        # Calcular centroide
        xs = [kp[0] for kp in keypoints]
        ys = [kp[1] for kp in keypoints]
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)

        # Calcular escala (distancia máxima desde el centroide)
        max_dist = 0.0
        for kp in keypoints:
            dist = ((kp[0] - centroid_x) ** 2 + (kp[1] - centroid_y) ** 2) ** 0.5
            max_dist = max(max_dist, dist)

        # Normalizar respecto al centroide
        normalized = []
        scale = max_dist if max_dist > 0 else 1.0
        for kp in keypoints:
            normalized.append([
                (kp[0] - centroid_x) / scale,  # x centrado y escalado
                (kp[1] - centroid_y) / scale,  # y centrado y escalado
                kp[2] if len(kp) > 2 else 0.0,  # z sin cambios
                max(0.0, min(1.0, kp[3] if len(kp) > 3 else 1.0))  # confidence
            ])
        return normalized

    else:
        raise ValueError(f"Método de normalización desconocido: {method}")


def process_video_clip(
    video_path: str,
    output_path: Optional[str] = None,
    fps: Optional[float] = None,
    normalize: bool = True,
    format: str = "json"
) -> FeatureClip:
    """
    Procesa un clip de video completo: extrae keypoints y los guarda

    Args:
        video_path: Ruta al archivo de video
        output_path: Ruta donde guardar las features (None = auto)
        fps: FPS objetivo para procesamiento
        normalize: Si normalizar los keypoints
        format: Formato de salida ('json' o 'parquet')

    Returns:
        FeatureClip procesado
    """
    cfg = config.load()
    
    # Extraer keypoints
    extractor = KeypointExtractor()
    feature_clip = extractor.extract_from_video(video_path, fps=fps)

    # Normalizar si está habilitado
    if normalize:
        normalization_method = cfg.get('preprocessing.normalization.method', 'relative')
        logger.info(f"Normalizando keypoints con método: {normalization_method}")

        normalized_frames = []
        for frame in feature_clip.frames:
            normalized_frame = FrameKeypoints(
                t=frame.t,
                hand_keypoints=normalize_keypoints(frame.hand_keypoints, method=normalization_method),
                body_keypoints=normalize_keypoints(frame.body_keypoints, method=normalization_method),
                face_keypoints=normalize_keypoints(frame.face_keypoints, method=normalization_method)
            )
            normalized_frames.append(normalized_frame)

        feature_clip.frames = normalized_frames

    # Determinar ruta de salida
    if output_path is None:
        output_dir = Path(cfg.get('preprocessing.output_dir', 'data/features'))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{feature_clip.clip_id}.{format}"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar features
    if format == "json":
        save_features_json(feature_clip, output_path)
    elif format == "parquet":
        save_features_parquet(feature_clip, output_path)
    else:
        raise ValueError(f"Formato no soportado: {format}")

    logger.info(f"Features guardadas en: {output_path}")
    return feature_clip


def save_features_json(feature_clip: FeatureClip, output_path: Path) -> None:
    """Guarda features en formato JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(feature_clip.to_dict(), f, indent=2, ensure_ascii=False)


def save_features_parquet(feature_clip: FeatureClip, output_path: Path) -> None:
    """Guarda features en formato Parquet"""
    # Convertir a formato tabular para Parquet
    rows = []
    for frame in feature_clip.frames:
        row = {
            'clip_id': feature_clip.clip_id,
            't': frame.t,
            'hand_keypoints': frame.hand_keypoints,
            'body_keypoints': frame.body_keypoints,
            'face_keypoints': frame.face_keypoints,
            'fps': feature_clip.fps
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False, engine='pyarrow')

