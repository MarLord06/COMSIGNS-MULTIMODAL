"""
Utilidades para el encoder: conversión de features a tensores
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from ..schemas import FeatureClip, FrameKeypoints

logger = logging.getLogger(__name__)


def keypoints_to_tensor(
    keypoints: List[List[float]],
    expected_size: Optional[int] = None,
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Convierte una lista de keypoints a tensor con tamaño fijo.

    Notas:
    - Cada keypoint debe ser [x, y, z]. Cualquier canal extra (p.ej. confidence/visibility)
      será ignorado por esta función.
    - El tensor resultante está aplanado y tiene forma (num_keypoints * 3,) o (expected_size,)

    Args:
        keypoints: Lista de keypoints [[x, y, z], ...] (o con columas extra que serán truncadas)
        expected_size: Tamaño esperado del tensor aplanado (num_keypoints * 3)
        pad_value: Valor para padding si hay menos keypoints

    Returns:
        Tensor aplanado de forma (expected_size,) o (num_keypoints * 3,)
    """
    if not keypoints:
        # Si no hay keypoints, retornar tensor de ceros del tamaño esperado
        if expected_size:
            return torch.zeros(expected_size, dtype=torch.float32)
        else:
            return torch.zeros(0, dtype=torch.float32)

    # Convertir a numpy y luego a tensor
    kp_array = np.array(keypoints, dtype=np.float32)


    # Handle 1D flattened legacy arrays and ensure we end with shape (num_points, 3)
    if kp_array.ndim == 1:
        L = kp_array.shape[0]
        if L % 4 == 0:
            # Legacy format: flattened [x,y,z,conf] per point
            kp_array = kp_array.reshape(-1, 4)
        elif L % 3 == 0:
            # New format: flattened [x,y,z] per point
            kp_array = kp_array.reshape(-1, 3)
        else:
            raise ValueError(f"Unexpected flattened keypoints length: {L}")

    # If 2D, normalize columns: keep only first 3 (x,y,z). If fewer than 3, pad zeros.
    if kp_array.shape[1] >= 3:
        kp_array = kp_array[:, :3]
    else:
        padded = np.zeros((kp_array.shape[0], 3), dtype=np.float32)
        padded[:, :kp_array.shape[1]] = kp_array
        kp_array = padded

    # Aplanar para tener (num_keypoints * 3,)
    kp_flat = kp_array.flatten()

    # Si se especifica un tamaño esperado, hacer padding o truncar
    if expected_size:
        if len(kp_flat) < expected_size:
            # Padding
            padding = np.full(expected_size - len(kp_flat), pad_value, dtype=np.float32)
            kp_flat = np.concatenate([kp_flat, padding])
        elif len(kp_flat) > expected_size:
            # Truncar
            kp_flat = kp_flat[:expected_size]

    return torch.from_numpy(kp_flat)


def feature_clip_to_tensors(
    feature_clip: FeatureClip,
    max_hands: int = 2,
    hand_keypoints_per_hand: int = 21
) -> Dict[str, torch.Tensor]:
    """
    Convierte un FeatureClip a tensores para el encoder

    Args:
        feature_clip: FeatureClip con keypoints
        max_hands: Número máximo de manos a procesar
        hand_keypoints_per_hand: Keypoints por mano

    Returns:
        Diccionario con tensores: {
            'hand': (seq_len, hand_input_dim),
            'body': (seq_len, body_input_dim),
            'face': (seq_len, face_input_dim)
        }
    """
    seq_len = len(feature_clip.frames)

    # Inicializar listas para cada tipo de keypoint
    hand_tensors = []
    body_tensors = []
    face_tensors = []

    # Tamaños fijos esperados para cada tipo de keypoint (3 canales: x,y,z)
    hand_expected_size = max_hands * hand_keypoints_per_hand * 3  # 2 manos * 21 * 3 = 126
    body_expected_size = 33 * 3  # 33 keypoints * 3 = 99
    face_expected_size = 468 * 3  # 468 keypoints * 3 = 1404

    for frame in feature_clip.frames:
        # Procesar keypoints de manos con tamaño fijo
        hand_tensor = keypoints_to_tensor(
            frame.hand_keypoints,
            expected_size=hand_expected_size,
            pad_value=0.0
        )
        hand_tensors.append(hand_tensor)

        # Procesar keypoints del cuerpo con tamaño fijo
        body_tensor = keypoints_to_tensor(
            frame.body_keypoints,
            expected_size=body_expected_size,
            pad_value=0.0
        )
        body_tensors.append(body_tensor)

        # Procesar keypoints del rostro con tamaño fijo
        face_tensor = keypoints_to_tensor(
            frame.face_keypoints,
            expected_size=face_expected_size,
            pad_value=0.0
        )
        face_tensors.append(face_tensor)

    # Apilar en secuencias
    hand_seq = torch.stack(hand_tensors)  # (seq_len, hand_input_dim)
    body_seq = torch.stack(body_tensors)  # (seq_len, body_input_dim)
    face_seq = torch.stack(face_tensors)  # (seq_len, face_input_dim)

    return {
        'hand': hand_seq,
        'body': body_seq,
        'face': face_seq
    }

