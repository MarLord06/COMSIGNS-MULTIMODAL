"""
Encoder Multimodal para procesar keypoints de manos, cuerpo y rostro
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional, Dict

from ..config import config

logger = logging.getLogger(__name__)


class HandBranch(nn.Module):
    """Rama del encoder para procesar keypoints de manos"""

    def __init__(
        self,
        input_dim: int = 21 * 4,  # 21 keypoints * [x, y, z, confidence]
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Inicializa la rama de manos

        Args:
            input_dim: Dimensión de entrada (keypoints aplanados)
            hidden_dim: Dimensión oculta
            num_layers: Número de capas LSTM
            dropout: Tasa de dropout
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Proyección inicial
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM para procesamiento temporal
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Normalización
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa keypoints de manos

        Args:
            x: Tensor de forma (batch, seq_len, input_dim)

        Returns:
            Tensor de forma (batch, seq_len, hidden_dim)
        """
        # Proyección inicial
        x = self.input_proj(x)
        x = F.relu(x)

        # LSTM
        x, _ = self.lstm(x)

        # Normalización
        x = self.layer_norm(x)

        return x


class BodyBranch(nn.Module):
    """Rama del encoder para procesar keypoints del cuerpo"""

    def __init__(
        self,
        input_dim: int = 33 * 4,  # 33 keypoints * [x, y, z, visibility]
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Inicializa la rama del cuerpo

        Args:
            input_dim: Dimensión de entrada
            hidden_dim: Dimensión oculta
            num_layers: Número de capas LSTM
            dropout: Tasa de dropout
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Proyección inicial
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Normalización
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa keypoints del cuerpo

        Args:
            x: Tensor de forma (batch, seq_len, input_dim)

        Returns:
            Tensor de forma (batch, seq_len, hidden_dim)
        """
        x = self.input_proj(x)
        x = F.relu(x)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        return x


class FaceBranch(nn.Module):
    """Rama del encoder para procesar keypoints del rostro"""

    def __init__(
        self,
        input_dim: int = 468 * 4,  # 468 keypoints * [x, y, z, confidence]
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Inicializa la rama del rostro

        Args:
            input_dim: Dimensión de entrada
            hidden_dim: Dimensión oculta
            num_layers: Número de capas LSTM
            dropout: Tasa de dropout
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Reducción de dimensionalidad inicial (468 es mucho)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Normalización
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa keypoints del rostro

        Args:
            x: Tensor de forma (batch, seq_len, input_dim)

        Returns:
            Tensor de forma (batch, seq_len, hidden_dim)
        """
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        return x


class MultimodalEncoder(nn.Module):
    """
    Encoder multimodal que combina tres ramas (manos, cuerpo, rostro)
    """

    def __init__(
        self,
        hand_input_dim: int = 21 * 4 * 2,  # 2 manos * 21 keypoints * 4 valores
        body_input_dim: int = 33 * 4,
        face_input_dim: int = 468 * 4,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Inicializa el encoder multimodal

        Args:
            hand_input_dim: Dimensión de entrada de manos (por defecto 168 para 2 manos)
            body_input_dim: Dimensión de entrada del cuerpo (132)
            face_input_dim: Dimensión de entrada del rostro (1872)
            hidden_dim: Dimensión oculta de cada rama
            output_dim: Dimensión de salida final
            num_layers: Número de capas LSTM por rama
            dropout: Tasa de dropout
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Tres ramas
        self.hand_branch = HandBranch(hand_input_dim, hidden_dim, num_layers, dropout)
        self.body_branch = BodyBranch(body_input_dim, hidden_dim, num_layers, dropout)
        self.face_branch = FaceBranch(face_input_dim, hidden_dim, num_layers, dropout)

        # Fusión de embeddings
        # Cada rama produce hidden_dim, combinamos las 3
        fusion_dim = hidden_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        hand_keypoints: torch.Tensor,
        body_keypoints: torch.Tensor,
        face_keypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Procesa keypoints de las tres ramas y los fusiona

        Args:
            hand_keypoints: Tensor (batch, seq_len, hand_input_dim)
            body_keypoints: Tensor (batch, seq_len, body_input_dim)
            face_keypoints: Tensor (batch, seq_len, face_input_dim)

        Returns:
            Tensor fusionado (batch, seq_len, output_dim)
        """
        # Procesar cada rama
        hand_emb = self.hand_branch(hand_keypoints)
        body_emb = self.body_branch(body_keypoints)
        face_emb = self.face_branch(face_keypoints)

        # Concatenar embeddings
        fused = torch.cat([hand_emb, body_emb, face_emb], dim=-1)

        # Proyección final
        output = self.fusion(fused)

        return output

    def encode_features(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Método de conveniencia para codificar features desde diccionario

        Args:
            features: Diccionario con 'hand', 'body', 'face' keypoints

        Returns:
            Tensor fusionado
        """
        return self.forward(
            features['hand'],
            features['body'],
            features['face']
        )


def create_encoder(
    config_path: Optional[str] = None,
    **kwargs
) -> MultimodalEncoder:
    """
    Crea un encoder multimodal desde configuración

    Args:
        config_path: Ruta al archivo de configuración
        **kwargs: Parámetros adicionales para sobrescribir configuración

    Returns:
        MultimodalEncoder inicializado
    """
    cfg = config.load()
    encoder_config = cfg.get('encoder', {})

    # Parámetros del encoder
    params = {
        'hidden_dim': encoder_config.get('hidden_dim', 256),
        'output_dim': encoder_config.get('output_dim', 512),
        'num_layers': encoder_config.get('num_layers', 2),
        'dropout': encoder_config.get('dropout', 0.1),
    }

    # Sobrescribir con kwargs
    params.update(kwargs)

    encoder = MultimodalEncoder(**params)
    logger.info(f"Encoder creado: hidden_dim={params['hidden_dim']}, output_dim={params['output_dim']}")

    return encoder

