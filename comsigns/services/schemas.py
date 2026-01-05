"""
Esquemas Pydantic para validación de datos en COMSIGNS
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import uuid


class Keypoint(BaseModel):
    """Representa un keypoint individual con coordenadas y confianza"""
    x: float = Field(..., ge=0.0, le=1.0, description="Coordenada X normalizada [0,1]")
    y: float = Field(..., ge=0.0, le=1.0, description="Coordenada Y normalizada [0,1]")
    z: Optional[float] = Field(None, description="Coordenada Z (profundidad) si está disponible")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza del keypoint [0,1]")

    @field_validator('x', 'y')
    @classmethod
    def validate_coordinates(cls, v: float) -> float:
        """Valida que las coordenadas estén en el rango [0,1]"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Coordenada debe estar en [0,1], recibido: {v}")
        return v


class FrameKeypoints(BaseModel):
    """Keypoints de un frame individual"""
    t: float = Field(..., ge=0.0, description="Tiempo del frame en segundos")
    hand_keypoints: List[List[float]] = Field(
        default_factory=list,
        description="Keypoints de manos: lista de [x, y, confidence] o [x, y, z, confidence]"
    )
    body_keypoints: List[List[float]] = Field(
        default_factory=list,
        description="Keypoints del cuerpo: lista de [x, y, confidence] o [x, y, z, confidence]"
    )
    face_keypoints: List[List[float]] = Field(
        default_factory=list,
        description="Keypoints del rostro: lista de [x, y, confidence] o [x, y, z, confidence]"
    )

    @field_validator('hand_keypoints', 'body_keypoints', 'face_keypoints')
    @classmethod
    def validate_keypoints_format(cls, v: List[List[float]]) -> List[List[float]]:
        """Valida que cada keypoint tenga al menos 3 valores [x, y, confidence]"""
        for kp in v:
            if len(kp) < 3:
                raise ValueError(f"Keypoint debe tener al menos [x, y, confidence], recibido: {kp}")
            if not all(isinstance(val, (int, float)) for val in kp):
                raise ValueError(f"Keypoint debe contener solo números, recibido: {kp}")
        return v


class ClipMetadata(BaseModel):
    """Metadatos del clip de video"""
    camera_id: Optional[str] = Field(None, description="ID de la cámara")
    user_id: Optional[str] = Field(None, description="ID del usuario")
    duration: Optional[float] = Field(None, ge=0.0, description="Duración del clip en segundos")
    resolution: Optional[Dict[str, int]] = Field(
        None,
        description="Resolución del video: {'width': int, 'height': int}"
    )
    source: Optional[str] = Field(None, description="Fuente del video (webcam, file, etc.)")


class FeatureClip(BaseModel):
    """Esquema completo de features de un clip de video"""
    clip_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="ID único del clip"
    )
    fps: float = Field(..., gt=0.0, description="Frames por segundo del video")
    frames: List[FrameKeypoints] = Field(..., min_length=1, description="Lista de frames con keypoints")
    meta: ClipMetadata = Field(default_factory=ClipMetadata, description="Metadatos del clip")

    @field_validator('fps')
    @classmethod
    def validate_fps(cls, v: float) -> float:
        """Valida que el FPS sea razonable"""
        if v <= 0 or v > 120:
            raise ValueError(f"FPS debe estar en (0, 120], recibido: {v}")
        return v

    @field_validator('frames')
    @classmethod
    def validate_frames_ordered(cls, v: List[FrameKeypoints]) -> List[FrameKeypoints]:
        """Valida que los frames estén ordenados por tiempo"""
        if len(v) > 1:
            times = [f.t for f in v]
            if times != sorted(times):
                raise ValueError("Los frames deben estar ordenados por tiempo (t)")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el esquema a diccionario"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureClip':
        """Crea un FeatureClip desde un diccionario"""
        return cls(**data)


class VideoManifest(BaseModel):
    """Manifest JSON para videos procesados"""
    video_id: str = Field(..., description="ID único del video")
    video_path: str = Field(..., description="Ruta al archivo de video")
    features_path: Optional[str] = Field(None, description="Ruta al archivo de features")
    duration: float = Field(..., ge=0.0, description="Duración en segundos")
    fps: float = Field(..., gt=0.0, description="Frames por segundo")
    resolution: Dict[str, int] = Field(..., description="Resolución del video")
    created_at: str = Field(..., description="Timestamp de creación")
    status: str = Field(..., description="Estado: 'pending', 'processing', 'completed', 'failed'")

