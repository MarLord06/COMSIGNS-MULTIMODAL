"""
Extracción de keypoints usando MediaPipe Tasks (Hands, Pose, Face Mesh)
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Importación de MediaPipe Tasks
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    logging.error(f"MediaPipe no está disponible: {e}")
    MEDIAPIPE_AVAILABLE = False
    mp = None
    vision = None
    base_options = None

from ..config import config
from ..schemas import FrameKeypoints, FeatureClip, ClipMetadata

logger = logging.getLogger(__name__)


class KeypointExtractor:
    """
    Extractor de keypoints usando MediaPipe Tasks para manos, cuerpo y rostro
    """

    def __init__(self, model_paths: Optional[Dict[str, str]] = None):
        """
        Inicializa los modelos de MediaPipe Tasks
        
        Args:
            model_paths: Diccionario opcional con rutas a modelos personalizados:
                {'hand': 'path/to/hand.task', 'pose': 'path/to/pose.task', 'face': 'path/to/face.task'}
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe no está instalado. Instala con: pip install mediapipe"
            )
        
        cfg = config.load()
        mp_config = cfg.get('preprocessing.mediapipe', {})
        
        min_detection_confidence = mp_config.get('min_detection_confidence', 0.5)
        min_tracking_confidence = mp_config.get('min_tracking_confidence', 0.5)
        static_image_mode = mp_config.get('static_image_mode', False)
        
        # Usar modelos personalizados si se proporcionan
        if model_paths:
            hand_model_path = model_paths.get('hand')
            pose_model_path = model_paths.get('pose')
            face_model_path = model_paths.get('face')
        else:
            # Intentar descargar modelos automáticamente (URLs correctas del bucket)
            hand_model_path = self._download_model_if_needed(
                "hand_landmarker.task",
                [
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                ]
            )
            pose_model_path = self._download_model_if_needed(
                "pose_landmarker_lite.task",
                [
                    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
                ]
            )
            face_model_path = self._download_model_if_needed(
                "face_landmarker.task",
                [
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
                ]
            )

        # Configurar modo de ejecución
        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO

        # Inicializar HandLandmarker
        hand_base_opts = base_options.BaseOptions(model_asset_path=hand_model_path)
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_opts,
            running_mode=running_mode,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # Inicializar PoseLandmarker
        pose_base_opts = base_options.BaseOptions(model_asset_path=pose_model_path)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_opts,
            running_mode=running_mode,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # Inicializar FaceLandmarker
        face_base_opts = base_options.BaseOptions(model_asset_path=face_model_path)
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_opts,
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        logger.info("KeypointExtractor inicializado con MediaPipe Tasks")

    def _download_model_if_needed(self, local_name: str, download_urls) -> str:
        """
        Descarga el modelo si no existe localmente
        
        Args:
            local_name: Nombre del archivo local
            download_urls: URL o lista de URLs para intentar descargar el modelo
            
        Returns:
            Ruta al archivo del modelo
        """
        from pathlib import Path
        import urllib.request
        
        # Convertir a lista si es string
        if isinstance(download_urls, str):
            download_urls = [download_urls]
        
        # Directorio para modelos
        models_dir = Path(__file__).parent.parent.parent.parent / "models" / "mediapipe"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / local_name
        
        # Si el modelo no existe, descargarlo
        if not model_path.exists():
            logger.info(f"Descargando modelo {local_name}...")
            last_error = None
            
            # Intentar cada URL hasta que una funcione
            for download_url in download_urls:
                try:
                    logger.info(f"Intentando descargar desde: {download_url}")
                    urllib.request.urlretrieve(download_url, model_path)
                    logger.info(f"Modelo descargado exitosamente: {model_path}")
                    return str(model_path)
                except Exception as e:
                    last_error = e
                    logger.warning(f"Error con URL {download_url}: {e}")
                    continue
            
            # Si todas las URLs fallaron
            logger.error(f"Error descargando modelo {local_name} desde todas las URLs")
            raise RuntimeError(
                f"No se pudo descargar el modelo {local_name}.\n"
                f"URLs intentadas: {download_urls}\n"
                f"Último error: {str(last_error)}\n\n"
                f"Soluciones:\n"
                f"1. Verifica tu conexión a internet\n"
                f"2. Descarga manualmente el modelo desde:\n"
                f"   https://developers.google.com/mediapipe/solutions/vision\n"
                f"   y colócalo en: {model_path}\n"
                f"3. O verifica que las URLs de MediaPipe estén disponibles"
            )
        
        return str(model_path)

    def extract_from_frame(self, frame: np.ndarray, timestamp_ms: int = 0) -> FrameKeypoints:
        """
        Extrae keypoints de un frame individual

        Args:
            frame: Frame de video como array numpy (BGR)
            timestamp_ms: Timestamp en milisegundos (para modo VIDEO)

        Returns:
            FrameKeypoints con los keypoints extraídos
        """
        # Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crear imagen de MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Extraer keypoints de manos (máximo 2 manos)
        hand_keypoints = self._extract_hand_keypoints(mp_image, timestamp_ms)

        # Extraer keypoints del cuerpo
        body_keypoints = self._extract_body_keypoints(mp_image, timestamp_ms)

        # Extraer keypoints del rostro
        face_keypoints = self._extract_face_keypoints(mp_image, timestamp_ms)

        # Crear FrameKeypoints (t se establecerá externamente)
        return FrameKeypoints(
            t=0.0,  # Se actualizará con el tiempo real
            hand_keypoints=hand_keypoints,
            body_keypoints=body_keypoints,
            face_keypoints=face_keypoints
        )

    def _extract_hand_keypoints(self, mp_image: mp.Image, timestamp_ms: int) -> List[List[float]]:
        """
        Extrae keypoints de las manos (21 puntos por mano)

        Returns:
            Lista de keypoints: [[x, y, z, confidence], ...] para ambas manos
        """
        # Detectar landmarks de manos
        if hasattr(self.hand_landmarker, 'detect_for_video'):
            detection_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            detection_result = self.hand_landmarker.detect(mp_image)
        
        keypoints = []

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # MediaPipe Hands tiene 21 landmarks por mano
                for landmark in hand_landmarks:
                    # x, y, z están normalizados [0, 1]
                    # z es la profundidad relativa
                    keypoints.append([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        1.0  # MediaPipe no proporciona confidence por landmark
                    ])

        return keypoints

    def _extract_body_keypoints(self, mp_image: mp.Image, timestamp_ms: int) -> List[List[float]]:
        """
        Extrae keypoints del cuerpo (33 puntos de MediaPipe Pose)

        Returns:
            Lista de keypoints: [[x, y, z, visibility], ...]
        """
        # Detectar landmarks de pose
        if hasattr(self.pose_landmarker, 'detect_for_video'):
            detection_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            detection_result = self.pose_landmarker.detect(mp_image)
        
        keypoints = []

        if detection_result.pose_landmarks:
            for landmark in detection_result.pose_landmarks[0]:
                # MediaPipe Pose tiene 33 landmarks
                keypoints.append([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility  # MediaPipe usa visibility en lugar de confidence
                ])

        return keypoints

    def _extract_face_keypoints(self, mp_image: mp.Image, timestamp_ms: int) -> List[List[float]]:
        """
        Extrae keypoints del rostro (468 puntos de MediaPipe Face Landmarker)

        Returns:
            Lista de keypoints: [[x, y, z], ...]
        """
        # Detectar landmarks faciales
        if hasattr(self.face_landmarker, 'detect_for_video'):
            detection_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            detection_result = self.face_landmarker.detect(mp_image)
        
        keypoints = []

        if detection_result.face_landmarks:
            # Solo procesar la primera cara
            face_landmarks = detection_result.face_landmarks[0]
            for landmark in face_landmarks:
                # MediaPipe Face Landmarker tiene 468 landmarks
                keypoints.append([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    1.0  # Face Landmarker no proporciona confidence
                ])

        return keypoints

    def extract_from_video(
        self,
        video_path: str,
        fps: Optional[float] = None
    ) -> FeatureClip:
        """
        Extrae keypoints de todos los frames de un video

        Args:
            video_path: Ruta al archivo de video
            fps: FPS objetivo para procesamiento (None = todos los frames)

        Returns:
            FeatureClip con todos los keypoints
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = fps or video_fps
        frame_interval = max(1, int(video_fps / target_fps)) if target_fps < video_fps else 1

        # Obtener información del video antes de procesar
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        frame_count = 0
        processed_count = 0

        logger.info(f"Extrayendo keypoints de {video_path} (FPS objetivo: {target_fps})")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Procesar frame según intervalo
                if frame_count % frame_interval == 0:
                    # Calcular timestamp en milisegundos
                    timestamp_ms = int((frame_count / video_fps) * 1000) if video_fps > 0 else frame_count * 33
                    
                    frame_kp = self.extract_from_frame(frame, timestamp_ms=timestamp_ms)
                    frame_kp.t = frame_count / video_fps  # Tiempo en segundos
                    frames.append(frame_kp)
                    processed_count += 1

                frame_count += 1

        finally:
            cap.release()

        # Crear metadata
        duration = frame_count / video_fps if video_fps > 0 else 0

        metadata = ClipMetadata(
            duration=duration,
            resolution={'width': width, 'height': height},
            source='file'
        )

        # Crear FeatureClip
        feature_clip = FeatureClip(
            clip_id=video_path.stem,
            fps=target_fps,
            frames=frames,
            meta=metadata
        )

        logger.info(f"Extraídos keypoints de {len(frames)} frames")
        return feature_clip

    def __del__(self):
        """Libera recursos de MediaPipe"""
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()
        if hasattr(self, 'face_landmarker'):
            self.face_landmarker.close()


def extract_keypoints_from_video(
    video_path: str,
    fps: Optional[float] = None
) -> FeatureClip:
    """
    Función de conveniencia para extraer keypoints de un video

    Args:
        video_path: Ruta al archivo de video
        fps: FPS objetivo

    Returns:
        FeatureClip con los keypoints
    """
    extractor = KeypointExtractor()
    return extractor.extract_from_video(video_path, fps=fps)
