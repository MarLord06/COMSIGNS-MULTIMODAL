"""
Módulo de captura de video desde webcam o archivo
"""

import cv2
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

from ..config import config
from ..schemas import VideoManifest

logger = logging.getLogger(__name__)


class VideoCapture:
    """
    Clase para capturar video desde webcam o archivo y convertirlo a frames
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Inicializa el capturador de video

        Args:
            output_dir: Directorio donde guardar los videos capturados
        """
        self.config = config.load()
        self.output_dir = Path(output_dir or self.config.get('ingestion.video_dir', 'data/raw/videos'))
        self.manifest_dir = Path(
            self.config.get('ingestion.manifest_dir', 'data/raw/manifests')
        )
        self.max_duration = self.config.get('ingestion.max_duration', 60)
        self.supported_formats = self.config.get('ingestion.supported_formats', ['mp4', 'avi', 'mov', 'mkv'])

        # Crear directorios si no existen
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def capture_from_webcam(
        self,
        camera_id: int = 0,
        duration: Optional[float] = None,
        fps: int = 30
    ) -> Tuple[str, VideoManifest]:
        """
        Captura video desde webcam

        Args:
            camera_id: ID de la cámara (por defecto 0)
            duration: Duración de la captura en segundos (None = hasta interrupción)
            fps: Frames por segundo

        Returns:
            Tupla con (ruta_al_video, manifest)
        """
        video_id = str(uuid.uuid4())
        video_path = self.output_dir / f"{video_id}.mp4"

        logger.info(f"Iniciando captura desde cámara {camera_id}")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {camera_id}")

        # Configurar codec y VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        frame_count = 0
        max_frames = int((duration or self.max_duration) * fps) if duration else None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("No se pudo leer frame de la cámara")
                    break

                out.write(frame)
                frame_count += 1

                if max_frames and frame_count >= max_frames:
                    logger.info(f"Captura completada: {frame_count} frames")
                    break

                # Mostrar frame (opcional, para debug)
                cv2.imshow('Captura', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Captura interrumpida por usuario")
                    break

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        actual_duration = frame_count / fps
        logger.info(f"Video guardado: {video_path} ({actual_duration:.2f}s)")

        # Crear manifest
        manifest = VideoManifest(
            video_id=video_id,
            video_path=str(video_path),
            duration=actual_duration,
            fps=fps,
            resolution={'width': width, 'height': height},
            created_at=datetime.now().isoformat(),
            status='completed'
        )

        self._save_manifest(manifest)

        return str(video_path), manifest

    def capture_from_file(
        self,
        file_path: str,
        copy: bool = True
    ) -> Tuple[str, VideoManifest]:
        """
        Procesa un archivo de video existente

        Args:
            file_path: Ruta al archivo de video
            copy: Si True, copia el archivo al directorio de salida

        Returns:
            Tupla con (ruta_al_video, manifest)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo de video no encontrado: {file_path}")

        # Validar formato
        if file_path.suffix[1:].lower() not in self.supported_formats:
            raise ValueError(
                f"Formato no soportado: {file_path.suffix}. "
                f"Formatos soportados: {self.supported_formats}"
            )

        logger.info(f"Procesando archivo: {file_path}")

        # Validar video
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {file_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        video_id = str(uuid.uuid4())

        if copy:
            video_path = self.output_dir / f"{video_id}{file_path.suffix}"
            import shutil
            shutil.copy2(file_path, video_path)
            logger.info(f"Video copiado a: {video_path}")
        else:
            video_path = file_path

        # Crear manifest
        manifest = VideoManifest(
            video_id=video_id,
            video_path=str(video_path),
            duration=duration,
            fps=fps,
            resolution={'width': width, 'height': height},
            created_at=datetime.now().isoformat(),
            status='completed'
        )

        self._save_manifest(manifest)

        return str(video_path), manifest

    def _save_manifest(self, manifest: VideoManifest) -> None:
        """Guarda el manifest en JSON"""
        manifest_path = self.manifest_dir / f"{manifest.video_id}.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(f"Manifest guardado: {manifest_path}")


def capture_from_webcam(
    camera_id: int = 0,
    duration: Optional[float] = None,
    output_dir: Optional[str] = None
) -> Tuple[str, VideoManifest]:
    """
    Función de conveniencia para capturar desde webcam

    Args:
        camera_id: ID de la cámara
        duration: Duración en segundos
        output_dir: Directorio de salida

    Returns:
        Tupla con (ruta_al_video, manifest)
    """
    capture = VideoCapture(output_dir=output_dir)
    return capture.capture_from_webcam(camera_id=camera_id, duration=duration)


def capture_from_file(
    file_path: str,
    copy: bool = True,
    output_dir: Optional[str] = None
) -> Tuple[str, VideoManifest]:
    """
    Función de conveniencia para procesar archivo de video

    Args:
        file_path: Ruta al archivo
        copy: Si copiar el archivo
        output_dir: Directorio de salida

    Returns:
        Tupla con (ruta_al_video, manifest)
    """
    capture = VideoCapture(output_dir=output_dir)
    return capture.capture_from_file(file_path, copy=copy)

