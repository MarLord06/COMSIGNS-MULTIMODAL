"""
Módulo de ingestion: captura y procesamiento de video
"""

from .capture import VideoCapture, capture_from_webcam, capture_from_file
from .utils import validate_video, extract_frames, get_video_info

__all__ = [
    'VideoCapture',
    'capture_from_webcam',
    'capture_from_file',
    'validate_video',
    'extract_frames',
    'get_video_info',
]

