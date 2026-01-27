"""Routes package for the ComSigns API."""

from .inference import router as inference_router
from .video import router as video_router

__all__ = ["inference_router", "video_router"]
