"""Routes package for the ComSigns API."""

from .inference import router as inference_router

__all__ = ["inference_router"]
