"""
Tests unitarios para módulo de ingestion
"""

import pytest
from pathlib import Path
import tempfile
import cv2
import numpy as np

from comsigns.services.ingestion.utils import validate_video, get_video_info


class TestValidateVideo:
    """Tests para validación de video"""

    def test_validate_nonexistent_file(self):
        """Test validación de archivo inexistente"""
        is_valid, error = validate_video("nonexistent_file.mp4")
        assert not is_valid
        assert error is not None

    def test_create_and_validate_video(self):
        """Test crear video temporal y validarlo"""
        # Crear video temporal simple
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Crear video con OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_path, fourcc, 30.0, (640, 480))

            # Escribir algunos frames
            for _ in range(30):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                out.write(frame)

            out.release()

            # Validar
            is_valid, error = validate_video(tmp_path)
            assert is_valid, f"Video debería ser válido: {error}"

            # Verificar información
            info = get_video_info(tmp_path)
            assert info['fps'] > 0
            assert info['width'] == 640
            assert info['height'] == 480
            assert info['frame_count'] > 0

        finally:
            Path(tmp_path).unlink(missing_ok=True)

