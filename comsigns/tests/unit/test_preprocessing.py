"""
Tests unitarios para módulo de preprocessing
"""

import pytest
import numpy as np
from comsigns.services.preprocessing.process_clip import normalize_keypoints


class TestNormalizeKeypoints:
    """Tests para normalización de keypoints"""

    def test_normalize_relative(self):
        """Test normalización relativa"""
        keypoints = [
            [0.5, 0.5, 0.1, 0.9],
            [0.6, 0.6, 0.2, 0.8]
        ]

        normalized = normalize_keypoints(keypoints, method="relative")
        assert len(normalized) == 2
        # En método relativo, los valores deben estar en [0, 1]
        assert all(0.0 <= kp[0] <= 1.0 for kp in normalized)
        assert all(0.0 <= kp[1] <= 1.0 for kp in normalized)

    def test_normalize_absolute(self):
        """Test normalización absoluta (centroide)"""
        keypoints = [
            [0.5, 0.5, 0.1, 0.9],
            [0.6, 0.6, 0.2, 0.8],
            [0.4, 0.4, 0.0, 0.9]
        ]

        normalized = normalize_keypoints(keypoints, method="absolute")
        assert len(normalized) == 3

        # En método absoluto, el centroide debería estar cerca de (0, 0)
        xs = [kp[0] for kp in normalized]
        ys = [kp[1] for kp in normalized]
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)

        # El centroide debería estar cerca de 0
        assert abs(centroid_x) < 0.1
        assert abs(centroid_y) < 0.1

    def test_normalize_empty(self):
        """Test con lista vacía"""
        normalized = normalize_keypoints([], method="relative")
        assert normalized == []

    def test_normalize_invalid_method(self):
        """Test método inválido"""
        keypoints = [[0.5, 0.5, 0.1, 0.9]]
        with pytest.raises(ValueError):
            normalize_keypoints(keypoints, method="invalid")

