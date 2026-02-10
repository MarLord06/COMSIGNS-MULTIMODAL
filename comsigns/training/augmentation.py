"""Keypoint augmentation utilities for ComSigns training."""

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class AugmentConfig:
    time_shift: int = 2  # max frames to shift
    noise_std: float = 0.01  # gaussian noise std
    mirror_prob: float = 0.0  # probability to mirror


class KeypointAugmenter:
    """Applies lightweight, geometry-safe augmentations to keypoints."""

    def __init__(self, config: Optional[AugmentConfig] = None):
        self.config = config or AugmentConfig()

    def _add_noise(self, arr: np.ndarray) -> np.ndarray:
        if self.config.noise_std <= 0:
            return arr
        # reshape to (T, N, 4) to avoid perturbing confidence channel
        reshaped = arr.reshape(arr.shape[0], -1, 4)
        noise = np.random.normal(0, self.config.noise_std, size=reshaped[..., :3].shape)
        reshaped[..., :3] = reshaped[..., :3] + noise
        return reshaped.reshape(arr.shape)

    def _time_shift(self, arr: np.ndarray) -> np.ndarray:
        if self.config.time_shift <= 0:
            return arr
        shift = np.random.randint(-self.config.time_shift, self.config.time_shift + 1)
        if shift == 0:
            return arr
        if shift > 0:
            pad = arr[:1].repeat(shift, axis=0)
            return np.concatenate([pad, arr[:-shift]], axis=0)
        shift = abs(shift)
        pad = arr[-1:].repeat(shift, axis=0)
        return np.concatenate([arr[shift:], pad], axis=0)

    def _mirror(self, hand: np.ndarray, body: np.ndarray, face: np.ndarray) -> tuple:
        # Flip X coordinate: x' = 1 - x (assuming normalized coordinates)
        def flip_x(arr: np.ndarray) -> np.ndarray:
            original_shape = arr.shape
            reshaped = arr.reshape(arr.shape[0], -1, 4)
            reshaped[..., 0] = 1.0 - reshaped[..., 0]
            return reshaped.reshape(original_shape)

        # Swap left/right hands (hand has 2*21*4 features)
        original_hand_shape = hand.shape
        hand = hand.reshape(hand.shape[0], 2, -1)
        hand = hand[:, ::-1, :].reshape(original_hand_shape)

        return flip_x(hand), flip_x(body), flip_x(face)

    def apply(self, sample):
        """Return augmented EncoderReadySample (in-place on arrays)."""
        hand = sample.hand_keypoints.copy()
        body = sample.body_keypoints.copy()
        face = sample.face_keypoints.copy()

        hand = self._time_shift(hand)
        body = self._time_shift(body)
        face = self._time_shift(face)

        hand = self._add_noise(hand)
        body = self._add_noise(body)
        face = self._add_noise(face)

        if self.config.mirror_prob > 0:
            if np.random.rand() < self.config.mirror_prob:
                hand, body, face = self._mirror(hand, body, face)

        sample.hand_keypoints = hand
        sample.body_keypoints = body
        sample.face_keypoints = face
        return sample
