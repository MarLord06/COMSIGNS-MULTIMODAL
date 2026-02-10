"""
Pure, stateless converters for AEC keypoints format.

These functions convert AEC-specific keypoint structures to the format
expected by the MultimodalEncoder. They have NO dependencies on the
dataset or resolver classes, making them reusable for debugging,
visualization, and dataset comparison.

AEC Format (per frame):
    {
        'pose': {'x': [33 values], 'y': [33 values]},
        'left_hand': {'x': [21 values], 'y': [21 values]},
        'right_hand': {'x': [21 values], 'y': [21 values]},
        'face': {'x': [468 values], 'y': [468 values]}
    }

Encoder Format (updated):
    - hand: [126] = 2 hands × 21 keypoints × 3 values [x, y, z]
    - body: [99] = 33 keypoints × 3 values [x, y, z]
    - face: [1404] = 468 keypoints × 3 values [x, y, z]
"""

import numpy as np
from typing import Dict, List


# Default padding value for missing z
DEFAULT_Z = 0.0


def _xy_to_xyz(
    x_values: List[float],
    y_values: List[float],
    z_value: float = DEFAULT_Z
) -> np.ndarray:
    """
    Convert separate x, y lists to flattened [x, y, z, ...] array.

    Args:
        x_values: List of x coordinates
        y_values: List of y coordinates
        z_value: Value to use for z (depth), default 0.0

    Returns:
        Flattened numpy array: [x0, y0, z, x1, y1, z, ...]
    """
    num_points = len(x_values)
    result = np.zeros(num_points * 3, dtype=np.float32)

    for i in range(num_points):
        base_idx = i * 3
        result[base_idx] = x_values[i]
        result[base_idx + 1] = y_values[i]
        result[base_idx + 2] = z_value

    return result


def aec_frame_to_encoder_arrays(frame: Dict) -> Dict[str, np.ndarray]:
    """
    Convert a single AEC frame dict to encoder-compatible numpy arrays.
    
    PURE FUNCTION - no side effects, no dependencies on dataset/resolver.
    
    Args:
        frame: AEC frame dict with keys: pose, left_hand, right_hand, face
               Each has {'x': [...], 'y': [...]}
    
    Returns:
        Dict with keys 'hand', 'body', 'face' as flattened numpy arrays:
        - hand: [168] (2 hands × 21 keypoints × 4 values)
        - body: [132] (33 keypoints × 4 values)
        - face: [1872] (468 keypoints × 4 values)
    
    Padding strategy:
        z = 0.0 (depth unknown in AEC)
        confidence = 1.0 (assume valid keypoints)
    """
    # Process hands: concatenate left_hand + right_hand (each => 21 * 3)
    left_hand = frame.get('left_hand', {'x': [0.0] * 21, 'y': [0.0] * 21})
    right_hand = frame.get('right_hand', {'x': [0.0] * 21, 'y': [0.0] * 21})

    left_array = _xy_to_xyz(left_hand['x'], left_hand['y'])
    right_array = _xy_to_xyz(right_hand['x'], right_hand['y'])
    hand_array = np.concatenate([left_array, right_array])  # [126]

    # Process body (pose) -> 33 * 3
    pose = frame.get('pose', {'x': [0.0] * 33, 'y': [0.0] * 33})
    body_array = _xy_to_xyz(pose['x'], pose['y'])  # [99]

    # Process face -> 468 * 3
    face = frame.get('face', {'x': [0.0] * 468, 'y': [0.0] * 468})
    face_array = _xy_to_xyz(face['x'], face['y'])  # [1404]
    
    return {
        'hand': hand_array,
        'body': body_array,
        'face': face_array
    }


def aec_keypoints_to_encoder_format(frames: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Convert list of AEC frames to encoder-ready tensors.
    
    PURE FUNCTION - stateless, suitable for debugging and visualization.
    
    Args:
        frames: List of AEC frame dicts (as loaded from .pkl file)
    
    Returns:
        Dict with keys 'hand', 'body', 'face' as numpy arrays:
        - hand: [T, 168]
        - body: [T, 132]
        - face: [T, 1872]
    
    Raises:
        ValueError: If frames list is empty
    """
    if not frames:
        raise ValueError("Cannot convert empty frames list")
    
    T = len(frames)
    
    # Pre-allocate arrays
    hand_array = np.zeros((T, 126), dtype=np.float32)
    body_array = np.zeros((T, 99), dtype=np.float32)
    face_array = np.zeros((T, 1404), dtype=np.float32)
    
    for t, frame in enumerate(frames):
        converted = aec_frame_to_encoder_arrays(frame)
        hand_array[t] = converted['hand']
        body_array[t] = converted['body']
        face_array[t] = converted['face']
    
    return {
        'hand': hand_array,
        'body': body_array,
        'face': face_array
    }


def validate_aec_frame(frame: Dict) -> bool:
    """
    Validate that an AEC frame has the expected structure.
    
    Args:
        frame: Frame dict to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = {'pose', 'left_hand', 'right_hand', 'face'}
    
    if not isinstance(frame, dict):
        return False
    
    if not required_keys.issubset(frame.keys()):
        return False
    
    for key in required_keys:
        if 'x' not in frame[key] or 'y' not in frame[key]:
            return False
    
    # Check expected sizes
    expected_sizes = {
        'pose': 33,
        'left_hand': 21,
        'right_hand': 21,
        'face': 468
    }
    
    for key, expected_size in expected_sizes.items():
        if len(frame[key]['x']) != expected_size:
            return False
        if len(frame[key]['y']) != expected_size:
            return False
    
    return True
