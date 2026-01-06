"""
Sample dataclasses for sign language datasets.

Defines data structures that separate raw dataset format from
encoder-ready tensors, enabling clean separation of concerns.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class Sample:
    """
    Represents a single raw sign language sample.
    
    The keypoints field preserves the original dataset format (list of frame dicts)
    to enable debugging, visualization, and comparison across different datasets.
    Tensorization is NOT performed here - that's the responsibility of EncoderReadySample.
    
    Attributes:
        gloss: The sign label/class name (e.g., "hola", "yo", "gracias")
        keypoints: Raw AEC frames as list of dicts. Each dict contains:
                   {'pose': {'x': [...], 'y': [...]}, 
                    'left_hand': {'x': [...], 'y': [...]}, 
                    'right_hand': {...}, 
                    'face': {...}}
        frame_start: Starting frame index in the original video
        frame_end: Ending frame index in the original video
        unique_name: Unique identifier for this sample (e.g., "yo_303")
        metadata: Optional additional metadata from the dataset
    """
    gloss: str
    keypoints: List[Dict]  # Raw AEC frames, NOT tensorized
    frame_start: int
    frame_end: int
    unique_name: str
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def num_frames(self) -> int:
        """Return the number of frames in this sample."""
        return len(self.keypoints)


@dataclass
class EncoderReadySample:
    """
    Sample with keypoints pre-formatted for the MultimodalEncoder.
    
    All arrays are numpy float32 and ready for torch conversion.
    This is the format returned by AECDataset.__getitem__() and
    consumed directly by the training pipeline.
    
    Attributes:
        gloss: The sign label/class name
        gloss_id: Integer class ID for CrossEntropyLoss training
        hand_keypoints: Shape [T, 168] - 2 hands × 21 keypoints × 4 values [x,y,z,conf]
        body_keypoints: Shape [T, 132] - 33 keypoints × 4 values
        face_keypoints: Shape [T, 1872] - 468 keypoints × 4 values
        unique_name: Unique identifier for this sample
        metadata: Optional additional metadata
    """
    gloss: str
    gloss_id: int
    hand_keypoints: np.ndarray  # [T, 168]
    body_keypoints: np.ndarray  # [T, 132]
    face_keypoints: np.ndarray  # [T, 1872]
    unique_name: str
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def num_frames(self) -> int:
        """Return the number of frames in this sample."""
        return self.hand_keypoints.shape[0]
    
    def to_tensors(self) -> Dict[str, 'np.ndarray']:
        """
        Return keypoints as a dictionary of arrays.
        
        Useful for direct integration with encoder's encode_features method.
        """
        return {
            'hand': self.hand_keypoints,
            'body': self.body_keypoints,
            'face': self.face_keypoints
        }
