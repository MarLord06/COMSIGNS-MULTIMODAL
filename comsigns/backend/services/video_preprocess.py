"""
Video preprocessing pipeline for sign language inference.

Extracts frames from video, runs keypoint extraction,
and normalizes to tensors compatible with the model.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, BinaryIO
import tempfile
import io

import numpy as np

logger = logging.getLogger(__name__)

# Import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


# Default configuration
DEFAULT_MAX_FRAMES = 150  # Maximum frames to process
DEFAULT_MIN_FRAMES = 5    # Minimum frames required
DEFAULT_TARGET_FPS = 30   # Target FPS for sampling


class VideoPreprocessor:
    """Preprocesses video files for sign language inference.
    
    Pipeline:
    1. Read video file
    2. Extract frames at target FPS
    3. Run keypoint extraction per frame
    4. Normalize and pad/truncate to fixed length
    5. Return tensors compatible with model
    
    Example:
        >>> preprocessor = VideoPreprocessor(max_frames=150)
        >>> 
        >>> # Process video file
        >>> features = preprocessor.process_video("sign.mp4")
        >>> print(features["hand"].shape)   # torch.Size([150, 168])
        >>> print(features["body"].shape)   # torch.Size([150, 132])
        >>> print(features["face"].shape)   # torch.Size([150, 1872])
        >>> print(features["lengths"])      # tensor([actual_frames])
    """
    
    def __init__(
        self,
        max_frames: int = DEFAULT_MAX_FRAMES,
        min_frames: int = DEFAULT_MIN_FRAMES,
        target_fps: Optional[float] = DEFAULT_TARGET_FPS,
        keypoint_extractor=None
    ):
        """Initialize the video preprocessor.
        
        Args:
            max_frames: Maximum number of frames (pad/truncate to this)
            min_frames: Minimum frames required (reject shorter videos)
            target_fps: Target FPS for sampling (None = use video FPS)
            keypoint_extractor: Optional KeypointExtractor instance
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is not installed. Install with: pip install opencv-python")
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")
        
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.target_fps = target_fps
        self._extractor = keypoint_extractor
    
    @property
    def extractor(self):
        """Get or create keypoint extractor."""
        if self._extractor is None:
            from backend.services.keypoint_extractor import get_keypoint_extractor
            self._extractor = get_keypoint_extractor()
        return self._extractor
    
    def process_video(
        self,
        video_source: Union[str, Path, bytes, BinaryIO]
    ) -> Dict[str, torch.Tensor]:
        """Process video and return model-ready features.
        
        Args:
            video_source: Video file path, bytes, or file-like object
        
        Returns:
            Dictionary with tensors:
            - "hand": (max_frames, 168)
            - "body": (max_frames, 132)
            - "face": (max_frames, 1872)
            - "lengths": (1,) actual frame count before padding
        
        Raises:
            ValueError: If video is too short or invalid format
            RuntimeError: If video cannot be read
        """
        # Handle different input types
        if isinstance(video_source, (str, Path)):
            video_path = str(video_source)
            temp_file = None
        elif isinstance(video_source, bytes):
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_file.write(video_source)
            temp_file.flush()
            video_path = temp_file.name
        elif hasattr(video_source, 'read'):
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_file.write(video_source.read())
            temp_file.flush()
            video_path = temp_file.name
        else:
            raise ValueError(f"Unsupported video source type: {type(video_source)}")
        
        try:
            # Extract frames and keypoints
            frame_keypoints = self._extract_frames_and_keypoints(video_path)
            
            # Validate frame count
            actual_frames = len(frame_keypoints)
            if actual_frames < self.min_frames:
                raise ValueError(
                    f"Video too short: {actual_frames} frames (minimum: {self.min_frames})"
                )
            
            # Convert to tensors with padding/truncation
            return self._to_tensors(frame_keypoints)
        
        finally:
            # Clean up temp file
            if temp_file:
                import os
                temp_file.close()
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass
    
    def _extract_frames_and_keypoints(
        self,
        video_path: str
    ) -> list:
        """Extract frames from video and run keypoint extraction.
        
        Args:
            video_path: Path to video file
        
        Returns:
            List of keypoint dictionaries per frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        try:
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame sampling
            if self.target_fps and self.target_fps < video_fps:
                frame_interval = max(1, int(video_fps / self.target_fps))
            else:
                frame_interval = 1
            
            logger.debug(
                f"Video: {total_frames} frames @ {video_fps:.1f} FPS, "
                f"sampling every {frame_interval} frame(s)"
            )
            
            keypoints_list = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Extract keypoints
                    kp = self.extractor.extract_from_frame(rgb_frame)
                    keypoints_list.append(kp)
                    
                    # Stop if we have enough frames
                    if len(keypoints_list) >= self.max_frames:
                        break
                
                frame_idx += 1
            
            logger.debug(f"Extracted keypoints from {len(keypoints_list)} frames")
            return keypoints_list
        
        finally:
            cap.release()
    
    def _to_tensors(
        self,
        frame_keypoints: list
    ) -> Dict[str, torch.Tensor]:
        """Convert keypoints list to padded tensors.
        
        Args:
            frame_keypoints: List of keypoint dicts per frame
        
        Returns:
            Dictionary with padded tensors
        """
        actual_frames = len(frame_keypoints)
        
        # Get dimensions from keypoint extractor
        from backend.services.keypoint_extractor import HAND_DIM, BODY_DIM, FACE_DIM
        
        # Initialize tensors with zeros (padding)
        hand_tensor = torch.zeros(self.max_frames, HAND_DIM, dtype=torch.float32)
        body_tensor = torch.zeros(self.max_frames, BODY_DIM, dtype=torch.float32)
        face_tensor = torch.zeros(self.max_frames, FACE_DIM, dtype=torch.float32)
        
        # Fill with actual data (truncate if needed)
        frames_to_use = min(actual_frames, self.max_frames)
        
        for i in range(frames_to_use):
            kp = frame_keypoints[i]
            hand_tensor[i] = torch.from_numpy(kp["hand"])
            body_tensor[i] = torch.from_numpy(kp["body"])
            face_tensor[i] = torch.from_numpy(kp["face"])
        
        return {
            "hand": hand_tensor,
            "body": body_tensor,
            "face": face_tensor,
            "lengths": torch.tensor([frames_to_use], dtype=torch.long)
        }
    
    def get_video_info(self, video_path: Union[str, Path]) -> Dict:
        """Get video metadata without processing.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with video info
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        try:
            return {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                    if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
            }
        finally:
            cap.release()


# Singleton instance
_preprocessor_instance: Optional[VideoPreprocessor] = None


def get_video_preprocessor(
    max_frames: int = DEFAULT_MAX_FRAMES,
    min_frames: int = DEFAULT_MIN_FRAMES
) -> VideoPreprocessor:
    """Get or create the global video preprocessor instance."""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = VideoPreprocessor(
            max_frames=max_frames,
            min_frames=min_frames
        )
    return _preprocessor_instance


def process_video_bytes(
    video_bytes: bytes,
    max_frames: int = DEFAULT_MAX_FRAMES
) -> Dict[str, torch.Tensor]:
    """Convenience function to process video bytes.
    
    Args:
        video_bytes: Raw video file bytes
        max_frames: Maximum frames to extract
    
    Returns:
        Dictionary with model-ready tensors
    """
    preprocessor = VideoPreprocessor(max_frames=max_frames)
    return preprocessor.process_video(video_bytes)
