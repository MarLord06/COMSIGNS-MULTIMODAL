"""
Keypoint extractor using MediaPipe for video frames.

Extracts hand, pose, and face landmarks from RGB frames
and returns normalized tensors compatible with the model.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe imports
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    logger.warning(f"MediaPipe not available: {e}")
    MEDIAPIPE_AVAILABLE = False
    mp = None
    vision = None
    base_options = None


# Keypoint dimensions
HAND_LANDMARKS = 21  # per hand
POSE_LANDMARKS = 33
FACE_LANDMARKS = 468
COORDS_PER_POINT = 4  # x, y, z, visibility

# Output dimensions
HAND_DIM = HAND_LANDMARKS * 2 * COORDS_PER_POINT  # 21 * 2 hands * 4 = 168
BODY_DIM = POSE_LANDMARKS * COORDS_PER_POINT  # 33 * 4 = 132
FACE_DIM = FACE_LANDMARKS * COORDS_PER_POINT  # 468 * 4 = 1872


class KeypointExtractor:
    """Extracts keypoints from video frames using MediaPipe.
    
    Extracts:
    - Hands: 21 keypoints per hand (max 2 hands) → [168] per frame
    - Pose: 33 keypoints → [132] per frame  
    - Face: 468 keypoints → [1872] per frame
    
    Each keypoint has [x, y, z, visibility] coordinates.
    
    Example:
        >>> extractor = KeypointExtractor()
        >>> 
        >>> # Extract from single frame (RGB numpy array)
        >>> result = extractor.extract_from_frame(frame)
        >>> print(result["hand"].shape)  # (168,)
        >>> print(result["body"].shape)  # (132,)
        >>> print(result["face"].shape)  # (1872,)
    """
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = True
    ):
        """Initialize the keypoint extractor.
        
        Args:
            model_dir: Directory containing MediaPipe .task models.
                       If None, uses default location and downloads if needed.
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            static_image_mode: If True, treat each frame independently
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is not installed. Install with: pip install mediapipe"
            )
        
        self.model_dir = model_dir or self._get_default_model_dir()
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        
        # Lazy-loaded landmarkers
        self._hand_landmarker = None
        self._pose_landmarker = None
        self._face_landmarker = None
        
        logger.info(f"KeypointExtractor initialized (model_dir={self.model_dir})")
    
    def _get_default_model_dir(self) -> Path:
        """Get default model directory."""
        # Look relative to this file: backend/services/ -> comsigns/models/mediapipe
        return Path(__file__).parent.parent.parent / "models" / "mediapipe"
    
    def _ensure_model_exists(self, model_name: str) -> Path:
        """Ensure model file exists, downloading if needed.
        
        Args:
            model_name: Name of the model file (e.g., "hand_landmarker.task")
        
        Returns:
            Path to the model file
        """
        import urllib.request
        
        model_path = self.model_dir / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        if model_path.exists():
            return model_path
        
        # Download URLs for MediaPipe models
        download_urls = {
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        }
        
        if model_name not in download_urls:
            raise FileNotFoundError(f"Unknown model: {model_name}")
        
        url = download_urls[model_name]
        logger.info(f"Downloading {model_name} from {url}...")
        
        try:
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Downloaded {model_name} to {model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {model_name}: {e}\n"
                f"Please download manually from {url} and place in {self.model_dir}"
            )
        
        return model_path
    
    @property
    def hand_landmarker(self):
        """Get or create hand landmarker."""
        if self._hand_landmarker is None:
            model_path = self._ensure_model_exists("hand_landmarker.task")
            
            opts = vision.HandLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self._hand_landmarker = vision.HandLandmarker.create_from_options(opts)
        
        return self._hand_landmarker
    
    @property
    def pose_landmarker(self):
        """Get or create pose landmarker."""
        if self._pose_landmarker is None:
            model_path = self._ensure_model_exists("pose_landmarker_lite.task")
            
            opts = vision.PoseLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self._pose_landmarker = vision.PoseLandmarker.create_from_options(opts)
        
        return self._pose_landmarker
    
    @property
    def face_landmarker(self):
        """Get or create face landmarker."""
        if self._face_landmarker is None:
            model_path = self._ensure_model_exists("face_landmarker.task")
            
            opts = vision.FaceLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self._face_landmarker = vision.FaceLandmarker.create_from_options(opts)
        
        return self._face_landmarker
    
    def extract_from_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract keypoints from a single RGB frame.
        
        Args:
            frame: RGB frame as numpy array (H, W, 3)
        
        Returns:
            Dictionary with normalized keypoint arrays:
            - "hand": (168,) - both hands flattened
            - "body": (132,) - pose keypoints
            - "face": (1872,) - face keypoints
        """
        # Ensure RGB format
        if frame.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame with 3 channels, got {frame.shape}")
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Extract all keypoints with error handling
        try:
            hand_kp = self._extract_hand_keypoints(mp_image)
        except Exception as e:
            logger.warning(f"Hand extraction failed: {e}, using zeros")
            hand_kp = np.zeros(HAND_DIM, dtype=np.float32)
        
        try:
            body_kp = self._extract_body_keypoints(mp_image)
        except Exception as e:
            logger.warning(f"Body extraction failed: {e}, using zeros")
            body_kp = np.zeros(BODY_DIM, dtype=np.float32)
        
        try:
            face_kp = self._extract_face_keypoints(mp_image)
        except Exception as e:
            logger.warning(f"Face extraction failed: {e}, using zeros")
            face_kp = np.zeros(FACE_DIM, dtype=np.float32)
        
        # Final shape validation
        if hand_kp.shape != (HAND_DIM,):
            logger.warning(f"Hand shape mismatch: {hand_kp.shape}, padding with zeros")
            hand_kp = np.zeros(HAND_DIM, dtype=np.float32)
        
        if body_kp.shape != (BODY_DIM,):
            logger.warning(f"Body shape mismatch: {body_kp.shape}, padding with zeros")
            body_kp = np.zeros(BODY_DIM, dtype=np.float32)
        
        if face_kp.shape != (FACE_DIM,):
            logger.warning(f"Face shape mismatch: {face_kp.shape}, padding with zeros")
            face_kp = np.zeros(FACE_DIM, dtype=np.float32)
        
        return {
            "hand": hand_kp,
            "body": body_kp,
            "face": face_kp
        }
    
    def _extract_hand_keypoints(self, mp_image: "mp.Image") -> np.ndarray:
        """Extract hand keypoints (21 per hand, max 2 hands).
        
        Returns:
            np.ndarray of shape (168,) - [left_hand(84), right_hand(84)]
        """
        result = self.hand_landmarker.detect(mp_image)
        
        # Initialize with zeros (no detection)
        keypoints = np.zeros((2, HAND_LANDMARKS, COORDS_PER_POINT), dtype=np.float32)
        
        if result.hand_landmarks:
            # Determine handedness and fill appropriately
            for i, (landmarks, handedness) in enumerate(
                zip(result.hand_landmarks, result.handedness)
            ):
                if i >= 2:
                    break
                
                # Get hand index (0=left, 1=right) based on handedness
                # Note: MediaPipe reports from camera's perspective, so we flip
                hand_label = handedness[0].category_name.lower()
                hand_idx = 0 if hand_label == "left" else 1
                
                # Limit to expected number of landmarks
                for j, landmark in enumerate(landmarks):
                    if j >= HAND_LANDMARKS:
                        break
                    keypoints[hand_idx, j] = [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        1.0  # visibility = 1 when detected
                    ]
        
        # Flatten: [2, 21, 4] -> [168]
        result = keypoints.flatten()
        assert result.shape == (HAND_DIM,), f"Hand keypoints shape mismatch: {result.shape}"
        return result
    
    def _extract_body_keypoints(self, mp_image: "mp.Image") -> np.ndarray:
        """Extract pose keypoints (33 landmarks).
        
        Returns:
            np.ndarray of shape (132,) - [33 landmarks * 4 coords]
        """
        result = self.pose_landmarker.detect(mp_image)
        
        # Initialize with zeros (no detection)
        keypoints = np.zeros((POSE_LANDMARKS, COORDS_PER_POINT), dtype=np.float32)
        
        if result.pose_landmarks:
            # Limit to expected number of landmarks (defensive)
            landmarks = result.pose_landmarks[0]
            num_landmarks = min(len(landmarks), POSE_LANDMARKS)
            
            for j in range(num_landmarks):
                landmark = landmarks[j]
                keypoints[j] = [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ]
        
        # Flatten: [33, 4] -> [132]
        result_array = keypoints.flatten()
        assert result_array.shape == (BODY_DIM,), f"Body keypoints shape mismatch: {result_array.shape}"
        return result_array
    
    def _extract_face_keypoints(self, mp_image: "mp.Image") -> np.ndarray:
        """Extract face keypoints (468 landmarks).
        
        Returns:
            np.ndarray of shape (1872,) - [468 landmarks * 4 coords]
        
        Note:
            MediaPipe Face Mesh has exactly 468 landmarks (indices 0-467).
            This method includes defensive bounds checking to prevent
            IndexError if MediaPipe returns unexpected data.
        """
        result = self.face_landmarker.detect(mp_image)
        
        # Initialize with zeros (no detection = zero tensor)
        keypoints = np.zeros((FACE_LANDMARKS, COORDS_PER_POINT), dtype=np.float32)
        
        if result.face_landmarks and len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]
            # CRITICAL: Limit to exactly FACE_LANDMARKS (468) to prevent IndexError
            # Valid indices are 0 to 467, never access index 468
            num_landmarks = min(len(landmarks), FACE_LANDMARKS)
            
            for j in range(num_landmarks):
                landmark = landmarks[j]
                keypoints[j] = [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    1.0  # visibility = 1 when detected
                ]
            
            if len(landmarks) != FACE_LANDMARKS:
                logger.warning(
                    f"Face landmarks count mismatch: got {len(landmarks)}, expected {FACE_LANDMARKS}. "
                    f"Using first {num_landmarks} landmarks."
                )
        
        # Flatten: [468, 4] -> [1872]
        result_array = keypoints.flatten()
        
        # Final validation - ensure correct shape
        if result_array.shape != (FACE_DIM,):
            logger.error(f"Face keypoints shape error: {result_array.shape}, expected ({FACE_DIM},)")
            return np.zeros(FACE_DIM, dtype=np.float32)
        
        return result_array
    
    def close(self):
        """Close MediaPipe resources."""
        if self._hand_landmarker:
            self._hand_landmarker.close()
            self._hand_landmarker = None
        if self._pose_landmarker:
            self._pose_landmarker.close()
            self._pose_landmarker = None
        if self._face_landmarker:
            self._face_landmarker.close()
            self._face_landmarker = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Singleton instance for reuse
_extractor_instance: Optional[KeypointExtractor] = None


def get_keypoint_extractor() -> KeypointExtractor:
    """Get or create the global keypoint extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = KeypointExtractor()
    return _extractor_instance
