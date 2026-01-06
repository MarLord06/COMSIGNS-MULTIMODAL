"""
Path resolver for AEC (Asociación de Estudio del Conocimiento) dataset.

Translates internal keypoints_path from dict.json to actual filesystem paths.
"""

from pathlib import Path
import logging

from ..base import KeypointResolverInterface

logger = logging.getLogger(__name__)


class AECKeypointResolver(KeypointResolverInterface):
    """
    Resolves AEC keypoints_path from dict.json to actual filesystem paths.
    
    The AEC dataset stores paths in the format:
        "./Data/AEC/Keypoints/pkl/ira_alegria/yo_1.pkl"
    
    This resolver translates them to actual paths relative to the dataset root:
        /path/to/data/raw/lsp_aec/Keypoints/pkl/ira_alegria/yo_1.pkl
    
    Example:
        >>> resolver = AECKeypointResolver(Path("data/raw/lsp_aec"))
        >>> path = resolver.resolve("./Data/AEC/Keypoints/pkl/ira_alegria/yo_1.pkl")
        >>> print(path)
        data/raw/lsp_aec/Keypoints/pkl/ira_alegria/yo_1.pkl
    
    Attributes:
        dataset_root: Base directory of the AEC dataset (e.g., data/raw/lsp_aec/)
    """
    
    # Prefix used in dict.json that needs to be stripped
    DEFAULT_PREFIX = "./Data/AEC/"
    
    def __init__(
        self, 
        dataset_root: Path,
        prefix_to_strip: str = DEFAULT_PREFIX
    ):
        """
        Initialize the AEC keypoint resolver.
        
        Args:
            dataset_root: Path to the AEC dataset root (e.g., data/raw/lsp_aec/)
            prefix_to_strip: Prefix in dict.json paths to remove (default: "./Data/AEC/")
        """
        self.dataset_root = Path(dataset_root).resolve()
        self._prefix_to_strip = prefix_to_strip
        
        if not self.dataset_root.exists():
            logger.warning(f"Dataset root does not exist: {self.dataset_root}")
    
    def resolve(self, raw_path: str) -> Path:
        """
        Translate a dict.json keypoints_path to an absolute filesystem path.
        
        Args:
            raw_path: Path as stored in dict.json 
                      (e.g., "./Data/AEC/Keypoints/pkl/ira_alegria/yo_1.pkl")
        
        Returns:
            Absolute Path to the keypoints file
            
        Raises:
            ValueError: If raw_path doesn't start with the expected prefix
        """
        # Normalize path separators
        normalized = raw_path.replace("\\", "/")
        
        # Strip the prefix
        if normalized.startswith(self._prefix_to_strip):
            relative = normalized[len(self._prefix_to_strip):]
        elif normalized.startswith("./"):
            # Fallback: try to extract after "AEC/"
            try:
                aec_idx = normalized.index("/AEC/")
                relative = normalized[aec_idx + 5:]  # Skip "/AEC/"
            except ValueError:
                # Last resort: use as-is without "./"
                relative = normalized[2:]
        else:
            # Use path as-is
            relative = normalized
        
        return self.dataset_root / relative
    
    def exists(self, raw_path: str) -> bool:
        """
        Check if the resolved path exists on the filesystem.
        
        Args:
            raw_path: Path as stored in dict.json
            
        Returns:
            True if the file exists, False otherwise
        """
        try:
            resolved = self.resolve(raw_path)
            return resolved.exists()
        except Exception as e:
            logger.debug(f"Error checking existence of {raw_path}: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"AECKeypointResolver(dataset_root={self.dataset_root})"
