"""
Base interfaces for sign language datasets.

Provides abstract base classes that ensure consistent behavior
across different sign language datasets (AEC, LSM, LSC, LIBRAS, etc.)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Dict


class KeypointResolverInterface(ABC):
    """
    Interface for resolving keypoint paths from dataset-specific formats.
    
    Each dataset may store paths in its own format (relative paths,
    different prefixes, etc.). The resolver translates these to actual
    filesystem paths.
    """
    
    @abstractmethod
    def resolve(self, raw_path: str) -> Path:
        """
        Translate a dataset-internal path to an absolute filesystem path.
        
        Args:
            raw_path: Path as stored in the dataset metadata (e.g., dict.json)
            
        Returns:
            Absolute Path object pointing to the actual file
        """
        pass
    
    @abstractmethod
    def exists(self, raw_path: str) -> bool:
        """
        Check if the resolved path exists on the filesystem.
        
        Args:
            raw_path: Path as stored in the dataset metadata
            
        Returns:
            True if the file exists, False otherwise
        """
        pass


class BaseDataset(ABC):
    """
    Abstract base class for sign language datasets.
    
    Defines the interface that all dataset implementations must follow
    to ensure compatibility with the ComSigns training pipeline.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Get a single sample by index.
        
        Args:
            idx: Sample index (0 to len-1)
            
        Returns:
            Sample data (typically EncoderReadySample)
        """
        pass
    
    @abstractmethod
    def get_gloss_labels(self) -> List[str]:
        """
        Return list of unique gloss labels in the dataset.
        
        Returns:
            List of gloss strings (vocabulary)
        """
        pass
    
    @property
    @abstractmethod
    def gloss_to_id(self) -> Dict[str, int]:
        """
        Mapping from gloss string to integer ID.
        
        Used for converting labels to class indices for CrossEntropyLoss.
        """
        pass
    
    @property
    @abstractmethod
    def id_to_gloss(self) -> Dict[int, str]:
        """
        Mapping from integer ID to gloss string.
        
        Used for converting predictions back to human-readable labels.
        """
        pass
    
    @property
    def num_classes(self) -> int:
        """Return the number of unique classes (glosses) in the dataset."""
        return len(self.gloss_to_id)
