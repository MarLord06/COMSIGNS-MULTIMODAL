"""
AEC Dataset loader for Peruvian Sign Language (Lengua de Señas Peruana).

This module provides the main AECDataset class that loads and iterates
over the AEC dataset, returning samples compatible with the ComSigns
MultimodalEncoder pipeline.

Dataset source: Asociación de Estudio del Conocimiento (AEC)
Location: data/raw/lsp_aec/

IMPORTANT: The AEC dataset contains only two source videos and no valid signer
information (signer_id is always -1). Any train/val split should use gloss
stratification for technical pipeline validation, not as a true evaluation
of model generalization.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, Literal, Set

import numpy as np

from ..base import BaseDataset, KeypointResolverInterface
from ..sample import Sample, EncoderReadySample
from .resolver import AECKeypointResolver
from .converters import aec_keypoints_to_encoder_format

logger = logging.getLogger(__name__)


class AECDataset(BaseDataset):
    """
    Dataset loader for AEC (Peruvian Sign Language) dataset.
    
    Responsibilities:
    - Parse dict.json to extract gloss entries and instances
    - Build gloss ↔ id vocabularies for training
    - Iterate over instances with lazy loading
    - Load .pkl keypoints files on demand
    - Convert raw format to encoder-ready tensors
    
    The dataset flattens all instances across all glosses for simple
    sequential access, while maintaining the gloss vocabulary for
    class-balanced sampling if needed.
    
    Example:
        >>> dataset = AECDataset(Path("data/raw/lsp_aec"))
        >>> print(f"Total samples: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(f"Gloss: {sample.gloss}, ID: {sample.gloss_id}")
        >>> print(f"Hand shape: {sample.hand_keypoints.shape}")
    
    Attributes:
        dataset_root: Path to the AEC dataset root directory
        gloss_to_id: Mapping from gloss string to integer ID
        id_to_gloss: Mapping from integer ID to gloss string
    """
    
    DEFAULT_DICT_FILENAME = "dict.json"
    
    def __init__(
        self,
        dataset_root: Path,
        dict_path: Optional[Path] = None,
        resolver: Optional[KeypointResolverInterface] = None,
        transform: Optional[Callable[[EncoderReadySample], Any]] = None,
        lazy_load: bool = True,
        skip_missing: bool = True,
        split_file: Optional[Path] = None,
        split: Optional[Literal["train", "val"]] = None
    ):
        """
        Initialize the AEC dataset.
        
        Args:
            dataset_root: Path to the AEC dataset root (e.g., data/raw/lsp_aec/)
            dict_path: Optional explicit path to dict.json. If None, uses
                       dataset_root/dict.json
            resolver: Optional custom KeypointResolverInterface. If None,
                      uses AECKeypointResolver with default settings
            transform: Optional transform function to apply to samples
            lazy_load: If True, load keypoints on-demand. If False, load all
                       at initialization (memory intensive)
            skip_missing: If True, skip instances with missing .pkl files
            split_file: Optional path to a JSON file containing split definitions.
                       The file should have format: {"train": [...], "val": [...]}
                       where values are lists of unique_name strings.
            split: Which split to load ("train" or "val"). Required if split_file
                  is provided. If split_file is None, this is ignored.
        
        Note:
            The AEC dataset contains only two source videos and no valid signer
            information (signer_id=-1). Split files should use gloss stratification
            for technical pipeline validation, not as true generalization evaluation.
        """
        self.dataset_root = Path(dataset_root).resolve()
        self.transform = transform
        self.lazy_load = lazy_load
        self.skip_missing = skip_missing
        self.split_file = Path(split_file) if split_file else None
        self.split = split
        
        # Validate split parameters
        if self.split_file is not None and self.split is None:
            raise ValueError("split parameter is required when split_file is provided")
        
        # Setup resolver
        self.resolver = resolver or AECKeypointResolver(self.dataset_root)
        
        # Setup dict path
        self.dict_path = dict_path or (self.dataset_root / self.DEFAULT_DICT_FILENAME)
        
        # Load and parse the dictionary
        self._raw_dict: Dict[str, Any] = {}
        self._flat_instances: List[Dict[str, Any]] = []
        self._gloss_to_id: Dict[str, int] = {}
        self._id_to_gloss: Dict[int, str] = {}
        
        self._load_dictionary()
        self._build_vocabulary()
        self._flatten_instances()
        
        # Apply external split if provided
        if self.split_file is not None:
            self._apply_split()
        
        logger.info(
            f"AECDataset initialized: {len(self)} samples, "
            f"{len(self._gloss_to_id)} glosses"
            + (f" (split={self.split})" if self.split else "")
        )
    
    def _load_dictionary(self) -> None:
        """Load and parse dict.json."""
        if not self.dict_path.exists():
            raise FileNotFoundError(f"Dictionary not found: {self.dict_path}")
        
        logger.debug(f"Loading dictionary from {self.dict_path}")
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            self._raw_dict = json.load(f)
        
        logger.debug(f"Loaded {len(self._raw_dict)} gloss entries")
    
    def _build_vocabulary(self) -> None:
        """Build gloss ↔ id mappings from the dictionary."""
        glosses = set()
        
        for entry in self._raw_dict.values():
            gloss = entry.get('gloss', '')
            if gloss:
                glosses.add(gloss)
        
        # Sort for deterministic ordering
        sorted_glosses = sorted(glosses)
        
        self._gloss_to_id = {gloss: idx for idx, gloss in enumerate(sorted_glosses)}
        self._id_to_gloss = {idx: gloss for gloss, idx in self._gloss_to_id.items()}
        
        logger.debug(f"Built vocabulary with {len(self._gloss_to_id)} unique glosses")
    
    def _flatten_instances(self) -> None:
        """
        Flatten all instances across all glosses into a single list.
        
        Each entry in _flat_instances contains:
        - gloss: The gloss label
        - instance: The instance dict from dict.json
        - gloss_id: The integer ID for the gloss
        """
        self._flat_instances = []
        missing_count = 0
        
        for entry in self._raw_dict.values():
            gloss = entry.get('gloss', '')
            if not gloss:
                continue
            
            gloss_id = self._gloss_to_id[gloss]
            
            for instance in entry.get('instances', []):
                keypoints_path = instance.get('keypoints_path', '')
                
                # Check if file exists (if skip_missing is enabled)
                if self.skip_missing and keypoints_path:
                    if not self.resolver.exists(keypoints_path):
                        missing_count += 1
                        continue
                
                self._flat_instances.append({
                    'gloss': gloss,
                    'gloss_id': gloss_id,
                    'instance': instance
                })
        
        if missing_count > 0:
            logger.warning(f"Skipped {missing_count} instances with missing keypoint files")
    
    def _apply_split(self) -> None:
        """
        Filter instances based on external split file.
        
        The split file should be a JSON with format:
        {
            "train": ["unique_name_1", "unique_name_2", ...],
            "val": ["unique_name_3", ...]
        }
        
        Only instances whose unique_name appears in the specified split
        will be kept. The vocabulary is NOT rebuilt - it contains all
        glosses regardless of split for consistent label encoding.
        """
        if self.split_file is None or self.split is None:
            return
        
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        
        with open(self.split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        
        if self.split not in split_data:
            available = [k for k in split_data.keys() if k != "metadata"]
            raise ValueError(
                f"Split '{self.split}' not found in split file. "
                f"Available splits: {available}"
            )
        
        valid_names: Set[str] = set(split_data[self.split])
        original_count = len(self._flat_instances)
        
        self._flat_instances = [
            entry for entry in self._flat_instances
            if entry['instance'].get('unique_name') in valid_names
        ]
        
        # Count glosses in this split
        glosses_in_split = set(entry['gloss'] for entry in self._flat_instances)
        
        logger.info(
            f"Applied '{self.split}' split: {original_count} → {len(self._flat_instances)} samples "
            f"({len(glosses_in_split)} glosses)"
        )
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self._flat_instances)
    
    def __getitem__(self, idx: int) -> EncoderReadySample:
        """
        Get a single encoder-ready sample by index.
        
        Clean separation: load → parse → convert
        
        Args:
            idx: Sample index (0 to len-1)
            
        Returns:
            EncoderReadySample with tensorized keypoints
            
        Raises:
            IndexError: If idx is out of range
            FileNotFoundError: If keypoints file is missing and skip_missing is False
        """
        raw = self.get_raw_sample(idx)
        sample = self._to_encoder_ready(raw)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def get_raw_sample(self, idx: int) -> Sample:
        """
        Load and return raw sample (AEC format, no tensorization).
        
        Useful for debugging, visualization, and data inspection.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample with raw keypoints as list of dicts
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        entry = self._flat_instances[idx]
        instance = entry['instance']
        gloss = entry['gloss']
        
        # Resolve and load keypoints
        keypoints_path = instance.get('keypoints_path', '')
        resolved_path = self.resolver.resolve(keypoints_path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"Keypoints file not found: {resolved_path}")
        
        with open(resolved_path, 'rb') as f:
            keypoints = pickle.load(f)
        
        # Build metadata
        metadata = {
            'signer_id': instance.get('signer_id'),
            'source': instance.get('source'),
            'source_video_name': instance.get('source_video_name'),
            'image_dimension': instance.get('image_dimention'),  # Note: typo in original data
            'keypoints_path': keypoints_path,
            'resolved_path': str(resolved_path)
        }
        
        return Sample(
            gloss=gloss,
            keypoints=keypoints,  # List[Dict] - raw AEC format
            frame_start=instance.get('frame_start', 0),
            frame_end=instance.get('frame_end', len(keypoints) - 1),
            unique_name=instance.get('unique_name', f"{gloss}_{idx}"),
            metadata=metadata
        )
    
    def _to_encoder_ready(self, sample: Sample) -> EncoderReadySample:
        """
        Convert raw Sample to EncoderReadySample.
        
        Uses pure converter functions from converters.py.
        
        Args:
            sample: Raw sample with AEC-format keypoints
            
        Returns:
            EncoderReadySample with tensorized keypoints
        """
        # Use the pure stateless converter
        converted = aec_keypoints_to_encoder_format(sample.keypoints)
        
        return EncoderReadySample(
            gloss=sample.gloss,
            gloss_id=self._gloss_to_id[sample.gloss],
            hand_keypoints=converted['hand'],
            body_keypoints=converted['body'],
            face_keypoints=converted['face'],
            unique_name=sample.unique_name,
            metadata=sample.metadata
        )
    
    @property
    def gloss_to_id(self) -> Dict[str, int]:
        """Mapping from gloss string to integer ID."""
        return self._gloss_to_id
    
    @property
    def id_to_gloss(self) -> Dict[int, str]:
        """Mapping from integer ID to gloss string."""
        return self._id_to_gloss
    
    def get_gloss_labels(self) -> List[str]:
        """Return list of unique gloss labels in the dataset."""
        return list(self._gloss_to_id.keys())
    
    def get_instances_by_gloss(self, gloss: str) -> List[int]:
        """
        Get all sample indices for a specific gloss.
        
        Useful for class-balanced sampling.
        
        Args:
            gloss: The gloss label to filter by
            
        Returns:
            List of sample indices that have this gloss
        """
        return [
            idx for idx, entry in enumerate(self._flat_instances)
            if entry['gloss'] == gloss
        ]
    
    def get_sample_counts_per_gloss(self) -> Dict[str, int]:
        """
        Get the number of samples for each gloss.
        
        Returns:
            Dict mapping gloss -> sample count
        """
        counts: Dict[str, int] = {}
        for entry in self._flat_instances:
            gloss = entry['gloss']
            counts[gloss] = counts.get(gloss, 0) + 1
        return counts
    
    def __repr__(self) -> str:
        return (
            f"AECDataset(samples={len(self)}, "
            f"glosses={len(self._gloss_to_id)}, "
            f"root={self.dataset_root})"
        )
