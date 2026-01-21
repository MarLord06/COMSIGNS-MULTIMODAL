"""
Dataset Coverage Analysis module.

Analyzes coverage between a vocabulary file (dict.json) and actual dataset samples,
providing insights into class imbalance, missing classes, and dataset health.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class DatasetCoverageAnalyzer:
    """
    Analyzer for dataset coverage against a vocabulary/dictionary file.
    
    Compares:
    - Classes defined in vocabulary vs classes present in data
    - Sample distribution across classes
    - Identifies missing/orphan classes
    
    Example:
        >>> analyzer = DatasetCoverageAnalyzer("data/raw/lsp_aec/dict.json")
        >>> analyzer.load_vocabulary()
        >>> analyzer.analyze_from_labels([0, 1, 1, 2, 2, 2])
        >>> print(analyzer.format_report())
    """
    
    def __init__(
        self,
        vocab_path: Optional[Union[str, Path]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize analyzer.
        
        Args:
            vocab_path: Path to vocabulary file (dict.json format)
            class_names: Alternatively, provide class names directly as list
        """
        self.vocab_path = Path(vocab_path) if vocab_path else None
        self._class_names: List[str] = class_names or []
        self._class_to_idx: Dict[str, int] = {}
        self._idx_to_class: Dict[int, str] = {}
        
        # Analysis results
        self._label_counts: Counter = Counter()
        self._total_samples: int = 0
        self._classes_in_data: Set[int] = set()
        
        if class_names:
            self._build_mappings_from_names(class_names)
    
    def load_vocabulary(self) -> int:
        """
        Load vocabulary from JSON file.
        
        Expected format: {"gloss_name": index, ...}
        
        Returns:
            Number of classes loaded
        
        Raises:
            FileNotFoundError: If vocab file doesn't exist
            ValueError: If file format is invalid
        """
        if not self.vocab_path:
            raise ValueError("No vocabulary path specified")
        
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")
        
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        
        if not isinstance(vocab_data, dict):
            raise ValueError(f"Expected dict format, got {type(vocab_data)}")
        
        # Build mappings
        self._class_names = []
        self._class_to_idx = {}
        self._idx_to_class = {}
        
        for name, idx in vocab_data.items():
            if not isinstance(idx, int):
                logger.warning(f"Skipping non-integer index for {name}: {idx}")
                continue
            
            self._class_to_idx[name] = idx
            self._idx_to_class[idx] = name
        
        # Sort by index to build class_names list
        max_idx = max(self._idx_to_class.keys()) if self._idx_to_class else -1
        self._class_names = ["" for _ in range(max_idx + 1)]
        
        for idx, name in self._idx_to_class.items():
            self._class_names[idx] = name
        
        logger.info(f"Loaded vocabulary with {len(self._class_to_idx)} classes")
        return len(self._class_to_idx)
    
    def _build_mappings_from_names(self, names: List[str]) -> None:
        """Build mappings from a list of class names."""
        self._class_names = names
        self._class_to_idx = {name: idx for idx, name in enumerate(names)}
        self._idx_to_class = {idx: name for idx, name in enumerate(names)}
    
    @property
    def num_classes(self) -> int:
        """Number of classes in vocabulary."""
        return len(self._class_names)
    
    @property
    def class_names(self) -> List[str]:
        """List of class names indexed by class ID."""
        return self._class_names
    
    def get_class_name(self, idx: int) -> str:
        """Get class name by index."""
        return self._idx_to_class.get(idx, f"Unknown_{idx}")
    
    def get_class_index(self, name: str) -> int:
        """Get class index by name."""
        return self._class_to_idx.get(name, -1)
    
    def analyze_from_labels(
        self,
        labels: Union[List[int], np.ndarray],
        reset: bool = True
    ) -> Dict:
        """
        Analyze coverage from a list of labels.
        
        Args:
            labels: List or array of class indices
            reset: Whether to reset previous analysis
        
        Returns:
            Analysis results dictionary
        """
        if reset:
            self.reset()
        
        labels_arr = np.asarray(labels)
        
        self._total_samples += len(labels_arr)
        self._classes_in_data.update(labels_arr.tolist())
        self._label_counts.update(labels_arr.tolist())
        
        return self.get_analysis()
    
    def analyze_from_dataset(
        self,
        dataset,
        label_key: str = "label"
    ) -> Dict:
        """
        Analyze coverage from a PyTorch dataset.
        
        Args:
            dataset: Dataset with __len__ and __getitem__
            label_key: Key or attribute to extract label from sample
        
        Returns:
            Analysis results dictionary
        """
        self.reset()
        
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, dict):
                label = sample.get(label_key)
            elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
                label = sample[1]  # Assume (data, label) format
            elif hasattr(sample, label_key):
                label = getattr(sample, label_key)
            else:
                raise ValueError(f"Cannot extract label from sample: {type(sample)}")
            
            if hasattr(label, 'item'):
                label = label.item()
            labels.append(label)
        
        return self.analyze_from_labels(labels, reset=False)
    
    def reset(self) -> None:
        """Reset analysis state."""
        self._label_counts = Counter()
        self._total_samples = 0
        self._classes_in_data = set()
    
    def get_analysis(self) -> Dict:
        """
        Get comprehensive analysis results.
        
        Returns:
            Dictionary with coverage analysis:
            - total_samples: Total samples analyzed
            - vocab_size: Number of classes in vocabulary
            - classes_in_data: Number of unique classes found
            - coverage_ratio: Proportion of vocabulary covered
            - missing_classes: List of classes not in data
            - orphan_classes: Classes in data but not in vocab
            - class_distribution: Dict of class -> count
            - imbalance_metrics: Imbalance statistics
        """
        vocab_classes = set(range(self.num_classes))
        
        missing_classes = vocab_classes - self._classes_in_data
        orphan_classes = self._classes_in_data - vocab_classes
        
        # Convert missing classes to names
        missing_class_info = []
        for cls in sorted(missing_classes):
            missing_class_info.append({
                "class_id": cls,
                "name": self.get_class_name(cls)
            })
        
        # Compute imbalance metrics
        if self._label_counts:
            counts = list(self._label_counts.values())
            min_count = min(counts)
            max_count = max(counts)
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            median_count = np.median(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        else:
            min_count = max_count = mean_count = std_count = median_count = 0
            imbalance_ratio = 0.0
        
        # Class distribution with names
        class_distribution = {}
        for cls_idx, count in self._label_counts.most_common():
            name = self.get_class_name(cls_idx)
            class_distribution[name] = {
                "class_id": cls_idx,
                "count": count,
                "percentage": (count / self._total_samples * 100) if self._total_samples > 0 else 0
            }
        
        coverage_ratio = (
            len(self._classes_in_data & vocab_classes) / self.num_classes
            if self.num_classes > 0 else 0.0
        )
        
        return {
            "total_samples": self._total_samples,
            "vocab_size": self.num_classes,
            "classes_in_data": len(self._classes_in_data),
            "coverage_ratio": coverage_ratio,
            "coverage_percentage": coverage_ratio * 100,
            "missing_classes": missing_class_info,
            "num_missing": len(missing_classes),
            "orphan_classes": list(orphan_classes),
            "num_orphans": len(orphan_classes),
            "class_distribution": class_distribution,
            "imbalance_metrics": {
                "min_count": int(min_count),
                "max_count": int(max_count),
                "mean_count": float(mean_count),
                "std_count": float(std_count),
                "median_count": float(median_count),
                "imbalance_ratio": float(imbalance_ratio)
            }
        }
    
    def get_underrepresented_classes(
        self,
        threshold: int = 5
    ) -> List[Dict]:
        """
        Get classes with fewer than threshold samples.
        
        Args:
            threshold: Minimum samples threshold
        
        Returns:
            List of class info dicts for underrepresented classes
        """
        underrepresented = []
        
        for cls_idx in range(self.num_classes):
            count = self._label_counts.get(cls_idx, 0)
            if count < threshold:
                underrepresented.append({
                    "class_id": cls_idx,
                    "name": self.get_class_name(cls_idx),
                    "count": count,
                    "missing": count == 0
                })
        
        # Sort by count ascending
        underrepresented.sort(key=lambda x: x["count"])
        
        return underrepresented
    
    def get_overrepresented_classes(
        self,
        threshold_percentile: float = 95.0
    ) -> List[Dict]:
        """
        Get classes above the threshold percentile in sample count.
        
        Args:
            threshold_percentile: Percentile threshold (e.g., 95.0)
        
        Returns:
            List of class info dicts for overrepresented classes
        """
        if not self._label_counts:
            return []
        
        counts = list(self._label_counts.values())
        threshold = np.percentile(counts, threshold_percentile)
        
        overrepresented = []
        for cls_idx, count in self._label_counts.items():
            if count >= threshold:
                overrepresented.append({
                    "class_id": cls_idx,
                    "name": self.get_class_name(cls_idx),
                    "count": count,
                    "percentile": float(
                        (np.array(counts) <= count).sum() / len(counts) * 100
                    )
                })
        
        # Sort by count descending
        overrepresented.sort(key=lambda x: x["count"], reverse=True)
        
        return overrepresented
    
    def format_report(
        self,
        show_distribution: bool = True,
        top_n: int = 20,
        show_missing: bool = True
    ) -> str:
        """
        Format analysis as human-readable report.
        
        Args:
            show_distribution: Show class distribution table
            top_n: Number of top/bottom classes to show
            show_missing: Show list of missing classes
        
        Returns:
            Formatted report string
        """
        analysis = self.get_analysis()
        
        lines = []
        lines.append("=" * 80)
        lines.append("DATASET COVERAGE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary statistics
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total samples:         {analysis['total_samples']:,}")
        lines.append(f"Vocabulary size:       {analysis['vocab_size']:,}")
        lines.append(f"Classes in data:       {analysis['classes_in_data']:,}")
        lines.append(f"Coverage:              {analysis['coverage_percentage']:.1f}%")
        lines.append(f"Missing classes:       {analysis['num_missing']:,}")
        lines.append(f"Orphan classes:        {analysis['num_orphans']:,}")
        lines.append("")
        
        # Imbalance metrics
        imb = analysis["imbalance_metrics"]
        lines.append("IMBALANCE METRICS")
        lines.append("-" * 40)
        lines.append(f"Min samples/class:     {imb['min_count']:,}")
        lines.append(f"Max samples/class:     {imb['max_count']:,}")
        lines.append(f"Mean samples/class:    {imb['mean_count']:.1f}")
        lines.append(f"Std samples/class:     {imb['std_count']:.1f}")
        lines.append(f"Median samples/class:  {imb['median_count']:.1f}")
        lines.append(f"Imbalance ratio:       {imb['imbalance_ratio']:.1f}x")
        lines.append("")
        
        # Class distribution
        if show_distribution and analysis["class_distribution"]:
            lines.append(f"CLASS DISTRIBUTION (Top {top_n} & Bottom {top_n})")
            lines.append("-" * 40)
            
            dist_items = list(analysis["class_distribution"].items())
            
            # Top classes
            lines.append("Most frequent:")
            for name, info in dist_items[:top_n]:
                display_name = name[:25] + "..." if len(name) > 28 else name
                lines.append(f"  {display_name:<30} {info['count']:>6} ({info['percentage']:>5.1f}%)")
            
            if len(dist_items) > top_n * 2:
                lines.append(f"  ... ({len(dist_items) - top_n * 2} more classes) ...")
            
            # Bottom classes
            if len(dist_items) > top_n:
                lines.append("Least frequent:")
                for name, info in dist_items[-top_n:]:
                    display_name = name[:25] + "..." if len(name) > 28 else name
                    lines.append(f"  {display_name:<30} {info['count']:>6} ({info['percentage']:>5.1f}%)")
            
            lines.append("")
        
        # Missing classes
        if show_missing and analysis["missing_classes"]:
            lines.append(f"MISSING CLASSES ({analysis['num_missing']} total)")
            lines.append("-" * 40)
            missing_to_show = analysis["missing_classes"][:30]
            for info in missing_to_show:
                lines.append(f"  [{info['class_id']:>3}] {info['name']}")
            if len(analysis["missing_classes"]) > 30:
                lines.append(f"  ... and {len(analysis['missing_classes']) - 30} more")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def export_to_json(self, output_path: Union[str, Path]) -> None:
        """Export analysis to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        analysis = self.get_analysis()
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis exported to {output_path}")
