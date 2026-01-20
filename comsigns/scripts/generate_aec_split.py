#!/usr/bin/env python3
"""
Generate stratified train/val split for AEC dataset.

This script creates a split file based on gloss stratification:
- Each gloss's instances are split 80/20 (configurable)
- Glosses with only 1 instance go to train
- The split is reproducible via seed

Output: data/splits/aec_stratified.json

Usage:
    python scripts/generate_aec_split.py
    python scripts/generate_aec_split.py --train-ratio 0.9 --seed 123
    python scripts/generate_aec_split.py --output custom_split.json

IMPORTANT: The AEC dataset contains only two source videos and no valid signer
information. This stratified split is intended for **technical pipeline validation**,
not as a true evaluation of model generalization.
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Setup path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_dict_json(dict_path: Path) -> Dict[str, Any]:
    """Load and parse dict.json."""
    if not dict_path.exists():
        raise FileNotFoundError(f"dict.json not found at: {dict_path}")
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_instances_by_gloss(dict_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Group all unique_names by their gloss.
    
    Args:
        dict_data: Parsed dict.json content
        
    Returns:
        Dict mapping gloss -> list of unique_names
    """
    gloss_to_instances: Dict[str, List[str]] = defaultdict(list)
    
    for entry in dict_data.values():
        gloss = entry.get('gloss', '')
        if not gloss:
            continue
        
        for instance in entry.get('instances', []):
            unique_name = instance.get('unique_name')
            if unique_name:
                gloss_to_instances[gloss].append(unique_name)
    
    return dict(gloss_to_instances)


def generate_stratified_split(
    gloss_to_instances: Dict[str, List[str]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Generate stratified train/val split by gloss.
    
    Each gloss's instances are independently shuffled and split.
    Glosses with only 1 instance go entirely to train.
    
    Args:
        gloss_to_instances: Dict mapping gloss -> list of unique_names
        train_ratio: Fraction of instances for training (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_names, val_names, stats_dict)
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    
    random.seed(seed)
    
    train_names: List[str] = []
    val_names: List[str] = []
    single_instance_glosses = 0
    
    # Sort glosses for deterministic iteration
    sorted_glosses = sorted(gloss_to_instances.keys())
    
    for gloss in sorted_glosses:
        instances = gloss_to_instances[gloss].copy()
        random.shuffle(instances)
        
        if len(instances) == 1:
            # Single instance glosses go to train (can't split)
            train_names.extend(instances)
            single_instance_glosses += 1
        else:
            # Split with at least 1 in train
            split_idx = max(1, int(train_ratio * len(instances)))
            train_names.extend(instances[:split_idx])
            val_names.extend(instances[split_idx:])
    
    stats = {
        "total_glosses": len(sorted_glosses),
        "single_instance_glosses": single_instance_glosses,
        "total_instances": len(train_names) + len(val_names),
        "train_count": len(train_names),
        "val_count": len(val_names),
        "actual_train_ratio": len(train_names) / (len(train_names) + len(val_names))
    }
    
    return train_names, val_names, stats


def validate_split(
    train_names: List[str],
    val_names: List[str],
    gloss_to_instances: Dict[str, List[str]]
) -> None:
    """
    Validate the generated split.
    
    Checks:
    - No overlap between train and val
    - All instances are included
    - Every gloss has at least one instance in train
    
    Raises:
        ValueError: If validation fails
    """
    train_set = set(train_names)
    val_set = set(val_names)
    
    # Check no overlap
    overlap = train_set & val_set
    if overlap:
        raise ValueError(f"Split has {len(overlap)} overlapping instances: {list(overlap)[:5]}...")
    
    # Check all instances included
    all_instances = set()
    for instances in gloss_to_instances.values():
        all_instances.update(instances)
    
    split_instances = train_set | val_set
    
    missing = all_instances - split_instances
    if missing:
        raise ValueError(f"Split is missing {len(missing)} instances: {list(missing)[:5]}...")
    
    extra = split_instances - all_instances
    if extra:
        raise ValueError(f"Split has {len(extra)} extra instances: {list(extra)[:5]}...")
    
    # Check every gloss has train instances
    gloss_in_train = defaultdict(int)
    for gloss, instances in gloss_to_instances.items():
        for inst in instances:
            if inst in train_set:
                gloss_in_train[gloss] += 1
    
    missing_glosses = [g for g in gloss_to_instances.keys() if gloss_in_train[g] == 0]
    if missing_glosses:
        raise ValueError(f"Glosses without train instances: {missing_glosses[:5]}...")
    
    logger.info("✅ Split validation passed")


def save_split(
    train_names: List[str],
    val_names: List[str],
    output_path: Path,
    seed: int,
    train_ratio: float,
    stats: Dict[str, Any]
) -> None:
    """Save split to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    split_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "seed": seed,
            "train_ratio": train_ratio,
            "strategy": "stratified_by_gloss",
            "total_instances": stats["total_instances"],
            "total_glosses": stats["total_glosses"],
            "single_instance_glosses": stats["single_instance_glosses"],
            "note": (
                "The AEC dataset contains only two source videos and no valid signer "
                "information. This split is for technical pipeline validation, not "
                "as a true evaluation of model generalization."
            )
        },
        "train": sorted(train_names),
        "val": sorted(val_names)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved split to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified train/val split for AEC dataset"
    )
    parser.add_argument(
        "--dict-path",
        type=Path,
        default=PROJECT_ROOT.parent / "data" / "raw" / "lsp_aec" / "dict.json",
        help="Path to dict.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT.parent / "data" / "splits" / "aec_stratified.json",
        help="Output path for split file"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of instances for training (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Generating Stratified Split for AEC Dataset")
    logger.info("=" * 60)
    
    # Load dict.json
    logger.info(f"Loading dict.json from: {args.dict_path}")
    dict_data = load_dict_json(args.dict_path)
    
    # Group by gloss
    gloss_to_instances = group_instances_by_gloss(dict_data)
    logger.info(f"Found {len(gloss_to_instances)} glosses")
    
    # Count total instances
    total_instances = sum(len(v) for v in gloss_to_instances.values())
    logger.info(f"Total instances: {total_instances}")
    
    # Generate split
    logger.info(f"Generating split (train_ratio={args.train_ratio}, seed={args.seed})...")
    train_names, val_names, stats = generate_stratified_split(
        gloss_to_instances,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    # Log stats
    logger.info(f"  Single-instance glosses: {stats['single_instance_glosses']} (all to train)")
    logger.info(f"  Train: {stats['train_count']} ({stats['actual_train_ratio']*100:.1f}%)")
    logger.info(f"  Val: {stats['val_count']} ({(1-stats['actual_train_ratio'])*100:.1f}%)")
    
    # Validate
    validate_split(train_names, val_names, gloss_to_instances)
    
    # Save
    save_split(
        train_names,
        val_names,
        args.output,
        args.seed,
        args.train_ratio,
        stats
    )
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
