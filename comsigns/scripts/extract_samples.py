#!/usr/bin/env python3
"""
Extract inference-ready samples from the AEC dataset.

Exports individual samples in a format directly consumable by the inference
script (scripts/infer.py).

Usage:
    # Extract samples for specific classes
    python scripts/extract_samples.py \
      --class-ids 18 26 31 51 \
      --output-dir samples/

    # Extract first N samples (one per class)
    python scripts/extract_samples.py \
      --num-samples 10 \
      --output-dir samples/

    # Use validation split
    python scripts/extract_samples.py \
      --split validation \
      --split-file experiments/run_xxx/split.json \
      --class-ids 18 26 \
      --output-dir samples/

Output format (.pkl):
    {
        "hand": torch.Tensor,      # shape [1, T, 168]
        "body": torch.Tensor,      # shape [1, T, 132]
        "face": torch.Tensor,      # shape [1, T, 1872]
        "lengths": torch.Tensor,   # shape [1]
        "class_id": int,
        "label": str,
        "bucket": str,             # HEAD | MID | TAIL
        "unique_name": str
    }
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.datasets.aec import AECDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract inference-ready samples from the AEC dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract specific classes
  python scripts/extract_samples.py --class-ids 18 26 31 51

  # Extract first 10 classes
  python scripts/extract_samples.py --num-samples 10

  # From validation split
  python scripts/extract_samples.py --split val --split-file split.json --class-ids 5 10
        """
    )
    
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("../data/raw/lsp_aec"),
        help="Path to AEC dataset root (default: ../data/raw/lsp_aec)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "validation"],
        default=None,
        help="Which split to use (train or val). If not specified, uses full dataset."
    )
    
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Path to split JSON file (required if --split is used)"
    )
    
    parser.add_argument(
        "--class-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific class IDs to extract (e.g., --class-ids 18 26 31)"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to extract if --class-ids not provided (default: 5)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("samples"),
        help="Output directory for extracted samples (default: samples/)"
    )
    
    parser.add_argument(
        "--head-threshold",
        type=int,
        default=10,
        help="Threshold for HEAD bucket classification (default: 10)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)"
    )
    
    return parser.parse_args()


def compute_bucket(support: int, head_threshold: int = 10) -> str:
    """Classify a class into a bucket based on sample count.
    
    Args:
        support: Number of samples for the class
        head_threshold: Minimum samples to be HEAD
    
    Returns:
        Bucket string: "HEAD", "MID", or "TAIL"
    """
    if support >= head_threshold:
        return "HEAD"
    elif 3 <= support <= 9:
        return "MID"
    else:
        return "TAIL"


def is_valid_sample(sample) -> bool:
    """Check if a sample has valid features (no NaNs, non-empty).
    
    Args:
        sample: EncoderReadySample from dataset
    
    Returns:
        True if valid, False otherwise
    """
    for arr in [sample.hand_keypoints, sample.body_keypoints, sample.face_keypoints]:
        if arr is None or arr.size == 0:
            return False
        if np.isnan(arr).any():
            return False
    return True


def find_valid_sample_for_class(
    dataset: AECDataset,
    class_id: int,
    seed: int = 42
) -> Optional[int]:
    """Find a valid sample index for a given class ID.
    
    Args:
        dataset: AECDataset instance
        class_id: Target class ID
        seed: Random seed for deterministic selection
    
    Returns:
        Sample index or None if no valid sample found
    """
    # Get gloss name for this class ID
    gloss = dataset.id_to_gloss.get(class_id)
    if gloss is None:
        return None
    
    # Get all indices for this gloss
    indices = dataset.get_instances_by_gloss(gloss)
    if not indices:
        return None
    
    # Set seed for deterministic selection
    np.random.seed(seed + class_id)  # Different seed per class for variety
    np.random.shuffle(indices)
    
    # Find first valid sample
    for idx in indices:
        try:
            sample = dataset[idx]
            if is_valid_sample(sample):
                return idx
        except Exception:
            continue
    
    return None


def extract_sample(
    dataset: AECDataset,
    idx: int,
    class_support: Dict[int, int],
    head_threshold: int = 10
) -> Dict:
    """Extract a single sample in inference-ready format.
    
    Args:
        dataset: AECDataset instance
        idx: Sample index
        class_support: Mapping from class_id to sample count
        head_threshold: Threshold for HEAD bucket
    
    Returns:
        Dictionary with features and metadata
    """
    sample = dataset[idx]
    
    # Convert to tensors with batch dimension
    hand = torch.from_numpy(sample.hand_keypoints).float().unsqueeze(0)  # [1, T, 168]
    body = torch.from_numpy(sample.body_keypoints).float().unsqueeze(0)  # [1, T, 132]
    face = torch.from_numpy(sample.face_keypoints).float().unsqueeze(0)  # [1, T, 1872]
    lengths = torch.tensor([sample.num_frames], dtype=torch.long)
    
    # Determine bucket
    support = class_support.get(sample.gloss_id, 0)
    bucket = compute_bucket(support, head_threshold)
    
    return {
        "hand": hand,
        "body": body,
        "face": face,
        "lengths": lengths,
        "class_id": sample.gloss_id,
        "label": sample.gloss,
        "bucket": bucket,
        "unique_name": sample.unique_name,
        "num_frames": sample.num_frames
    }


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Set global seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # =========================================================================
    # 1. Load Dataset
    # =========================================================================
    print(f"Loading dataset from {args.dataset_path}...")
    
    # Handle split argument
    split = args.split
    if split == "validation":
        split = "val"
    
    # Validate split arguments
    if split is not None and args.split_file is None:
        print("Warning: --split specified without --split-file, ignoring split")
        split = None
    
    try:
        if split is not None and args.split_file is not None:
            dataset = AECDataset(
                args.dataset_path,
                split_file=args.split_file,
                split=split
            )
        else:
            dataset = AECDataset(args.dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"  Loaded {len(dataset)} samples, {len(dataset.gloss_to_id)} classes")
    
    # =========================================================================
    # 2. Compute Class Support (for bucket assignment)
    # =========================================================================
    class_support = {}
    counts = dataset.get_sample_counts_per_gloss()
    for gloss, count in counts.items():
        class_id = dataset.gloss_to_id[gloss]
        class_support[class_id] = count
    
    # =========================================================================
    # 3. Determine Classes to Extract
    # =========================================================================
    if args.class_ids is not None:
        target_classes = args.class_ids
        print(f"Extracting samples for classes: {target_classes}")
    else:
        # Get first N unique classes that have valid samples
        all_classes = sorted(dataset.gloss_to_id.values())
        target_classes = all_classes[:args.num_samples]
        print(f"Extracting first {args.num_samples} classes: {target_classes}")
    
    # =========================================================================
    # 4. Extract Samples
    # =========================================================================
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted = 0
    skipped = 0
    
    for class_id in target_classes:
        # Find a valid sample for this class
        idx = find_valid_sample_for_class(dataset, class_id, args.seed)
        
        if idx is None:
            gloss = dataset.id_to_gloss.get(class_id, "UNKNOWN")
            print(f"  [SKIP] Class {class_id} ({gloss}): no valid sample found")
            skipped += 1
            continue
        
        # Extract the sample
        try:
            sample_data = extract_sample(
                dataset, idx, class_support, args.head_threshold
            )
        except Exception as e:
            print(f"  [ERROR] Class {class_id}: {e}")
            skipped += 1
            continue
        
        # Save to file
        output_path = args.output_dir / f"class_{class_id:03d}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(sample_data, f)
        
        print(
            f"  [OK] Class {class_id:3d} ({sample_data['label']:<15}) "
            f"→ {output_path.name}  "
            f"[{sample_data['bucket']}, T={sample_data['num_frames']}]"
        )
        extracted += 1
    
    # =========================================================================
    # 5. Summary
    # =========================================================================
    print()
    print("=" * 60)
    print(f"Extraction complete: {extracted} samples saved, {skipped} skipped")
    print(f"Output directory: {args.output_dir.resolve()}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
