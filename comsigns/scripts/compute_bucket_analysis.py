#!/usr/bin/env python3
"""
Script to compute training support and run bucket analysis.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.datasets.aec import AECDataset
from training.analysis.bucket_analysis import BucketAnalyzer, Bucket


def main():
    # Paths
    dataset_root = Path("../data/raw/lsp_aec")
    split_file = Path("../data/splits/aec_stratified.json")
    metrics_file = Path("experiments/run_20260120_162424/metrics_by_class.json")
    output_file = Path("experiments/run_20260120_162424/bucket_analysis_with_train_support.json")
    
    # Load training set
    print("Loading training dataset...")
    train_ds = AECDataset(
        dataset_root=dataset_root,
        split_file=split_file,
        split='train'
    )
    
    # Compute training support
    print("Computing training support...")
    train_support = Counter()
    for i in range(len(train_ds)):
        sample = train_ds[i]
        train_support[sample.gloss_id] += 1
    
    # Print training distribution
    print(f"\n{'='*60}")
    print("TRAINING SET DISTRIBUTION")
    print(f"{'='*60}")
    print(f"Total samples: {len(train_ds)}")
    print(f"Total classes with samples: {len(train_support)}")
    
    # Bucket counts
    head = [k for k, v in train_support.items() if v >= 10]
    mid = [k for k, v in train_support.items() if 3 <= v <= 9]
    tail = [k for k, v in train_support.items() if v <= 2]
    
    print(f"\nHEAD (≥10 samples): {len(head)} classes, {sum(train_support[k] for k in head)} samples")
    print(f"MID (3-9 samples):  {len(mid)} classes, {sum(train_support[k] for k in mid)} samples")
    print(f"TAIL (1-2 samples): {len(tail)} classes, {sum(train_support[k] for k in tail)} samples")
    
    # Support distribution
    print("\nSupport distribution (top):")
    for count in sorted(set(train_support.values()), reverse=True)[:10]:
        classes_with_count = sum(1 for v in train_support.values() if v == count)
        print(f"  {count:>3} samples: {classes_with_count} classes")
    
    # Run bucket analysis with training support
    print(f"\n{'='*60}")
    print("BUCKET ANALYSIS (using training support)")
    print(f"{'='*60}")
    
    analyzer = BucketAnalyzer()
    analyzer.load_from_files(metrics_file)
    
    # Override with training support
    analyzer._class_support = dict(train_support)
    
    result = analyzer.analyze()
    analyzer.print_report()
    
    # Export
    analyzer.export_results(output_file)
    print(f"\nResults exported to {output_file}")


if __name__ == "__main__":
    main()
