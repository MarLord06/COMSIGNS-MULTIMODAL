#!/usr/bin/env python3
"""
Bucket Analysis CLI Script.

Analyzes model performance by support buckets (HEAD/MID/TAIL)
and computes coverage metrics for long-tail classification.

This script is READ-ONLY - it does NOT modify:
- Model
- Trainer  
- Dataset
- Split

Bucket Definitions:
- HEAD: ≥ 10 training samples
- MID: 3-9 training samples
- TAIL: 1-2 training samples

Usage:
    # Basic analysis (uses validation support from metrics)
    python scripts/run_bucket_analysis.py \\
        --metrics experiments/run_001/metrics_by_class.json

    # With training support from dataset
    python scripts/run_bucket_analysis.py \\
        --metrics experiments/run_001/metrics_by_class.json \\
        --dataset-root data/raw/lsp_aec \\
        --split-file data/processed/stratified_split.json

    # With custom thresholds
    python scripts/run_bucket_analysis.py \\
        --metrics experiments/run_001/metrics_by_class.json \\
        --head-threshold 8 \\
        --mid-range 3 7 \\
        --output experiments/run_001/bucket_analysis.json
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.analysis.bucket_analysis import BucketAnalyzer, Bucket, run_bucket_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_training_support(
    dataset_root: Path,
    split_file: Path,
    split: str = "train"
) -> dict:
    """
    Compute support counts from training split.
    
    Args:
        dataset_root: Path to AEC dataset
        split_file: Path to stratified split JSON
        split: Which split to count ("train" or "val")
    
    Returns:
        Dict mapping class_id to sample count
    """
    from core.data.datasets.aec import AECDataset
    
    logger.info(f"Loading dataset from {dataset_root}")
    logger.info(f"Using split file: {split_file}, split={split}")
    
    dataset = AECDataset(
        dataset_root=dataset_root,
        split_file=split_file,
        split=split
    )
    
    # Count samples per class
    support = Counter()
    for i in range(len(dataset)):
        sample = dataset[i]
        support[sample.gloss_id] += 1
    
    logger.info(f"Computed support for {len(support)} classes, {sum(support.values())} samples")
    
    return dict(support)


def main():
    parser = argparse.ArgumentParser(
        description="Bucket Analysis for Long-Tail Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic analysis:
    python scripts/run_bucket_analysis.py --metrics experiments/run_001/metrics_by_class.json

  With training support:
    python scripts/run_bucket_analysis.py \\
        --metrics experiments/run_001/metrics_by_class.json \\
        --dataset-root data/raw/lsp_aec \\
        --split-file data/processed/stratified_split.json

  Custom bucket thresholds:
    python scripts/run_bucket_analysis.py \\
        --metrics experiments/run_001/metrics_by_class.json \\
        --head-threshold 8 \\
        --mid-range 3 7
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to metrics_by_class.json from evaluation"
    )
    
    # Optional: Dataset for training support
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to AEC dataset root (for computing training support)"
    )
    
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to stratified split JSON file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output JSON file for results"
    )
    
    parser.add_argument(
        "--head-threshold",
        type=int,
        default=10,
        help="Minimum samples for HEAD bucket (default: 10)"
    )
    
    parser.add_argument(
        "--mid-range",
        type=int,
        nargs=2,
        default=[3, 9],
        metavar=("MIN", "MAX"),
        help="Sample range for MID bucket (default: 3 9)"
    )
    
    parser.add_argument(
        "--tail-range",
        type=int,
        nargs=2,
        default=[1, 2],
        metavar=("MIN", "MAX"),
        help="Sample range for TAIL bucket (default: 1 2)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output (still writes to file)"
    )
    
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output raw JSON to stdout instead of formatted report"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        sys.exit(1)
    
    # Compute training support if dataset is provided
    training_support = None
    if args.dataset_root and args.split_file:
        dataset_root = Path(args.dataset_root)
        split_file = Path(args.split_file)
        
        if not dataset_root.exists():
            logger.error(f"Dataset root not found: {dataset_root}")
            sys.exit(1)
        if not split_file.exists():
            logger.error(f"Split file not found: {split_file}")
            sys.exit(1)
        
        training_support = compute_training_support(dataset_root, split_file, "train")
    elif args.dataset_root or args.split_file:
        logger.warning("Both --dataset-root and --split-file required for training support")
    
    # Initialize analyzer with custom thresholds
    analyzer = BucketAnalyzer(
        head_threshold=args.head_threshold,
        mid_range=tuple(args.mid_range),
        tail_range=tuple(args.tail_range)
    )
    
    # Load data
    logger.info(f"Loading metrics from {metrics_path}")
    analyzer.load_from_files(metrics_path)
    
    # Override with training support if available
    if training_support:
        analyzer._class_support = training_support
        logger.info("Using training set support for bucket classification")
    else:
        logger.warning("Using validation support (may differ from training distribution)")
    
    # Run analysis
    logger.info("Running bucket analysis...")
    result = analyzer.analyze()
    
    # Output
    if args.json_only:
        # Raw JSON to stdout
        print(json.dumps(result.to_dict(), indent=2))
    elif not args.quiet:
        # Formatted report
        analyzer.print_report()
    
    # Export to file if requested
    if args.output:
        output_path = Path(args.output)
        analyzer.export_results(output_path)
        logger.info(f"Results exported to {output_path}")
    
    # Summary for logging
    tail_analysis = result.tail_analysis
    head_bucket = analyzer.classify_bucket(10)
    mid_bucket = analyzer.classify_bucket(5)
    tail_bucket = analyzer.classify_bucket(1)
    
    logger.info(
        f"Analysis complete: "
        f"HEAD={result.bucket_metrics[head_bucket].num_classes} classes, "
        f"MID={result.bucket_metrics[mid_bucket].num_classes} classes, "
        f"TAIL={result.bucket_metrics[tail_bucket].num_classes} classes"
    )
    logger.info(f"Recommended strategy: {tail_analysis['diagnosis']['recommendation']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
