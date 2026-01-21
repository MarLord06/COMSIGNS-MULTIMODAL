#!/usr/bin/env python3
"""
Dataset Analysis Script.

Analyzes AEC dataset coverage, class distribution, and generates reports.

Usage:
    python scripts/analyze_dataset.py --data-dir data/raw/lsp_aec
    python scripts/analyze_dataset.py --data-dir data/raw/lsp_aec --output-dir reports/
    python scripts/analyze_dataset.py --data-dir data/raw/lsp_aec --split-file data/splits/aec_stratified.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.analysis.coverage import DatasetCoverageAnalyzer
from core.data.datasets.aec import AECDataset


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def analyze_vocabulary(vocab_path: Path) -> dict:
    """Analyze the vocabulary file."""
    print("\n" + "=" * 70)
    print("VOCABULARY ANALYSIS")
    print("=" * 70)
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    print(f"Vocabulary file: {vocab_path}")
    print(f"Total glosses defined: {len(vocab)}")
    
    # Check for gaps in indices
    indices = sorted(vocab.values())
    expected = list(range(len(vocab)))
    gaps = set(expected) - set(indices)
    duplicates = len(indices) - len(set(indices))
    
    if gaps:
        print(f"WARNING: Missing indices: {sorted(gaps)[:10]}{'...' if len(gaps) > 10 else ''}")
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate indices found")
    
    # Sample glosses
    sample_glosses = list(vocab.keys())[:10]
    print(f"\nSample glosses: {sample_glosses}")
    
    return {
        "path": str(vocab_path),
        "total_glosses": len(vocab),
        "index_range": (min(indices), max(indices)) if indices else (0, 0),
        "has_gaps": len(gaps) > 0,
        "has_duplicates": duplicates > 0
    }


def analyze_dataset(
    data_dir: Path,
    vocab_path: Path,
    split_file: Optional[Path] = None,
    features_dir: Optional[Path] = None
) -> dict:
    """Analyze dataset using DatasetCoverageAnalyzer."""
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    # Initialize analyzer with vocabulary
    analyzer = DatasetCoverageAnalyzer(vocab_path=vocab_path)
    analyzer.load_vocabulary()
    
    # Load dataset
    try:
        if split_file:
            # Analyze train split
            train_dataset = AECDataset(
                root_dir=data_dir,
                split_file=split_file,
                split="train",
                features_dir=features_dir
            )
            print(f"\nTraining split: {len(train_dataset)} samples")
            
            train_labels = [train_dataset[i]["label"].item() for i in range(len(train_dataset))]
            train_analysis = analyzer.analyze_from_labels(train_labels)
            
            print(f"  Classes covered: {train_analysis['classes_in_data']}/{train_analysis['vocab_size']}")
            print(f"  Coverage: {train_analysis['coverage_percentage']:.1f}%")
            
            # Reset and analyze validation
            val_dataset = AECDataset(
                root_dir=data_dir,
                split_file=split_file,
                split="val",
                features_dir=features_dir
            )
            print(f"\nValidation split: {len(val_dataset)} samples")
            
            val_labels = [val_dataset[i]["label"].item() for i in range(len(val_dataset))]
            analyzer.reset()
            val_analysis = analyzer.analyze_from_labels(val_labels)
            
            print(f"  Classes covered: {val_analysis['classes_in_data']}/{val_analysis['vocab_size']}")
            print(f"  Coverage: {val_analysis['coverage_percentage']:.1f}%")
            
            # Combined analysis
            all_labels = train_labels + val_labels
            analyzer.reset()
            full_analysis = analyzer.analyze_from_labels(all_labels)
            
        else:
            # Analyze full dataset
            dataset = AECDataset(
                root_dir=data_dir,
                features_dir=features_dir
            )
            print(f"\nFull dataset: {len(dataset)} samples")
            
            all_labels = [dataset[i]["label"].item() for i in range(len(dataset))]
            full_analysis = analyzer.analyze_from_labels(all_labels)
        
        # Print detailed report
        print("\n" + analyzer.format_report(show_distribution=True, top_n=15))
        
        return full_analysis
        
    except Exception as e:
        logging.error(f"Error analyzing dataset: {e}")
        raise


def export_reports(
    analyzer: DatasetCoverageAnalyzer,
    analysis: dict,
    output_dir: Path
) -> None:
    """Export analysis reports to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export JSON analysis
    json_path = output_dir / "coverage_analysis.json"
    analyzer.export_to_json(json_path)
    print(f"\nExported JSON report: {json_path}")
    
    # Export text report
    report_path = output_dir / "coverage_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(analyzer.format_report(show_distribution=True, top_n=50, show_missing=True))
    print(f"Exported text report: {report_path}")
    
    # Export underrepresented classes
    underrep = analyzer.get_underrepresented_classes(threshold=5)
    underrep_path = output_dir / "underrepresented_classes.json"
    with open(underrep_path, "w", encoding="utf-8") as f:
        json.dump(underrep, f, indent=2, ensure_ascii=False)
    print(f"Exported underrepresented classes: {underrep_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze AEC dataset coverage and distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python scripts/analyze_dataset.py --data-dir data/raw/lsp_aec
    
    # With split file
    python scripts/analyze_dataset.py --data-dir data/raw/lsp_aec \\
        --split-file data/splits/aec_stratified.json
    
    # Export reports
    python scripts/analyze_dataset.py --data-dir data/raw/lsp_aec \\
        --output-dir reports/coverage/
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/lsp_aec",
        help="Path to AEC dataset directory"
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        default=None,
        help="Path to vocabulary file (default: data-dir/dict.json)"
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to split file for train/val analysis"
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Path to precomputed features directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to export reports (optional)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Resolve paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    vocab_path = Path(args.vocab_file) if args.vocab_file else data_dir / "dict.json"
    if not vocab_path.exists():
        print(f"ERROR: Vocabulary file not found: {vocab_path}")
        sys.exit(1)
    
    split_file = Path(args.split_file) if args.split_file else None
    features_dir = Path(args.features_dir) if args.features_dir else None
    
    print("=" * 70)
    print("AEC DATASET ANALYSIS")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Vocabulary: {vocab_path}")
    if split_file:
        print(f"Split file: {split_file}")
    
    # Run analysis
    vocab_info = analyze_vocabulary(vocab_path)
    
    # Initialize analyzer for later export
    analyzer = DatasetCoverageAnalyzer(vocab_path=vocab_path)
    analyzer.load_vocabulary()
    
    analysis = analyze_dataset(
        data_dir=data_dir,
        vocab_path=vocab_path,
        split_file=split_file,
        features_dir=features_dir
    )
    
    # Re-run analysis on full dataset for export
    try:
        if split_file:
            train_ds = AECDataset(data_dir, split_file=split_file, split="train", features_dir=features_dir)
            val_ds = AECDataset(data_dir, split_file=split_file, split="val", features_dir=features_dir)
            all_labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
            all_labels += [val_ds[i]["label"].item() for i in range(len(val_ds))]
        else:
            ds = AECDataset(data_dir, features_dir=features_dir)
            all_labels = [ds[i]["label"].item() for i in range(len(ds))]
        
        analyzer.analyze_from_labels(all_labels)
    except Exception as e:
        logging.warning(f"Could not re-analyze for export: {e}")
    
    # Export if output directory specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        export_reports(analyzer, analysis, output_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


# Add Optional import at top
from typing import Optional

if __name__ == "__main__":
    main()
