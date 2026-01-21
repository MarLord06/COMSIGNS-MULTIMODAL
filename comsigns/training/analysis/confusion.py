"""
Confusion Matrix utilities.

Provides functions to export confusion matrices in various formats
and analyze the most confused class pairs.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def export_confusion_matrix_csv(
    confusion_matrix: np.ndarray,
    output_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    include_totals: bool = True
) -> Path:
    """
    Export confusion matrix to CSV file.
    
    Args:
        confusion_matrix: Square confusion matrix [C x C]
        output_path: Path to output CSV file
        class_names: Optional class name labels
        include_totals: Whether to include row/column totals
    
    Returns:
        Path to created CSV file
    
    Example CSV format:
        ,class_0,class_1,...,Total
        class_0,10,2,...,12
        class_1,1,15,...,16
        ...
        Total,11,17,...,
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_classes = confusion_matrix.shape[0]
    
    # Generate class names if not provided
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    elif len(class_names) < n_classes:
        # Pad with generic names
        class_names = list(class_names) + [
            f"class_{i}" for i in range(len(class_names), n_classes)
        ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header row
        header = ["true\\predicted"] + class_names[:n_classes]
        if include_totals:
            header.append("Total")
        writer.writerow(header)
        
        # Data rows
        for i in range(n_classes):
            row = [class_names[i]] + [int(confusion_matrix[i, j]) for j in range(n_classes)]
            if include_totals:
                row.append(int(confusion_matrix[i].sum()))
            writer.writerow(row)
        
        # Total row
        if include_totals:
            total_row = ["Total"] + [int(confusion_matrix[:, j].sum()) for j in range(n_classes)]
            total_row.append(int(confusion_matrix.sum()))
            writer.writerow(total_row)
    
    logger.info(f"Confusion matrix exported to {output_path}")
    return output_path


def export_confusion_matrix_heatmap(
    confusion_matrix: np.ndarray,
    output_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    normalize: bool = False,
    show_values: bool = True,
    value_format: str = ".2f",
    subset_indices: Optional[List[int]] = None
) -> Optional[Path]:
    """
    Export confusion matrix as heatmap image.
    
    Args:
        confusion_matrix: Square confusion matrix [C x C]
        output_path: Path to output image (PNG, PDF, etc.)
        class_names: Optional class name labels
        title: Plot title
        figsize: Figure size (width, height) in inches
        cmap: Matplotlib colormap name
        normalize: Whether to normalize rows (true class)
        show_values: Whether to display values in cells
        value_format: Format string for values
        subset_indices: Only show these class indices (for large matrices)
    
    Returns:
        Path to created image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping heatmap export")
        return None
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle subset
    if subset_indices is not None:
        confusion_matrix = confusion_matrix[np.ix_(subset_indices, subset_indices)]
        if class_names:
            class_names = [class_names[i] for i in subset_indices if i < len(class_names)]
    
    n_classes = confusion_matrix.shape[0]
    
    # Generate class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    # Normalize if requested
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = np.divide(
            confusion_matrix.astype(float),
            row_sums,
            where=row_sums != 0
        )
        title = f"{title} (Normalized)"
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xlabel='Predicted label',
        ylabel='True label',
        title=title
    )
    
    # Handle tick labels for large matrices
    if n_classes <= 30:
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
    else:
        # Show every nth label
        step = max(1, n_classes // 30)
        ax.set_xticks(np.arange(0, n_classes, step))
        ax.set_yticks(np.arange(0, n_classes, step))
        ax.set_xticklabels([class_names[i] for i in range(0, n_classes, step)], rotation=45, ha='right')
        ax.set_yticklabels([class_names[i] for i in range(0, n_classes, step)])
    
    # Add text annotations
    if show_values and n_classes <= 50:
        thresh = confusion_matrix.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                value = confusion_matrix[i, j]
                if value > 0 or not normalize:
                    text = format(value, value_format) if normalize else str(int(value))
                    ax.text(
                        j, i, text,
                        ha="center", va="center",
                        color="white" if value > thresh else "black",
                        fontsize=8 if n_classes > 20 else 10
                    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix heatmap exported to {output_path}")
    return output_path


def get_most_confused_pairs(
    confusion_matrix: np.ndarray,
    k: int = 10,
    class_names: Optional[List[str]] = None,
    exclude_diagonal: bool = True
) -> List[Dict]:
    """
    Get the K most confused class pairs (off-diagonal entries).
    
    Args:
        confusion_matrix: Square confusion matrix [C x C]
        k: Number of pairs to return
        class_names: Optional class name labels
        exclude_diagonal: Whether to exclude correct predictions
    
    Returns:
        List of dicts with:
        - true_class: True class index
        - true_name: True class name
        - pred_class: Predicted class index
        - pred_name: Predicted class name
        - count: Number of confusions
        - confusion_rate: Proportion of true class confused as pred
    """
    n_classes = confusion_matrix.shape[0]
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    
    # Get all off-diagonal entries
    pairs = []
    for i in range(n_classes):
        row_sum = confusion_matrix[i].sum()
        for j in range(n_classes):
            if exclude_diagonal and i == j:
                continue
            
            count = int(confusion_matrix[i, j])
            if count > 0:
                pairs.append({
                    "true_class": i,
                    "true_name": class_names[i] if i < len(class_names) else f"class_{i}",
                    "pred_class": j,
                    "pred_name": class_names[j] if j < len(class_names) else f"class_{j}",
                    "count": count,
                    "confusion_rate": count / row_sum if row_sum > 0 else 0.0
                })
    
    # Sort by count descending
    pairs.sort(key=lambda x: x["count"], reverse=True)
    
    return pairs[:k]


def get_confusion_summary(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Get summary statistics from confusion matrix.
    
    Args:
        confusion_matrix: Square confusion matrix [C x C]
        class_names: Optional class name labels
    
    Returns:
        Dictionary with summary statistics:
        - total_samples: Total predictions
        - correct_predictions: Diagonal sum
        - accuracy: Overall accuracy
        - per_class_accuracy: Dict of class -> accuracy
        - most_accurate_class: Class with highest accuracy
        - least_accurate_class: Class with lowest accuracy (min support > 0)
    """
    n_classes = confusion_matrix.shape[0]
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    
    total = confusion_matrix.sum()
    correct = np.trace(confusion_matrix)
    accuracy = correct / total if total > 0 else 0.0
    
    # Per-class accuracy (recall)
    per_class_acc = {}
    for i in range(n_classes):
        row_sum = confusion_matrix[i].sum()
        cls_acc = confusion_matrix[i, i] / row_sum if row_sum > 0 else 0.0
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        per_class_acc[name] = {
            "accuracy": float(cls_acc),
            "support": int(row_sum),
            "correct": int(confusion_matrix[i, i])
        }
    
    # Find best/worst (with support > 0)
    valid_classes = [(name, info) for name, info in per_class_acc.items() if info["support"] > 0]
    
    if valid_classes:
        most_accurate = max(valid_classes, key=lambda x: x[1]["accuracy"])
        least_accurate = min(valid_classes, key=lambda x: x[1]["accuracy"])
    else:
        most_accurate = least_accurate = (None, {"accuracy": 0.0, "support": 0})
    
    return {
        "total_samples": int(total),
        "correct_predictions": int(correct),
        "accuracy": float(accuracy),
        "per_class_accuracy": per_class_acc,
        "most_accurate_class": {
            "name": most_accurate[0],
            **most_accurate[1]
        },
        "least_accurate_class": {
            "name": least_accurate[0],
            **least_accurate[1]
        },
        "num_classes_with_support": len(valid_classes)
    }


def format_confusion_report(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    top_confused: int = 10
) -> str:
    """
    Format confusion analysis as human-readable report.
    
    Args:
        confusion_matrix: Square confusion matrix
        class_names: Optional class name labels
        top_confused: Number of confused pairs to show
    
    Returns:
        Formatted report string
    """
    summary = get_confusion_summary(confusion_matrix, class_names)
    confused_pairs = get_most_confused_pairs(confusion_matrix, top_confused, class_names)
    
    lines = []
    lines.append("=" * 70)
    lines.append("CONFUSION MATRIX ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total predictions:     {summary['total_samples']:,}")
    lines.append(f"Correct predictions:   {summary['correct_predictions']:,}")
    lines.append(f"Overall accuracy:      {summary['accuracy']:.2%}")
    lines.append(f"Classes with support:  {summary['num_classes_with_support']}")
    lines.append("")
    
    most = summary["most_accurate_class"]
    least = summary["least_accurate_class"]
    lines.append(f"Most accurate class:   {most['name']} ({most['accuracy']:.2%}, n={most['support']})")
    lines.append(f"Least accurate class:  {least['name']} ({least['accuracy']:.2%}, n={least['support']})")
    lines.append("")
    
    if confused_pairs:
        lines.append(f"TOP {top_confused} CONFUSED PAIRS")
        lines.append("-" * 70)
        lines.append(f"{'True Class':<25} -> {'Pred Class':<25} Count   Rate")
        lines.append("-" * 70)
        
        for pair in confused_pairs:
            true_name = pair["true_name"][:23] if len(pair["true_name"]) > 23 else pair["true_name"]
            pred_name = pair["pred_name"][:23] if len(pair["pred_name"]) > 23 else pair["pred_name"]
            lines.append(
                f"{true_name:<25} -> {pred_name:<25} "
                f"{pair['count']:>5}  {pair['confusion_rate']:>5.1%}"
            )
        
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)
