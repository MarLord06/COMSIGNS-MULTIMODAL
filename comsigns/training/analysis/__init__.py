"""
Analysis module for dataset coverage, bucket analysis, and confusion matrix utilities.

This module provides tools for:
- Dataset coverage analysis (dict.json vs actual data)
- Bucket analysis (HEAD/MID/TAIL classification)
- Coverage metrics (Coverage@K, Vocabulary Coverage)
- Confusion matrix generation and export
- Class distribution visualization
"""

from .coverage import DatasetCoverageAnalyzer
from .confusion import (
    export_confusion_matrix_csv,
    export_confusion_matrix_heatmap,
    get_most_confused_pairs
)
from .bucket_analysis import (
    Bucket,
    BucketMetrics,
    BucketAnalysisResult,
    BucketAnalyzer,
    run_bucket_analysis
)
from .coverage_metrics import (
    CoverageResult,
    CoverageAnalyzer,
    compute_coverage_metrics
)
from .remapped_metrics import (
    BucketMetricsResult,
    PredictionDiagnostics,
    BucketConfusionMatrix,
    OtherDiagnostics,
    RemappedMetricsTracker,
    create_comparison_report
)
from .learned_words import (
    LearnedWordCriteria,
    ClassMetrics,
    BucketLearnedStats,
    LearnedWordsReport,
    LearnedWordsAnalyzer,
    compute_class_metrics_from_predictions,
    analyze_learned_words
)

__all__ = [
    # Coverage (dict.json analysis)
    "DatasetCoverageAnalyzer",
    # Confusion matrix
    "export_confusion_matrix_csv",
    "export_confusion_matrix_heatmap",
    "get_most_confused_pairs",
    # Bucket analysis
    "Bucket",
    "BucketMetrics",
    "BucketAnalysisResult",
    "BucketAnalyzer",
    "run_bucket_analysis",
    # Coverage metrics
    "CoverageResult",
    "CoverageAnalyzer",
    "compute_coverage_metrics",
    # Remapped metrics
    "BucketMetricsResult",
    "PredictionDiagnostics",
    "BucketConfusionMatrix",
    "OtherDiagnostics",
    "RemappedMetricsTracker",
    "create_comparison_report",
    # Learned words analysis
    "LearnedWordCriteria",
    "ClassMetrics",
    "BucketLearnedStats",
    "LearnedWordsReport",
    "LearnedWordsAnalyzer",
    "compute_class_metrics_from_predictions",
    "analyze_learned_words"
]
