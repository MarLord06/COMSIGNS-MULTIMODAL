"""
Tests for Bucket Analysis module.

Tests cover:
- Bucket classification logic
- Metrics computation per bucket
- Coverage metrics
- Tail analysis
- Export/import functionality
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from training.analysis.bucket_analysis import (
    Bucket,
    BucketMetrics,
    BucketAnalysisResult,
    BucketAnalyzer,
    run_bucket_analysis,
    BUCKET_THRESHOLDS
)


class TestBucketClassification:
    """Tests for bucket classification logic."""
    
    def test_classify_head_exactly_10(self):
        """10 samples should be HEAD."""
        analyzer = BucketAnalyzer()
        assert analyzer.classify_bucket(10) == Bucket.HEAD
    
    def test_classify_head_above_10(self):
        """More than 10 samples should be HEAD."""
        analyzer = BucketAnalyzer()
        assert analyzer.classify_bucket(50) == Bucket.HEAD
        assert analyzer.classify_bucket(100) == Bucket.HEAD
    
    def test_classify_mid_range(self):
        """3-9 samples should be MID."""
        analyzer = BucketAnalyzer()
        assert analyzer.classify_bucket(3) == Bucket.MID
        assert analyzer.classify_bucket(5) == Bucket.MID
        assert analyzer.classify_bucket(9) == Bucket.MID
    
    def test_classify_tail_range(self):
        """1-2 samples should be TAIL."""
        analyzer = BucketAnalyzer()
        assert analyzer.classify_bucket(1) == Bucket.TAIL
        assert analyzer.classify_bucket(2) == Bucket.TAIL
    
    def test_classify_custom_thresholds(self):
        """Custom thresholds should work."""
        analyzer = BucketAnalyzer(
            head_threshold=5,
            mid_range=(2, 4),
            tail_range=(1, 1)
        )
        assert analyzer.classify_bucket(5) == Bucket.HEAD
        assert analyzer.classify_bucket(4) == Bucket.MID
        assert analyzer.classify_bucket(2) == Bucket.MID
        assert analyzer.classify_bucket(1) == Bucket.TAIL


class TestBucketMetrics:
    """Tests for BucketMetrics dataclass."""
    
    def test_to_dict(self):
        """to_dict should convert all fields."""
        metrics = BucketMetrics(
            bucket=Bucket.HEAD,
            num_classes=10,
            num_samples=100,
            accuracy_at_1=0.5,
            accuracy_at_5=0.8,
            coverage_at_5=0.8,
            coverage_at_10=0.9,
            recall_at_5=0.8,
            class_ids=[0, 1, 2]
        )
        
        d = metrics.to_dict()
        
        assert d["bucket"] == "HEAD"
        assert d["num_classes"] == 10
        assert d["num_samples"] == 100
        assert d["accuracy_at_1"] == 0.5
        assert d["accuracy_at_5"] == 0.8
        assert d["class_ids"] == [0, 1, 2]


class TestBucketAnalysisResult:
    """Tests for BucketAnalysisResult dataclass."""
    
    def test_to_dict(self):
        """to_dict should convert complete result."""
        bucket_metrics = {
            Bucket.HEAD: BucketMetrics(
                bucket=Bucket.HEAD,
                num_classes=5,
                num_samples=50,
                accuracy_at_1=0.6,
                accuracy_at_5=0.85,
                coverage_at_5=0.85,
                coverage_at_10=0.92,
                recall_at_5=0.85,
                class_ids=[0, 1]
            )
        }
        
        result = BucketAnalysisResult(
            bucket_metrics=bucket_metrics,
            class_to_bucket={0: Bucket.HEAD, 1: Bucket.HEAD},
            class_support={0: 25, 1: 25},
            global_coverage_at_5=0.75,
            global_coverage_at_10=0.85,
            tail_analysis={"tail_class_count": 0}
        )
        
        d = result.to_dict()
        
        assert "HEAD" in d["bucket_metrics"]
        assert d["global_coverage_at_5"] == 0.75
        assert d["class_to_bucket"]["0"] == "HEAD"


class TestBucketAnalyzer:
    """Tests for BucketAnalyzer class."""
    
    @pytest.fixture
    def sample_metrics(self) -> dict:
        """Create sample metrics data."""
        return {
            "0": {"recall": 0.8, "precision": 0.75, "f1": 0.77, "support": 20},  # HEAD
            "1": {"recall": 0.7, "precision": 0.65, "f1": 0.67, "support": 15},  # HEAD
            "2": {"recall": 0.5, "precision": 0.45, "f1": 0.47, "support": 5},   # MID
            "3": {"recall": 0.4, "precision": 0.35, "f1": 0.37, "support": 3},   # MID
            "4": {"recall": 0.1, "precision": 0.08, "f1": 0.09, "support": 2},   # TAIL
            "5": {"recall": 0.05, "precision": 0.04, "f1": 0.04, "support": 1},  # TAIL
        }
    
    @pytest.fixture
    def metrics_file(self, sample_metrics, tmp_path) -> Path:
        """Create temporary metrics file."""
        path = tmp_path / "metrics_by_class.json"
        with open(path, "w") as f:
            json.dump(sample_metrics, f)
        return path
    
    def test_load_from_files(self, metrics_file):
        """Should load metrics from JSON file."""
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(metrics_file)
        
        assert len(analyzer._metrics_by_class) == 6
        assert analyzer._class_support[0] == 20
        assert analyzer._class_support[5] == 1
    
    def test_load_from_files_nested_format(self, tmp_path):
        """Should handle nested metrics_by_class format."""
        data = {
            "metrics_by_class": {
                "0": {"recall": 0.5, "support": 10},
                "1": {"recall": 0.3, "support": 2}
            }
        }
        
        path = tmp_path / "metrics.json"
        with open(path, "w") as f:
            json.dump(data, f)
        
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(path)
        
        assert len(analyzer._metrics_by_class) == 2
    
    def test_analyze_creates_result(self, metrics_file):
        """analyze() should create BucketAnalysisResult."""
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(metrics_file)
        
        result = analyzer.analyze()
        
        assert isinstance(result, BucketAnalysisResult)
        assert Bucket.HEAD in result.bucket_metrics
        assert Bucket.MID in result.bucket_metrics
        assert Bucket.TAIL in result.bucket_metrics
    
    def test_analyze_bucket_class_counts(self, metrics_file):
        """analyze() should correctly count classes per bucket."""
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(metrics_file)
        
        result = analyzer.analyze()
        
        # Based on sample_metrics: 2 HEAD, 2 MID, 2 TAIL
        assert result.bucket_metrics[Bucket.HEAD].num_classes == 2
        assert result.bucket_metrics[Bucket.MID].num_classes == 2
        assert result.bucket_metrics[Bucket.TAIL].num_classes == 2
    
    def test_analyze_bucket_sample_counts(self, metrics_file):
        """analyze() should correctly count samples per bucket."""
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(metrics_file)
        
        result = analyzer.analyze()
        
        # HEAD: 20 + 15 = 35
        assert result.bucket_metrics[Bucket.HEAD].num_samples == 35
        # MID: 5 + 3 = 8
        assert result.bucket_metrics[Bucket.MID].num_samples == 8
        # TAIL: 2 + 1 = 3
        assert result.bucket_metrics[Bucket.TAIL].num_samples == 3
    
    def test_analyze_tail_analysis(self, metrics_file):
        """analyze() should generate tail analysis."""
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(metrics_file)
        
        result = analyzer.analyze()
        tail = result.tail_analysis
        
        assert "tail_class_count" in tail
        assert "tail_class_percentage" in tail
        assert "tail_sample_count" in tail
        assert "tail_sample_percentage" in tail
        assert "comparison" in tail
        assert "diagnosis" in tail
    
    def test_analyze_class_to_bucket_mapping(self, metrics_file):
        """analyze() should create correct class-to-bucket mapping."""
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(metrics_file)
        
        result = analyzer.analyze()
        
        assert result.class_to_bucket[0] == Bucket.HEAD
        assert result.class_to_bucket[1] == Bucket.HEAD
        assert result.class_to_bucket[2] == Bucket.MID
        assert result.class_to_bucket[3] == Bucket.MID
        assert result.class_to_bucket[4] == Bucket.TAIL
        assert result.class_to_bucket[5] == Bucket.TAIL
    
    def test_export_results(self, metrics_file, tmp_path):
        """export_results() should create valid JSON."""
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(metrics_file)
        analyzer.analyze()
        
        output_path = tmp_path / "bucket_analysis.json"
        analyzer.export_results(output_path)
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert "bucket_metrics" in data
        assert "HEAD" in data["bucket_metrics"]
        assert "tail_analysis" in data
    
    def test_get_bucket_classes(self, metrics_file):
        """get_bucket_classes() should return correct class IDs."""
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(metrics_file)
        analyzer.analyze()
        
        head_classes = analyzer.get_bucket_classes(Bucket.HEAD)
        
        assert len(head_classes) == 2
        assert 0 in head_classes
        assert 1 in head_classes


class TestTailRecommendation:
    """Tests for tail recommendation logic."""
    
    def test_recommend_tail_to_other_for_noise(self, tmp_path):
        """Should recommend TAIL_TO_OTHER when tail is noise."""
        # All tail classes have very low accuracy
        metrics = {
            "0": {"recall": 0.8, "support": 20},  # HEAD
            "1": {"recall": 0.02, "support": 1},  # TAIL - noise
            "2": {"recall": 0.01, "support": 2},  # TAIL - noise
        }
        
        path = tmp_path / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f)
        
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(path)
        result = analyzer.analyze()
        
        # Low accuracy tail should get TAIL_TO_OTHER
        assert result.tail_analysis["diagnosis"]["tail_is_noise"] == True
    
    def test_recommend_keep_tail_when_learning(self, tmp_path):
        """Should recommend KEEP_TAIL when model is learning."""
        # Tail classes have reasonable accuracy
        metrics = {
            "0": {"recall": 0.8, "support": 20},   # HEAD
            "1": {"recall": 0.35, "support": 2},   # TAIL - learning
            "2": {"recall": 0.30, "support": 1},   # TAIL - learning
        }
        
        path = tmp_path / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f)
        
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(path)
        result = analyzer.analyze()
        
        assert result.tail_analysis["diagnosis"]["tail_is_learning"] == True


class TestCoverageMetrics:
    """Tests for coverage-related metrics."""
    
    def test_global_coverage_computed(self, tmp_path):
        """Global coverage should be computed."""
        metrics = {
            "0": {"recall": 0.5, "support": 10},
            "1": {"recall": 0.3, "support": 10},
        }
        
        path = tmp_path / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f)
        
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(path)
        result = analyzer.analyze()
        
        # Coverage should be some value (estimated from recall)
        assert result.global_coverage_at_5 >= 0
        assert result.global_coverage_at_10 >= 0


class TestRunBucketAnalysis:
    """Tests for run_bucket_analysis convenience function."""
    
    def test_run_bucket_analysis(self, tmp_path):
        """run_bucket_analysis should work end-to-end."""
        metrics = {
            "0": {"recall": 0.5, "support": 15},  # HEAD
            "1": {"recall": 0.3, "support": 5},   # MID
            "2": {"recall": 0.1, "support": 2},   # TAIL
        }
        
        metrics_path = tmp_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        
        output_path = tmp_path / "output.json"
        
        result = run_bucket_analysis(
            metrics_file=metrics_path,
            output_file=output_path,
            print_report=False
        )
        
        assert isinstance(result, BucketAnalysisResult)
        assert output_path.exists()


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_bucket(self, tmp_path):
        """Should handle buckets with no classes."""
        # Only HEAD classes, no MID or TAIL
        metrics = {
            "0": {"recall": 0.5, "support": 50},
            "1": {"recall": 0.6, "support": 40},
        }
        
        path = tmp_path / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f)
        
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(path)
        result = analyzer.analyze()
        
        # Empty buckets should have 0 classes
        assert result.bucket_metrics[Bucket.MID].num_classes == 0
        assert result.bucket_metrics[Bucket.TAIL].num_classes == 0
    
    def test_all_tail_classes(self, tmp_path):
        """Should handle dataset with only tail classes."""
        metrics = {
            "0": {"recall": 0.1, "support": 1},
            "1": {"recall": 0.2, "support": 2},
            "2": {"recall": 0.15, "support": 1},
        }
        
        path = tmp_path / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f)
        
        analyzer = BucketAnalyzer()
        analyzer.load_from_files(path)
        result = analyzer.analyze()
        
        assert result.bucket_metrics[Bucket.HEAD].num_classes == 0
        assert result.bucket_metrics[Bucket.MID].num_classes == 0
        assert result.bucket_metrics[Bucket.TAIL].num_classes == 3
    
    def test_analyze_without_load_raises(self):
        """analyze() without loading data should still work (empty)."""
        analyzer = BucketAnalyzer()
        result = analyzer.analyze()
        
        # Should have empty results
        assert result.bucket_metrics[Bucket.HEAD].num_classes == 0
    
    def test_export_without_analyze_raises(self, tmp_path):
        """export_results() without analyze() should raise."""
        analyzer = BucketAnalyzer()
        
        with pytest.raises(RuntimeError):
            analyzer.export_results(tmp_path / "output.json")
    
    def test_get_bucket_classes_without_analyze_raises(self):
        """get_bucket_classes() without analyze() should raise."""
        analyzer = BucketAnalyzer()
        
        with pytest.raises(RuntimeError):
            analyzer.get_bucket_classes(Bucket.HEAD)
