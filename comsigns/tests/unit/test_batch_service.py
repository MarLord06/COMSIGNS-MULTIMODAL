"""
Unit tests for the batch inference service.

Tests batch processing, sequence construction, and error handling.
"""

import pytest
from unittest.mock import Mock, MagicMock
from backend.services.batch_service import (
    BatchPrediction,
    BatchFileResult,
    SequenceWord,
    SemanticSequence,
    BatchInferenceResult,
    BatchInferenceService
)


# ============================================================
# Test Data Classes
# ============================================================

class TestBatchPrediction:
    """Tests for BatchPrediction dataclass."""
    
    def test_creation(self):
        pred = BatchPrediction(
            class_id=74,
            gloss="MID_169",
            bucket="MID",
            confidence=0.3078,
            accepted=True,
            reason="confidence_passed"
        )
        assert pred.class_id == 74
        assert pred.gloss == "MID_169"
        assert pred.bucket == "MID"
        assert pred.accepted is True
    
    def test_to_dict(self):
        pred = BatchPrediction(
            class_id=74,
            gloss="MID_169",
            bucket="MID",
            confidence=0.3078,
            accepted=True,
            reason="confidence_passed"
        )
        d = pred.to_dict()
        
        assert d["class_id"] == 74
        assert d["gloss"] == "MID_169"
        assert d["bucket"] == "MID"
        assert d["confidence"] == 0.3078
        assert d["accepted"] is True
        assert d["reason"] == "confidence_passed"
    
    def test_confidence_rounding(self):
        pred = BatchPrediction(
            class_id=1,
            gloss="test",
            bucket="HEAD",
            confidence=0.30781234567,
            accepted=True,
            reason="test"
        )
        d = pred.to_dict()
        assert d["confidence"] == 0.3078  # Rounded to 4 decimal places


class TestBatchFileResult:
    """Tests for BatchFileResult dataclass."""
    
    def test_success_result(self):
        pred = BatchPrediction(
            class_id=74,
            gloss="MID_169",
            bucket="MID",
            confidence=0.31,
            accepted=True,
            reason="confidence_passed"
        )
        result = BatchFileResult(
            file_name="sample_01.pkl",
            prediction=pred
        )
        
        assert result.file_name == "sample_01.pkl"
        assert result.prediction is not None
        assert result.error is None
    
    def test_error_result(self):
        result = BatchFileResult(
            file_name="bad_file.pkl",
            error="Invalid file format"
        )
        
        assert result.file_name == "bad_file.pkl"
        assert result.prediction is None
        assert result.error == "Invalid file format"
    
    def test_to_dict_success(self):
        pred = BatchPrediction(
            class_id=74,
            gloss="MID_169",
            bucket="MID",
            confidence=0.31,
            accepted=True,
            reason="confidence_passed"
        )
        result = BatchFileResult(
            file_name="sample_01.pkl",
            prediction=pred
        )
        d = result.to_dict()
        
        assert d["file_name"] == "sample_01.pkl"
        assert "prediction" in d
        assert "error" not in d
    
    def test_to_dict_error(self):
        result = BatchFileResult(
            file_name="bad_file.pkl",
            error="Invalid file format"
        )
        d = result.to_dict()
        
        assert d["file_name"] == "bad_file.pkl"
        assert d["error"] == "Invalid file format"
        assert "prediction" not in d


class TestSemanticSequence:
    """Tests for SemanticSequence dataclass."""
    
    def test_empty_sequence(self):
        seq = SemanticSequence()
        assert seq.length == 0
        assert seq.accepted == []
    
    def test_append_word(self):
        seq = SemanticSequence()
        seq.append("HEAD_259", 0.62)
        
        assert seq.length == 1
        assert seq.accepted[0].gloss == "HEAD_259"
        assert seq.accepted[0].confidence == 0.62
    
    def test_append_multiple_words(self):
        seq = SemanticSequence()
        seq.append("HEAD_259", 0.62)
        seq.append("MID_169", 0.31)
        seq.append("HEAD_402", 0.55)
        
        assert seq.length == 3
        assert seq.accepted[0].gloss == "HEAD_259"
        assert seq.accepted[1].gloss == "MID_169"
        assert seq.accepted[2].gloss == "HEAD_402"
    
    def test_to_dict(self):
        seq = SemanticSequence()
        seq.append("HEAD_259", 0.62)
        seq.append("MID_169", 0.31)
        
        d = seq.to_dict()
        
        assert d["length"] == 2
        assert len(d["accepted"]) == 2
        assert d["accepted"][0]["gloss"] == "HEAD_259"
        assert d["accepted"][1]["gloss"] == "MID_169"


class TestBatchInferenceResult:
    """Tests for BatchInferenceResult dataclass."""
    
    def test_empty_result(self):
        result = BatchInferenceResult()
        assert len(result.results) == 0
        assert result.sequence.length == 0
    
    def test_to_dict_with_mixed_results(self):
        result = BatchInferenceResult()
        
        # Add success result
        pred1 = BatchPrediction(
            class_id=74,
            gloss="MID_169",
            bucket="MID",
            confidence=0.31,
            accepted=True,
            reason="confidence_passed"
        )
        result.results.append(BatchFileResult("file1.pkl", prediction=pred1))
        
        # Add error result
        result.results.append(BatchFileResult("file2.pkl", error="Invalid format"))
        
        # Add sequence
        result.sequence.append("MID_169", 0.31)
        
        d = result.to_dict()
        
        assert len(d["results"]) == 1  # Only success results
        assert len(d["errors"]) == 1  # Only error results
        assert d["sequence"]["length"] == 1


# ============================================================
# Test Batch Service
# ============================================================

class TestBatchInferenceService:
    """Tests for BatchInferenceService."""
    
    @pytest.fixture
    def mock_inference_service(self):
        """Create a mock inference service."""
        service = Mock()
        
        # Mock inference result
        mock_result = Mock()
        mock_result.to_dict.return_value = {
            "top1": {
                "new_class_id": 74,
                "gloss": "MID_169",
                "bucket": "MID",
                "confidence": 0.3078,
                "is_other": False
            },
            "topk": [
                {"new_class_id": 74, "confidence": 0.3078},
                {"new_class_id": 50, "confidence": 0.15}
            ]
        }
        service.infer_from_bytes.return_value = mock_result
        
        return service
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock decision evaluator."""
        evaluator = Mock()
        
        evaluator.process_from_inference_result.return_value = {
            "prediction": {
                "gloss": "MID_169",
                "accepted": True,
                "confidence": 0.3078,
                "bucket": "MID",
                "reason": "confidence_passed",
                "rule_applied": "accepted"
            },
            "sequence": {
                "accepted": [],
                "length": 0
            }
        }
        
        evaluator.get_sequence_state.return_value = {
            "accepted": [],
            "length": 0,
            "glosses": []
        }
        
        return evaluator
    
    def test_process_single_file_success(self, mock_inference_service, mock_evaluator):
        batch_service = BatchInferenceService(mock_inference_service, mock_evaluator)
        
        result = batch_service.process_single_file(
            file_name="sample_01.pkl",
            file_contents=b"dummy_content",
            topk=5
        )
        
        assert result.file_name == "sample_01.pkl"
        assert result.error is None
        assert result.prediction is not None
        assert result.prediction.class_id == 74
        assert result.prediction.gloss == "MID_169"
        assert result.prediction.accepted is True
    
    def test_process_single_file_error(self, mock_inference_service, mock_evaluator):
        # Make inference fail
        mock_inference_service.infer_from_bytes.side_effect = ValueError("Invalid file")
        
        batch_service = BatchInferenceService(mock_inference_service, mock_evaluator)
        
        result = batch_service.process_single_file(
            file_name="bad_file.pkl",
            file_contents=b"invalid_content",
            topk=5
        )
        
        assert result.file_name == "bad_file.pkl"
        assert result.error is not None
        assert "Invalid file" in result.error
        assert result.prediction is None
    
    def test_process_batch_all_accepted(self, mock_inference_service, mock_evaluator):
        batch_service = BatchInferenceService(mock_inference_service, mock_evaluator)
        
        files = [
            ("file1.pkl", b"content1"),
            ("file2.pkl", b"content2"),
            ("file3.pkl", b"content3")
        ]
        
        result = batch_service.process_batch(files, topk=5)
        
        assert len(result.results) == 3
        assert result.sequence.length == 3  # All accepted
        
        for i, file_result in enumerate(result.results):
            assert file_result.file_name == files[i][0]
            assert file_result.prediction is not None
            assert file_result.prediction.accepted is True
    
    def test_process_batch_mixed_acceptance(self, mock_inference_service, mock_evaluator):
        """Test batch with some accepted and some rejected."""
        
        # Configure evaluator to alternate acceptance
        call_count = [0]
        def alternate_acceptance(*args, **kwargs):
            call_count[0] += 1
            accepted = call_count[0] % 2 == 1  # Odd = accepted, Even = rejected
            return {
                "prediction": {
                    "gloss": f"GLOSS_{call_count[0]}",
                    "accepted": accepted,
                    "confidence": 0.5,
                    "bucket": "HEAD",
                    "reason": "confidence_passed" if accepted else "low_confidence",
                    "rule_applied": "accepted" if accepted else "low_confidence"
                },
                "sequence": {"accepted": [], "length": 0}
            }
        
        mock_evaluator.process_from_inference_result.side_effect = alternate_acceptance
        
        batch_service = BatchInferenceService(mock_inference_service, mock_evaluator)
        
        files = [
            ("file1.pkl", b"content1"),
            ("file2.pkl", b"content2"),
            ("file3.pkl", b"content3"),
            ("file4.pkl", b"content4")
        ]
        
        result = batch_service.process_batch(files, topk=5)
        
        assert len(result.results) == 4
        assert result.sequence.length == 2  # Only odd files accepted
        
        assert result.results[0].prediction.accepted is True
        assert result.results[1].prediction.accepted is False
        assert result.results[2].prediction.accepted is True
        assert result.results[3].prediction.accepted is False
    
    def test_process_batch_preserves_order(self, mock_inference_service, mock_evaluator):
        """Test that sequence order matches file order."""
        
        call_count = [0]
        def mock_inference(*args, **kwargs):
            call_count[0] += 1
            result = Mock()
            result.to_dict.return_value = {
                "top1": {
                    "new_class_id": call_count[0],
                    "gloss": f"GLOSS_{call_count[0]}",
                    "bucket": "HEAD",
                    "confidence": 0.6,
                    "is_other": False
                },
                "topk": [{"new_class_id": call_count[0], "confidence": 0.6}]
            }
            return result
        
        mock_inference_service.infer_from_bytes.side_effect = mock_inference
        
        def mock_decision(result):
            gloss = result["top1"]["gloss"]
            return {
                "prediction": {
                    "gloss": gloss,
                    "accepted": True,
                    "confidence": 0.6,
                    "bucket": "HEAD",
                    "reason": "confidence_passed",
                    "rule_applied": "accepted"
                },
                "sequence": {"accepted": [], "length": 0}
            }
        
        mock_evaluator.process_from_inference_result.side_effect = mock_decision
        
        batch_service = BatchInferenceService(mock_inference_service, mock_evaluator)
        
        files = [
            ("first.pkl", b"1"),
            ("second.pkl", b"2"),
            ("third.pkl", b"3")
        ]
        
        result = batch_service.process_batch(files, topk=5)
        
        # Check order is preserved
        assert result.sequence.accepted[0].gloss == "GLOSS_1"
        assert result.sequence.accepted[1].gloss == "GLOSS_2"
        assert result.sequence.accepted[2].gloss == "GLOSS_3"
    
    def test_process_batch_handles_errors(self, mock_inference_service, mock_evaluator):
        """Test that errors don't cancel other files."""
        
        call_count = [0]
        def sometimes_fail(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Second file fails
                raise ValueError("Simulated error")
            
            result = Mock()
            result.to_dict.return_value = {
                "top1": {
                    "new_class_id": call_count[0],
                    "gloss": f"GLOSS_{call_count[0]}",
                    "bucket": "HEAD",
                    "confidence": 0.6,
                    "is_other": False
                },
                "topk": []
            }
            return result
        
        mock_inference_service.infer_from_bytes.side_effect = sometimes_fail
        
        batch_service = BatchInferenceService(mock_inference_service, mock_evaluator)
        
        files = [
            ("file1.pkl", b"content1"),
            ("file2.pkl", b"content2"),  # This will fail
            ("file3.pkl", b"content3")
        ]
        
        result = batch_service.process_batch(files, topk=5)
        
        # All files processed
        assert len(result.results) == 3
        
        # First and third succeeded
        assert result.results[0].prediction is not None
        assert result.results[2].prediction is not None
        
        # Second failed
        assert result.results[1].error is not None
        assert "Simulated error" in result.results[1].error
        
        # Sequence only has accepted files
        assert result.sequence.length == 2
    
    def test_reset_sequence_state(self, mock_inference_service, mock_evaluator):
        batch_service = BatchInferenceService(mock_inference_service, mock_evaluator)
        
        batch_service.reset_sequence_state()
        
        mock_evaluator.reset_sequence.assert_called_once()
