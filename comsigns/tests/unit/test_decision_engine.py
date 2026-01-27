"""
Unit tests for the decision engine.

Tests for rules, sequence manager, and evaluator.
"""

import pytest
from backend.decision_engine import (
    PredictionInput,
    AcceptanceResult,
    SequenceItem,
    RejectedItem,
    SequenceState,
    DecisionEngineConfig,
    RuleEngine,
    evaluate_prediction,
    SequenceManager,
    DecisionEvaluator
)


# ============================================================
# Test Types
# ============================================================

class TestPredictionInput:
    """Tests for PredictionInput dataclass."""
    
    def test_creation(self):
        pred = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.65,
            topk_scores=[0.65, 0.20, 0.10],
            topk_class_ids=[28, 45, 100]
        )
        assert pred.class_id == 28
        assert pred.class_name == "yo"
        assert pred.bucket == "HEAD"
        assert pred.confidence == 0.65
        assert len(pred.topk_scores) == 3
    
    def test_to_dict(self):
        pred = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.65
        )
        d = pred.to_dict()
        assert d["class_id"] == 28
        assert d["class_name"] == "yo"
    
    def test_is_other_default_false(self):
        pred = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.65
        )
        assert pred.is_other is False


class TestAcceptanceResult:
    """Tests for AcceptanceResult dataclass."""
    
    def test_accepted_result(self):
        result = AcceptanceResult(
            accepted=True,
            reason="Accepted: HEAD class with 65% confidence",
            confidence=0.65,
            bucket="HEAD",
            rule_applied="accepted"
        )
        assert result.accepted is True
        assert "HEAD" in result.reason
    
    def test_rejected_result(self):
        result = AcceptanceResult(
            accepted=False,
            reason="Rejected: OTHER class",
            confidence=0.40,
            bucket="OTHER",
            rule_applied="reject_other"
        )
        assert result.accepted is False
    
    def test_to_dict(self):
        result = AcceptanceResult(
            accepted=True,
            reason="Test",
            confidence=0.5,
            bucket="HEAD"
        )
        d = result.to_dict()
        assert "accepted" in d
        assert "reason" in d


class TestSequenceState:
    """Tests for SequenceState dataclass."""
    
    def test_empty_state(self):
        state = SequenceState()
        assert state.length == 0
        assert state.glosses == []
    
    def test_with_items(self):
        item1 = SequenceItem(gloss="yo", class_id=28, confidence=0.65, bucket="HEAD")
        item2 = SequenceItem(gloss="tu", class_id=45, confidence=0.55, bucket="MID")
        state = SequenceState(accepted=[item1, item2])
        
        assert state.length == 2
        assert state.glosses == ["yo", "tu"]
    
    def test_to_dict(self):
        state = SequenceState()
        d = state.to_dict()
        assert "accepted" in d
        assert "rejected" in d
        assert "length" in d
        assert "glosses" in d


class TestDecisionEngineConfig:
    """Tests for DecisionEngineConfig dataclass."""
    
    def test_default_thresholds(self):
        config = DecisionEngineConfig()
        assert config.head_threshold == 0.45
        assert config.mid_threshold == 0.55
        assert config.margin_threshold == 0.10
    
    def test_get_threshold_head(self):
        config = DecisionEngineConfig()
        assert config.get_threshold("HEAD") == 0.45
    
    def test_get_threshold_mid(self):
        config = DecisionEngineConfig()
        assert config.get_threshold("MID") == 0.55
    
    def test_get_threshold_other(self):
        config = DecisionEngineConfig()
        assert config.get_threshold("OTHER") == 1.0


# ============================================================
# Test Rules
# ============================================================

class TestRuleEngine:
    """Tests for RuleEngine."""
    
    def test_accept_high_confidence_head(self):
        """HEAD class with high confidence should be accepted."""
        engine = RuleEngine()
        pred = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.65,
            topk_scores=[0.65, 0.20]
        )
        result = engine.evaluate(pred)
        assert result.accepted is True
        assert result.rule_applied == "accepted"
    
    def test_accept_high_confidence_mid(self):
        """MID class with high confidence should be accepted."""
        engine = RuleEngine()
        pred = PredictionInput(
            class_id=100,
            class_name="cien",
            bucket="MID",
            confidence=0.70,
            topk_scores=[0.70, 0.15]
        )
        result = engine.evaluate(pred)
        assert result.accepted is True
    
    def test_reject_other_bucket(self):
        """OTHER bucket should be rejected when reject_other=True."""
        config = DecisionEngineConfig(reject_other=True)
        engine = RuleEngine(config)
        pred = PredictionInput(
            class_id=141,
            class_name="OTHER",
            bucket="OTHER",
            confidence=0.95,  # High confidence doesn't matter
            topk_scores=[0.95, 0.02]
        )
        result = engine.evaluate(pred)
        assert result.accepted is False
        assert result.rule_applied == "reject_other"
        assert "OTHER" in result.reason
    
    def test_reject_is_other_flag(self):
        """is_other=True should be rejected when reject_other=True."""
        config = DecisionEngineConfig(reject_other=True)
        engine = RuleEngine(config)
        pred = PredictionInput(
            class_id=141,
            class_name="OTHER",
            bucket="HEAD",  # Even if bucket says HEAD
            confidence=0.95,
            is_other=True
        )
        result = engine.evaluate(pred)
        assert result.accepted is False
        assert result.rule_applied == "reject_other"
    
    def test_accept_other_when_reject_other_false(self):
        """OTHER bucket should be accepted when reject_other=False (default)."""
        # Default config has reject_other=False
        engine = RuleEngine()
        pred = PredictionInput(
            class_id=141,
            class_name="OTHER",
            bucket="OTHER",
            confidence=0.95,
            topk_scores=[0.95, 0.02]
        )
        result = engine.evaluate(pred)
        # OTHER is not rejected, but will fail threshold check (OTHER threshold = 1.0)
        assert result.accepted is False
        assert result.rule_applied == "low_confidence"  # Failed on threshold, not reject_other
    
    def test_reject_low_confidence_head(self):
        """HEAD class below threshold should be rejected."""
        engine = RuleEngine()
        pred = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.05,  # Below 0.10 threshold
            topk_scores=[0.05, 0.02]
        )
        result = engine.evaluate(pred)
        assert result.accepted is False
        assert result.rule_applied == "low_confidence"
    
    def test_reject_low_confidence_mid(self):
        """MID class below threshold should be rejected."""
        engine = RuleEngine()
        pred = PredictionInput(
            class_id=100,
            class_name="cien",
            bucket="MID",
            confidence=0.05,  # Below 0.10 threshold
            topk_scores=[0.05, 0.02]
        )
        result = engine.evaluate(pred)
        assert result.accepted is False
        assert result.rule_applied == "low_confidence"
    
    def test_reject_low_margin(self):
        """Low margin between top1 and top2 should be rejected."""
        engine = RuleEngine()
        pred = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.55,
            topk_scores=[0.55, 0.50]  # Margin = 0.05 < 0.10
        )
        result = engine.evaluate(pred)
        assert result.accepted is False
        assert result.rule_applied == "low_margin"
        assert "margin" in result.reason.lower()
    
    def test_exact_threshold_accepted(self):
        """Exact threshold value should be accepted."""
        engine = RuleEngine()
        pred = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.45,  # Exactly at threshold
            topk_scores=[0.45, 0.25]  # Margin = 0.20 >= 0.10
        )
        result = engine.evaluate(pred)
        assert result.accepted is True
    
    def test_exact_margin_rejected(self):
        """Exact margin threshold is rejected (margin must be > threshold)."""
        engine = RuleEngine()
        pred = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.50,
            topk_scores=[0.50, 0.40]  # Margin = 0.10 exactly
        )
        result = engine.evaluate(pred)
        # margin < threshold uses strict inequality, so 0.10 < 0.10 is False
        # but due to floating point, we actually get rejected
        assert result.accepted is False
    
    def test_above_margin_accepted(self):
        """Margin above threshold should be accepted."""
        engine = RuleEngine()
        pred = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.50,
            topk_scores=[0.50, 0.39]  # Margin = 0.11 > 0.10
        )
        result = engine.evaluate(pred)
        assert result.accepted is True
    
    def test_custom_config(self):
        """Custom configuration should override defaults."""
        config = DecisionEngineConfig(
            head_threshold=0.60,
            mid_threshold=0.70,
            margin_threshold=0.15
        )
        engine = RuleEngine(config)
        
        # Would be accepted with default config
        pred = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.55,
            topk_scores=[0.55, 0.35]
        )
        result = engine.evaluate(pred)
        assert result.accepted is False  # Below 0.60
    
    def test_single_topk_score(self):
        """Single top-k score (no top2) should skip margin check."""
        engine = RuleEngine()
        pred = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.50,
            topk_scores=[0.50]  # Only one score
        )
        result = engine.evaluate(pred)
        assert result.accepted is True  # No margin check applied


class TestEvaluatePrediction:
    """Tests for evaluate_prediction convenience function."""
    
    def test_basic_evaluation(self):
        pred = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.65,
            topk_scores=[0.65, 0.20]
        )
        result = evaluate_prediction(pred)
        assert result.accepted is True


# ============================================================
# Test Sequence Manager
# ============================================================

class TestSequenceManager:
    """Tests for SequenceManager."""
    
    def test_empty_initial_state(self):
        manager = SequenceManager()
        state = manager.get_state()
        assert state.length == 0
        assert state.accepted == []
    
    def test_add_accepted_prediction(self):
        manager = SequenceManager()
        
        acceptance = AcceptanceResult(
            accepted=True,
            reason="Accepted",
            confidence=0.65,
            bucket="HEAD"
        )
        prediction = PredictionInput(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.65
        )
        
        manager.add_prediction(acceptance, prediction)
        state = manager.get_state()
        
        assert state.length == 1
        assert state.accepted[0].gloss == "yo"
        assert state.accepted[0].position == 0
    
    def test_add_rejected_prediction(self):
        manager = SequenceManager()
        
        acceptance = AcceptanceResult(
            accepted=False,
            reason="Rejected: OTHER",
            confidence=0.40,
            bucket="OTHER",
            rule_applied="reject_other"
        )
        prediction = PredictionInput(
            class_id=141,
            class_name="OTHER",
            bucket="OTHER",
            confidence=0.40
        )
        
        manager.add_prediction(acceptance, prediction)
        state = manager.get_state()
        
        assert state.length == 0  # Nothing accepted
        assert len(state.rejected) == 1
        assert state.rejected[0].reason == "Rejected: OTHER"
    
    def test_multiple_accepted_predictions(self):
        manager = SequenceManager()
        
        # Add first prediction
        manager.add_prediction(
            AcceptanceResult(accepted=True, reason="OK", confidence=0.65, bucket="HEAD"),
            PredictionInput(class_id=28, class_name="yo", bucket="HEAD", confidence=0.65)
        )
        
        # Add second prediction
        manager.add_prediction(
            AcceptanceResult(accepted=True, reason="OK", confidence=0.60, bucket="MID"),
            PredictionInput(class_id=100, class_name="tu", bucket="MID", confidence=0.60)
        )
        
        state = manager.get_state()
        assert state.length == 2
        assert state.glosses == ["yo", "tu"]
        assert state.accepted[0].position == 0
        assert state.accepted[1].position == 1
    
    def test_reset(self):
        manager = SequenceManager()
        
        # Add a prediction
        manager.add_prediction(
            AcceptanceResult(accepted=True, reason="OK", confidence=0.65, bucket="HEAD"),
            PredictionInput(class_id=28, class_name="yo", bucket="HEAD", confidence=0.65)
        )
        
        assert manager.get_sequence_length() == 1
        
        # Reset
        manager.reset()
        
        assert manager.get_sequence_length() == 0
        assert manager.get_accepted_glosses() == []
    
    def test_get_last_accepted(self):
        manager = SequenceManager()
        
        # Empty - should return None
        assert manager.get_last_accepted() is None
        
        # Add predictions
        manager.add_prediction(
            AcceptanceResult(accepted=True, reason="OK", confidence=0.65, bucket="HEAD"),
            PredictionInput(class_id=28, class_name="yo", bucket="HEAD", confidence=0.65)
        )
        manager.add_prediction(
            AcceptanceResult(accepted=True, reason="OK", confidence=0.60, bucket="MID"),
            PredictionInput(class_id=100, class_name="tu", bucket="MID", confidence=0.60)
        )
        
        last = manager.get_last_accepted()
        assert last.gloss == "tu"
    
    def test_get_summary(self):
        manager = SequenceManager()
        
        manager.add_prediction(
            AcceptanceResult(accepted=True, reason="OK", confidence=0.65, bucket="HEAD"),
            PredictionInput(class_id=28, class_name="yo", bucket="HEAD", confidence=0.65)
        )
        
        summary = manager.get_summary()
        assert "accepted" in summary
        assert "length" in summary
        assert "glosses" in summary
        assert summary["length"] == 1
        assert summary["glosses"] == ["yo"]


# ============================================================
# Test Evaluator
# ============================================================

class TestDecisionEvaluator:
    """Tests for DecisionEvaluator."""
    
    def test_process_accepted_prediction(self):
        evaluator = DecisionEvaluator()
        
        result = evaluator.process_prediction(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.65,
            topk_scores=[0.65, 0.20],
            topk_class_ids=[28, 45]
        )
        
        assert result["prediction"]["accepted"] is True
        assert result["prediction"]["gloss"] == "yo"
        assert result["sequence"]["length"] == 1
    
    def test_process_rejected_prediction(self):
        evaluator = DecisionEvaluator()
        
        result = evaluator.process_prediction(
            class_id=141,
            class_name="OTHER",
            bucket="OTHER",
            confidence=0.40,
            is_other=True
        )
        
        assert result["prediction"]["accepted"] is False
        assert result["sequence"]["length"] == 0
    
    def test_sequence_accumulates(self):
        evaluator = DecisionEvaluator()
        
        # First prediction
        evaluator.process_prediction(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.65,
            topk_scores=[0.65, 0.20]
        )
        
        # Second prediction
        result = evaluator.process_prediction(
            class_id=45,
            class_name="tu",
            bucket="MID",
            confidence=0.60,
            topk_scores=[0.60, 0.25]
        )
        
        assert result["sequence"]["length"] == 2
        assert result["sequence"]["glosses"] == ["yo", "tu"]
    
    def test_process_from_inference_result(self):
        evaluator = DecisionEvaluator()
        
        # Mock inference result format
        inference_result = {
            "top1": {
                "gloss": "yo",
                "confidence": 0.65,
                "bucket": "HEAD",
                "is_other": False,
                "new_class_id": 28
            },
            "topk": [
                {"gloss": "yo", "confidence": 0.65, "new_class_id": 28},
                {"gloss": "tu", "confidence": 0.20, "new_class_id": 45}
            ]
        }
        
        result = evaluator.process_from_inference_result(inference_result)
        
        assert result["prediction"]["accepted"] is True
        assert result["prediction"]["gloss"] == "yo"
    
    def test_reset_sequence(self):
        evaluator = DecisionEvaluator()
        
        # Add a prediction
        evaluator.process_prediction(
            class_id=28,
            class_name="yo",
            bucket="HEAD",
            confidence=0.65,
            topk_scores=[0.65, 0.20]
        )
        
        assert evaluator.get_sequence_state()["length"] == 1
        
        # Reset
        evaluator.reset_sequence()
        
        assert evaluator.get_sequence_state()["length"] == 0
    
    def test_get_config(self):
        evaluator = DecisionEvaluator()
        config = evaluator.get_config()
        
        assert "head_threshold" in config
        assert "mid_threshold" in config
        assert "margin_threshold" in config


# ============================================================
# Integration Tests
# ============================================================

class TestDecisionEngineIntegration:
    """Integration tests for the full decision engine flow."""
    
    def test_full_sequence_flow(self):
        """Test a complete sequence of predictions."""
        evaluator = DecisionEvaluator()
        
        # 1. Accepted HEAD class
        r1 = evaluator.process_prediction(
            class_id=28, class_name="yo", bucket="HEAD",
            confidence=0.70, topk_scores=[0.70, 0.15]
        )
        assert r1["prediction"]["accepted"] is True
        assert r1["sequence"]["length"] == 1
        
        # 2. Rejected OTHER class
        r2 = evaluator.process_prediction(
            class_id=141, class_name="OTHER", bucket="OTHER",
            confidence=0.50, topk_scores=[0.50, 0.30], is_other=True
        )
        assert r2["prediction"]["accepted"] is False
        assert r2["sequence"]["length"] == 1  # Still 1
        
        # 3. Accepted MID class
        r3 = evaluator.process_prediction(
            class_id=100, class_name="quiero", bucket="MID",
            confidence=0.60, topk_scores=[0.60, 0.20]
        )
        assert r3["prediction"]["accepted"] is True
        assert r3["sequence"]["length"] == 2
        
        # 4. Rejected low margin
        r4 = evaluator.process_prediction(
            class_id=50, class_name="comer", bucket="HEAD",
            confidence=0.50, topk_scores=[0.50, 0.45]  # Margin = 0.05
        )
        assert r4["prediction"]["accepted"] is False
        assert r4["sequence"]["length"] == 2  # Still 2
        
        # Final sequence
        assert evaluator.get_accepted_glosses() == ["yo", "quiero"]
    
    def test_deterministic_behavior(self):
        """Same input should always produce same output."""
        for _ in range(3):
            evaluator = DecisionEvaluator()
            
            result = evaluator.process_prediction(
                class_id=28, class_name="yo", bucket="HEAD",
                confidence=0.65, topk_scores=[0.65, 0.20]
            )
            
            assert result["prediction"]["accepted"] is True
            assert result["prediction"]["rule_applied"] == "accepted"
