from unittest.mock import patch

from django.test import TestCase

from fraud_checks.models import FraudCheck
from fraud_checks.services import apply_ml_escalation


class ApplyMlEscalationInvariantTests(TestCase):
    """ML never lowers rule decisions (REVIEW/BLOCKED stay unchanged)."""

    def test_blocked_stays_blocked_ml_not_called(self):
        rule_result = {"risk_score": 60, "decision": "BLOCKED", "reasons": ["limit_exceeded"]}
        result = apply_ml_escalation(None, rule_result)
        self.assertEqual(result["decision"], "BLOCKED")
        self.assertEqual(result["reasons"], ["limit_exceeded"])
        self.assertIsNone(result["ml_proba"])

    def test_review_stays_review_ml_not_called(self):
        rule_result = {"risk_score": 40, "decision": "REVIEW", "reasons": ["high_amount"]}
        result = apply_ml_escalation(None, rule_result)
        self.assertEqual(result["decision"], "REVIEW")
        self.assertEqual(result["reasons"], ["high_amount"])
        self.assertIsNone(result["ml_proba"])

    @patch("ml.inference.predict_ml_proba", return_value=0.95)
    def test_approved_escalated_when_proba_above_threshold(self, mock_predict):
        rule_result = {"risk_score": 10, "decision": "APPROVED", "reasons": []}
        result = apply_ml_escalation(None, rule_result)
        self.assertEqual(result["decision"], "REVIEW")
        self.assertIn("ml_escalation", result["reasons"])
        self.assertEqual(result["ml_proba"], 0.95)

    @patch("ml.inference.predict_ml_proba", return_value=0.01)
    def test_approved_stays_approved_when_proba_below_threshold(self, mock_predict):
        rule_result = {"risk_score": 10, "decision": "APPROVED", "reasons": []}
        result = apply_ml_escalation(None, rule_result)
        self.assertEqual(result["decision"], "APPROVED")
        self.assertNotIn("ml_escalation", result["reasons"])
        self.assertEqual(result["ml_proba"], 0.01)

    @patch("ml.inference.predict_ml_proba", return_value=0.95)
    def test_ml_escalation_does_not_lower_existing_review(self, mock_predict):
        rule_result = {"risk_score": 35, "decision": "REVIEW", "reasons": ["high_frequency"]}
        result = apply_ml_escalation(None, rule_result)
        self.assertEqual(result["decision"], "REVIEW")
        self.assertNotIn("ml_escalation", result["reasons"])
        self.assertIsNone(result["ml_proba"])
