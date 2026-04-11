"""Tests for src.runtime.workflows — workflow decomposition."""
from __future__ import annotations

import pytest

from src.runtime.workflows import (
    WorkflowContext,
    diagnose_order_risk,
    assess_primary_fulfillment,
    evaluate_alternate_recovery_paths,
    synthesize_recommendation,
    validate_skill_transition,
    run_diagnostic_flow,
    SKILL_NAMES,
    SKILL_TOOL_PATTERNS,
    SKILL_TRANSITIONS,
)


class TestWorkflowConstants:
    def test_four_skills(self):
        assert len(SKILL_NAMES) == 4

    def test_skill_names(self):
        assert "diagnose_order_risk" in SKILL_NAMES
        assert "assess_primary_fulfillment" in SKILL_NAMES
        assert "evaluate_alternate_recovery_paths" in SKILL_NAMES
        assert "synthesize_recommendation" in SKILL_NAMES

    def test_each_skill_has_tools(self):
        for skill in SKILL_NAMES:
            assert skill in SKILL_TOOL_PATTERNS
            assert len(SKILL_TOOL_PATTERNS[skill]) >= 1


class TestValidateSkillTransition:
    def test_start_to_diagnose(self):
        ok, reason = validate_skill_transition(None, "diagnose_order_risk")
        assert ok is True

    def test_diagnose_to_assess(self):
        ok, reason = validate_skill_transition("diagnose_order_risk", "assess_primary_fulfillment")
        assert ok is True

    def test_skip_ahead_rejected(self):
        ok, reason = validate_skill_transition("diagnose_order_risk", "synthesize_recommendation")
        assert ok is False
        assert len(reason) > 0


class TestDiagnoseOrderRisk:
    def test_diagnose(self):
        ctx = WorkflowContext(order_id="SO-10482")
        result = diagnose_order_risk(ctx)
        assert result.order_id == "SO-10482"
        assert result.is_at_risk is True
        assert result.sku == "SKU-4090"
        assert result.qty == 1200
        assert len(ctx.tool_calls) == 2  # get_order + get_shipment_status


class TestAssessPrimaryFulfillment:
    def test_assess(self):
        ctx = WorkflowContext(order_id="SO-10482")
        diagnose_order_risk(ctx)
        result = assess_primary_fulfillment(ctx)
        assert result.can_fulfill is False
        assert result.shortfall > 0
        assert len(ctx.tool_calls) == 4


class TestEvaluateAlternateRecoveryPaths:
    def test_paths(self):
        ctx = WorkflowContext(order_id="SO-10482")
        diagnose_order_risk(ctx)
        assess_primary_fulfillment(ctx)
        paths = evaluate_alternate_recovery_paths(ctx)
        assert len(paths) > 0
        assert any(p.path_type == "dc_transfer" for p in paths)


class TestSynthesizeRecommendation:
    def test_full_flow(self):
        ctx = WorkflowContext(order_id="SO-10482")
        diagnose_order_risk(ctx)
        assess_primary_fulfillment(ctx)
        evaluate_alternate_recovery_paths(ctx)
        rec = synthesize_recommendation(ctx)
        assert rec.confidence > 0
        assert len(rec.action) > 0
        assert len(rec.rationale) > 0


class TestRunDiagnosticFlow:
    def test_full_flow(self):
        ctx = run_diagnostic_flow("SO-10482")
        assert ctx.recommendation is not None
        assert len(ctx.tool_calls) >= 5
        assert len(ctx.workflows_executed) == 4
