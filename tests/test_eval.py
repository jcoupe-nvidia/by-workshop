"""Tests for src.eval — offline evaluation metrics."""
from __future__ import annotations

import pytest

from src.eval.metrics import (
    evaluate_trajectory,
    eval_skill_selection,
    eval_tool_validity,
    eval_tool_accuracy,
    eval_sequence_correctness,
    eval_task_success,
    eval_recovery_quality,
    eval_efficiency,
    EVAL_DIMENSIONS,
    DIMENSION_WEIGHTS,
    PASS_THRESHOLD,
    TrajectoryEvaluation,
    DimensionScore,
)
from src.rollouts.trace_types import Episode, EpisodeMetrics


class TestEvalDimensions:
    def test_seven_dimensions(self):
        assert len(EVAL_DIMENSIONS) == 7

    def test_weights_sum_to_one(self):
        total = sum(DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_dimensions_have_weights(self):
        for dim in EVAL_DIMENSIONS:
            assert dim in DIMENSION_WEIGHTS


class TestEvaluateSuccessfulTrajectory:
    def test_perfect_score(self, successful_episode):
        evaluation = evaluate_trajectory(successful_episode)
        assert evaluation.overall == pytest.approx(1.0, abs=0.01)
        assert evaluation.passed is True
        assert len(evaluation.scores) == 7

    def test_all_dimensions_present(self, successful_episode):
        evaluation = evaluate_trajectory(successful_episode)
        dims_scored = {s.dimension for s in evaluation.scores}
        assert dims_scored == set(EVAL_DIMENSIONS)


class TestEvaluateRepairTrajectory:
    def test_still_passes(self, repair_episode):
        evaluation = evaluate_trajectory(repair_episode)
        assert evaluation.passed is True
        assert evaluation.overall >= PASS_THRESHOLD

    def test_recovery_quality_reflects_repairs(self, repair_episode):
        score = eval_recovery_quality(repair_episode)
        # Should be positive (repairs succeeded) but not necessarily 1.0
        assert score.score > 0


class TestIndividualEvaluators:
    def test_skill_selection_perfect(self, successful_episode):
        score = eval_skill_selection(successful_episode)
        assert score.score == pytest.approx(1.0, abs=0.01)
        assert score.dimension == "skill_selection"

    def test_tool_validity_all_valid(self, successful_episode):
        score = eval_tool_validity(successful_episode)
        assert score.score == 1.0

    def test_tool_accuracy(self, successful_episode):
        score = eval_tool_accuracy(successful_episode)
        assert score.score == pytest.approx(1.0, abs=0.01)

    def test_sequence_correctness_perfect(self, successful_episode):
        score = eval_sequence_correctness(successful_episode)
        assert score.score == 1.0

    def test_task_success(self, successful_episode):
        score = eval_task_success(successful_episode)
        assert score.score == 1.0

    def test_recovery_quality_no_repairs(self, successful_episode):
        score = eval_recovery_quality(successful_episode)
        assert score.score == 1.0  # no repairs needed = perfect

    def test_efficiency(self, successful_episode):
        score = eval_efficiency(successful_episode)
        assert score.score == pytest.approx(1.0, abs=0.01)


class TestSequenceViolation:
    def test_detects_dependency_violation(self):
        """Build an episode that violates get_transfer_eta's prerequisite."""
        from src.rollouts.trace_types import (
            Event, EventType, ToolCallPayload, ToolResultPayload,
        )
        from src.runtime.tools import TOOL_REGISTRY

        events = []
        step = 0
        for name, args in [
            ("get_order", {"order_id": "SO-10482"}),
            ("get_shipment_status", {"order_id": "SO-10482"}),
            ("get_inventory", {"sku": "SKU-4090", "dc_id": "DC-WEST-01"}),
            # VIOLATION: get_transfer_eta before find_alternate_inventory
            ("get_transfer_eta", {"from_dc": "DC-EAST-02", "to_dc": "DC-WEST-01",
                                  "sku": "SKU-4090", "qty": 900}),
        ]:
            fn, _, _ = TOOL_REGISTRY[name]
            result = fn(**args)
            events.append(Event(event_type=EventType.TOOL_CALL, step_index=step,
                                payload=ToolCallPayload(tool_name=name, arguments=args)))
            step += 1
            events.append(Event(event_type=EventType.TOOL_RESULT, step_index=step,
                                payload=ToolResultPayload(tool_name=name, result=result)))
            step += 1

        bad = Episode(
            task_id="SO-10482",
            task_prompt="bad sequence test",
            events=events,
            metrics=EpisodeMetrics(valid_tool_calls=4),
        )
        score = eval_sequence_correctness(bad)
        assert score.score < 1.0
