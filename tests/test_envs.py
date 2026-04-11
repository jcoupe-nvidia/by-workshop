"""Tests for src.envs — environment state, transitions, rewards, validators."""
from __future__ import annotations

import pytest

from src.envs.state import (
    LateOrderEnvState,
    Subgoal,
    SUBGOAL_ORDER,
    make_initial_state,
)
from src.envs.transitions import (
    apply_tool_call,
    check_preconditions,
    record_invalid_action,
    apply_terminal,
)
from src.envs.rewards import (
    RewardSignal,
    EpisodeRewardSummary,
    compute_step_reward,
    compute_terminal_reward,
    summarize_episode_rewards,
    EXPECTED_ARGUMENTS,
    OPTIMAL_TOOL_SEQUENCE,
    REWARD_WEIGHTS,
)
from src.envs.validators import check_dependencies, check_dependencies_from_state
from src.envs.late_order_env import LateOrderRecoveryEnv
from src.runtime.tools import TOOL_REGISTRY, TOOL_DEPENDENCIES


class TestMakeInitialState:
    def test_known_order(self):
        state = make_initial_state("SO-10482")
        assert state.order_id == "SO-10482"
        assert state.sku == "SKU-4090"
        assert state.qty == 1200
        assert state.is_terminal is False
        assert len(state.tools_called) == 0

    def test_unknown_order_raises(self):
        with pytest.raises(ValueError, match="Unknown order"):
            make_initial_state("SO-99999")


class TestSubgoalOrder:
    def test_four_subgoals(self):
        assert len(SUBGOAL_ORDER) == 4

    def test_order(self):
        assert SUBGOAL_ORDER[0] == Subgoal.ORDER_DIAGNOSED
        assert SUBGOAL_ORDER[-1] == Subgoal.RECOMMENDATION_SYNTHESIZED


class TestCheckPreconditions:
    def test_get_order_always_valid(self):
        state = make_initial_state("SO-10482")
        ok, _, _ = check_preconditions(state, "get_order")
        assert ok is True

    def test_transfer_eta_needs_alternate(self):
        state = make_initial_state("SO-10482")
        ok, error_type, msg = check_preconditions(state, "get_transfer_eta")
        assert ok is False
        assert error_type is not None


class TestApplyToolCall:
    def test_valid_first_call(self):
        state = make_initial_state("SO-10482")
        fn, _, _ = TOOL_REGISTRY["get_order"]
        result = fn(order_id="SO-10482")
        step = apply_tool_call(state, "get_order", {"order_id": "SO-10482"}, result)
        assert step.valid is True
        assert "get_order" in state.tools_called
        assert state.order_found is True

    def test_redundant_call(self):
        state = make_initial_state("SO-10482")
        fn, _, _ = TOOL_REGISTRY["get_order"]
        result = fn(order_id="SO-10482")
        apply_tool_call(state, "get_order", {"order_id": "SO-10482"}, result)
        step2 = apply_tool_call(state, "get_order", {"order_id": "SO-10482"}, result)
        assert step2.is_redundant is True


class TestRecordInvalidAction:
    def test_increments_counter(self):
        state = make_initial_state("SO-10482")
        assert state.invalid_action_count == 0
        record_invalid_action(state, "no_json", "No JSON found")
        assert state.invalid_action_count == 1


class TestApplyTerminal:
    def test_marks_terminal(self):
        state = make_initial_state("SO-10482")
        step = apply_terminal(state, "final_answer", {"action": "transfer"})
        assert state.is_terminal is True
        assert state.terminal_reason == "final_answer"
        assert step.is_terminal is True


class TestCheckDependencies:
    def test_no_deps(self):
        ok, msg = check_dependencies("get_order", [], TOOL_DEPENDENCIES)
        assert ok is True

    def test_satisfied_deps(self):
        ok, msg = check_dependencies(
            "get_shipment_status", ["get_order"], TOOL_DEPENDENCIES,
        )
        assert ok is True

    def test_missing_deps(self):
        ok, msg = check_dependencies(
            "get_transfer_eta", ["get_order"], TOOL_DEPENDENCIES,
        )
        assert ok is False
        assert "find_alternate_inventory" in msg

    def test_check_from_state(self):
        state = make_initial_state("SO-10482")
        ok, msg = check_dependencies_from_state("get_order", state)
        assert ok is True


class TestRewardSignal:
    def test_total_is_weighted_sum(self):
        sig = RewardSignal(
            valid_call=1.0, correct_tool=1.0, correct_arguments=1.0,
            dependency_satisfied=1.0, non_redundant=1.0, progress=1.0,
            efficiency=1.0, terminal_quality=0.0,
        )
        assert sig.total > 0
        assert abs(sig.total - sum(
            REWARD_WEIGHTS[k] * getattr(sig, k)
            for k in REWARD_WEIGHTS
        )) < 1e-6


class TestComputeStepReward:
    def test_valid_step_positive(self):
        state = make_initial_state("SO-10482")
        fn, _, _ = TOOL_REGISTRY["get_order"]
        result = fn(order_id="SO-10482")
        step = apply_tool_call(state, "get_order", {"order_id": "SO-10482"}, result)
        reward = compute_step_reward(step, state, {"order_id": "SO-10482"})
        assert reward.total > 0
        assert reward.valid_call == 1.0


class TestComputeTerminalReward:
    def test_with_final_answer(self, enriched_success):
        """Use an enriched episode to verify terminal reward is positive."""
        reward = compute_terminal_reward(
            make_initial_state("SO-10482"),  # just need a state object
        )
        # The state hasn't gone through the full flow, so terminal_quality
        # depends on subgoals. Use the enriched result instead.
        assert enriched_success.reward_summary.terminal_reward is not None
        assert enriched_success.reward_summary.terminal_reward.terminal_quality > 0


class TestEpisodeRewardSummary:
    def test_summary_from_signals(self):
        sig1 = RewardSignal(valid_call=1.0, correct_tool=1.0, correct_arguments=1.0,
                            dependency_satisfied=1.0, non_redundant=1.0, progress=1.0,
                            efficiency=1.0, terminal_quality=0.0)
        sig2 = RewardSignal(valid_call=1.0, correct_tool=0.5, correct_arguments=0.5,
                            dependency_satisfied=1.0, non_redundant=1.0, progress=0.5,
                            efficiency=1.0, terminal_quality=0.0)
        summary = summarize_episode_rewards([sig1, sig2])
        assert summary.total_reward == pytest.approx(sig1.total + sig2.total, abs=1e-4)
        assert summary.avg_step_reward == pytest.approx(
            (sig1.total + sig2.total) / 2, abs=1e-4,
        )


class TestExpectedArguments:
    def test_covers_key_tools(self):
        assert "get_order" in EXPECTED_ARGUMENTS
        assert "get_inventory" in EXPECTED_ARGUMENTS


class TestOptimalToolSequence:
    def test_length(self):
        assert len(OPTIMAL_TOOL_SEQUENCE) == 9

    def test_starts_with_get_order(self):
        assert OPTIMAL_TOOL_SEQUENCE[0] == "get_order"

    def test_ends_with_recommend_action(self):
        assert OPTIMAL_TOOL_SEQUENCE[-1] == "recommend_action"


class TestLateOrderRecoveryEnv:
    def test_reset(self):
        env = LateOrderRecoveryEnv()
        state = env.reset("SO-10482")
        assert state.order_id == "SO-10482"
        assert env.is_terminal is False

    def test_step_valid(self):
        env = LateOrderRecoveryEnv()
        env.reset("SO-10482")
        fn, _, _ = TOOL_REGISTRY["get_order"]
        result = fn(order_id="SO-10482")
        step = env.step("get_order", {"order_id": "SO-10482"}, result)
        assert step.valid is True

    def test_full_optimal_run(self, enriched_success):
        """The scripted trace runs the full optimal sequence through the env."""
        summary = enriched_success.reward_summary
        assert summary.total_reward > 0
        assert len(summary.step_rewards) == 9
        assert enriched_success.episode.is_complete is True

    def test_record_invalid(self):
        env = LateOrderRecoveryEnv()
        env.reset("SO-10482")
        step = env.record_invalid("no_json", "No JSON found")
        assert step.valid is False

    def test_get_allowed_tools(self):
        env = LateOrderRecoveryEnv()
        env.reset("SO-10482")
        allowed = env.get_allowed_tools()
        assert "get_order" in allowed

    def test_state_snapshot(self):
        env = LateOrderRecoveryEnv()
        env.reset("SO-10482")
        snapshot = env.get_state_snapshot()
        assert "order_id" in snapshot
        assert snapshot["order_id"] == "SO-10482"
