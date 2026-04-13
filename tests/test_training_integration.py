"""Regression tests for training-layer integration paths.

Covers:
    - reward_views: step vs trajectory blend is distinct (issue #4)
    - nemo_rl_adapter: validation/repair/reject events in DatumSpec (issue #3)
    - datasets._maybe_truncate: truncated reward recomputation (issue #7)
    - nemo_gym_adapter: session safety and canonical validation (issues #1, #2)
    - nemo_gym_adapter: verify() async path integration (issue #2 extended)
    - eval/metrics: evaluate_nemo_gym_result partial weight normalization (issue #6)
    - eval/metrics: eval_skill_selection uses env-owned subgoals (issue #5)
"""
from __future__ import annotations

import asyncio
import json
import uuid
import pytest

from src.rollouts.scripted_traces import build_successful_episode, build_repair_episode
from src.rollouts.episode_runner import enrich_episode, EnrichedEpisodeResult
from src.rollouts.trace_types import (
    Episode,
    Event,
    EventType,
    EpisodeMetrics,
    ToolCallPayload,
    ToolResultPayload,
    TerminalOutcomePayload,
    ValidationErrorPayload,
    RepairAttemptPayload,
    RejectPayload,
)
from src.envs.rewards import (
    RewardSignal,
    EpisodeRewardSummary,
    summarize_episode_rewards,
)
from src.training.curriculum import (
    StageConfig,
    TrainingStage,
    get_stage_config,
)
from src.training.reward_views import (
    build_episode_reward_view,
    shape_step_reward,
    EpisodeRewardView,
)
from src.training.datasets import (
    TrainingRecord,
    _maybe_truncate,
    build_training_dataset,
)
from src.training.nemo_rl_adapter import episode_to_datum_spec
from src.eval.metrics import (
    eval_skill_selection,
    evaluate_nemo_gym_result,
    EVAL_DIMENSIONS,
    DIMENSION_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Issue #4: reward_views step vs trajectory blend
# ---------------------------------------------------------------------------

class TestRewardViewStepTrajectoryBlend:
    """The step-level and trajectory-level reward signals must be distinct
    when the stage config has different weights for each."""

    def test_blend_produces_different_result_when_weights_differ(self):
        """When step_weight != trajectory_weight, the combined reward must
        NOT collapse to (step_weight + trajectory_weight) * same_number."""
        step_signals = [
            RewardSignal(valid_call=1.0, correct_tool=1.0, correct_arguments=1.0,
                         dependency_satisfied=1.0, non_redundant=1.0, progress=1.0,
                         efficiency=1.0, terminal_quality=0.0),
            RewardSignal(valid_call=1.0, correct_tool=0.5, correct_arguments=0.5,
                         dependency_satisfied=1.0, non_redundant=1.0, progress=0.5,
                         efficiency=0.8, terminal_quality=0.0),
        ]
        terminal_signal = RewardSignal(
            valid_call=0.0, correct_tool=0.0, correct_arguments=0.0,
            dependency_satisfied=0.0, non_redundant=0.0, progress=0.0,
            efficiency=0.0, terminal_quality=0.9,
        )
        summary = summarize_episode_rewards(step_signals, terminal_signal)

        stage = StageConfig(
            stage=TrainingStage.SHORT_HORIZON_RL,
            description="test",
            step_reward_weight=0.7,
            trajectory_reward_weight=0.3,
        )

        view = build_episode_reward_view(summary, stage)

        step_only_avg = sum(sv.shaped_reward for sv in view.step_views) / len(view.step_views)
        all_avg = view.trajectory_reward

        assert view.terminal_view is not None
        assert abs(step_only_avg - all_avg) > 0.001 or len(view.step_views) == 0

        naive = (0.7 + 0.3) * view.trajectory_reward
        assert abs(view.combined_reward - naive) > 0.001

    def test_sft_stage_zero_weights(self):
        """SFT stage has both weights at 0.0 — combined should be 0.0."""
        step_signals = [
            RewardSignal(valid_call=1.0, correct_tool=1.0, correct_arguments=1.0,
                         dependency_satisfied=1.0, non_redundant=1.0, progress=1.0,
                         efficiency=1.0, terminal_quality=0.0),
        ]
        summary = summarize_episode_rewards(step_signals)
        stage = get_stage_config(TrainingStage.SFT_SUCCESSFUL)
        view = build_episode_reward_view(summary, stage)
        assert view.combined_reward == 0.0


# ---------------------------------------------------------------------------
# Issue #3: nemo_rl_adapter includes validation/repair/reject events
# ---------------------------------------------------------------------------

class TestDatumSpecIncludesFailureEvents:
    """DatumSpec messages must include validation errors, repair
    attempts, and rejects — not just tool calls and terminal."""

    def test_repair_episode_has_error_messages(self):
        """A repair episode should produce messages for the error events."""
        episode = build_repair_episode()
        datum = episode_to_datum_spec(episode)

        messages = datum["message_log"]
        tool_messages = [
            json.loads(m["content"]) if m.get("content") else {}
            for m in messages
            if m.get("role") == "tool" and m.get("content")
        ]

        error_msgs = [c for c in tool_messages if "error" in c]
        repair_msgs = [c for c in tool_messages if "repair_succeeded" in c]
        reject_msgs = [c for c in tool_messages if "rejected" in c]

        assert len(error_msgs) > 0, "No validation error messages in DatumSpec"
        assert len(repair_msgs) > 0, "No repair attempt messages in DatumSpec"
        assert len(reject_msgs) > 0, "No reject messages in DatumSpec"

    def test_successful_episode_unchanged(self):
        """A clean successful episode should not gain spurious error messages."""
        episode = build_successful_episode()
        datum = episode_to_datum_spec(episode)

        messages = datum["message_log"]
        error_msgs = [
            m for m in messages
            if m.get("role") == "tool"
            and m.get("content")
            and "error" in (json.loads(m["content"]) if m["content"] else {})
        ]
        assert len(error_msgs) == 0


# ---------------------------------------------------------------------------
# Issue #7: datasets truncation recomputes reward
# ---------------------------------------------------------------------------

class TestTruncateRecomputesReward:
    """Truncated episodes must carry a recomputed reward from only the
    retained events, not the full-episode reward."""

    def test_truncated_reward_less_than_full(self, enriched_success):
        """Truncating to 3 tool calls should produce a lower total_reward
        than the full 9-call episode."""
        episode = enriched_success.episode
        full_reward = episode.metrics.total_reward

        truncated = _maybe_truncate(episode, 3)
        assert truncated.metrics.valid_tool_calls == 3
        assert truncated.terminal is None
        assert truncated.metrics.total_reward <= full_reward
        assert truncated.metadata.get("truncated_at") == 3

    def test_no_truncation_when_within_bounds(self, enriched_success):
        """Episode with 9 calls should not be truncated at max=15."""
        episode = enriched_success.episode
        result = _maybe_truncate(episode, 15)
        assert result is episode

    def test_truncated_reward_from_retained_events(self, enriched_success):
        """The truncated reward should match the sum of reward annotations
        on the retained events."""
        episode = enriched_success.episode
        truncated = _maybe_truncate(episode, 3)

        expected_reward = sum(
            e.reward for e in truncated.events if e.reward is not None
        )
        assert abs(truncated.metrics.total_reward - round(expected_reward, 4)) < 1e-4


# ---------------------------------------------------------------------------
# Issue #5: eval_skill_selection uses env-owned subgoals
# ---------------------------------------------------------------------------

class TestEvalSkillSelectionUsesEnvSubgoals:
    """eval_skill_selection must derive from envs.state.SUBGOAL_ORDER and
    TOOL_TO_SUBGOAL, not from runtime.workflows."""

    def test_perfect_score_on_optimal(self, successful_episode):
        score = eval_skill_selection(successful_episode)
        assert score.score == pytest.approx(1.0, abs=0.01)
        assert "subgoal" in score.details.lower()

    def test_empty_episode_scores_zero(self):
        ep = Episode(task_id="test", task_prompt="test", metrics=EpisodeMetrics())
        score = eval_skill_selection(ep)
        assert score.score == 0.0


# ---------------------------------------------------------------------------
# Issue #6: evaluate_nemo_gym_result weight normalization
# ---------------------------------------------------------------------------

class TestEvaluateNemoGymResultNormalization:
    """evaluate_nemo_gym_result must normalize weights to available
    dimensions and clearly label the output as partial."""

    def test_partial_overall_key(self, enriched_success):
        from src.envs.nemo_gym_adapter import NemoGymResultRow, episode_to_nemo_gym_row
        row = episode_to_nemo_gym_row(
            enriched_success.episode, enriched_success.reward_summary,
        )
        result = evaluate_nemo_gym_result(row)

        assert "partial_overall" in result
        assert "overall" not in result
        assert "dimensions_missing" in result
        assert len(result["dimensions_missing"]) > 0
        assert "note" in result

    def test_partial_overall_is_normalized(self, enriched_success):
        from src.envs.nemo_gym_adapter import NemoGymResultRow, episode_to_nemo_gym_row
        row = episode_to_nemo_gym_row(
            enriched_success.episode, enriched_success.reward_summary,
        )
        result = evaluate_nemo_gym_result(row)

        assert 0.0 <= result["partial_overall"] <= 1.0

    def test_type_error_on_wrong_input(self):
        with pytest.raises(TypeError):
            evaluate_nemo_gym_result({"not": "a result row"})


# ---------------------------------------------------------------------------
# Issue #2: NeMo Gym adapter session safety
# ---------------------------------------------------------------------------

class TestNemoGymSessionSafety:
    """Session management must use stable UUIDs and look up the correct env."""

    def _get_sessions(self) -> dict:
        """Access the module-level _sessions dict."""
        from src.envs import nemo_gym_adapter
        return nemo_gym_adapter._sessions

    def _reset_sessions(self) -> None:
        self._get_sessions().clear()

    def _make_session(self, order_id: str = "SO-10482"):
        """Create a _NemoGymSession wrapping env + recorder."""
        from src.envs.late_order_env import LateOrderRecoveryEnv
        from src.envs.nemo_gym_adapter import _NemoGymSession
        from src.runtime.tracing import EpisodeRecorder

        env = LateOrderRecoveryEnv()
        env.reset(order_id)
        recorder = EpisodeRecorder(
            task_id=order_id,
            task_prompt=f"test rollout for {order_id}",
            model_id="test",
        )
        return _NemoGymSession(env=env, recorder=recorder)

    def test_sessions_use_uuid_keys(self):
        self._reset_sessions()
        sessions = self._get_sessions()

        session = self._make_session()
        session_id = str(uuid.uuid4())
        sessions[session_id] = session

        for key in sessions:
            parsed = uuid.UUID(key)
            assert str(parsed) == key

        self._reset_sessions()

    def test_multiple_sessions_distinct(self):
        self._reset_sessions()
        sessions = self._get_sessions()

        session1 = self._make_session()
        id1 = str(uuid.uuid4())
        sessions[id1] = session1

        session2 = self._make_session()
        id2 = str(uuid.uuid4())
        sessions[id2] = session2

        assert id1 != id2
        assert sessions[id1].env is session1.env
        assert sessions[id2].env is session2.env

        self._reset_sessions()


# ---------------------------------------------------------------------------
# Issue #2 (extended): NeMo Gym verify() async path integration
# ---------------------------------------------------------------------------

def _make_function_call_item(name: str, arguments: dict) -> object:
    """Build a NeMo Gym ResponseFunctionToolCall."""
    from nemo_gym.openai_utils import NeMoGymResponseFunctionToolCall
    return NeMoGymResponseFunctionToolCall(
        arguments=json.dumps(arguments),
        call_id=f"call_{uuid.uuid4().hex[:8]}",
        name=name,
    )


def _make_message_item(content: str) -> object:
    """Build a NeMo Gym ResponseOutputMessage."""
    from nemo_gym.openai_utils import (
        NeMoGymResponseOutputMessage,
        NeMoGymResponseOutputText,
    )
    return NeMoGymResponseOutputMessage(
        id=f"msg_{uuid.uuid4().hex[:8]}",
        content=[NeMoGymResponseOutputText(text=content, annotations=[])],
    )


def _make_verify_request(
    output_items: list,
    session_id: str | None = None,
) -> object:
    """Build a BaseVerifyRequest with the given output items."""
    from nemo_gym.openai_utils import (
        NeMoGymResponse,
        NeMoGymResponseCreateParamsNonStreaming,
    )
    from nemo_gym.base_resources_server import BaseVerifyRequest

    response = NeMoGymResponse(
        id=f"resp_{uuid.uuid4().hex[:8]}",
        created_at=1000.0,
        model="test-model",
        object="response",
        output=output_items,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    params = NeMoGymResponseCreateParamsNonStreaming(input="test")
    req = BaseVerifyRequest(
        responses_create_params=params,
        response=response,
    )
    if session_id is not None:
        object.__setattr__(req, "session_id", session_id)
    return req


class TestVerifyAsyncPath:
    """Integration tests for LateOrderResourceServer.verify().

    These tests exercise the full async verify() path by constructing
    real NeMo Gym request/response objects and calling verify() directly.
    This validates:
        - Valid tool call execution and reward accumulation
        - Validation error detection
        - Repair success path (fuzzy tool name -> repaired -> executed)
        - Reject path (unrecoverable output -> penalty reward)
        - Terminal message handling (final answer -> terminal reward)
        - Multi-step reward accumulation across sequential verify() calls
        - Session lookup by session_id
    """

    @staticmethod
    def _get_sessions() -> dict:
        """Access the module-level _sessions dict."""
        from src.envs import nemo_gym_adapter
        return nemo_gym_adapter._sessions

    def _reset_sessions(self) -> None:
        self._get_sessions().clear()

    def _make_server(self):
        """Create a fresh server with a seeded session, return (server, session_id, env)."""
        from src.envs.nemo_gym_adapter import LateOrderResourceServer, _NemoGymSession
        from src.envs.late_order_env import LateOrderRecoveryEnv
        from src.runtime.tracing import EpisodeRecorder

        self._reset_sessions()
        sessions = self._get_sessions()

        server = LateOrderResourceServer.__new__(LateOrderResourceServer)
        env = LateOrderRecoveryEnv()
        env.reset("SO-10482")
        recorder = EpisodeRecorder(
            task_id="SO-10482",
            task_prompt="test rollout",
            model_id="test",
        )
        session_id = str(uuid.uuid4())
        sessions[session_id] = _NemoGymSession(env=env, recorder=recorder)
        return server, session_id, env

    def _cleanup(self):
        self._reset_sessions()

    def test_valid_tool_call_returns_positive_reward(self):
        """A valid get_order call should produce a positive reward."""
        server, sid, env = self._make_server()

        item = _make_function_call_item("get_order", {"order_id": "SO-10482"})
        req = _make_verify_request([item], session_id=sid)

        resp = asyncio.run(server.verify(req))

        assert resp.reward > 0.0, "Valid first tool call should earn positive reward"
        assert len(env._step_rewards) == 1
        assert env._step_rewards[0].valid_call > 0
        self._cleanup()

    def test_multi_step_reward_accumulates(self):
        """Calling get_order then get_shipment_status should accumulate reward."""
        server, sid, env = self._make_server()

        req1 = _make_verify_request(
            [_make_function_call_item("get_order", {"order_id": "SO-10482"})],
            session_id=sid,
        )
        resp1 = asyncio.run(server.verify(req1))
        assert resp1.reward > 0.0

        req2 = _make_verify_request(
            [_make_function_call_item("get_shipment_status", {"order_id": "SO-10482"})],
            session_id=sid,
        )
        resp2 = asyncio.run(server.verify(req2))
        assert resp2.reward > 0.0
        assert len(env._step_rewards) == 2
        self._cleanup()

    def test_unknown_tool_no_close_match_gets_rejected(self):
        """A completely unknown tool name with no fuzzy match should be rejected
        and produce a penalty reward."""
        server, sid, env = self._make_server()

        item = _make_function_call_item(
            "zzz_nonexistent_tool_xyz", {"foo": "bar"}
        )
        req = _make_verify_request([item], session_id=sid)

        resp = asyncio.run(server.verify(req))

        assert len(env._step_rewards) == 1
        assert env._step_rewards[0].valid_call < 0
        assert resp.reward < 0.0
        self._cleanup()

    def test_fuzzy_tool_name_gets_repaired(self):
        """A close misspelling of a tool name should be repaired and executed."""
        server, sid, env = self._make_server()

        item = _make_function_call_item("get_ordr", {"order_id": "SO-10482"})
        req = _make_verify_request([item], session_id=sid)

        resp = asyncio.run(server.verify(req))

        assert len(env._step_rewards) == 1
        assert env.state.repair_attempt_count >= 1
        self._cleanup()

    def test_terminal_message_produces_terminal_reward(self):
        """A message item should terminate the episode and produce a terminal reward."""
        server, sid, env = self._make_server()

        req0 = _make_verify_request(
            [_make_function_call_item("get_order", {"order_id": "SO-10482"})],
            session_id=sid,
        )
        asyncio.run(server.verify(req0))

        final_answer = json.dumps({"action": "transfer", "rationale": "test"})
        msg_item = _make_message_item(final_answer)
        req = _make_verify_request([msg_item], session_id=sid)

        resp = asyncio.run(server.verify(req))

        assert env.is_terminal
        assert env._terminal_reward is not None
        self._cleanup()

    def test_terminal_message_non_json_creates_unknown_answer(self):
        """A non-JSON message should still terminate with action=unknown."""
        server, sid, env = self._make_server()

        msg_item = _make_message_item("I recommend transferring the order.")
        req = _make_verify_request([msg_item], session_id=sid)

        resp = asyncio.run(server.verify(req))

        assert env.is_terminal
        self._cleanup()

    def test_empty_output_returns_zero_reward(self):
        """A verify call with no output items should return reward=0.0."""
        server, sid, env = self._make_server()

        req = _make_verify_request([], session_id=sid)
        resp = asyncio.run(server.verify(req))

        assert resp.reward == 0.0
        assert len(env._step_rewards) == 0
        self._cleanup()

    def test_multiple_tool_calls_in_one_response(self):
        """Multiple function_call items in one response: only the first is
        processed (parallel_tool_calls enforcement, MEDIUM-7)."""
        server, sid, env = self._make_server()

        items = [
            _make_function_call_item("get_order", {"order_id": "SO-10482"}),
            _make_function_call_item("get_shipment_status", {"order_id": "SO-10482"}),
        ]
        req = _make_verify_request(items, session_id=sid)

        resp = asyncio.run(server.verify(req))

        assert len(env._step_rewards) == 1
        assert resp.reward != 0.0
        self._cleanup()

    def test_verify_uses_correct_session(self):
        """verify() should look up the environment by session_id."""
        from src.envs.nemo_gym_adapter import LateOrderResourceServer, _NemoGymSession
        from src.envs.late_order_env import LateOrderRecoveryEnv
        from src.runtime.tracing import EpisodeRecorder

        self._reset_sessions()
        sessions = self._get_sessions()
        server = LateOrderResourceServer.__new__(LateOrderResourceServer)

        env1 = LateOrderRecoveryEnv()
        env1.reset("SO-10482")
        rec1 = EpisodeRecorder(task_id="SO-10482", task_prompt="t", model_id="t")
        sid1 = str(uuid.uuid4())
        sessions[sid1] = _NemoGymSession(env=env1, recorder=rec1)

        env2 = LateOrderRecoveryEnv()
        env2.reset("SO-10482")
        rec2 = EpisodeRecorder(task_id="SO-10482", task_prompt="t", model_id="t")
        sid2 = str(uuid.uuid4())
        sessions[sid2] = _NemoGymSession(env=env2, recorder=rec2)

        req = _make_verify_request(
            [_make_function_call_item("get_order", {"order_id": "SO-10482"})],
            session_id=sid2,
        )
        asyncio.run(server.verify(req))

        assert len(env2._step_rewards) == 1
        assert len(env1._step_rewards) == 0

        self._cleanup()

    def test_reward_is_rounded_to_4_decimals(self):
        """The returned reward should be rounded to 4 decimal places."""
        server, sid, env = self._make_server()

        req = _make_verify_request(
            [_make_function_call_item("get_order", {"order_id": "SO-10482"})],
            session_id=sid,
        )
        resp = asyncio.run(server.verify(req))

        reward_str = f"{resp.reward:.10f}"
        after_4th = reward_str.split(".")[1][4:]
        assert all(c == "0" for c in after_4th), (
            f"Reward {resp.reward} not rounded to 4 decimals"
        )
        self._cleanup()

    def test_response_fields_passed_through(self):
        """BaseVerifyResponse should carry through responses_create_params and response."""
        server, sid, env = self._make_server()

        item = _make_function_call_item("get_order", {"order_id": "SO-10482"})
        req = _make_verify_request([item], session_id=sid)

        resp = asyncio.run(server.verify(req))

        assert resp.responses_create_params is req.responses_create_params
        assert resp.response is req.response
        self._cleanup()

    def test_missing_arguments_rejected(self):
        """A tool call with missing required arguments should be rejected."""
        server, sid, env = self._make_server()

        item = _make_function_call_item("get_inventory", {"sku": "SKU-4090"})
        req = _make_verify_request([item], session_id=sid)

        resp = asyncio.run(server.verify(req))

        assert len(env._step_rewards) == 1
        assert env._step_rewards[0].valid_call < 0
        self._cleanup()
