"""
NeMo Gym adapter: resource server, result rows, and rollout-compatible exports.

This module bridges the repo-owned environment (LateOrderRecoveryEnv) with
NeMo Gym's training infrastructure:

    - LateOrderResourceServer: SimpleResourcesServer subclass implementing
      seed_session() and verify() for NeMo Gym rollout collection
    - NemoGymResultRow / episode_to_nemo_gym_row: JSONL row records with
      numeric reward fields for nemo-gym's RewardProfiler
    - build_rollout_input_row: task -> rollout input formatting

Owns:
    - NeMo Gym resource server adapter (seed + verify protocol)
    - Episode -> nemo-gym-compatible result row conversion
    - Task -> rollout input row formatting
    - Reward field extraction for profiling and training inspection

Does NOT own:
    - Environment state or transitions (see envs/)
    - Episode types (see rollouts.trace_types)
    - Rollout orchestration scheduling (NeMo Gym owns that)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    SimpleResourcesServer,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)

from src.envs.rewards import EpisodeRewardSummary, RewardSignal
from src.rollouts.trace_types import Episode, EpisodeMetrics
from src.runtime.tracing import EpisodeRecorder


# ---------------------------------------------------------------------------
# NeMo Gym resource server (seed + verify protocol)
# ---------------------------------------------------------------------------

@dataclass
class _NemoGymSession:
    """Per-session state holding both the environment and an EpisodeRecorder.

    The EpisodeRecorder captures canonical Event records so that the NeMo Gym
    training-time path produces the same inspectable Episode artifacts as the
    interactive runtime. This closes the trace-contract gap identified in the
    code review: verify() now emits canonical events, not just scalar rewards.
    """
    env: Any  # LateOrderRecoveryEnv
    recorder: EpisodeRecorder


# Per-session state, keyed by stable session_id (UUID).
# Module-level dict rather than class attribute because SimpleResourcesServer
# is a Pydantic BaseModel, and Pydantic treats underscore-prefixed class
# attributes as ModelPrivateAttr descriptors — which breaks dict operations
# when accessed on the class.
_sessions: dict[str, _NemoGymSession] = {}


class LateOrderResourceServer(SimpleResourcesServer):
    """NeMo Gym resource server for the late-order-recovery environment.

    Implements the two NeMo Gym resource-server endpoints:

        - seed_session(): initializes a fresh LateOrderRecoveryEnv for the
          order specified in the session config.
        - verify(): receives an agent response, extracts tool calls or
          terminal actions, steps the environment, and returns a scalar
          reward for NeMo Gym's reward profiling and rollout collection.

    This wires the repo-owned environment (state, transitions, rewards)
    into NeMo Gym's training-time infrastructure without duplicating
    task semantics or reward logic.
    """

    async def seed_session(
        self, body: BaseSeedSessionRequest,
    ) -> BaseSeedSessionResponse:
        """Initialize a fresh environment for a rollout session.

        NeMo Gym calls this once per rollout episode before the agent
        begins producing responses.  Returns a stable session_id so
        verify() can look up the correct environment even under
        concurrent rollout collection.

        Each session gets both a LateOrderRecoveryEnv and an
        EpisodeRecorder so that verify() emits canonical Event records
        alongside scalar rewards.
        """
        # Import here to avoid circular dependency
        from src.envs.late_order_env import LateOrderRecoveryEnv

        env = LateOrderRecoveryEnv()
        # Default order for the workshop scenario
        env.reset("SO-10482")

        recorder = EpisodeRecorder(
            task_id="SO-10482",
            task_prompt="NeMo Gym rollout for SO-10482",
            model_id="nemo-gym-rollout",
        )
        recorder.record_user_task("NeMo Gym rollout for SO-10482")

        # Use a UUID so session identity is stable and unique across
        # concurrent rollout workers (fixes parallel-safety issue).
        session_id = str(uuid.uuid4())
        _sessions[session_id] = _NemoGymSession(env=env, recorder=recorder)

        return BaseSeedSessionResponse(session_id=session_id)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """Compute a reward for the agent's response.

        NeMo Gym calls this after each agent response during rollout
        collection. The response may contain tool calls (agent actions)
        or a final message (terminal action).

        This method extracts agent actions from the NeMo Gym response
        and delegates to the rollouts-owned execution pipeline, which
        runs the canonical validate → repair → reject → execute → record
        loop. This keeps envs/ focused on environment validation
        surfaces while rollouts/ owns the execution plumbing.
        """
        from src.rollouts.nemo_gym_rollouts import (
            AgentAction,
            process_agent_actions,
        )

        session = self._get_session(body)
        actions = _extract_actions_from_response(body.response)
        reward = process_agent_actions(actions, session.env, session.recorder)

        # Clean up terminated sessions to prevent memory leaks during
        # long training runs (RL_ARCHITECTURE.md line 337).
        if session.env.is_terminal:
            session_id = getattr(body, "session_id", None)
            if session_id and session_id in _sessions:
                _sessions.pop(session_id, None)

        return BaseVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=round(reward, 4),
        )

    def _get_session(self, body: Any) -> _NemoGymSession:
        """Look up the session for the request.

        Uses session_id from the request when available. Falls back to
        the sole active session for single-session use. In all other
        cases, fails closed to prevent misrouted verify() traffic.

        Fallback sessions are registered in _sessions under a stable id
        so that export/read APIs can retrieve them afterward.
        """
        # Try to extract session_id from the request body.
        session_id = getattr(body, "session_id", None)
        if session_id and session_id in _sessions:
            return _sessions[session_id]

        # Fallback for tests or single-session use: return the only
        # seeded session if exactly one exists.
        if len(_sessions) == 1:
            return next(iter(_sessions.values()))

        # No session found and multiple (or zero) sessions exist.
        # Fail closed rather than silently creating an untracked session.
        raise ValueError(
            f"session_id '{session_id}' not found among "
            f"{len(_sessions)} active session(s). "
            f"Call seed_session() before verify()."
        )

    @staticmethod
    def get_session_episode(session_id: str) -> Episode | None:
        """Retrieve the canonical Episode accumulated during a rollout session.

        This is the primary way to inspect training-time traces in the
        same format used by the interactive runtime and offline eval.
        Populates env_state_init from the session's environment so that
        the episode carries the initial state snapshot for offline
        analysis and durable serialization.

        Returns None if the session_id is not found.
        """
        session = _sessions.get(session_id)
        if session is None:
            return None
        episode = session.recorder.build_episode()
        if not episode.env_state_init:
            episode.env_state_init = session.env.get_initial_state_snapshot()
        return episode

    @staticmethod
    def get_session_reward_summary(session_id: str):
        """Retrieve the EpisodeRewardSummary from a rollout session's environment.

        Returns None if the session_id is not found. This provides the
        reward breakdown computed during real-time execution, avoiding
        the need for a second enrichment pass.
        """
        session = _sessions.get(session_id)
        if session is None:
            return None
        return session.env.get_episode_reward_summary()

    @staticmethod
    def get_session_env_snapshot(session_id: str) -> dict | None:
        """Retrieve the final environment state snapshot from a rollout session.

        Returns None if the session_id is not found.
        """
        session = _sessions.get(session_id)
        if session is None:
            return None
        return session.env.get_state_snapshot()


def _extract_actions_from_response(response: Any) -> list:
    """Convert NeMo Gym response output items to AgentAction objects.

    This is NeMo Gym protocol-specific extraction: it reads the
    framework's response item types and converts them to the
    framework-agnostic AgentAction format used by the rollouts-owned
    execution pipeline.
    """
    from src.rollouts.nemo_gym_rollouts import AgentAction

    actions: list[AgentAction] = []
    if not hasattr(response, "output") or not response.output:
        return actions

    for item in response.output:
        item_type = getattr(item, "type", None)

        if item_type == "function_call":
            tool_name = getattr(item, "name", "")
            arguments = getattr(item, "arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            actions.append(AgentAction(
                action_type="function_call",
                tool_name=tool_name,
                arguments=arguments,
            ))

        elif item_type == "message":
            content = getattr(item, "content", "")
            if isinstance(content, list):
                content = " ".join(
                    getattr(c, "text", str(c)) for c in content
                )
            actions.append(AgentAction(
                action_type="message",
                content=content,
            ))

    return actions


def build_resource_server_config(
    host: str = "0.0.0.0",
    port: int = 8001,
    name: str = "late-order-recovery-env",
) -> BaseResourcesServerConfig:
    """Build a NeMo Gym resource server config for the late-order env."""
    return BaseResourcesServerConfig(
        host=host,
        port=port,
        name=name,
        entrypoint="src.envs.nemo_gym_adapter:LateOrderResourceServer",
    )


# ---------------------------------------------------------------------------
# nemo-gym input row format
# ---------------------------------------------------------------------------

def build_rollout_input_row(
    order_id: str,
    task_prompt: str,
    agent_name: str = "late-order-recovery-agent",
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
) -> dict[str, Any]:
    """Build a nemo-gym input row for a rollout collection run.

    This is the format expected by nemo-gym's RolloutCollectionHelper.
    Each row describes one task for the agent to solve.

    Args:
        order_id: The order ID to investigate.
        task_prompt: The full task prompt for the agent.
        agent_name: Agent server name for routing.
        temperature: Sampling temperature.
        max_output_tokens: Max tokens per model response.
    """
    return {
        "agent_ref": {"name": agent_name},
        "responses_create_params": {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        },
        "order_id": order_id,
        "task_prompt": task_prompt,
    }


# ---------------------------------------------------------------------------
# nemo-gym result row format (from enriched episodes)
# ---------------------------------------------------------------------------

@dataclass
class NemoGymResultRow:
    """A single result row in nemo-gym's expected format.

    Numeric fields are automatically aggregated by nemo-gym's RewardProfiler
    (mean, max, min, median, std, histogram). Non-numeric fields are preserved
    in the "response" dict for inspection.
    """
    # Identity
    episode_id: str
    order_id: str
    agent_ref: dict[str, str]

    # Numeric reward fields (profiled by nemo-gym)
    total_reward: float
    avg_step_reward: float
    valid_tool_calls: int
    invalid_tool_calls: int
    repair_attempts: int
    rejects: int
    episode_length: int
    task_success: int  # 1 if final_answer with correct action, 0 otherwise

    # Reward component averages (for detailed profiling)
    avg_valid_call: float
    avg_correct_tool: float
    avg_correct_arguments: float
    avg_dependency_satisfied: float
    avg_non_redundant: float
    avg_progress: float
    avg_efficiency: float
    terminal_quality: float

    # Penalty counts
    total_penalties: int

    # Non-numeric metadata (in "response" dict for nemo-gym)
    response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict suitable for nemo-gym JSONL output."""
        d: dict[str, Any] = {
            "episode_id": self.episode_id,
            "agent_ref": self.agent_ref,
            "order_id": self.order_id,
            "total_reward": self.total_reward,
            "avg_step_reward": self.avg_step_reward,
            "valid_tool_calls": self.valid_tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
            "repair_attempts": self.repair_attempts,
            "rejects": self.rejects,
            "episode_length": self.episode_length,
            "task_success": self.task_success,
            "avg_valid_call": self.avg_valid_call,
            "avg_correct_tool": self.avg_correct_tool,
            "avg_correct_arguments": self.avg_correct_arguments,
            "avg_dependency_satisfied": self.avg_dependency_satisfied,
            "avg_non_redundant": self.avg_non_redundant,
            "avg_progress": self.avg_progress,
            "avg_efficiency": self.avg_efficiency,
            "terminal_quality": self.terminal_quality,
            "total_penalties": self.total_penalties,
            "response": self.response,
        }
        return d


def episode_to_nemo_gym_row(
    episode: Episode,
    reward_summary: EpisodeRewardSummary,
    agent_name: str = "late-order-recovery-agent",
) -> NemoGymResultRow:
    """Convert an enriched Episode + reward summary to a nemo-gym result row.

    The numeric fields in the result row are what nemo-gym's RewardProfiler
    will aggregate across rollouts.
    """
    metrics = episode.metrics

    # Compute average reward components across steps
    step_rewards = reward_summary.step_rewards
    n_steps = max(len(step_rewards), 1)

    avg_valid_call = sum(r.valid_call for r in step_rewards) / n_steps
    avg_correct_tool = sum(r.correct_tool for r in step_rewards) / n_steps
    avg_correct_arguments = sum(r.correct_arguments for r in step_rewards) / n_steps
    avg_dependency_satisfied = sum(r.dependency_satisfied for r in step_rewards) / n_steps
    avg_non_redundant = sum(r.non_redundant for r in step_rewards) / n_steps
    avg_progress = sum(r.progress for r in step_rewards) / n_steps
    avg_efficiency = sum(r.efficiency for r in step_rewards) / n_steps

    terminal_quality = (
        reward_summary.terminal_reward.terminal_quality
        if reward_summary.terminal_reward else 0.0
    )

    total_penalties = sum(reward_summary.penalty_counts.values())

    # Determine task success
    task_success = 0
    if episode.terminal and episode.terminal.final_answer:
        action = episode.terminal.final_answer.get("action", "")
        if action and action != "escalate":
            task_success = 1

    # Build response metadata
    response: dict[str, Any] = {
        "usage": {
            "total_steps": metrics.total_steps,
            "model_calls": metrics.model_calls,
            "wall_time_seconds": metrics.wall_time_seconds,
        },
        "terminal_reason": episode.terminal.reason if episode.terminal else "unknown",
        "final_answer": episode.terminal.final_answer if episode.terminal else None,
        "penalty_counts": reward_summary.penalty_counts,
    }

    return NemoGymResultRow(
        episode_id=episode.episode_id,
        order_id=episode.task_id,
        agent_ref={"name": agent_name},
        total_reward=round(reward_summary.total_reward, 4),
        avg_step_reward=round(reward_summary.avg_step_reward, 4),
        valid_tool_calls=metrics.valid_tool_calls,
        invalid_tool_calls=metrics.invalid_tool_calls,
        repair_attempts=metrics.repair_attempts,
        rejects=metrics.rejects,
        episode_length=metrics.total_steps,
        task_success=task_success,
        avg_valid_call=round(avg_valid_call, 4),
        avg_correct_tool=round(avg_correct_tool, 4),
        avg_correct_arguments=round(avg_correct_arguments, 4),
        avg_dependency_satisfied=round(avg_dependency_satisfied, 4),
        avg_non_redundant=round(avg_non_redundant, 4),
        avg_progress=round(avg_progress, 4),
        avg_efficiency=round(avg_efficiency, 4),
        terminal_quality=round(terminal_quality, 4),
        total_penalties=total_penalties,
        response=response,
    )


# ---------------------------------------------------------------------------
# JSONL serialization
# ---------------------------------------------------------------------------

def save_nemo_gym_rows_jsonl(
    rows: list[NemoGymResultRow],
    path: str,
) -> None:
    """Write nemo-gym result rows to a JSONL file."""
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict()) + "\n")


def save_nemo_gym_inputs_jsonl(
    input_rows: list[dict[str, Any]],
    path: str,
) -> None:
    """Write nemo-gym input rows to a JSONL file."""
    with open(path, "w") as f:
        for row in input_rows:
            f.write(json.dumps(row) + "\n")
