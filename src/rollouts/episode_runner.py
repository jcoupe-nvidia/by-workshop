"""
Episode runner: orchestrates runtime agent loop + environment + reward enrichment.

Owns:
    - Running one episode through the runtime agent loop
    - Replaying episode events through the environment for reward computation
    - Attaching per-step rewards to canonical Event records
    - Populating env_state_init on the Episode
    - Producing fully enriched Episodes ready for serialization or training

Does NOT own:
    - Tool implementations or prompt policy (see runtime/)
    - Environment state transitions or reward formulas (see envs/)
    - Serialization formats (see rollouts.serializers)
    - Training dataset views (see training/)
    - Offline evaluation metrics (see eval/)

The episode runner is the integration point between runtime and environment.
The runtime produces a raw Episode (events without rewards), and this module
replays those events through the environment to compute dense rewards and
attach them back to the Episode. This keeps the runtime independent of the
environment while giving downstream consumers (serializers, export adapters,
training) fully enriched episodes.

Usage::

    episode = run_enriched_episode("SO-10482")
    # episode.events now have .reward populated
    # episode.env_state_init is populated
    # episode.metrics.total_reward reflects env rewards
"""
from __future__ import annotations

from typing import Any

from src.rollouts.trace_types import (
    Episode,
    EpisodeMetrics,
    Event,
    EventType,
    ToolCallPayload,
    ToolResultPayload,
    ValidationErrorPayload,
    RepairAttemptPayload,
    RejectPayload,
    TerminalOutcomePayload,
)
from src.envs.late_order_env import LateOrderRecoveryEnv
from src.envs.rewards import RewardSignal, EpisodeRewardSummary


def run_enriched_episode(
    order_id: str,
    max_iterations: int = 15,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    verbose: bool = True,
) -> EnrichedEpisodeResult:
    """Run a full episode and enrich it with environment rewards.

    This is the primary entry point for the rollout layer. It:
        1. Calls run_agent_episode() to get the raw Episode from runtime
        2. Replays events through LateOrderRecoveryEnv for reward computation
        3. Attaches rewards to Event records
        4. Returns the enriched Episode with full reward metadata

    Args:
        order_id: The order to investigate (e.g. "SO-10482").
        max_iterations: Safety bound on loop iterations.
        max_tokens: Max tokens per model response.
        temperature: Sampling temperature.
        verbose: Print each step as it happens.

    Returns:
        EnrichedEpisodeResult with the episode and reward summary.
    """
    from src.runtime.agent import run_agent_episode

    raw_episode = run_agent_episode(
        order_id=order_id,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=verbose,
    )

    return enrich_episode(raw_episode, order_id)


def enrich_episode(
    episode: Episode,
    order_id: str | None = None,
) -> EnrichedEpisodeResult:
    """Replay an Episode through the environment to attach rewards.

    This can be used on any Episode — whether produced by run_agent_episode()
    live, or loaded from serialized storage for re-scoring.

    The function walks the episode's events in order and feeds each relevant
    event to the environment:
        - TOOL_CALL + TOOL_RESULT pairs → env.step()
        - TOOL_VALIDATION_ERROR → env.record_invalid()
        - TOOL_REPAIR_ATTEMPT → env.record_fallback_repair()
        - TOOL_REJECT → env.record_fallback_reject()
        - TERMINAL_OUTCOME → env.terminate()

    After replay, each tool-call Event gets its .reward field populated
    with the environment's step reward, and the terminal event gets the
    terminal reward.

    Args:
        episode: A raw or previously enriched Episode.
        order_id: Override for the order ID. If None, uses episode.task_id.

    Returns:
        EnrichedEpisodeResult with reward-annotated episode and summary.
    """
    task_id = order_id or episode.task_id
    env = LateOrderRecoveryEnv()
    env.reset(task_id)

    # Populate env_state_init if not already set
    if not episode.env_state_init:
        episode.env_state_init = env.get_initial_state_snapshot()

    # Walk events and feed to environment
    # Track tool calls and their results so we can pair them for env.step()
    pending_tool_call: _PendingToolCall | None = None
    tool_call_event_indices: list[int] = []

    for i, event in enumerate(episode.events):
        if event.event_type == EventType.TOOL_CALL:
            payload = event.payload
            if isinstance(payload, ToolCallPayload):
                # Check if previous tool call had a repair
                was_repaired = _was_preceding_repair(episode.events, i)
                pending_tool_call = _PendingToolCall(
                    event_index=i,
                    tool_name=payload.tool_name,
                    arguments=payload.arguments,
                    was_repaired=was_repaired,
                )
                tool_call_event_indices.append(i)

        elif event.event_type == EventType.TOOL_RESULT:
            payload = event.payload
            if isinstance(payload, ToolResultPayload) and pending_tool_call is not None:
                # Feed tool call + result to environment
                step_result = env.step(
                    tool_name=pending_tool_call.tool_name,
                    tool_arguments=pending_tool_call.arguments,
                    tool_result=payload.result,
                    was_repaired=pending_tool_call.was_repaired,
                )
                # Get the reward for this step (last one added)
                step_idx = len(env._step_rewards) - 1
                reward = env.get_step_reward(step_idx)
                if reward is not None:
                    # Attach reward to the TOOL_CALL event
                    episode.events[pending_tool_call.event_index].reward = reward.total
                pending_tool_call = None

        elif event.event_type == EventType.TOOL_VALIDATION_ERROR:
            payload = event.payload
            if isinstance(payload, ValidationErrorPayload):
                env.record_invalid(
                    error_type=payload.error_type,
                    error_message=payload.message,
                )
                # Get the penalty reward
                step_idx = len(env._step_rewards) - 1
                reward = env.get_step_reward(step_idx)
                if reward is not None:
                    event.reward = reward.total

        elif event.event_type == EventType.TOOL_REPAIR_ATTEMPT:
            payload = event.payload
            if isinstance(payload, RepairAttemptPayload):
                env.record_fallback_repair(succeeded=payload.succeeded)

        elif event.event_type == EventType.TOOL_REJECT:
            env.record_fallback_reject()

        elif event.event_type == EventType.TERMINAL_OUTCOME:
            payload = event.payload
            if isinstance(payload, TerminalOutcomePayload):
                env.terminate(
                    reason=payload.reason,
                    final_answer=payload.final_answer,
                )
                terminal_reward = env.get_terminal_reward()
                if terminal_reward is not None:
                    event.reward = terminal_reward.total

    # Get the full reward summary
    reward_summary = env.get_episode_reward_summary()

    # Update episode metrics with total reward
    episode.metrics.total_reward = round(reward_summary.total_reward, 4)

    return EnrichedEpisodeResult(
        episode=episode,
        reward_summary=reward_summary,
        env_final_state=env.get_state_snapshot(),
    )


def _was_preceding_repair(events: list[Event], tool_call_index: int) -> bool:
    """Check if the event immediately before a TOOL_CALL was a successful repair."""
    if tool_call_index == 0:
        return False
    # Look backwards for a repair attempt
    for j in range(tool_call_index - 1, max(0, tool_call_index - 3) - 1, -1):
        prev = events[j]
        if prev.event_type == EventType.TOOL_REPAIR_ATTEMPT:
            if isinstance(prev.payload, RepairAttemptPayload):
                return prev.payload.succeeded
        # Stop looking if we hit a different tool call or result
        if prev.event_type in (EventType.TOOL_CALL, EventType.TOOL_RESULT):
            break
    return False


class _PendingToolCall:
    """Transient state for pairing TOOL_CALL with its TOOL_RESULT."""
    __slots__ = ("event_index", "tool_name", "arguments", "was_repaired")

    def __init__(
        self,
        event_index: int,
        tool_name: str,
        arguments: dict[str, Any],
        was_repaired: bool,
    ) -> None:
        self.event_index = event_index
        self.tool_name = tool_name
        self.arguments = arguments
        self.was_repaired = was_repaired


# -- Result container ---------------------------------------------------------

class EnrichedEpisodeResult:
    """Result of running or enriching an episode with environment rewards.

    Attributes:
        episode: The Episode with .reward fields populated on events.
        reward_summary: Aggregated reward breakdown from the environment.
        env_final_state: Snapshot of environment state at episode end.
    """
    __slots__ = ("episode", "reward_summary", "env_final_state")

    def __init__(
        self,
        episode: Episode,
        reward_summary: EpisodeRewardSummary,
        env_final_state: dict[str, Any],
    ) -> None:
        self.episode = episode
        self.reward_summary = reward_summary
        self.env_final_state = env_final_state

    def print_summary(self) -> None:
        """Print a compact summary of the enriched episode."""
        ep = self.episode
        rs = self.reward_summary

        print(f"Episode: {ep.task_id}")
        print(f"  Model:          {ep.model_id}")
        print(f"  Tool calls:     {ep.metrics.valid_tool_calls}")
        print(f"  Invalid calls:  {ep.metrics.invalid_tool_calls}")
        print(f"  Repairs:        {ep.metrics.repair_attempts} ({ep.metrics.repair_successes} succeeded)")
        print(f"  Rejects:        {ep.metrics.rejects}")
        print(f"  Total reward:   {rs.total_reward:+.4f}")
        print(f"  Avg step reward:{rs.avg_step_reward:+.4f}")
        if rs.penalty_counts:
            print(f"  Penalties:      {rs.penalty_counts}")

        terminal = ep.terminal
        if terminal:
            print(f"  Outcome:        {terminal.reason}")
            if terminal.final_answer:
                action = terminal.final_answer.get("action", "N/A")
                print(f"  Recommendation: {action}")

        # Per-step reward breakdown
        print()
        print("  Step rewards:")
        for i, sr in enumerate(rs.step_rewards):
            penalties = f" [{', '.join(sr.penalties)}]" if sr.penalties else ""
            print(f"    {i:2d}. {sr.total:+.4f}{penalties}")
        if rs.terminal_reward:
            penalties = f" [{', '.join(rs.terminal_reward.penalties)}]" if rs.terminal_reward.penalties else ""
            print(f"    T.  {rs.terminal_reward.total:+.4f}{penalties}")
