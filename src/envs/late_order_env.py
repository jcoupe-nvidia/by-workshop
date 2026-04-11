"""
Main environment class for the late-order recovery task.

Combines state, transitions, and rewards into a single environment that
serves as the authoritative source of truth for:
    - What has happened so far in the episode
    - What actions are valid right now
    - What the reward signal is for each action
    - Whether the episode is terminal

Owns:
    - Episode lifecycle (reset, step, terminate)
    - Scenario-specific knowledge (expected arguments, optimal sequence)
    - State snapshots for rollout capture

Does NOT own:
    - Tool implementations (see runtime.tools)
    - Prompt policy or orchestration (see runtime/)
    - Rollout batching or serialization (see rollouts/)
    - Training dataset views (see training/)

Usage::

    env = LateOrderRecoveryEnv()
    state = env.reset("SO-10482")

    # Agent calls a tool...
    step = env.step("get_order", {"order_id": "SO-10482"}, tool_result)
    reward = env.get_step_reward(step, {"order_id": "SO-10482"})

    # Agent finishes...
    terminal_step = env.terminate("final_answer", final_answer={...})
    terminal_reward = env.get_terminal_reward()
    summary = env.get_episode_reward_summary()
"""
from __future__ import annotations

from typing import Any

from src.envs.state import (
    LateOrderEnvState,
    Subgoal,
    SUBGOAL_ORDER,
    TOOL_TO_SUBGOAL,
    TOOL_COMPLETES_SUBGOAL,
    make_initial_state,
)
from src.envs.transitions import (
    StepResult,
    apply_tool_call,
    apply_terminal,
    record_invalid_action,
    record_repair_attempt,
    record_reject,
    check_preconditions,
    should_force_terminate,
    MAX_EPISODE_STEPS,
)
from src.envs.rewards import (
    RewardSignal,
    EpisodeRewardSummary,
    compute_step_reward,
    compute_terminal_reward,
    summarize_episode_rewards,
    EXPECTED_ARGUMENTS,
    OPTIMAL_TOOL_SEQUENCE,
    OPTIMAL_STEP_COUNT,
    CORRECT_TOOLS_BY_SUBGOAL,
)
from src.runtime.tools import TOOL_REGISTRY, TOOL_DEPENDENCIES


class LateOrderRecoveryEnv:
    """Explicit RL environment for the late-order recovery task.

    This class is the single authority on task state and validity. The
    runtime calls step() after each tool execution, and the environment
    returns transition metadata and reward signals.

    The environment does not execute tools — it observes what the runtime
    did and updates its own state accordingly.
    """

    def __init__(self) -> None:
        self._state: LateOrderEnvState | None = None
        self._step_rewards: list[RewardSignal] = []
        self._terminal_reward: RewardSignal | None = None
        self._step_results: list[StepResult] = []

    # -- Lifecycle -------------------------------------------------------------

    def reset(self, order_id: str) -> LateOrderEnvState:
        """Initialize the environment for a new episode.

        Args:
            order_id: The order to investigate (e.g. "SO-10482").

        Returns:
            The initial environment state.
        """
        self._state = make_initial_state(order_id)
        self._step_rewards = []
        self._terminal_reward = None
        self._step_results = []
        return self._state

    @property
    def state(self) -> LateOrderEnvState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    @property
    def is_terminal(self) -> bool:
        return self._state is not None and self._state.is_terminal

    # -- Step processing -------------------------------------------------------

    def step(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
        tool_result: dict[str, Any],
        was_repaired: bool = False,
    ) -> StepResult:
        """Process a tool call and advance environment state.

        The runtime should call this after executing a valid tool call.
        The environment updates its state, computes a reward signal, and
        returns transition metadata.

        Args:
            tool_name: Name of the tool that was called.
            tool_arguments: Arguments the agent provided.
            tool_result: Deterministic result from the tool.
            was_repaired: Whether this call required fallback repair.

        Returns:
            StepResult with transition metadata.
        """
        result = apply_tool_call(self.state, tool_name, tool_arguments, tool_result)

        # Compute and store the step reward
        reward = compute_step_reward(result, self.state, tool_arguments, was_repaired)
        self._step_rewards.append(reward)
        self._step_results.append(result)

        # Check for forced termination
        if result.valid and should_force_terminate(self.state):
            self.terminate("max_iterations")

        return result

    def record_invalid(
        self,
        error_type: str,
        error_message: str,
    ) -> StepResult:
        """Record an invalid action (malformed, schema error, etc.).

        Use this when the runtime's parse/validate step fails before
        a tool can be identified or executed.
        """
        result = record_invalid_action(self.state, error_type, error_message)

        # Compute penalty reward for the invalid step
        reward = compute_step_reward(result, self.state)
        self._step_rewards.append(reward)
        self._step_results.append(result)

        return result

    def record_fallback_repair(self, succeeded: bool) -> None:
        """Record that a fallback repair was attempted."""
        record_repair_attempt(self.state, succeeded)

    def record_fallback_reject(self) -> None:
        """Record that a fallback repair was rejected."""
        record_reject(self.state)

    def terminate(
        self,
        reason: str,
        final_answer: dict[str, Any] | None = None,
    ) -> StepResult:
        """Mark the episode as terminal.

        Args:
            reason: Why the episode ended ("final_answer", "max_iterations", "error").
            final_answer: The agent's final recommendation, if any.
        """
        result = apply_terminal(self.state, reason, final_answer)
        self._terminal_reward = compute_terminal_reward(self.state)
        self._step_results.append(result)
        return result

    # -- Reward access ---------------------------------------------------------

    def get_step_reward(self, step_index: int) -> RewardSignal | None:
        """Get the reward signal for a specific step."""
        if 0 <= step_index < len(self._step_rewards):
            return self._step_rewards[step_index]
        return None

    def get_terminal_reward(self) -> RewardSignal | None:
        """Get the terminal step reward signal."""
        return self._terminal_reward

    def get_episode_reward_summary(self) -> EpisodeRewardSummary:
        """Get the aggregated episode reward summary."""
        return summarize_episode_rewards(self._step_rewards, self._terminal_reward)

    def get_step_reward_totals(self) -> list[float]:
        """Get just the total reward values for each step (for training export)."""
        return [r.total for r in self._step_rewards]

    # -- Query methods ---------------------------------------------------------

    def is_tool_allowed(self, tool_name: str) -> tuple[bool, str | None]:
        """Check if a tool call is currently allowed.

        Returns (is_allowed, reason_if_not).
        """
        ok, err_type, err_msg = check_preconditions(self.state, tool_name)
        return ok, err_msg

    def get_allowed_tools(self) -> list[str]:
        """Return the list of tools that can be called right now."""
        allowed = []
        for tool_name in TOOL_REGISTRY:
            ok, _, _ = check_preconditions(self.state, tool_name)
            if ok:
                allowed.append(tool_name)
        return allowed

    def get_next_expected_tools(self) -> list[str]:
        """Return tools that would represent optimal forward progress.

        These are the tools belonging to the next incomplete subgoal.
        """
        next_sg = self.state.next_expected_subgoal
        if next_sg is None:
            return []
        correct = CORRECT_TOOLS_BY_SUBGOAL.get(next_sg, set())
        # Filter to only those whose dependencies are satisfied
        return [
            t for t in correct
            if t not in self.state.tools_called_set
            and check_preconditions(self.state, t)[0]
        ]

    # -- State snapshots -------------------------------------------------------

    def get_state_snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the current state."""
        return self.state.to_dict()

    def get_initial_state_snapshot(self) -> dict[str, Any]:
        """Return a snapshot suitable for Episode.env_state_init."""
        return {
            "order_id": self.state.order_id,
            "sku": self.state.sku,
            "qty": self.state.qty,
            "source_dc": self.state.source_dc,
            "committed_date": self.state.committed_date,
            "region": self.state.region,
        }

    # -- Scenario knowledge (class-level) --------------------------------------

    @staticmethod
    def expected_arguments() -> dict[str, dict[str, Any]]:
        """Return the expected tool arguments for the SO-10482 scenario."""
        return dict(EXPECTED_ARGUMENTS)

    @staticmethod
    def optimal_tool_sequence() -> list[str]:
        """Return the optimal tool call sequence for SO-10482."""
        return list(OPTIMAL_TOOL_SEQUENCE)

    @staticmethod
    def optimal_step_count() -> int:
        """Return the number of steps in the optimal trajectory."""
        return OPTIMAL_STEP_COUNT

    @staticmethod
    def tool_dependencies() -> dict[str, set[str]]:
        """Return the tool dependency graph."""
        return dict(TOOL_DEPENDENCIES)

    @staticmethod
    def subgoal_order() -> list[Subgoal]:
        """Return the canonical subgoal progression."""
        return list(SUBGOAL_ORDER)

    @staticmethod
    def max_episode_steps() -> int:
        return MAX_EPISODE_STEPS
