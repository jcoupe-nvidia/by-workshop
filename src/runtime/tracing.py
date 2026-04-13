"""
Structured event emission for the agent runtime.

Owns:
    - Construction of canonical Event records from runtime actions
    - Episode assembly from a sequence of events
    - Metrics computation from event sequences

Does NOT own:
    - Event/Episode type definitions (see rollouts.trace_types)
    - Rollout serialization or batching (see rollouts/)
    - Reward computation (see envs.rewards)
    - Tool execution or prompt policy

This module provides a thin EpisodeRecorder that the agent loop uses to
emit structured events as they happen.  The recorder produces a fully
populated Episode at the end, using the canonical types from
rollouts.trace_types.
"""
from __future__ import annotations

import time
from typing import Any

from src.rollouts.trace_types import (
    Episode,
    EpisodeMetrics,
    Event,
    EventType,
    RepairAttemptPayload,
    RejectPayload,
    TerminalOutcomePayload,
    ToolCallPayload,
    ToolResultPayload,
    ValidationErrorPayload,
)


class EpisodeRecorder:
    """Accumulates structured events and produces a canonical Episode.

    Usage::

        recorder = EpisodeRecorder(task_id="SO-10482", task_prompt="...", model_id="...")
        recorder.record_user_task(prompt)
        recorder.record_tool_call(tool_name, arguments, thought, raw)
        recorder.record_tool_result(tool_name, result)
        ...
        recorder.record_terminal(reason, final_answer=...)
        episode = recorder.build_episode()
    """

    def __init__(
        self,
        task_id: str,
        task_prompt: str,
        model_id: str = "",
        env_state_init: dict[str, Any] | None = None,
        episode_id: str | None = None,
    ) -> None:
        from src.rollouts.trace_types import _mint_episode_id

        self.episode_id = episode_id or _mint_episode_id()
        self.task_id = task_id
        self.task_prompt = task_prompt
        self.model_id = model_id
        self.env_state_init = env_state_init or {}
        self._events: list[Event] = []
        self._step_index = 0
        self._start_time = time.time()

        # Counters for metrics
        self._model_calls = 0
        self._valid_tool_calls = 0
        self._invalid_tool_calls = 0
        self._repair_attempts = 0
        self._repair_successes = 0
        self._rejects = 0

    def _next_step(self) -> int:
        idx = self._step_index
        self._step_index += 1
        return idx

    def record_user_task(self, prompt: str) -> None:
        """Record the initial user task event."""
        self._events.append(Event(
            event_type=EventType.USER_TASK,
            step_index=self._next_step(),
            payload=prompt,
        ))

    def record_model_thought(self, thought: str) -> None:
        """Record an intentionally preserved model thought."""
        self._events.append(Event(
            event_type=EventType.MODEL_THOUGHT,
            step_index=self._next_step(),
            payload=thought,
        ))

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        thought: str | None = None,
        raw_model_output: str = "",
    ) -> None:
        """Record a validated, dispatched tool call."""
        self._valid_tool_calls += 1
        self._events.append(Event(
            event_type=EventType.TOOL_CALL,
            step_index=self._next_step(),
            payload=ToolCallPayload(
                tool_name=tool_name,
                arguments=arguments,
                thought=thought,
                raw_model_output=raw_model_output,
            ),
        ))

    def record_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        """Record the deterministic result of a tool execution."""
        self._events.append(Event(
            event_type=EventType.TOOL_RESULT,
            step_index=self._next_step(),
            payload=ToolResultPayload(
                tool_name=tool_name,
                result=result,
            ),
        ))

    def record_validation_error(
        self,
        error_type: str,
        message: str,
        raw_model_output: str = "",
    ) -> None:
        """Record a tool call that failed schema or dependency validation."""
        self._invalid_tool_calls += 1
        self._events.append(Event(
            event_type=EventType.TOOL_VALIDATION_ERROR,
            step_index=self._next_step(),
            payload=ValidationErrorPayload(
                error_type=error_type,
                message=message,
                raw_model_output=raw_model_output,
            ),
        ))

    def record_repair_attempt(
        self,
        original_output: str,
        repaired_output: str | None,
        repairs_applied: list[str],
        succeeded: bool,
    ) -> None:
        """Record a fallback repair attempt."""
        self._repair_attempts += 1
        if succeeded:
            self._repair_successes += 1
        self._events.append(Event(
            event_type=EventType.TOOL_REPAIR_ATTEMPT,
            step_index=self._next_step(),
            payload=RepairAttemptPayload(
                original_output=original_output,
                repaired_output=repaired_output,
                repairs_applied=repairs_applied,
                succeeded=succeeded,
            ),
        ))

    def record_reject(
        self,
        reason: str,
        raw_model_output: str = "",
        repairs_attempted: list[str] | None = None,
    ) -> None:
        """Record a fallback rejection (unrecoverable malformed output)."""
        self._rejects += 1
        self._events.append(Event(
            event_type=EventType.TOOL_REJECT,
            step_index=self._next_step(),
            payload=RejectPayload(
                reason=reason,
                raw_model_output=raw_model_output,
                repairs_attempted=repairs_attempted or [],
            ),
        ))

    def record_agent_message(self, message: str) -> None:
        """Record a free-form agent message (rare)."""
        self._events.append(Event(
            event_type=EventType.AGENT_MESSAGE,
            step_index=self._next_step(),
            payload=message,
        ))

    def record_terminal(
        self,
        reason: str,
        final_answer: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        """Record the terminal outcome of the episode."""
        self._events.append(Event(
            event_type=EventType.TERMINAL_OUTCOME,
            step_index=self._next_step(),
            payload=TerminalOutcomePayload(
                reason=reason,
                final_answer=final_answer,
                error_message=error_message,
            ),
        ))

    def record_metadata(self, key: str, value: Any) -> None:
        """Attach arbitrary metadata to the episode for offline analysis.

        Metadata is stored on the Episode.metadata dict, making it
        available via session inspection and serialized artifacts.
        """
        if not hasattr(self, "_metadata"):
            self._metadata: dict[str, Any] = {}
        self._metadata[key] = value

    def increment_model_calls(self) -> None:
        """Increment the model call counter."""
        self._model_calls += 1

    def build_episode(self) -> Episode:
        """Assemble the recorded events into a canonical Episode."""
        wall_time = round(time.time() - self._start_time, 2)

        # Find terminal payload if present
        terminal = None
        for e in reversed(self._events):
            if e.event_type == EventType.TERMINAL_OUTCOME:
                if isinstance(e.payload, TerminalOutcomePayload):
                    terminal = e.payload
                break

        total_reward = sum(e.reward for e in self._events if e.reward is not None)

        metrics = EpisodeMetrics(
            total_steps=len(self._events),
            valid_tool_calls=self._valid_tool_calls,
            invalid_tool_calls=self._invalid_tool_calls,
            repair_attempts=self._repair_attempts,
            repair_successes=self._repair_successes,
            rejects=self._rejects,
            model_calls=self._model_calls,
            wall_time_seconds=wall_time,
            total_reward=total_reward,
        )

        ep = Episode(
            episode_id=self.episode_id,
            task_id=self.task_id,
            task_prompt=self.task_prompt,
            model_id=self.model_id,
            env_state_init=self.env_state_init,
            events=list(self._events),
            terminal=terminal,
            metrics=metrics,
        )
        if hasattr(self, "_metadata"):
            ep.metadata.update(self._metadata)

        from src.rollouts.trace_types import ASYNC_META_KEYS, ASYNC_META_REPLAY_STATUS
        for key in ASYNC_META_KEYS:
            if key not in ep.metadata:
                if key == ASYNC_META_REPLAY_STATUS:
                    ep.metadata[key] = "accepted"
                else:
                    ep.metadata[key] = 0

        return ep
