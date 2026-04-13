"""
Canonical structured episode and event model.

This module is the source of truth for the structured records that flow
through runtime, rollouts, training, and evaluation.  Every layer consumes
these types rather than ad-hoc dicts or unstructured trace text.

Event vocabulary
----------------
Each turn in an episode is represented as one or more ``Event`` records
with an explicit ``event_type``:

    user_task               -- the initial task prompt
    model_thought           -- optional preserved reasoning (when intentionally kept)
    tool_call               -- a validated, dispatched tool invocation
    tool_result             -- the deterministic output of a tool
    tool_validation_error   -- a tool call that failed schema or dependency validation
    tool_repair_attempt     -- a fallback repair was attempted on a malformed call
    tool_reject             -- the fallback layer rejected an unrecoverable call
    agent_message           -- free-form agent text (rare; prefer structured events)
    terminal_outcome        -- episode ended (final answer, max iterations, or error)

Episode structure
-----------------
An ``Episode`` groups:
    - task input and initial environment state metadata
    - an ordered list of ``Event`` records (the turn-by-turn trace)
    - step-level reward annotations (populated by envs/rewards or training)
    - a terminal outcome summary
    - episode-level summary metrics
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _mint_episode_id() -> str:
    """Generate a stable, unique episode identifier."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Async GRPO metadata key constants (RL_ARCHITECTURE.md lines 68-77)
# ---------------------------------------------------------------------------
# These keys define the async metadata contract that trajectories must carry
# when async GRPO is enabled. They make stale trajectories explicit and
# filterable instead of silently mixing them into training.
#
# Populated with placeholder values during synchronous runs so the contract
# is visible in serialized output and filterable by downstream tooling.

ASYNC_META_GEN_WEIGHT_VERSION = "gen_weight_version"
ASYNC_META_TRAIN_WEIGHT_VERSION = "train_weight_version"
ASYNC_META_TRAJECTORY_AGE_MS = "trajectory_age_ms"
ASYNC_META_REPLAY_STATUS = "replay_status"

ASYNC_META_KEYS = (
    ASYNC_META_GEN_WEIGHT_VERSION,
    ASYNC_META_TRAIN_WEIGHT_VERSION,
    ASYNC_META_TRAJECTORY_AGE_MS,
    ASYNC_META_REPLAY_STATUS,
)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    """Canonical event vocabulary for episode traces."""
    USER_TASK = "user_task"
    MODEL_THOUGHT = "model_thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_VALIDATION_ERROR = "tool_validation_error"
    TOOL_REPAIR_ATTEMPT = "tool_repair_attempt"
    TOOL_REJECT = "tool_reject"
    AGENT_MESSAGE = "agent_message"
    TERMINAL_OUTCOME = "terminal_outcome"


# ---------------------------------------------------------------------------
# Event payloads  (one dataclass per event type that carries data)
# ---------------------------------------------------------------------------

@dataclass
class ToolCallPayload:
    """Payload for a TOOL_CALL event."""
    tool_name: str
    arguments: dict[str, Any]
    thought: str | None = None
    raw_model_output: str = ""


@dataclass
class ToolResultPayload:
    """Payload for a TOOL_RESULT event."""
    tool_name: str
    result: dict[str, Any]


@dataclass
class ValidationErrorPayload:
    """Payload for a TOOL_VALIDATION_ERROR event."""
    error_type: str
    message: str
    raw_model_output: str = ""


@dataclass
class RepairAttemptPayload:
    """Payload for a TOOL_REPAIR_ATTEMPT event."""
    original_output: str
    repaired_output: str | None
    repairs_applied: list[str] = field(default_factory=list)
    succeeded: bool = False


@dataclass
class RejectPayload:
    """Payload for a TOOL_REJECT event."""
    reason: str
    raw_model_output: str = ""
    repairs_attempted: list[str] = field(default_factory=list)


@dataclass
class TerminalOutcomePayload:
    """Payload for a TERMINAL_OUTCOME event."""
    reason: str  # "final_answer", "max_iterations", "error"
    final_answer: dict[str, Any] | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Unified Event record
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """One structured event in an episode trace.

    Every event has:
        - ``event_type``  -- from the canonical vocabulary
        - ``step_index``  -- position in the episode (0-based)
        - ``payload``     -- typed data (dict or a payload dataclass)
        - ``reward``      -- optional step-level reward annotation

    The ``payload`` field holds a typed dataclass for structured events
    (ToolCallPayload, ToolResultPayload, etc.) or a plain dict/str for
    simpler events like user_task or agent_message.
    """
    event_type: EventType
    step_index: int
    payload: Any  # typed payload dataclass or dict/str
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Episode summary metrics
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    """Aggregate metrics for a completed episode."""
    total_steps: int = 0
    valid_tool_calls: int = 0
    invalid_tool_calls: int = 0
    repair_attempts: int = 0
    repair_successes: int = 0
    rejects: int = 0
    model_calls: int = 0
    wall_time_seconds: float = 0.0
    total_reward: float = 0.0


# ---------------------------------------------------------------------------
# Episode  (top-level container)
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A complete structured episode / trajectory.

    This is the canonical record that every downstream layer consumes:
        - ``rollouts/`` serializes episodes for dataset production
        - ``training/`` builds reward views and dataset adapters over episodes
        - ``eval/`` computes offline metrics from episodes

    Attributes:
        episode_id:     Stable unique identifier for this episode, minted at
                        creation time. Threaded through serialization, NeMo Gym
                        result rows, art trajectory metadata, and ATIF exports
                        so that replay, regression analysis, and trainer/eval
                        joins can key on a repo-owned identity.
        task_id:        Identifier for the task (e.g. "SO-10482").
        task_prompt:    The initial user-facing task description.
        model_id:       Which model produced this episode.
        env_state_init: Snapshot of relevant environment state at episode start.
        events:         Ordered list of Event records (the trace).
        terminal:       The terminal outcome event (also the last in events).
        metrics:        Aggregate summary metrics.
        metadata:       Arbitrary extra info (provenance, config, etc.).
    """
    episode_id: str = field(default_factory=_mint_episode_id)
    task_id: str = ""
    task_prompt: str = ""
    model_id: str = ""
    env_state_init: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    terminal: TerminalOutcomePayload | None = None
    metrics: EpisodeMetrics = field(default_factory=EpisodeMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- convenience helpers ------------------------------------------------

    @property
    def tool_calls(self) -> list[Event]:
        """All TOOL_CALL events in order."""
        return [e for e in self.events if e.event_type == EventType.TOOL_CALL]

    @property
    def tool_names_called(self) -> list[str]:
        """Ordered list of tool names from valid TOOL_CALL events."""
        names: list[str] = []
        for e in self.events:
            if e.event_type == EventType.TOOL_CALL and isinstance(e.payload, ToolCallPayload):
                names.append(e.payload.tool_name)
        return names

    @property
    def is_complete(self) -> bool:
        """Whether the episode reached a terminal outcome."""
        return self.terminal is not None

    @property
    def final_answer(self) -> dict[str, Any] | None:
        """The final answer dict, if the episode ended with one."""
        if self.terminal and self.terminal.final_answer:
            return self.terminal.final_answer
        return None
