"""
Stable serialization for canonical Episodes.

Owns:
    - Episode <-> dict/JSON conversion
    - JSONL file read/write for episode collections
    - Payload-type-aware event serialization
    - Preservation of exact turn order, validation failures, repairs,
      rejects, and terminal outcomes in the serialized form

Does NOT own:
    - Episode or Event type definitions (see rollouts.trace_types)
    - Reward computation (see envs.rewards)
    - Training-specific dataset views (see training/)
    - ProRL trajectory format (see rollouts.prorl_adapter)

Serialization contract:
    - Every event is serialized with its full payload, event_type, step_index,
      reward annotation, and metadata
    - Typed payload dataclasses are flattened to dicts
    - The round-trip Episode -> dict -> Episode preserves all data
    - JSONL format: one JSON object per line, one episode per line

Usage::

    from src.rollouts.serializers import episode_to_jsonl, save_episodes_jsonl

    jsonl_line = episode_to_jsonl(episode)
    save_episodes_jsonl([episode], "trajectories/batch_001.jsonl")
"""
from __future__ import annotations

import json
from dataclasses import asdict
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


# -- Event serialization ------------------------------------------------------

def _serialize_payload(event: Event) -> dict[str, Any] | str:
    """Convert a typed event payload to a serializable form.

    Typed payload dataclasses become dicts. Plain strings (user_task,
    model_thought, agent_message) pass through as-is.
    """
    payload = event.payload

    if isinstance(payload, ToolCallPayload):
        return {
            "tool_name": payload.tool_name,
            "arguments": payload.arguments,
            "thought": payload.thought,
            "raw_model_output": payload.raw_model_output,
        }

    if isinstance(payload, ToolResultPayload):
        return {
            "tool_name": payload.tool_name,
            "result": payload.result,
        }

    if isinstance(payload, ValidationErrorPayload):
        return {
            "error_type": payload.error_type,
            "message": payload.message,
            "raw_model_output": payload.raw_model_output,
        }

    if isinstance(payload, RepairAttemptPayload):
        return {
            "original_output": payload.original_output,
            "repaired_output": payload.repaired_output,
            "repairs_applied": list(payload.repairs_applied),
            "succeeded": payload.succeeded,
        }

    if isinstance(payload, RejectPayload):
        return {
            "reason": payload.reason,
            "raw_model_output": payload.raw_model_output,
            "repairs_attempted": list(payload.repairs_attempted),
        }

    if isinstance(payload, TerminalOutcomePayload):
        return {
            "reason": payload.reason,
            "final_answer": payload.final_answer,
            "error_message": payload.error_message,
        }

    # Plain string or dict payloads (user_task, model_thought, agent_message)
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        return payload

    # Fallback: try to convert to dict
    try:
        return asdict(payload)
    except TypeError:
        return str(payload)


def event_to_dict(event: Event) -> dict[str, Any]:
    """Serialize one Event to a plain dict."""
    return {
        "event_type": event.event_type.value,
        "step_index": event.step_index,
        "payload": _serialize_payload(event),
        "reward": event.reward,
        "metadata": event.metadata,
    }


# -- Episode serialization ----------------------------------------------------

def episode_to_dict(episode: Episode) -> dict[str, Any]:
    """Convert an Episode to a serializable dict.

    This is the canonical serialization format. It preserves:
        - All events with typed payloads flattened to dicts
        - Exact turn order (events list order is stable)
        - Terminal outcome (duplicated at top level for convenience)
        - Episode metrics
        - Environment initial state
        - All metadata
    """
    terminal_dict = None
    if episode.terminal is not None:
        terminal_dict = {
            "reason": episode.terminal.reason,
            "final_answer": episode.terminal.final_answer,
            "error_message": episode.terminal.error_message,
        }

    return {
        "task_id": episode.task_id,
        "task_prompt": episode.task_prompt,
        "model_id": episode.model_id,
        "env_state_init": episode.env_state_init,
        "events": [event_to_dict(e) for e in episode.events],
        "terminal": terminal_dict,
        "metrics": {
            "total_steps": episode.metrics.total_steps,
            "valid_tool_calls": episode.metrics.valid_tool_calls,
            "invalid_tool_calls": episode.metrics.invalid_tool_calls,
            "repair_attempts": episode.metrics.repair_attempts,
            "repair_successes": episode.metrics.repair_successes,
            "rejects": episode.metrics.rejects,
            "model_calls": episode.metrics.model_calls,
            "wall_time_seconds": episode.metrics.wall_time_seconds,
            "total_reward": episode.metrics.total_reward,
        },
        "metadata": episode.metadata,
    }


def episode_to_jsonl(episode: Episode) -> str:
    """Serialize an Episode to a single JSONL line."""
    return json.dumps(episode_to_dict(episode))


# -- Deserialization ----------------------------------------------------------

def _deserialize_payload(event_type: EventType, raw: Any) -> Any:
    """Reconstruct a typed payload from a serialized form."""
    if event_type == EventType.TOOL_CALL and isinstance(raw, dict):
        return ToolCallPayload(
            tool_name=raw["tool_name"],
            arguments=raw["arguments"],
            thought=raw.get("thought"),
            raw_model_output=raw.get("raw_model_output", ""),
        )

    if event_type == EventType.TOOL_RESULT and isinstance(raw, dict):
        return ToolResultPayload(
            tool_name=raw["tool_name"],
            result=raw["result"],
        )

    if event_type == EventType.TOOL_VALIDATION_ERROR and isinstance(raw, dict):
        return ValidationErrorPayload(
            error_type=raw["error_type"],
            message=raw["message"],
            raw_model_output=raw.get("raw_model_output", ""),
        )

    if event_type == EventType.TOOL_REPAIR_ATTEMPT and isinstance(raw, dict):
        return RepairAttemptPayload(
            original_output=raw["original_output"],
            repaired_output=raw.get("repaired_output"),
            repairs_applied=raw.get("repairs_applied", []),
            succeeded=raw.get("succeeded", False),
        )

    if event_type == EventType.TOOL_REJECT and isinstance(raw, dict):
        return RejectPayload(
            reason=raw["reason"],
            raw_model_output=raw.get("raw_model_output", ""),
            repairs_attempted=raw.get("repairs_attempted", []),
        )

    if event_type == EventType.TERMINAL_OUTCOME and isinstance(raw, dict):
        return TerminalOutcomePayload(
            reason=raw["reason"],
            final_answer=raw.get("final_answer"),
            error_message=raw.get("error_message"),
        )

    # Plain string or dict (user_task, model_thought, agent_message)
    return raw


def dict_to_event(d: dict[str, Any]) -> Event:
    """Reconstruct an Event from a serialized dict."""
    event_type = EventType(d["event_type"])
    payload = _deserialize_payload(event_type, d["payload"])

    return Event(
        event_type=event_type,
        step_index=d["step_index"],
        payload=payload,
        reward=d.get("reward"),
        metadata=d.get("metadata", {}),
    )


def dict_to_episode(d: dict[str, Any]) -> Episode:
    """Reconstruct an Episode from a serialized dict."""
    events = [dict_to_event(e) for e in d.get("events", [])]

    # Reconstruct terminal payload
    terminal = None
    terminal_raw = d.get("terminal")
    if terminal_raw is not None:
        terminal = TerminalOutcomePayload(
            reason=terminal_raw["reason"],
            final_answer=terminal_raw.get("final_answer"),
            error_message=terminal_raw.get("error_message"),
        )

    # Reconstruct metrics
    metrics_raw = d.get("metrics", {})
    metrics = EpisodeMetrics(
        total_steps=metrics_raw.get("total_steps", 0),
        valid_tool_calls=metrics_raw.get("valid_tool_calls", 0),
        invalid_tool_calls=metrics_raw.get("invalid_tool_calls", 0),
        repair_attempts=metrics_raw.get("repair_attempts", 0),
        repair_successes=metrics_raw.get("repair_successes", 0),
        rejects=metrics_raw.get("rejects", 0),
        model_calls=metrics_raw.get("model_calls", 0),
        wall_time_seconds=metrics_raw.get("wall_time_seconds", 0.0),
        total_reward=metrics_raw.get("total_reward", 0.0),
    )

    return Episode(
        task_id=d["task_id"],
        task_prompt=d.get("task_prompt", ""),
        model_id=d.get("model_id", ""),
        env_state_init=d.get("env_state_init", {}),
        events=events,
        terminal=terminal,
        metrics=metrics,
        metadata=d.get("metadata", {}),
    )


def jsonl_to_episode(line: str) -> Episode:
    """Deserialize a single JSONL line into an Episode."""
    return dict_to_episode(json.loads(line))


# -- File I/O -----------------------------------------------------------------

def save_episodes_jsonl(episodes: list[Episode], path: str) -> None:
    """Write a list of Episodes to a JSONL file.

    Each line is one complete episode with all events, rewards,
    and metadata preserved.

    Args:
        episodes: List of episodes to write.
        path: Output file path.
    """
    with open(path, "w") as f:
        for ep in episodes:
            f.write(episode_to_jsonl(ep) + "\n")


def load_episodes_jsonl(path: str) -> list[Episode]:
    """Load Episodes from a JSONL file.

    Args:
        path: Input file path.

    Returns:
        List of deserialized Episodes.
    """
    episodes: list[Episode] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(jsonl_to_episode(line))
    return episodes
