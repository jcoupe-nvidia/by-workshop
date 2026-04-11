"""
ATIF trajectory adapter: converts Episode to NAT ATIF format.

ATIF (Agent Trajectory Interchange Format) is NAT's canonical trajectory
schema. This module converts our Episode/Event types into ATIF Trajectory,
Step, ToolCall, and Observation records so NAT's finetuning and evaluation
infrastructure can consume them directly.

Owns:
    - Episode-to-ATIF Trajectory conversion
    - Event-to-ATIF Step mapping
    - OpenAI-style tool definition export for ATIF agent metadata

Does NOT own:
    - Episode structure or event types (see rollouts.trace_types)
    - Tool implementations (see runtime.tools)
    - Reward semantics or training concerns
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from nat.atif.agent import Agent as ATIFAgent
from nat.atif.final_metrics import FinalMetrics
from nat.atif.observation import Observation
from nat.atif.observation_result import ObservationResult
from nat.atif.step import Step as ATIFStep
from nat.atif.tool_call import ToolCall as ATIFToolCall
from nat.atif.trajectory import Trajectory

from src.rollouts.trace_types import (
    Episode,
    Event,
    EventType,
    ToolCallPayload,
    ToolResultPayload,
    ValidationErrorPayload,
    RepairAttemptPayload,
    RejectPayload,
    TerminalOutcomePayload,
)
from src.runtime.nat_tools import build_openai_tool_definitions


def episode_to_atif(
    episode: Episode,
    agent_name: str = "by-workshop-agent",
    agent_version: str = "0.1.0",
) -> Trajectory:
    """Convert an Episode to an ATIF Trajectory.

    Maps episode events to ATIF steps following this correspondence:
        USER_TASK             -> Step(source="user", message=prompt)
        MODEL_THOUGHT         -> Step(source="agent", reasoning_content=thought)
        TOOL_CALL             -> Step(source="agent", tool_calls=[...])
        TOOL_RESULT           -> Step(source="system", observation=...)
        TOOL_VALIDATION_ERROR -> Step(source="system", message=error_text)
        TOOL_REPAIR_ATTEMPT   -> Step(source="system", message=repair_info)
        TOOL_REJECT           -> Step(source="system", message=reject_reason)
        TERMINAL_OUTCOME      -> Step(source="agent" or "system", message=outcome)
    """
    agent = ATIFAgent(
        name=agent_name,
        version=agent_version,
        model_name=episode.model_id or "nvidia/nemotron-3-nano",
        tool_definitions=build_openai_tool_definitions(),
        extra={"task_id": episode.task_id},
    )

    steps: list[ATIFStep] = []
    step_id = 1
    # Track tool_call_ids so observations can reference them
    last_tool_call_id: str | None = None

    for event in episode.events:
        atif_step = _event_to_step(event, step_id, last_tool_call_id)
        if atif_step is None:
            continue

        steps.append(atif_step)

        # Update last_tool_call_id for TOOL_CALL events
        if event.event_type == EventType.TOOL_CALL:
            if atif_step.tool_calls:
                last_tool_call_id = atif_step.tool_calls[0].tool_call_id
        elif event.event_type == EventType.TOOL_RESULT:
            last_tool_call_id = None

        step_id += 1

    final_metrics = FinalMetrics(
        total_steps=episode.metrics.total_steps,
        extra={
            "valid_tool_calls": episode.metrics.valid_tool_calls,
            "invalid_tool_calls": episode.metrics.invalid_tool_calls,
            "repair_attempts": episode.metrics.repair_attempts,
            "repair_successes": episode.metrics.repair_successes,
            "rejects": episode.metrics.rejects,
            "model_calls": episode.metrics.model_calls,
            "wall_time_seconds": episode.metrics.wall_time_seconds,
            "total_reward": episode.metrics.total_reward,
        },
    )

    return Trajectory(
        agent=agent,
        steps=steps,
        final_metrics=final_metrics,
        extra={
            "task_id": episode.task_id,
            "task_prompt": episode.task_prompt,
            "env_state_init": episode.env_state_init,
        },
    )


def _event_to_step(
    event: Event,
    step_id: int,
    last_tool_call_id: str | None,
) -> ATIFStep | None:
    """Convert a single Event to an ATIF Step."""
    timestamp = datetime.now(timezone.utc).isoformat()
    extra: dict[str, Any] = {}

    if event.reward is not None:
        extra["reward"] = event.reward
    if event.metadata:
        extra.update(event.metadata)

    payload = event.payload

    if event.event_type == EventType.USER_TASK:
        message = payload if isinstance(payload, str) else str(payload)
        return ATIFStep(
            step_id=step_id,
            timestamp=timestamp,
            source="user",
            message=message,
            extra=extra or None,
        )

    if event.event_type == EventType.MODEL_THOUGHT:
        thought = payload if isinstance(payload, str) else str(payload)
        return ATIFStep(
            step_id=step_id,
            timestamp=timestamp,
            source="agent",
            message="",
            reasoning_content=thought,
            extra=extra or None,
        )

    if event.event_type == EventType.TOOL_CALL:
        if not isinstance(payload, ToolCallPayload):
            return None
        tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
        tool_call = ATIFToolCall(
            tool_call_id=tool_call_id,
            function_name=payload.tool_name,
            arguments=payload.arguments,
        )
        return ATIFStep(
            step_id=step_id,
            timestamp=timestamp,
            source="agent",
            message=payload.raw_model_output or "",
            reasoning_content=payload.thought,
            tool_calls=[tool_call],
            extra=extra or None,
        )

    if event.event_type == EventType.TOOL_RESULT:
        if not isinstance(payload, ToolResultPayload):
            return None
        # ATIF validator requires source_call_id to reference a tool_call_id
        # from the same step. Since tool results are separate system steps,
        # we omit source_call_id and use extra metadata to link them.
        obs_result = ObservationResult(
            source_call_id=None,
            content=json.dumps(payload.result),
        )
        result_extra = {**(extra or {}), "tool_name": payload.tool_name}
        if last_tool_call_id:
            result_extra["related_call_id"] = last_tool_call_id
        return ATIFStep(
            step_id=step_id,
            timestamp=timestamp,
            source="system",
            message="",
            observation=Observation(results=[obs_result]),
            extra=result_extra,
        )

    if event.event_type == EventType.TOOL_VALIDATION_ERROR:
        if isinstance(payload, ValidationErrorPayload):
            message = f"Validation error ({payload.error_type}): {payload.message}"
        else:
            message = str(payload)
        return ATIFStep(
            step_id=step_id,
            timestamp=timestamp,
            source="system",
            message=message,
            extra=extra or None,
        )

    if event.event_type == EventType.TOOL_REPAIR_ATTEMPT:
        if isinstance(payload, RepairAttemptPayload):
            status = "succeeded" if payload.succeeded else "failed"
            message = (
                f"Repair attempt ({status}). "
                f"Repairs applied: {payload.repairs_applied}"
            )
        else:
            message = str(payload)
        return ATIFStep(
            step_id=step_id,
            timestamp=timestamp,
            source="system",
            message=message,
            extra=extra or None,
        )

    if event.event_type == EventType.TOOL_REJECT:
        if isinstance(payload, RejectPayload):
            message = f"Rejected: {payload.reason}"
        else:
            message = str(payload)
        return ATIFStep(
            step_id=step_id,
            timestamp=timestamp,
            source="system",
            message=message,
            extra=extra or None,
        )

    if event.event_type == EventType.TERMINAL_OUTCOME:
        if isinstance(payload, TerminalOutcomePayload):
            if payload.final_answer:
                message = json.dumps(payload.final_answer)
                source = "agent"
            elif payload.error_message:
                message = f"Error: {payload.error_message}"
                source = "system"
            else:
                message = f"Episode ended: {payload.reason}"
                source = "system"
        else:
            message = str(payload)
            source = "system"
        return ATIFStep(
            step_id=step_id,
            timestamp=timestamp,
            source=source,
            message=message,
            extra=extra or None,
        )

    if event.event_type == EventType.AGENT_MESSAGE:
        message = payload if isinstance(payload, str) else str(payload)
        return ATIFStep(
            step_id=step_id,
            timestamp=timestamp,
            source="agent",
            message=message,
            extra=extra or None,
        )

    return None
