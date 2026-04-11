"""
Backward-compatibility shim for src.evaluation.

The canonical definitions now live in:
    - src.eval.metrics   (evaluators, scoring types, constants)
    - src.eval.reports   (display and report helpers)

This module re-exports everything so existing imports continue to work.
It also provides an adapter so code that passes the backward-compatible
AgentTrace type to evaluate_trajectory() keeps working during the
transition to canonical Episode traces.
"""
from __future__ import annotations

from typing import Any

from src.runtime.agent import AgentTrace
from src.rollouts.trace_types import (
    Episode,
    Event,
    EventType,
    EpisodeMetrics,
    ToolCallPayload,
    ToolResultPayload,
    TerminalOutcomePayload,
)

# Re-export canonical evaluation API
from src.eval.metrics import (  # noqa: F401
    EVAL_DIMENSIONS,
    DIMENSION_WEIGHTS,
    PASS_THRESHOLD,
    DimensionScore,
    TrajectoryEvaluation,
    EVALUATORS,
    eval_skill_selection,
    eval_tool_validity,
    eval_tool_accuracy,
    eval_sequence_correctness,
    eval_task_success,
    eval_recovery_quality,
    eval_efficiency,
)
from src.eval.metrics import evaluate_trajectory as _evaluate_episode
from src.eval.reports import print_evaluation  # noqa: F401

# Re-export scenario constants from their canonical home (envs.rewards)
from src.envs.rewards import (  # noqa: F401
    EXPECTED_ARGUMENTS,
    OPTIMAL_TOOL_SEQUENCE,
)


# -- AgentTrace -> Episode adapter -----------------------------------------

def _agent_trace_to_episode(trace: AgentTrace) -> Episode:
    """Convert a backward-compatible AgentTrace to a canonical Episode."""
    events: list[Event] = []
    step_idx = 0

    # Record each step as tool call + result events (or validation errors)
    for record in trace.steps:
        if record.valid:
            events.append(Event(
                event_type=EventType.TOOL_CALL,
                step_index=step_idx,
                payload=ToolCallPayload(
                    tool_name=record.tool_name,
                    arguments=record.arguments,
                    thought=record.thought,
                ),
            ))
            step_idx += 1
            events.append(Event(
                event_type=EventType.TOOL_RESULT,
                step_index=step_idx,
                payload=ToolResultPayload(
                    tool_name=record.tool_name,
                    result=record.result,
                ),
            ))
            step_idx += 1
        else:
            events.append(Event(
                event_type=EventType.TOOL_VALIDATION_ERROR,
                step_index=step_idx,
                payload={"error": record.validation_error or "unknown"},
            ))
            step_idx += 1

    # Terminal outcome
    terminal = None
    if trace.stop_reason:
        terminal = TerminalOutcomePayload(
            reason=trace.stop_reason,
            final_answer=trace.final_answer,
        )
        events.append(Event(
            event_type=EventType.TERMINAL_OUTCOME,
            step_index=step_idx,
            payload=terminal,
        ))

    valid_count = sum(1 for s in trace.steps if s.valid)
    invalid_count = sum(1 for s in trace.steps if not s.valid)

    metrics = EpisodeMetrics(
        total_steps=valid_count + invalid_count,
        valid_tool_calls=valid_count,
        invalid_tool_calls=invalid_count,
        repair_attempts=trace.fallback_repairs,
        repair_successes=trace.fallback_repairs,  # in AgentTrace, repairs = successes
        rejects=trace.fallback_rejects,
        model_calls=trace.model_calls,
        wall_time_seconds=trace.wall_time_seconds,
    )

    return Episode(
        task_id=trace.task,
        task_prompt=trace.task,
        events=events,
        terminal=terminal,
        metrics=metrics,
    )


def evaluate_trajectory(
    trace: AgentTrace | Episode,
    threshold: float = PASS_THRESHOLD,
) -> TrajectoryEvaluation:
    """Run all evaluators against a trace and produce a full evaluation.

    Accepts either the canonical Episode or the backward-compatible AgentTrace.
    AgentTrace inputs are converted to Episode automatically.
    """
    if isinstance(trace, AgentTrace):
        episode = _agent_trace_to_episode(trace)
    else:
        episode = trace
    return _evaluate_episode(episode, threshold=threshold)
