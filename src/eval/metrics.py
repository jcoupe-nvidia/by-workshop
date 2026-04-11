"""
Sequence-sensitive evaluators and scoring helpers for canonical Episode traces.

Evaluates agent trajectories across seven dimensions:
    - Skill selection quality:  Did the agent pick the right skills in order?
    - Tool validity:            Were all tool calls well-formed?
    - Tool accuracy:            Did tool arguments match expected values?
    - Sequence correctness:     Were dependency constraints respected?
    - Task success:             Did the agent reach a valid recommendation?
    - Recovery quality:         If fallback was needed, was the repair correct?
    - Efficiency:               How many steps vs. the optimal trajectory?

These evaluators consume canonical ``Episode`` records from
``src.rollouts.trace_types`` rather than the backward-compatible ``AgentTrace``.
Reward-relevant constants (expected arguments, optimal tool sequence) are
sourced from ``src.envs.rewards`` — the single authority for scenario-specific
ground truth used by both training and evaluation.

The sequence correctness evaluator is the most important for this workshop:
it checks that tools were called in an order consistent with the dependency
graph defined in ``src.runtime.tools.TOOL_DEPENDENCIES``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.rollouts.trace_types import (
    Episode,
    Event,
    EventType,
    ToolCallPayload,
    RepairAttemptPayload,
)
from src.envs.state import TOOL_DEPENDENCIES
from src.envs.state import SUBGOAL_ORDER, TOOL_TO_SUBGOAL, Subgoal
from src.envs.rewards import EXPECTED_ARGUMENTS, OPTIMAL_TOOL_SEQUENCE


# -- Evaluation dimensions -------------------------------------------------

EVAL_DIMENSIONS = [
    "skill_selection",
    "tool_validity",
    "tool_accuracy",
    "sequence_correctness",
    "task_success",
    "recovery_quality",
    "efficiency",
]

DIMENSION_WEIGHTS: dict[str, float] = {
    "skill_selection":      0.15,
    "tool_validity":        0.15,
    "tool_accuracy":        0.10,
    "sequence_correctness": 0.25,
    "task_success":         0.15,
    "recovery_quality":     0.10,
    "efficiency":           0.10,
}

PASS_THRESHOLD = 0.6


# -- Score types -----------------------------------------------------------

@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    dimension: str
    score: float        # 0.0 to 1.0
    max_score: float    # always 1.0 for normalized scores
    details: str        # human-readable explanation


@dataclass
class TrajectoryEvaluation:
    """Full evaluation of one agent trajectory."""
    scores: list[DimensionScore]
    overall: float              # weighted average
    passed: bool                # meets minimum threshold
    summary: str                # one-line summary


# -- Individual evaluators -------------------------------------------------

def eval_skill_selection(episode: Episode) -> DimensionScore:
    """Did the agent call tools that map to the correct subgoal sequence?

    Uses the environment-owned SUBGOAL_ORDER and TOOL_TO_SUBGOAL mappings
    (from envs.state) as the single source of truth for task progression,
    rather than maintaining a separate ordering in the eval layer.
    """
    tool_names = episode.tool_names_called
    if not tool_names:
        return DimensionScore("skill_selection", 0.0, 1.0, "No valid tool calls.")

    # Map each tool call to its subgoal using the env-owned mapping
    seen: set[Subgoal] = set()
    implied_subgoals: list[Subgoal] = []
    for t in tool_names:
        sg = TOOL_TO_SUBGOAL.get(t)
        if sg and sg not in seen:
            seen.add(sg)
            implied_subgoals.append(sg)

    # Score: fraction of canonical subgoal order covered in the right order
    expected = SUBGOAL_ORDER
    matched = 0
    exp_idx = 0
    for sg in implied_subgoals:
        if exp_idx < len(expected) and sg == expected[exp_idx]:
            matched += 1
            exp_idx += 1

    score = matched / len(expected) if expected else 0.0
    implied_names = [sg.value for sg in implied_subgoals]
    expected_names = [sg.value for sg in expected]
    details = (
        f"Implied subgoals: {implied_names}. "
        f"Expected: {expected_names}. "
        f"Matched {matched}/{len(expected)} in order."
    )
    return DimensionScore("skill_selection", round(score, 3), 1.0, details)


def eval_tool_validity(episode: Episode) -> DimensionScore:
    """Were all tool calls well-formed (valid JSON, known tool, correct args)?

    Scores as the fraction of tool-related events (calls + validation errors)
    that were successful tool calls.
    """
    valid_calls = episode.metrics.valid_tool_calls
    invalid_calls = episode.metrics.invalid_tool_calls
    total = valid_calls + invalid_calls

    if total == 0:
        return DimensionScore("tool_validity", 0.0, 1.0, "No tool calls recorded.")

    score = valid_calls / total
    details = f"{valid_calls}/{total} tool calls valid. {invalid_calls} invalid."
    return DimensionScore("tool_validity", round(score, 3), 1.0, details)


def eval_tool_accuracy(episode: Episode) -> DimensionScore:
    """Did tool arguments match the expected values for the scenario?

    Only checks tools that have expected arguments defined in
    ``envs.rewards.EXPECTED_ARGUMENTS``. Scores as the fraction of checked
    tools whose arguments matched.
    """
    tool_names = episode.tool_names_called
    if not tool_names:
        return DimensionScore("tool_accuracy", 0.0, 1.0, "No valid tool calls.")

    checked = 0
    correct = 0
    mismatches: list[str] = []

    for event in episode.tool_calls:
        payload = event.payload
        if not isinstance(payload, ToolCallPayload):
            continue
        expected = EXPECTED_ARGUMENTS.get(payload.tool_name)
        if expected is None:
            continue
        checked += 1
        all_match = True
        for k, v in expected.items():
            actual = payload.arguments.get(k)
            if str(actual) != str(v):
                all_match = False
                mismatches.append(
                    f"{payload.tool_name}.{k}: expected {v!r}, got {actual!r}"
                )
        if all_match:
            correct += 1

    if checked == 0:
        return DimensionScore("tool_accuracy", 1.0, 1.0, "No checkable tools called.")

    score = correct / checked
    details = f"{correct}/{checked} tools had correct arguments."
    if mismatches:
        details += f" Mismatches: {mismatches}"
    return DimensionScore("tool_accuracy", round(score, 3), 1.0, details)


def eval_sequence_correctness(episode: Episode) -> DimensionScore:
    """Were dependency constraints respected in the tool call order?

    This is the most important evaluator for the workshop. It walks the
    tool call sequence and checks that every tool's prerequisites (from
    TOOL_DEPENDENCIES) were called earlier in the sequence.
    """
    tool_names = episode.tool_names_called
    if not tool_names:
        return DimensionScore("sequence_correctness", 0.0, 1.0, "No valid tool calls.")

    called_so_far: set[str] = set()
    total_checks = 0
    violations: list[str] = []

    for tool in tool_names:
        deps = TOOL_DEPENDENCIES.get(tool, set())
        for dep in deps:
            total_checks += 1
            if dep not in called_so_far:
                violations.append(f"{tool} called before prerequisite {dep}")
        called_so_far.add(tool)

    if total_checks == 0:
        return DimensionScore(
            "sequence_correctness", 1.0, 1.0,
            "No dependency checks needed (only root tools called)."
        )

    satisfied = total_checks - len(violations)
    score = satisfied / total_checks
    details = f"{satisfied}/{total_checks} dependency checks passed."
    if violations:
        details += f" Violations: {violations}"
    return DimensionScore("sequence_correctness", round(score, 3), 1.0, details)


def eval_task_success(episode: Episode) -> DimensionScore:
    """Did the agent reach a valid final recommendation?

    Checks: episode completed, final_answer present, required fields exist.
    """
    if not episode.is_complete or episode.final_answer is None:
        reason = "incomplete" if not episode.is_complete else "no final answer"
        return DimensionScore("task_success", 0.0, 1.0, f"Agent did not complete: {reason}.")

    answer = episode.final_answer
    required_fields = {"action", "rationale"}
    present = set(answer.keys()) & required_fields
    missing = required_fields - present

    if missing:
        score = len(present) / len(required_fields)
        details = f"Final answer missing fields: {missing}."
        return DimensionScore("task_success", round(score, 3), 1.0, details)

    # Bonus: check if confidence and delivery date are present
    has_confidence = "confidence" in answer
    has_delivery = "expected_delivery" in answer
    bonus_count = sum([has_confidence, has_delivery])

    score = 1.0
    details = f"Valid final answer with action='{answer.get('action', '?')}'."
    if bonus_count < 2:
        extras = []
        if not has_confidence:
            extras.append("confidence")
        if not has_delivery:
            extras.append("expected_delivery")
        details += f" Missing optional: {extras}."
    return DimensionScore("task_success", score, 1.0, details)


def eval_recovery_quality(episode: Episode) -> DimensionScore:
    """If fallback repairs were applied, were they followed by successful execution?

    Scores based on whether repaired tool calls led to valid results.
    If no fallbacks were needed, returns a perfect score.
    """
    repair_events = [
        e for e in episode.events
        if e.event_type == EventType.TOOL_REPAIR_ATTEMPT
    ]
    reject_count = episode.metrics.rejects

    if not repair_events:
        if reject_count > 0:
            return DimensionScore(
                "recovery_quality", 0.5, 1.0,
                f"{reject_count} fallback rejects, 0 repairs. "
                "Agent recovered by retrying after error feedback."
            )
        return DimensionScore(
            "recovery_quality", 1.0, 1.0,
            "No fallback repairs needed."
        )

    successful_repairs = sum(
        1 for e in repair_events
        if isinstance(e.payload, RepairAttemptPayload) and e.payload.succeeded
    )
    total_repairs = len(repair_events)
    score = successful_repairs / total_repairs
    details = (
        f"{successful_repairs}/{total_repairs} repaired calls succeeded. "
        f"Total rejects: {reject_count}."
    )
    return DimensionScore("recovery_quality", round(score, 3), 1.0, details)


def eval_efficiency(episode: Episode) -> DimensionScore:
    """How many tool-call attempts did the agent make vs. the optimal trajectory?

    The optimal SO-10482 trajectory uses 9 tool calls (all tools once).
    Extra attempts (retries, invalid calls) reduce the score.

    Compares optimal tool-call count against total tool-call attempts
    (valid + invalid) to keep units consistent.  metrics.total_steps is
    the raw event count (which includes results, thoughts, etc.) and
    should not be compared against a tool-call count.
    """
    optimal_count = len(OPTIMAL_TOOL_SEQUENCE)
    valid_calls = episode.metrics.valid_tool_calls
    invalid_calls = episode.metrics.invalid_tool_calls
    total_attempts = valid_calls + invalid_calls

    if total_attempts == 0:
        return DimensionScore("efficiency", 0.0, 1.0, "No tool call attempts.")

    # Ratio of optimal to actual attempts (capped at 1.0)
    # Uses total tool-call attempts (valid + invalid) so the units match
    score = min(1.0, optimal_count / total_attempts)

    details = (
        f"{valid_calls} valid tool calls, {invalid_calls} invalid, "
        f"{total_attempts} total attempts "
        f"(optimal: {optimal_count}). "
        f"Efficiency ratio: {score:.2f}."
    )
    return DimensionScore("efficiency", round(score, 3), 1.0, details)


# -- Main evaluation entry point -------------------------------------------

EVALUATORS: dict[str, Any] = {
    "skill_selection": eval_skill_selection,
    "tool_validity": eval_tool_validity,
    "tool_accuracy": eval_tool_accuracy,
    "sequence_correctness": eval_sequence_correctness,
    "task_success": eval_task_success,
    "recovery_quality": eval_recovery_quality,
    "efficiency": eval_efficiency,
}


def evaluate_trajectory(
    episode: Episode,
    threshold: float = PASS_THRESHOLD,
) -> TrajectoryEvaluation:
    """Run all evaluators against an Episode and produce a full evaluation.

    Args:
        episode: The canonical Episode to evaluate.
        threshold: Minimum overall score to pass.

    Returns:
        TrajectoryEvaluation with per-dimension scores and an overall result.
    """
    scores: list[DimensionScore] = []
    for dim in EVAL_DIMENSIONS:
        evaluator = EVALUATORS[dim]
        scores.append(evaluator(episode))

    # Weighted average
    overall = sum(
        s.score * DIMENSION_WEIGHTS[s.dimension]
        for s in scores
    )
    overall = round(overall, 3)

    passed = overall >= threshold

    # Build summary
    perfect = [s.dimension for s in scores if s.score == 1.0]
    weak = [s.dimension for s in scores if s.score < 0.5]
    summary = f"Overall: {overall:.2f} ({'PASS' if passed else 'FAIL'})."
    if weak:
        summary += f" Weak: {', '.join(weak)}."
    if perfect:
        summary += f" Perfect: {', '.join(perfect)}."

    return TrajectoryEvaluation(
        scores=scores,
        overall=overall,
        passed=passed,
        summary=summary,
    )


# -- NAT ATIF trajectory evaluation ----------------------------------------

def evaluate_atif_trajectory(
    atif_trajectory: Any,
    threshold: float = PASS_THRESHOLD,
) -> TrajectoryEvaluation:
    """Evaluate a NAT ATIF Trajectory by converting it to a canonical Episode.

    Accepts a nat.atif.trajectory.Trajectory, reconstructs a minimal Episode
    from ATIF steps, and runs the standard evaluators.

    Args:
        atif_trajectory: A nat.atif.trajectory.Trajectory object.
        threshold: Minimum overall score to pass.

    Returns:
        TrajectoryEvaluation with per-dimension scores.
    """
    from nat.atif.trajectory import Trajectory as ATIFTrajectory

    if not isinstance(atif_trajectory, ATIFTrajectory):
        raise TypeError(f"Expected nat.atif.trajectory.Trajectory, got {type(atif_trajectory)}")

    episode = _atif_to_episode(atif_trajectory)
    return evaluate_trajectory(episode, threshold=threshold)


def _atif_to_episode(atif_trajectory: Any) -> Episode:
    """Reconstruct a minimal Episode from a NAT ATIF Trajectory.

    Extracts tool calls and terminal outcomes from ATIF steps to build
    an Episode with enough structure for the evaluators to score.
    """
    from src.rollouts.trace_types import (
        EpisodeMetrics,
        Event,
        EventType,
        ToolCallPayload,
        ToolResultPayload,
        TerminalOutcomePayload,
    )

    events: list[Event] = []
    step_idx = 0
    valid_calls = 0
    invalid_calls = 0

    task_id = atif_trajectory.extra.get("task_id", "") if atif_trajectory.extra else ""
    task_prompt = atif_trajectory.extra.get("task_prompt", "") if atif_trajectory.extra else ""

    for atif_step in atif_trajectory.steps:
        source = atif_step.source

        # Tool calls from agent steps
        if source == "agent" and atif_step.tool_calls:
            for tc in atif_step.tool_calls:
                events.append(Event(
                    event_type=EventType.TOOL_CALL,
                    step_index=step_idx,
                    payload=ToolCallPayload(
                        tool_name=tc.function_name,
                        arguments=tc.arguments if isinstance(tc.arguments, dict) else {},
                        thought=atif_step.reasoning_content,
                        raw_model_output=atif_step.message or "",
                    ),
                    reward=atif_step.extra.get("reward") if atif_step.extra else None,
                ))
                valid_calls += 1
                step_idx += 1

        # Tool results from system observation steps
        elif source == "system" and atif_step.observation:
            for obs_result in atif_step.observation.results:
                import json as _json
                try:
                    result_data = _json.loads(obs_result.content) if obs_result.content else {}
                except (ValueError, TypeError):
                    result_data = {"raw": obs_result.content}
                tool_name = atif_step.extra.get("tool_name", "") if atif_step.extra else ""
                events.append(Event(
                    event_type=EventType.TOOL_RESULT,
                    step_index=step_idx,
                    payload=ToolResultPayload(
                        tool_name=tool_name,
                        result=result_data,
                    ),
                ))
                step_idx += 1

        # Terminal agent message (final answer)
        elif source == "agent" and not atif_step.tool_calls and atif_step.message:
            import json as _json
            try:
                final_answer = _json.loads(atif_step.message)
                if isinstance(final_answer, dict) and "action" in final_answer:
                    events.append(Event(
                        event_type=EventType.TERMINAL_OUTCOME,
                        step_index=step_idx,
                        payload=TerminalOutcomePayload(
                            reason="final_answer",
                            final_answer=final_answer,
                        ),
                        reward=atif_step.extra.get("reward") if atif_step.extra else None,
                    ))
                    step_idx += 1
            except (ValueError, TypeError):
                pass

        # Validation error system messages
        elif source == "system" and not atif_step.observation:
            if atif_step.message and "error" in atif_step.message.lower():
                invalid_calls += 1

    # Extract metrics from ATIF final_metrics
    extra = atif_trajectory.final_metrics.extra if atif_trajectory.final_metrics else {}
    total_reward = extra.get("total_reward", 0.0) if extra else 0.0
    model_calls = extra.get("model_calls", 0) if extra else 0
    repair_attempts = extra.get("repair_attempts", 0) if extra else 0
    rejects = extra.get("rejects", 0) if extra else 0

    # Find terminal
    terminal = None
    for event in reversed(events):
        if event.event_type == EventType.TERMINAL_OUTCOME:
            terminal = event.payload
            break

    metrics = EpisodeMetrics(
        total_steps=step_idx,
        valid_tool_calls=valid_calls,
        invalid_tool_calls=invalid_calls,
        repair_attempts=repair_attempts,
        repair_successes=0,
        rejects=rejects,
        model_calls=model_calls,
        wall_time_seconds=extra.get("wall_time_seconds", 0.0) if extra else 0.0,
        total_reward=total_reward,
    )

    return Episode(
        task_id=task_id,
        task_prompt=task_prompt,
        model_id=atif_trajectory.agent.model_name if atif_trajectory.agent else "",
        events=events,
        terminal=terminal,
        metrics=metrics,
    )


# -- NeMo Gym result row evaluation ----------------------------------------

def evaluate_nemo_gym_result(
    result_row: Any,
    threshold: float = PASS_THRESHOLD,
) -> dict[str, Any]:
    """Produce a lightweight evaluation summary from a NeMo Gym result row.

    NeMo Gym result rows (NemoGymResultRow) contain pre-computed reward
    components from the environment. This function maps those components
    to a *subset* of the canonical eval dimensions without re-running the
    full evaluators.

    Because ``skill_selection`` and ``recovery_quality`` are not available
    from the result row, the weights are renormalized over the available
    dimensions so that ``partial_overall`` is on the same 0-1 scale as
    ``evaluate_trajectory().overall`` for the covered dimensions. The two
    scores are still not directly comparable — use ``evaluate_trajectory``
    for full regression tracking.

    Args:
        result_row: A NemoGymResultRow (from envs.nemo_gym_adapter).
        threshold: Minimum overall score to pass.

    Returns:
        Dict with dimension scores and a partial pass/fail.
    """
    from src.envs.nemo_gym_adapter import NemoGymResultRow

    if not isinstance(result_row, NemoGymResultRow):
        raise TypeError(f"Expected NemoGymResultRow, got {type(result_row)}")

    total_calls = result_row.valid_tool_calls + result_row.invalid_tool_calls
    tool_validity = (
        result_row.valid_tool_calls / total_calls if total_calls > 0 else 0.0
    )

    # Map NeMo Gym reward components to the eval dimensions that are
    # available from the result row.  skill_selection and recovery_quality
    # are not computable here — they require the full event stream.
    scores = {
        "tool_validity": round(tool_validity, 3),
        "tool_accuracy": round(result_row.avg_correct_arguments, 3),
        "sequence_correctness": round(result_row.avg_dependency_satisfied, 3),
        "task_success": float(result_row.task_success),
        "efficiency": round(result_row.avg_efficiency, 3),
    }

    # Renormalize canonical weights to the available dimensions so the
    # partial overall is on a comparable 0-1 scale.
    available_weight_sum = sum(
        DIMENSION_WEIGHTS.get(dim, 0.0) for dim in scores
    )
    if available_weight_sum > 0:
        partial_overall = sum(
            scores[dim] * DIMENSION_WEIGHTS.get(dim, 0.0)
            for dim in scores
        ) / available_weight_sum
    else:
        partial_overall = 0.0
    partial_overall = round(partial_overall, 3)

    return {
        "order_id": result_row.order_id,
        "dimension_scores": scores,
        "dimensions_missing": sorted(
            set(EVAL_DIMENSIONS) - set(scores.keys())
        ),
        "total_reward": result_row.total_reward,
        "avg_step_reward": result_row.avg_step_reward,
        "partial_overall": partial_overall,
        "passed": partial_overall >= threshold,
        "note": (
            "Partial evaluation — missing dimensions: "
            f"{sorted(set(EVAL_DIMENSIONS) - set(scores.keys()))}. "
            "Use evaluate_trajectory() for full regression tracking."
        ),
    }
