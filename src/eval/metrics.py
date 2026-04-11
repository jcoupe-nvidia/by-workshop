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
from src.runtime.tools import TOOL_DEPENDENCIES
from src.runtime.workflows import WORKFLOW_TOOL_PATTERNS, WORKFLOW_ORDER
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
    """Did the agent call tools that map to the correct skill sequence?

    Maps each valid tool call to its parent workflow/skill and checks whether
    the implied skill order matches the canonical WORKFLOW_ORDER.
    """
    tool_names = episode.tool_names_called
    if not tool_names:
        return DimensionScore("skill_selection", 0.0, 1.0, "No valid tool calls.")

    # Build a reverse map: tool -> workflow
    tool_to_workflow: dict[str, str] = {}
    for workflow, tools in WORKFLOW_TOOL_PATTERNS.items():
        for t in tools:
            tool_to_workflow[t] = workflow

    # Extract the implied workflow sequence (deduplicated, in order of first appearance)
    seen: set[str] = set()
    implied_workflows: list[str] = []
    for t in tool_names:
        wf = tool_to_workflow.get(t)
        if wf and wf not in seen:
            seen.add(wf)
            implied_workflows.append(wf)

    # Score: fraction of canonical workflow order covered in the right order
    expected = WORKFLOW_ORDER
    matched = 0
    exp_idx = 0
    for wf in implied_workflows:
        if exp_idx < len(expected) and wf == expected[exp_idx]:
            matched += 1
            exp_idx += 1

    score = matched / len(expected) if expected else 0.0
    details = (
        f"Implied skills: {implied_workflows}. "
        f"Expected: {expected}. "
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
    """How many steps did the agent take vs. the optimal trajectory?

    The optimal SO-10482 trajectory uses 9 tool calls (all tools once).
    Extra steps (retries, invalid calls) reduce the score.
    """
    optimal_count = len(OPTIMAL_TOOL_SEQUENCE)
    valid_calls = episode.metrics.valid_tool_calls
    total_steps = episode.metrics.total_steps

    if total_steps == 0:
        return DimensionScore("efficiency", 0.0, 1.0, "No steps taken.")

    # Ratio of optimal to actual (capped at 1.0)
    # Using total_steps (not just valid) to penalize retries
    score = min(1.0, optimal_count / total_steps) if total_steps > 0 else 0.0

    details = (
        f"{valid_calls} valid tool calls, {total_steps} total steps "
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
