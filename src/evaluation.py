"""
Sequence-sensitive evaluators and scoring helpers.

Evaluates agent trajectories across multiple dimensions:
    - Skill selection quality:  Did the agent pick the right skills in order?
    - Tool validity:            Were all tool calls well-formed?
    - Tool accuracy:            Did tool arguments match expected values?
    - Sequence correctness:     Were dependency constraints respected?
    - Task success:             Did the agent reach a valid recommendation?
    - Recovery quality:         If fallback was needed, was the repair correct?
    - Efficiency:               How many steps vs. the optimal trajectory?

The sequence correctness evaluator is the most important for this workshop:
it checks that tools were called in an order consistent with the dependency
graph in tools.TOOL_DEPENDENCIES.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.tools import TOOL_REGISTRY, TOOL_DEPENDENCIES
from src.skills import SKILL_TOOL_PATTERNS, SKILL_ORDER
from src.agent_loop import AgentTrace

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

# Weights for the overall score (must sum to 1.0)
DIMENSION_WEIGHTS: dict[str, float] = {
    "skill_selection":      0.15,
    "tool_validity":        0.15,
    "tool_accuracy":        0.10,
    "sequence_correctness": 0.25,
    "task_success":         0.15,
    "recovery_quality":     0.10,
    "efficiency":           0.10,
}

# The optimal tool call sequence for SO-10482
OPTIMAL_TOOL_SEQUENCE = [
    "get_order",
    "get_shipment_status",
    "get_inventory",
    "get_fulfillment_capacity",
    "find_alternate_inventory",
    "get_transfer_eta",
    "get_supplier_expedite_options",
    "score_recovery_options",
    "recommend_action",
]

# Expected arguments for the SO-10482 scenario (used by tool_accuracy)
EXPECTED_ARGUMENTS: dict[str, dict[str, Any]] = {
    "get_order": {"order_id": "SO-10482"},
    "get_shipment_status": {"order_id": "SO-10482"},
    "get_inventory": {"sku": "SKU-4090", "dc_id": "DC-WEST-01"},
    "get_fulfillment_capacity": {"dc_id": "DC-WEST-01", "date": "2026-04-18"},
    "find_alternate_inventory": {"sku": "SKU-4090", "region": "ALL"},
    "get_supplier_expedite_options": {"sku": "SKU-4090"},
}


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

def eval_skill_selection(trace: AgentTrace) -> DimensionScore:
    """Did the agent call tools that map to the correct skill sequence?

    Maps each valid tool call to its parent skill and checks whether the
    implied skill order matches the canonical SKILL_ORDER.
    """
    tool_names = trace.tool_names_called
    if not tool_names:
        return DimensionScore("skill_selection", 0.0, 1.0, "No valid tool calls.")

    # Build a reverse map: tool -> skill
    tool_to_skill: dict[str, str] = {}
    for skill, tools in SKILL_TOOL_PATTERNS.items():
        for t in tools:
            tool_to_skill[t] = skill

    # Extract the implied skill sequence (deduplicated, in order of first appearance)
    seen: set[str] = set()
    implied_skills: list[str] = []
    for t in tool_names:
        skill = tool_to_skill.get(t)
        if skill and skill not in seen:
            seen.add(skill)
            implied_skills.append(skill)

    # Score: fraction of canonical skill order covered in the right order
    expected = SKILL_ORDER
    matched = 0
    exp_idx = 0
    for skill in implied_skills:
        if exp_idx < len(expected) and skill == expected[exp_idx]:
            matched += 1
            exp_idx += 1

    score = matched / len(expected) if expected else 0.0
    details = (
        f"Implied skills: {implied_skills}. "
        f"Expected: {expected}. "
        f"Matched {matched}/{len(expected)} in order."
    )
    return DimensionScore("skill_selection", round(score, 3), 1.0, details)


def eval_tool_validity(trace: AgentTrace) -> DimensionScore:
    """Were all tool calls well-formed (valid JSON, known tool, correct args)?

    Scores as the fraction of total steps (including invalid ones) that
    were valid.
    """
    total = len(trace.steps)
    if total == 0:
        return DimensionScore("tool_validity", 0.0, 1.0, "No tool calls recorded.")

    valid = sum(1 for s in trace.steps if s.valid)
    score = valid / total
    invalid = total - valid
    details = f"{valid}/{total} tool calls valid. {invalid} invalid."
    return DimensionScore("tool_validity", round(score, 3), 1.0, details)


def eval_tool_accuracy(trace: AgentTrace) -> DimensionScore:
    """Did tool arguments match the expected values for the scenario?

    Only checks tools that have expected arguments defined. Scores as
    the fraction of checked tools whose arguments matched exactly.
    """
    tool_names = trace.tool_names_called
    if not tool_names:
        return DimensionScore("tool_accuracy", 0.0, 1.0, "No valid tool calls.")

    checked = 0
    correct = 0
    mismatches: list[str] = []

    for step in trace.steps:
        if not step.valid:
            continue
        expected = EXPECTED_ARGUMENTS.get(step.tool_name)
        if expected is None:
            continue
        checked += 1
        # Check each expected key-value pair (allow extra args from the model)
        all_match = True
        for k, v in expected.items():
            actual = step.arguments.get(k)
            if str(actual) != str(v):
                all_match = False
                mismatches.append(
                    f"{step.tool_name}.{k}: expected {v!r}, got {actual!r}"
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


def eval_sequence_correctness(trace: AgentTrace) -> DimensionScore:
    """Were dependency constraints respected in the tool call order?

    This is the most important evaluator for the workshop. It walks the
    tool call sequence and checks that every tool's prerequisites (from
    TOOL_DEPENDENCIES) were called earlier in the sequence.

    Explicitly checks ordered pairs like:
    - get_inventory before find_alternate_inventory
    - find_alternate_inventory before get_transfer_eta
    - score_recovery_options before recommend_action
    """
    tool_names = trace.tool_names_called
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


def eval_task_success(trace: AgentTrace) -> DimensionScore:
    """Did the agent reach a valid final recommendation?

    Checks: completed flag, final_answer present, required fields exist.
    """
    if not trace.completed or trace.final_answer is None:
        reason = "incomplete" if not trace.completed else "no final answer"
        return DimensionScore("task_success", 0.0, 1.0, f"Agent did not complete: {reason}.")

    answer = trace.final_answer
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

    score = 1.0  # all required fields present
    details = f"Valid final answer with action='{answer.get('action', '?')}'."
    if bonus_count < 2:
        extras = []
        if not has_confidence:
            extras.append("confidence")
        if not has_delivery:
            extras.append("expected_delivery")
        details += f" Missing optional: {extras}."
    return DimensionScore("task_success", score, 1.0, details)


def eval_recovery_quality(trace: AgentTrace) -> DimensionScore:
    """If fallback repairs were applied, were they followed by successful execution?

    Scores based on whether repaired tool calls led to valid results.
    If no fallbacks were needed, returns a perfect score.
    """
    repaired_steps = [s for s in trace.steps if s.fallback_action == "repaired"]

    if not repaired_steps:
        if trace.fallback_rejects > 0:
            # Had rejects but no repairs -- partial score
            return DimensionScore(
                "recovery_quality", 0.5, 1.0,
                f"{trace.fallback_rejects} fallback rejects, 0 repairs. "
                "Agent recovered by retrying after error feedback."
            )
        return DimensionScore(
            "recovery_quality", 1.0, 1.0,
            "No fallback repairs needed."
        )

    successful_repairs = sum(1 for s in repaired_steps if s.valid)
    score = successful_repairs / len(repaired_steps)
    details = (
        f"{successful_repairs}/{len(repaired_steps)} repaired calls succeeded. "
        f"Total rejects: {trace.fallback_rejects}."
    )
    return DimensionScore("recovery_quality", round(score, 3), 1.0, details)


def eval_efficiency(trace: AgentTrace) -> DimensionScore:
    """How many steps did the agent take vs. the optimal trajectory?

    The optimal SO-10482 trajectory uses 9 tool calls (all tools once).
    Extra steps (retries, invalid calls) reduce the score.
    """
    optimal_count = len(OPTIMAL_TOOL_SEQUENCE)
    actual_valid = trace.total_tool_calls
    total_steps = len(trace.steps)  # includes invalid steps

    if total_steps == 0:
        return DimensionScore("efficiency", 0.0, 1.0, "No steps taken.")

    # Ratio of optimal to actual (capped at 1.0)
    # Using total_steps (not just valid) to penalize retries
    score = min(1.0, optimal_count / total_steps) if total_steps > 0 else 0.0

    details = (
        f"{actual_valid} valid tool calls, {total_steps} total steps "
        f"(optimal: {optimal_count}). "
        f"Efficiency ratio: {score:.2f}."
    )
    return DimensionScore("efficiency", round(score, 3), 1.0, details)


# -- Main evaluation entry point -------------------------------------------

EVALUATORS = {
    "skill_selection": eval_skill_selection,
    "tool_validity": eval_tool_validity,
    "tool_accuracy": eval_tool_accuracy,
    "sequence_correctness": eval_sequence_correctness,
    "task_success": eval_task_success,
    "recovery_quality": eval_recovery_quality,
    "efficiency": eval_efficiency,
}

PASS_THRESHOLD = 0.6


def evaluate_trajectory(
    trace: AgentTrace,
    threshold: float = PASS_THRESHOLD,
) -> TrajectoryEvaluation:
    """Run all evaluators against a trace and produce a full evaluation.

    Args:
        trace: The agent trace to evaluate.
        threshold: Minimum overall score to pass.

    Returns:
        TrajectoryEvaluation with per-dimension scores and an overall result.
    """
    scores: list[DimensionScore] = []
    for dim in EVAL_DIMENSIONS:
        evaluator = EVALUATORS[dim]
        scores.append(evaluator(trace))

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


# -- Display helpers -------------------------------------------------------

def print_evaluation(evaluation: TrajectoryEvaluation) -> None:
    """Pretty-print a trajectory evaluation."""
    print(f"{'Dimension':<25} {'Score':>6}  {'Weight':>6}  Details")
    print("-" * 90)
    for s in evaluation.scores:
        weight = DIMENSION_WEIGHTS[s.dimension]
        bar = "#" * int(s.score * 10) + "." * (10 - int(s.score * 10))
        print(f"{s.dimension:<25} {s.score:>5.2f}   {weight:>5.0%}   [{bar}]  {s.details}")
    print("-" * 90)
    tag = "PASS" if evaluation.passed else "FAIL"
    print(f"{'Overall':<25} {evaluation.overall:>5.2f}   {'':>6}  [{tag}] {evaluation.summary}")
