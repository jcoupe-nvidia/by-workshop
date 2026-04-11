"""
Human-facing display and reporting helpers for offline evaluation.

Owns:
    - Pretty-printing trajectory evaluations to the terminal
    - Formatting dimension scores as tables with visual bars
    - Summarizing evaluation results for notebook or CLI display

Does NOT own:
    - Evaluation logic or scoring (see eval.metrics)
    - Reward computation (see envs.rewards)
    - Training-oriented reward views (see training.reward_views)
"""
from __future__ import annotations

from src.eval.metrics import DimensionScore, TrajectoryEvaluation, DIMENSION_WEIGHTS


def print_evaluation(evaluation: TrajectoryEvaluation) -> None:
    """Pretty-print a trajectory evaluation as a dimension score table."""
    print(f"{'Dimension':<25} {'Score':>6}  {'Weight':>6}  Details")
    print("-" * 90)
    for s in evaluation.scores:
        weight = DIMENSION_WEIGHTS[s.dimension]
        bar = "#" * int(s.score * 10) + "." * (10 - int(s.score * 10))
        print(f"{s.dimension:<25} {s.score:>5.2f}   {weight:>5.0%}   [{bar}]  {s.details}")
    print("-" * 90)
    tag = "PASS" if evaluation.passed else "FAIL"
    print(f"{'Overall':<25} {evaluation.overall:>5.2f}   {'':>6}  [{tag}] {evaluation.summary}")


def format_evaluation_summary(evaluation: TrajectoryEvaluation) -> str:
    """Return a single-line summary string for an evaluation."""
    tag = "PASS" if evaluation.passed else "FAIL"
    return f"[{tag}] {evaluation.overall:.2f} — {evaluation.summary}"


def format_dimension_table(evaluation: TrajectoryEvaluation) -> str:
    """Return the full dimension table as a string (for notebook display)."""
    lines: list[str] = []
    lines.append(f"{'Dimension':<25} {'Score':>6}  {'Weight':>6}  Details")
    lines.append("-" * 90)
    for s in evaluation.scores:
        weight = DIMENSION_WEIGHTS[s.dimension]
        bar = "#" * int(s.score * 10) + "." * (10 - int(s.score * 10))
        lines.append(
            f"{s.dimension:<25} {s.score:>5.2f}   {weight:>5.0%}   [{bar}]  {s.details}"
        )
    lines.append("-" * 90)
    tag = "PASS" if evaluation.passed else "FAIL"
    lines.append(
        f"{'Overall':<25} {evaluation.overall:>5.2f}   {'':>6}  [{tag}] {evaluation.summary}"
    )
    return "\n".join(lines)
