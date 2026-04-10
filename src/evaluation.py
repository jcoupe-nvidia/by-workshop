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

Phase 6 will implement these evaluators. This file defines the scoring
interface and dimension names now.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
