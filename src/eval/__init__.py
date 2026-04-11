"""
Offline evaluation and reporting package.

Submodules:
    - metrics:  evaluators, scoring types, evaluate_trajectory()
    - reports:  human-facing display helpers
"""
from src.eval.metrics import (  # noqa: F401
    # Score types
    DimensionScore,
    TrajectoryEvaluation,
    # Constants
    EVAL_DIMENSIONS,
    DIMENSION_WEIGHTS,
    PASS_THRESHOLD,
    EVALUATORS,
    # Main entry point
    evaluate_trajectory,
    # NAT ATIF and NeMo Gym entry points
    evaluate_atif_trajectory,
    evaluate_nemo_gym_result,
    # Individual evaluators
    eval_skill_selection,
    eval_tool_validity,
    eval_tool_accuracy,
    eval_sequence_correctness,
    eval_task_success,
    eval_recovery_quality,
    eval_efficiency,
)
from src.eval.reports import (  # noqa: F401
    print_evaluation,
    format_evaluation_summary,
    format_dimension_table,
)
