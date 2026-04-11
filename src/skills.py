"""
Backward-compatibility shim for src.skills.

The canonical definitions now live in:
    - src.runtime.workflows  (workflow output types, registry, transitions, implementations)

This module re-exports everything so existing imports continue to work.
The "skill" naming is preserved for backward compatibility; the canonical
naming is now "workflow".
"""
from src.runtime.workflows import (  # noqa: F401
    # Output types
    OrderRiskDiagnosis,
    PrimaryAssessment,
    RecoveryPath,
    Recommendation,
    # Execution tracing (renamed: SkillToolCall -> WorkflowToolCall, SkillContext -> WorkflowContext)
    WorkflowToolCall as SkillToolCall,
    WorkflowContext as SkillContext,
    # Registries and constants (backward-compatible aliases)
    SKILL_NAMES,
    SKILL_TOOL_PATTERNS,
    SKILL_TRANSITIONS,
    SKILL_ORDER,
    SKILL_REGISTRY,
    # Validation
    validate_skill_transition,
    # Workflow implementations (same function names)
    diagnose_order_risk,
    assess_primary_fulfillment,
    evaluate_alternate_recovery_paths,
    synthesize_recommendation,
    # Full flow
    run_diagnostic_flow,
)
