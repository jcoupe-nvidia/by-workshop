"""
Backward-compatibility shim for src.skills.

The canonical definitions now live in:
    - src.runtime.workflows  (workflow output types, registry, transitions, implementations)
    - src.runtime.skills     (NAT-facing directory-backed skill discovery and execution)

This module re-exports the workflow layer so existing imports continue to work.
Note the distinction between two related concepts:
    - "workflows" are the higher-level tool-sequence decompositions
      (diagnose_order_risk, assess_primary_fulfillment, etc.)
    - "skills" are the NAT-facing directory-backed packages with SKILL.md files,
      discoverable via list_skills / search_skills / get_skill / run_skill_command
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
