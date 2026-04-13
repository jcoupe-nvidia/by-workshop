"""
Shared validate → repair → reject pipeline for tool-call processing.

Owns:
    - Canonical validation, fallback repair, and rejection logic
    - Recording repair/reject events through EpisodeRecorder

Does NOT own:
    - Tool implementations (see runtime.tools)
    - Environment state transitions (see envs/)
    - Agent loop control or prompt policy (see runtime.agent)
    - Training-time rollout orchestration (see rollouts/)

This module provides the single authoritative implementation of the
validate → repair → reject cycle. Both the interactive agent loop
(runtime.agent) and the training-time execution pipeline
(rollouts.nemo_gym_rollouts) call these functions to ensure identical
semantics across interactive and training-time execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.runtime.schemas import (
    ParsedToolCall,
    ParsedFinalAnswer,
    ValidationError,
    validate_tool_call,
)
from src.runtime.fallbacks import (
    FallbackAction,
    FallbackResult,
    try_repair,
)


@dataclass
class ValidateRepairResult:
    """Outcome of the validate → repair → reject pipeline.

    Attributes:
        parsed: The successfully parsed tool call or final answer,
                or None if the output was rejected or remains invalid.
        validation_error: The remaining ValidationError if the output
                         could not be parsed or repaired.
        fallback_result: The FallbackResult from repair attempts, if any.
        was_repaired: Whether the output was successfully repaired.
    """
    parsed: ParsedToolCall | ParsedFinalAnswer | None = None
    validation_error: ValidationError | None = None
    fallback_result: FallbackResult | None = None
    was_repaired: bool = False


def validate_and_repair(
    raw_output: str,
    tool_registry: dict[str, Any],
    recorder: Any | None = None,
) -> ValidateRepairResult:
    """Run the canonical validate → repair → reject pipeline on raw model output.

    This is the single source of truth for how raw model output is validated,
    repaired, or rejected. Both the interactive agent loop and training-time
    execution pipeline call this function.

    Args:
        raw_output: Raw string output from the model.
        tool_registry: The TOOL_REGISTRY dict mapping names to (fn, params, desc).
        recorder: Optional EpisodeRecorder to capture repair/reject events.

    Returns:
        ValidateRepairResult describing the outcome.
    """
    result = validate_tool_call(raw_output, tool_registry)
    fallback_result: FallbackResult | None = None
    was_repaired = False

    if isinstance(result, ValidationError):
        fb = try_repair(raw_output, tool_registry)

        if fb.action == FallbackAction.REPAIRED and fb.repaired:
            result = validate_tool_call(fb.repaired, tool_registry)
            re_valid = isinstance(result, (ParsedToolCall, ParsedFinalAnswer))
            if recorder is not None:
                recorder.record_repair_attempt(
                    original_output=raw_output,
                    repaired_output=fb.repaired,
                    repairs_applied=fb.repairs_applied,
                    succeeded=re_valid,
                )
            fallback_result = fb
            if re_valid:
                was_repaired = True

        elif fb.action == FallbackAction.REJECTED:
            if recorder is not None:
                recorder.record_repair_attempt(
                    original_output=raw_output,
                    repaired_output=None,
                    repairs_applied=fb.repairs_applied,
                    succeeded=False,
                )
                recorder.record_reject(
                    reason=fb.rejection_reason or "Unknown rejection",
                    raw_model_output=raw_output,
                    repairs_attempted=fb.repairs_applied,
                )
            fallback_result = fb

    if isinstance(result, ParsedToolCall) or isinstance(result, ParsedFinalAnswer):
        return ValidateRepairResult(
            parsed=result,
            fallback_result=fallback_result,
            was_repaired=was_repaired,
        )

    assert isinstance(result, ValidationError)
    return ValidateRepairResult(
        validation_error=result,
        fallback_result=fallback_result,
        was_repaired=was_repaired,
    )
