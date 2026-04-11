"""
Task-specific validity rules for the late-order recovery environment.

Owns:
    - Tool call dependency checking (sequence-sensitive prerequisites)
    - Environment-state-aware validation via check_preconditions
    - Any future environment-specific constraints on tool arguments

Does NOT own:
    - Structural JSON schema validation (see runtime.schemas)
    - Reward computation (see envs.rewards)
    - Runtime prompt or orchestration policy
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.envs.state import LateOrderEnvState


def check_dependencies(
    tool_name: str,
    called_tools: list[str],
    dependency_graph: dict[str, set[str]],
) -> tuple[bool, str]:
    """Check whether calling tool_name is valid given the tools already called.

    This encodes the task-specific sequencing rules: certain tools require
    prerequisite information that can only come from earlier tool calls.

    This is the backward-compatible interface used by the runtime agent
    loop. For environment-state-aware validation, use
    envs.transitions.check_preconditions instead.

    Args:
        tool_name: The tool about to be called.
        called_tools: List of tool names already called (in order).
        dependency_graph: Mapping tool -> set of prerequisite tools.

    Returns:
        (is_valid, reason) tuple.
    """
    required = dependency_graph.get(tool_name, set())
    called_set = set(called_tools)
    missing = required - called_set

    if missing:
        return False, (
            f"Cannot call '{tool_name}' yet. "
            f"Missing prerequisites: {missing}. "
            f"Tools called so far: {called_tools}."
        )
    return True, f"Dependencies satisfied for '{tool_name}'."


def check_dependencies_from_state(
    tool_name: str,
    state: LateOrderEnvState,
) -> tuple[bool, str]:
    """Check dependencies using environment state instead of raw lists.

    This is a convenience wrapper that extracts the called-tools set
    from the environment state and delegates to the transition layer's
    check_preconditions for the full validity check.

    Args:
        tool_name: The tool about to be called.
        state: Current environment state.

    Returns:
        (is_valid, reason) tuple.
    """
    from src.envs.transitions import check_preconditions
    ok, err_type, err_msg = check_preconditions(state, tool_name)
    if ok:
        return True, f"Dependencies satisfied for '{tool_name}'."
    return False, err_msg or f"Cannot call '{tool_name}'."
