"""
Task-specific validity rules for the late-order recovery environment.

Owns:
    - Tool call dependency checking (sequence-sensitive prerequisites)
    - Any future environment-specific constraints on tool arguments

Does NOT own:
    - Structural JSON schema validation (see runtime.schemas)
    - Reward computation (see envs.rewards, future)
    - Runtime prompt or orchestration policy
"""
from __future__ import annotations


def check_dependencies(
    tool_name: str,
    called_tools: list[str],
    dependency_graph: dict[str, set[str]],
) -> tuple[bool, str]:
    """Check whether calling tool_name is valid given the tools already called.

    This encodes the task-specific sequencing rules: certain tools require
    prerequisite information that can only come from earlier tool calls.

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
