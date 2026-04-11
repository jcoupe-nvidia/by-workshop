"""
Runtime-facing action schemas for the Nemotron-style structured tool-call format.

Owns:
    - Parsed action types (ParsedToolCall, ParsedFinalAnswer, ValidationError)
    - JSON extraction from raw model output
    - Schema validation against the tool registry (structural correctness)

Does NOT own:
    - Task-specific dependency/sequencing rules (see envs.validators)
    - Reward semantics
    - Rollout or training concerns

Canonical tool-call format:
    {
      "thought": "optional short reasoning summary",
      "tool_call": {
        "name": "get_inventory",
        "arguments": {
          "sku": "SKU-4090",
          "dc_id": "DC-WEST-01"
        }
      }
    }

Final-answer format (terminates the loop):
    {
      "thought": "optional reasoning",
      "final_answer": {
        "action": "...",
        "rationale": "..."
      }
    }
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


# -- Parsed action types ----------------------------------------------------

@dataclass
class ParsedToolCall:
    """A validated, structured tool call ready for execution."""
    thought: str | None
    tool_name: str
    arguments: dict[str, Any]
    raw: str  # original model output for tracing


@dataclass
class ParsedFinalAnswer:
    """A validated final answer that terminates the loop."""
    thought: str | None
    answer: dict[str, Any]
    raw: str


@dataclass
class ValidationError:
    """Describes why a tool call failed validation."""
    error_type: str   # e.g. "missing_field", "unknown_tool", "bad_arguments"
    message: str
    raw: str          # the original output that failed


# -- Schema constants -------------------------------------------------------

REQUIRED_KEYS = {"tool_call"}
OPTIONAL_KEYS = {"thought"}
ALLOWED_TOP_KEYS = {"thought", "tool_call", "final_answer"}
TOOL_CALL_REQUIRED_KEYS = {"name", "arguments"}


# -- JSON extraction --------------------------------------------------------

def extract_json(raw: str) -> str | None:
    """Extract the first JSON object from a string that may contain surrounding text.

    Handles common model outputs like:
        - Pure JSON
        - JSON inside a markdown code fence
        - Text before/after a JSON block
    Returns None if no JSON object is found.
    """
    text = raw.strip()

    # Try direct parse first
    if text.startswith("{"):
        return text

    # Try extracting from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Try extracting first { ... } block
    brace_start = text.find("{")
    if brace_start == -1:
        return None

    # Walk forward to find matching closing brace
    depth = 0
    in_string = False
    escape_next = False
    for i in range(brace_start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start:i + 1]

    return None


# -- Validation -------------------------------------------------------------

def validate_tool_call(
    raw: str,
    tool_registry: dict[str, Any],
) -> ParsedToolCall | ParsedFinalAnswer | ValidationError:
    """Parse and validate a raw model output string.

    Args:
        raw: The raw string from the model response.
        tool_registry: The TOOL_REGISTRY dict mapping tool_name -> (fn, params, desc).

    Returns:
        ParsedToolCall if the output is a valid tool call.
        ParsedFinalAnswer if the output is a final answer.
        ValidationError if validation fails.
    """
    # Step 1: Extract JSON from potentially messy output
    json_str = extract_json(raw)
    if json_str is None:
        return ValidationError(
            error_type="no_json",
            message="No JSON object found in model output.",
            raw=raw,
        )

    # Step 2: Parse JSON
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return ValidationError(
            error_type="invalid_json",
            message=f"JSON parse error: {e}",
            raw=raw,
        )

    if not isinstance(parsed, dict):
        return ValidationError(
            error_type="not_object",
            message=f"Expected JSON object, got {type(parsed).__name__}.",
            raw=raw,
        )

    # Step 3: Check top-level keys
    extra_keys = set(parsed.keys()) - ALLOWED_TOP_KEYS
    if extra_keys:
        return ValidationError(
            error_type="extra_keys",
            message=f"Unexpected top-level keys: {extra_keys}. Allowed: {ALLOWED_TOP_KEYS}.",
            raw=raw,
        )

    thought = parsed.get("thought")

    # Step 4: Check for final_answer
    if "final_answer" in parsed:
        answer = parsed["final_answer"]
        if not isinstance(answer, dict):
            return ValidationError(
                error_type="bad_final_answer",
                message=f"'final_answer' must be a dict, got {type(answer).__name__}.",
                raw=raw,
            )
        return ParsedFinalAnswer(thought=thought, answer=answer, raw=raw)

    # Step 5: Check tool_call is present
    if "tool_call" not in parsed:
        return ValidationError(
            error_type="missing_tool_call",
            message="Missing 'tool_call' key. Expected 'tool_call' or 'final_answer'.",
            raw=raw,
        )

    tool_call = parsed["tool_call"]
    if not isinstance(tool_call, dict):
        return ValidationError(
            error_type="bad_tool_call",
            message=f"'tool_call' must be a dict, got {type(tool_call).__name__}.",
            raw=raw,
        )

    # Step 6: Check tool_call required keys
    missing = TOOL_CALL_REQUIRED_KEYS - set(tool_call.keys())
    if missing:
        return ValidationError(
            error_type="missing_field",
            message=f"Missing required fields in tool_call: {missing}.",
            raw=raw,
        )

    tool_name = tool_call["name"]
    arguments = tool_call["arguments"]

    # Step 7: Validate tool name
    if not isinstance(tool_name, str):
        return ValidationError(
            error_type="bad_tool_name",
            message=f"'tool_call.name' must be a string, got {type(tool_name).__name__}.",
            raw=raw,
        )
    if tool_name not in tool_registry:
        return ValidationError(
            error_type="unknown_tool",
            message=f"Unknown tool '{tool_name}'. Known tools: {sorted(tool_registry.keys())}.",
            raw=raw,
        )

    # Step 8: Validate arguments
    if not isinstance(arguments, dict):
        return ValidationError(
            error_type="bad_arguments",
            message=f"'tool_call.arguments' must be a dict, got {type(arguments).__name__}.",
            raw=raw,
        )

    _fn, expected_params, _desc = tool_registry[tool_name]
    expected_keys = set(expected_params.keys())
    provided_keys = set(arguments.keys())

    missing_args = expected_keys - provided_keys
    if missing_args:
        return ValidationError(
            error_type="missing_arguments",
            message=f"Missing arguments for '{tool_name}': {missing_args}. Expected: {expected_keys}.",
            raw=raw,
        )

    extra_args = provided_keys - expected_keys
    if extra_args:
        return ValidationError(
            error_type="extra_arguments",
            message=f"Unexpected arguments for '{tool_name}': {extra_args}. Expected: {expected_keys}.",
            raw=raw,
        )

    return ParsedToolCall(
        thought=thought,
        tool_name=tool_name,
        arguments=arguments,
        raw=raw,
    )
