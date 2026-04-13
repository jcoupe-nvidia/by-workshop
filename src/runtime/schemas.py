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

from pydantic import ValidationError as PydanticValidationError


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

    # Try direct parse first — but only return the full text if it's valid JSON.
    # If it starts with '{' but has trailing prose, fall through to brace matching.
    if text.startswith("{"):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

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

    # Step 3: Validate envelope structure via Pydantic schemas.
    # The canonical envelope schemas in shared.tool_schemas provide
    # machine-checkable structural validation. Error types are mapped
    # to the same vocabulary the fallback/repair layer expects.
    from src.shared.tool_schemas import (
        NemotronToolCallEnvelope,
        NemotronFinalAnswerEnvelope,
    )

    extra_keys = set(parsed.keys()) - ALLOWED_TOP_KEYS
    if extra_keys:
        return ValidationError(
            error_type="extra_keys",
            message=f"Unexpected top-level keys: {extra_keys}. Allowed: {ALLOWED_TOP_KEYS}.",
            raw=raw,
        )

    thought = parsed.get("thought")

    # Step 4: Check for final_answer — validate via Pydantic envelope
    if "final_answer" in parsed:
        try:
            envelope = NemotronFinalAnswerEnvelope.model_validate(parsed)
        except PydanticValidationError as exc:
            first_err = exc.errors()[0] if exc.errors() else {}
            return ValidationError(
                error_type="bad_final_answer",
                message=f"Invalid final_answer envelope: {first_err.get('msg', str(exc))}",
                raw=raw,
            )
        return ParsedFinalAnswer(
            thought=envelope.thought,
            answer=envelope.final_answer.model_dump(),
            raw=raw,
        )

    # Step 5: Check tool_call is present
    if "tool_call" not in parsed:
        return ValidationError(
            error_type="missing_tool_call",
            message="Missing 'tool_call' key. Expected 'tool_call' or 'final_answer'.",
            raw=raw,
        )

    # Step 6: Validate tool_call envelope via Pydantic
    try:
        envelope = NemotronToolCallEnvelope.model_validate(parsed)
    except PydanticValidationError as exc:
        first_err = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(l) for l in first_err.get("loc", []))
        error_type = "bad_tool_call"
        if "name" in loc:
            error_type = "bad_tool_name"
        elif "arguments" in loc:
            error_type = "bad_arguments"
        elif loc == "tool_call":
            error_type = "bad_tool_call"
        else:
            error_type = "missing_field"
        return ValidationError(
            error_type=error_type,
            message=f"Invalid tool_call envelope at '{loc}': {first_err.get('msg', str(exc))}",
            raw=raw,
        )

    tool_name = envelope.tool_call.name
    arguments = envelope.tool_call.arguments

    # Step 7: Validate tool name against registry
    if tool_name not in tool_registry:
        return ValidationError(
            error_type="unknown_tool",
            message=f"Unknown tool '{tool_name}'. Known tools: {sorted(tool_registry.keys())}.",
            raw=raw,
        )

    # Step 8: Validate arguments against registry parameter spec
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
        thought=envelope.thought,
        tool_name=tool_name,
        arguments=arguments,
        raw=raw,
    )
