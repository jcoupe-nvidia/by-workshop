"""
Canonical Nemotron-style structured tool-call schema and validators.

Defines the JSON shape the model should emit, plus validation logic that
the agent loop uses before dispatching a tool call.

Canonical format:
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

Validation checks:
    - top-level keys are a subset of {"thought", "tool_call"}
    - "tool_call" is present and is a dict
    - "tool_call.name" is a string registered in TOOL_REGISTRY
    - "tool_call.arguments" is a dict with expected parameter names
    - no extra or unsafe argument values

Phase 4 will implement the full validation logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# -- Parsed tool call ------------------------------------------------------

@dataclass
class ParsedToolCall:
    """A validated, structured tool call ready for execution."""
    thought: str | None
    tool_name: str
    arguments: dict[str, Any]
    raw: str  # original model output for tracing

@dataclass
class ValidationError:
    """Describes why a tool call failed validation."""
    error_type: str   # e.g. "missing_field", "unknown_tool", "bad_arguments"
    message: str
    raw: str          # the original output that failed

# -- Schema constants ------------------------------------------------------

REQUIRED_KEYS = {"tool_call"}
OPTIONAL_KEYS = {"thought"}
TOOL_CALL_REQUIRED_KEYS = {"name", "arguments"}
