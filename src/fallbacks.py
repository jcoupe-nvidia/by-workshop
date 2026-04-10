"""
Fallback parsing and recovery for malformed model outputs.

Handles common failure modes when the model does not emit clean structured
tool calls:
    - Malformed JSON (missing braces, trailing commas, etc.)
    - Mixed text + JSON (reasoning text surrounding a JSON block)
    - Missing required fields (no "tool_call", no "name", etc.)
    - Unknown tool names
    - Unsafe or invalid argument values

Repair-vs-reject policy:
    REPAIR when the intent is unambiguous and the fix is mechanical:
        - strip surrounding text to extract JSON
        - fix minor JSON syntax errors
        - fill default values for optional missing fields
    REJECT when the intent is ambiguous or the error is structural:
        - unknown tool name (could mean anything)
        - missing tool_call entirely
        - arguments that reference disallowed values

Phase 5 will implement these handlers. This file defines the policy
categories and interface now.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

class FallbackAction(Enum):
    """What the fallback layer decided to do."""
    REPAIRED = "repaired"
    REJECTED = "rejected"
    NO_ACTION = "no_action"  # output was already valid

@dataclass
class FallbackResult:
    """Outcome of attempting to parse/repair a model output."""
    action: FallbackAction
    original: str
    repaired: str | None = None
    rejection_reason: str | None = None
    repairs_applied: list[str] | None = None  # e.g. ["stripped_surrounding_text", "fixed_trailing_comma"]
