"""
Fallback parsing and recovery for malformed model outputs.

Handles common failure modes when the model does not emit clean structured
tool calls:
    - Malformed JSON (missing braces, trailing commas, single quotes, etc.)
    - Mixed text + JSON (reasoning text surrounding a JSON block)
    - Missing required fields (no "tool_call", no "name", etc.)
    - Unknown tool names (close-match fuzzy correction)
    - Unsafe or invalid argument values

Repair-vs-reject policy:
    REPAIR when the intent is unambiguous and the fix is mechanical:
        - strip surrounding text to extract JSON
        - fix minor JSON syntax errors (trailing commas, single quotes)
        - fill default values for optional missing fields ("thought")
        - correct close-match tool names (edit distance <= 2)
    REJECT when the intent is ambiguous or the error is structural:
        - no JSON object found at all
        - missing "tool_call" and "final_answer" entirely
        - unknown tool name with no close match
        - arguments that reference disallowed values (e.g. SQL, shell commands)

Each repair is tagged with a short label so the full repair chain is
inspectable in traces and notebook output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
    repairs_applied: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON syntax repair helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON."""
    m = re.search(r"```(?:json)?\s*\n?(.+?)\s*\n?```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing braces/brackets.

    Handles patterns like:  {"a": 1,}  or  [1, 2,]
    """
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_single_quotes(text: str) -> str:
    """Replace single-quoted JSON keys/values with double quotes.

    This is a common model error.  We only fix when the text is clearly
    JSON-like (starts with { after stripping).
    """
    stripped = text.strip()
    if not stripped.startswith("{") and not stripped.startswith("["):
        return text

    # State machine: walk character by character and swap unescaped single
    # quotes for double quotes, but only outside of double-quoted strings.
    result: list[str] = []
    in_double = False
    in_single = False
    prev = ""
    for ch in stripped:
        if ch == '"' and not in_single and prev != "\\":
            in_double = not in_double
            result.append(ch)
        elif ch == "'" and not in_double and prev != "\\":
            in_single = not in_single
            result.append('"')          # swap to double quote
        else:
            result.append(ch)
        prev = ch
    return "".join(result)


def _fix_missing_closing_brace(text: str) -> str:
    """If the JSON is truncated with unbalanced braces, close them."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return text

    depth = 0
    in_string = False
    escape_next = False
    for ch in stripped:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

    if depth > 0:
        return stripped + ("}" * depth)
    return text


def _extract_json_object(text: str) -> str | None:
    """Extract the first balanced { ... } from text.

    Returns None if no opening brace is found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# ---------------------------------------------------------------------------
# Structural repair helpers
# ---------------------------------------------------------------------------

UNSAFE_PATTERNS = [
    re.compile(r"\b(DROP|DELETE|INSERT|UPDATE|ALTER)\b", re.IGNORECASE),
    re.compile(r";\s*(rm|sudo|curl|wget|sh|bash)\b", re.IGNORECASE),
    re.compile(r"__import__\s*\("),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\bos\.system\s*\("),
]


def _check_unsafe_arguments(arguments: dict[str, Any]) -> str | None:
    """Return a reason string if any argument value looks unsafe, else None."""
    for key, value in arguments.items():
        if not isinstance(value, str):
            continue
        for pattern in UNSAFE_PATTERNS:
            if pattern.search(value):
                return f"Unsafe argument value in '{key}': matched pattern {pattern.pattern!r}"
    return None


def _fuzzy_tool_name(name: str, known_tools: list[str]) -> str | None:
    """Find the closest known tool name within edit distance 2.

    Returns the match or None if nothing is close enough.
    """
    def _edit_distance(a: str, b: str) -> int:
        """Levenshtein distance (small strings only)."""
        if len(a) < len(b):
            return _edit_distance(b, a)
        if len(b) == 0:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                cost = 0 if ca == cb else 1
                curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        return prev[len(b)]

    best_match = None
    best_dist = 3  # threshold: must be <= 2
    for known in known_tools:
        d = _edit_distance(name.lower(), known.lower())
        if d < best_dist:
            best_dist = d
            best_match = known
    return best_match


def _fill_optional_defaults(parsed: dict[str, Any]) -> list[str]:
    """Fill optional fields with defaults.  Returns list of repair labels."""
    repairs = []
    # "thought" is optional -- add it as None if missing (for schema consistency)
    # No repair needed since it's already optional in the schema.

    # If tool_call.arguments exists but is a list, try to convert
    tc = parsed.get("tool_call", {})
    if isinstance(tc.get("arguments"), list):
        # Some models emit arguments as [[key, value], ...] pairs
        try:
            parsed["tool_call"]["arguments"] = dict(tc["arguments"])
            repairs.append("converted_arguments_list_to_dict")
        except (ValueError, TypeError):
            pass

    return repairs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def try_repair(
    raw: str,
    tool_registry: dict[str, Any],
) -> FallbackResult:
    """Attempt to repair a raw model output into valid structured JSON.

    This function applies a chain of mechanical repairs in order:
        1. Strip markdown fences
        2. Extract JSON object from mixed text
        3. Fix trailing commas
        4. Fix single quotes
        5. Fix missing closing braces
        6. Parse JSON
        7. Fill optional defaults
        8. Fuzzy-match unknown tool names
        9. Check for unsafe arguments

    If all repairs succeed and produce a parseable, valid tool call,
    the result is REPAIRED.  If the output cannot be salvaged, it is
    REJECTED with a reason.

    Args:
        raw: The raw model output string.
        tool_registry: TOOL_REGISTRY mapping tool_name -> (fn, params, desc).

    Returns:
        FallbackResult with the outcome.
    """
    repairs: list[str] = []
    text = raw

    # --- Phase 1: Text-level cleanup ---

    # Strip markdown fences
    cleaned = _strip_markdown_fences(text)
    if cleaned != text:
        repairs.append("stripped_markdown_fences")
        text = cleaned

    # Fix trailing commas (before extraction so brace matching works)
    fixed = _fix_trailing_commas(text)
    if fixed != text:
        repairs.append("fixed_trailing_commas")
        text = fixed

    # Fix single quotes (before extraction so JSON parse works)
    fixed = _fix_single_quotes(text)
    if fixed != text:
        repairs.append("fixed_single_quotes")
        text = fixed

    # Fix missing closing braces (before extraction so brace matching works)
    fixed = _fix_missing_closing_brace(text)
    if fixed != text:
        repairs.append("fixed_missing_closing_brace")
        text = fixed

    # Extract JSON from surrounding text
    json_obj = _extract_json_object(text)
    if json_obj is None:
        return FallbackResult(
            action=FallbackAction.REJECTED,
            original=raw,
            rejection_reason="No JSON object found in model output.",
            repairs_applied=repairs,
        )
    if json_obj != text.strip():
        repairs.append("extracted_json_from_text")
        text = json_obj

    # --- Phase 2: Parse JSON ---

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return FallbackResult(
            action=FallbackAction.REJECTED,
            original=raw,
            rejection_reason=f"JSON parse error after repairs: {e}",
            repairs_applied=repairs,
        )

    if not isinstance(parsed, dict):
        return FallbackResult(
            action=FallbackAction.REJECTED,
            original=raw,
            rejection_reason=f"Expected JSON object, got {type(parsed).__name__}.",
            repairs_applied=repairs,
        )

    # --- Phase 3: Structural repair ---

    # Fill optional defaults
    default_repairs = _fill_optional_defaults(parsed)
    repairs.extend(default_repairs)

    # Must have either tool_call or final_answer
    if "tool_call" not in parsed and "final_answer" not in parsed:
        # Check if the model emitted a flat tool call without the wrapper
        if "name" in parsed and "arguments" in parsed:
            parsed = {"tool_call": {"name": parsed["name"], "arguments": parsed["arguments"]}}
            if "thought" in parsed.get("tool_call", {}).get("arguments", {}):
                # Don't let thought leak into arguments
                pass
            repairs.append("wrapped_flat_tool_call")
        else:
            return FallbackResult(
                action=FallbackAction.REJECTED,
                original=raw,
                rejection_reason="Missing both 'tool_call' and 'final_answer'. Cannot determine intent.",
                repairs_applied=repairs,
            )

    # If it's a final_answer, no further repair needed
    if "final_answer" in parsed:
        if not repairs:
            return FallbackResult(action=FallbackAction.NO_ACTION, original=raw)
        return FallbackResult(
            action=FallbackAction.REPAIRED,
            original=raw,
            repaired=json.dumps(parsed),
            repairs_applied=repairs,
        )

    # Validate tool_call structure
    tc = parsed.get("tool_call", {})
    if not isinstance(tc, dict):
        return FallbackResult(
            action=FallbackAction.REJECTED,
            original=raw,
            rejection_reason=f"'tool_call' must be a dict, got {type(tc).__name__}.",
            repairs_applied=repairs,
        )

    if "name" not in tc:
        return FallbackResult(
            action=FallbackAction.REJECTED,
            original=raw,
            rejection_reason="Missing 'name' in tool_call.",
            repairs_applied=repairs,
        )

    if "arguments" not in tc:
        # Some models omit arguments when there are none -- add empty dict
        tc["arguments"] = {}
        repairs.append("added_empty_arguments")

    # --- Phase 4: Tool name validation / fuzzy match ---

    tool_name = tc["name"]
    known_tools = list(tool_registry.keys())

    if tool_name not in tool_registry:
        corrected = _fuzzy_tool_name(tool_name, known_tools)
        if corrected:
            repairs.append(f"corrected_tool_name:{tool_name}->{corrected}")
            tc["name"] = corrected
        else:
            return FallbackResult(
                action=FallbackAction.REJECTED,
                original=raw,
                rejection_reason=(
                    f"Unknown tool '{tool_name}' with no close match. "
                    f"Known tools: {sorted(known_tools)}."
                ),
                repairs_applied=repairs,
            )

    # --- Phase 5: Argument validation ---

    arguments = tc.get("arguments", {})
    if not isinstance(arguments, dict):
        return FallbackResult(
            action=FallbackAction.REJECTED,
            original=raw,
            rejection_reason=f"'arguments' must be a dict, got {type(arguments).__name__}.",
            repairs_applied=repairs,
        )

    # Check for unsafe argument values
    unsafe = _check_unsafe_arguments(arguments)
    if unsafe:
        return FallbackResult(
            action=FallbackAction.REJECTED,
            original=raw,
            rejection_reason=f"Unsafe arguments: {unsafe}",
            repairs_applied=repairs,
        )

    # Check argument names against the tool's expected parameters
    _fn, expected_params, _desc = tool_registry[tc["name"]]
    expected_keys = set(expected_params.keys())
    provided_keys = set(arguments.keys())

    # Remove unexpected extra arguments (repair, not reject)
    extra = provided_keys - expected_keys
    if extra:
        for k in extra:
            del arguments[k]
        repairs.append(f"removed_extra_arguments:{extra}")

    # Missing required arguments -- cannot repair, reject
    missing = expected_keys - provided_keys
    if missing:
        return FallbackResult(
            action=FallbackAction.REJECTED,
            original=raw,
            rejection_reason=(
                f"Missing required arguments for '{tc['name']}': {missing}. "
                f"Expected: {expected_keys}."
            ),
            repairs_applied=repairs,
        )

    # --- Done ---

    repaired_json = json.dumps(parsed)

    if not repairs:
        return FallbackResult(action=FallbackAction.NO_ACTION, original=raw)

    return FallbackResult(
        action=FallbackAction.REPAIRED,
        original=raw,
        repaired=repaired_json,
        repairs_applied=repairs,
    )


# ---------------------------------------------------------------------------
# Convenience: combined parse-or-repair entry point
# ---------------------------------------------------------------------------

def parse_with_fallback(
    raw: str,
    tool_registry: dict[str, Any],
) -> tuple[dict[str, Any] | None, FallbackResult]:
    """Parse a model output, attempting repair if the initial parse fails.

    Returns:
        (parsed_dict, fallback_result) where parsed_dict is the repaired
        JSON dict (or None if rejected).  The FallbackResult always
        describes what happened.
    """
    result = try_repair(raw, tool_registry)

    if result.action == FallbackAction.REJECTED:
        return None, result

    if result.action == FallbackAction.NO_ACTION:
        # Original was fine -- parse it directly
        from src.schema import extract_json
        json_str = extract_json(raw)
        if json_str:
            try:
                return json.loads(json_str), result
            except json.JSONDecodeError:
                pass
        return None, FallbackResult(
            action=FallbackAction.REJECTED,
            original=raw,
            rejection_reason="Original appeared valid but failed to parse.",
        )

    # REPAIRED -- parse the repaired version
    assert result.repaired is not None
    try:
        return json.loads(result.repaired), result
    except json.JSONDecodeError:
        return None, FallbackResult(
            action=FallbackAction.REJECTED,
            original=raw,
            rejection_reason="Repaired output still failed to parse as JSON.",
            repairs_applied=result.repairs_applied,
        )
