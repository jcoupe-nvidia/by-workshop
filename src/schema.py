"""
Backward-compatibility shim for src.schema.

The canonical definitions now live in:
    - src.runtime.schemas  (action types, JSON extraction, structural validation)
    - src.envs.validators  (task-specific dependency checking)

This module re-exports everything so existing imports continue to work.
"""
from src.runtime.schemas import (  # noqa: F401
    ParsedToolCall,
    ParsedFinalAnswer,
    ValidationError,
    REQUIRED_KEYS,
    OPTIONAL_KEYS,
    ALLOWED_TOP_KEYS,
    TOOL_CALL_REQUIRED_KEYS,
    extract_json,
    validate_tool_call,
)
from src.envs.validators import check_dependencies  # noqa: F401
