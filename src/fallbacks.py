"""
Backward-compatibility shim for src.fallbacks.

The canonical definitions now live in:
    - src.runtime.fallbacks  (fallback parsing, repair, and reject logic)

This module re-exports everything so existing imports continue to work.
"""
from src.runtime.fallbacks import (  # noqa: F401
    FallbackAction,
    FallbackResult,
    try_repair,
    parse_with_fallback,
)
