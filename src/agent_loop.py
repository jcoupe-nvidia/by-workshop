"""
OpenCode-style agent execution loop and model adapter.

Loop steps:
    1. Think   -- model receives context and emits a response
    2. Parse   -- extract structured tool call (or detect final answer)
    3. Validate -- check tool call against schema and dependency rules
    4. Execute -- dispatch to deterministic tool
    5. Observe -- append tool result to conversation history
    6. Continue -- loop back to step 1 (bounded by max iterations)

The model adapter is the single integration seam for the local LLM.
Everything else in this module is deterministic and testable.

Model adapter contract:
    - POST to http://0.0.0.0:8000/v1/chat/completions
    - model: "nvidia/nemotron-3-nano"
    - OpenAI-style messages list
    - Returns assistant message content as a string

Phase 4 will implement the full loop. This file defines the adapter
interface and trace data structures now so later phases can build on them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# -- Configuration ---------------------------------------------------------

MODEL_ENDPOINT = "http://0.0.0.0:8000/v1/chat/completions"
MODEL_NAME = "nvidia/nemotron-3-nano"
MAX_ITERATIONS = 15  # hard stop to prevent runaway loops
DEFAULT_MAX_TOKENS = 512

# -- Trace data structures -------------------------------------------------

@dataclass
class ToolCallRecord:
    """One step in the agent's trajectory."""
    iteration: int
    thought: str | None
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    valid: bool
    validation_error: str | None = None

@dataclass
class AgentTrace:
    """Full trajectory of a single agent run."""
    task: str
    steps: list[ToolCallRecord] = field(default_factory=list)
    final_answer: dict[str, Any] | None = None
    completed: bool = False
    stop_reason: str | None = None  # "final_answer", "max_iterations", "error"

# -- Model adapter interface -----------------------------------------------

def call_model(messages: list[dict[str, str]], max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """
    Send messages to the local Nemotron endpoint and return the response.

    This is the single integration point for the real model. Phase 4 will
    implement the HTTP call. For now, this raises NotImplementedError so
    that tests and notebook dry-runs fail explicitly rather than silently.
    """
    raise NotImplementedError(
        "Model adapter not yet implemented. See Phase 4 in PLAN.md."
    )
