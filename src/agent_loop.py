"""
Backward-compatibility shim for src.agent_loop.

The canonical definitions now live in:
    - src.runtime.agent    (agent loop, model adapter, trace types)
    - src.runtime.prompts  (system prompt, task message)
    - src.runtime.tracing  (structured event emission)

This module re-exports everything so existing imports continue to work.
"""
from src.runtime.agent import (  # noqa: F401
    # Configuration
    MODEL_ENDPOINT,
    MODEL_NAME,
    MAX_ITERATIONS,
    DEFAULT_MAX_TOKENS,
    REQUEST_TIMEOUT,
    # Trace types
    ToolCallRecord,
    AgentTrace,
    # Model adapter
    call_model,
    # Agent loop
    run_agent,
    run_agent_episode,
    # Helpers
    print_trace_summary,
    trace_to_trajectory,
)
from src.runtime.prompts import (  # noqa: F401
    build_system_prompt,
    build_task_message,
)
