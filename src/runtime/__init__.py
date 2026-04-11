"""
NAT-facing agent runtime package.

Submodules:
    tools       -- deterministic tool implementations and registry
    schemas     -- structured tool-call and final-answer parsing/validation
    workflows   -- higher-level workflow (skill) decomposition
    prompts     -- system prompt and task message construction
    fallbacks   -- fallback parsing, repair, and reject logic
    tracing     -- structured event emission using canonical trace_types
    agent       -- single-episode agent loop and model adapter
"""
