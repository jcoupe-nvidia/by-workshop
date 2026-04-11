"""
NAT-facing agent runtime package.

Submodules:
    tools         -- deterministic tool implementations and registry
    schemas       -- structured tool-call and final-answer parsing/validation
    workflows     -- higher-level workflow (skill) decomposition
    prompts       -- system prompt and task message construction
    fallbacks     -- fallback parsing, repair, and reject logic
    tracing       -- structured event emission using canonical trace_types
    agent         -- single-episode agent loop and model adapter

NAT integration modules:
    nat_tools     -- NAT Function wrappers with Pydantic schemas for tools
    nat_llm       -- NIMModelConfig for the local Nemotron endpoint
    atif_adapter  -- Episode-to-ATIF Trajectory conversion

Skills package:
    skills/       -- directory-backed skill definitions with SKILL.md files
    skills.api    -- list_skills, search_skills, get_skill, run_skill_command
"""
