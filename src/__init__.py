"""
Workshop helper modules for late-order recovery agentic workflow.

Modules:
    scenario_data  -- synthetic in-memory tables (orders, inventory, etc.)
    tools          -- deterministic tool implementations and registry
    skills         -- higher-level skills composed from tools
    schema         -- Nemotron-style structured tool-call schema and validators
    agent_loop     -- think/emit/validate/execute/observe loop and model adapter
    fallbacks      -- repair/reject parsing for malformed outputs
    evaluation     -- sequence-sensitive evaluators and scoring helpers
"""
