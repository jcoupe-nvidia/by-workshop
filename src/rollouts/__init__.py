"""ProRL-facing rollout collection package: episode runner, serializers, trace types.

Modules:
    trace_types      -- canonical Episode and Event types (source of truth)
    episode_runner   -- orchestrate runtime + environment to produce enriched episodes
    serializers      -- Episode <-> JSONL stable serialization
    prorl_adapter    -- Episode -> NeMo RL trajectory format
"""
