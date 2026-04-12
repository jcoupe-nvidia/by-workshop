"""Canonical rollout collection package: episode runner, serializers, trace types.

Modules:
    trace_types         -- canonical Episode and Event types (source of truth)
    episode_runner      -- orchestrate runtime + environment to produce enriched episodes
    serializers         -- Episode <-> JSONL stable serialization
    export_adapters     -- Episode -> training trajectory + ATIF trajectory formats
    scripted_traces     -- pre-built scripted episodes for workshop demonstration
    nemo_gym_rollouts   -- NeMo Gym rollout collection adapter (RolloutCollectionConfig,
                           enriched -> result row export)
"""
