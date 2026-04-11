"""
Minimal entrypoint for running one episode outside the notebook.

Usage:
    python -m src.main [--order ORDER_ID] [--check-imports] [--episode] [--rollout]

This validates the package structure imports correctly and can run
a full agent episode against the local model endpoint.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run one late-order recovery episode.",
    )
    parser.add_argument(
        "--order",
        default="SO-10482",
        help="Order ID to investigate (default: SO-10482).",
    )
    parser.add_argument(
        "--check-imports",
        action="store_true",
        help="Validate that all package imports resolve, then exit.",
    )
    parser.add_argument(
        "--episode",
        action="store_true",
        help="Run a full agent episode (requires local model endpoint).",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="Run via the rollout layer (enriched episode with env rewards).",
    )
    parser.add_argument(
        "--save-jsonl",
        default=None,
        help="Save the episode to a JSONL file (requires --rollout).",
    )
    args = parser.parse_args(argv)

    if args.check_imports:
        _check_imports()
        print("All package imports OK.")
        return

    if args.rollout:
        _run_rollout(args.order, save_path=args.save_jsonl)
        return

    if args.episode:
        _run_episode(args.order)
        return

    print(f"Order: {args.order}")
    print("Use --episode to run a full agent episode against the local model.")
    print("Use --rollout to run via the rollout layer with env rewards.")
    print("Use --check-imports to validate package structure.")


def _check_imports() -> None:
    """Import every package and key module to verify the skeleton is sound."""
    # Runtime package
    import src.runtime
    import src.runtime.schemas
    import src.runtime.tools
    import src.runtime.workflows
    import src.runtime.prompts
    import src.runtime.fallbacks
    import src.runtime.tracing
    import src.runtime.agent

    # Environment package
    import src.envs
    import src.envs.validators
    import src.envs.state
    import src.envs.transitions
    import src.envs.rewards
    import src.envs.late_order_env

    # Rollouts package
    import src.rollouts
    import src.rollouts.trace_types
    import src.rollouts.episode_runner
    import src.rollouts.serializers
    import src.rollouts.export_adapters
    import src.rollouts.scripted_traces

    # Training package
    import src.training
    import src.training.curriculum
    import src.training.reward_views
    import src.training.datasets
    import src.training.experiments

    # Evaluation package
    import src.eval
    import src.eval.metrics
    import src.eval.reports

    # Scenario data
    import src.scenario_data


def _run_rollout(order_id: str, save_path: str | None = None) -> None:
    """Run one episode via the rollout layer with environment rewards."""
    from src.rollouts.episode_runner import run_enriched_episode
    from src.rollouts.serializers import save_episodes_jsonl
    from src.rollouts.export_adapters import episode_to_training_trajectory

    result = run_enriched_episode(order_id)
    print()
    result.print_summary()

    # Show training trajectory conversion
    trajectory = episode_to_training_trajectory(
        result.episode,
        reward_summary=result.reward_summary,
    )
    print(f"\nTraining trajectory: {trajectory.episode_length} steps, "
          f"total reward {trajectory.total_reward:+.4f}")

    if save_path:
        save_episodes_jsonl([result.episode], save_path)
        print(f"Saved episode to {save_path}")


def _run_episode(order_id: str) -> None:
    """Run one agent episode and print results."""
    from src.runtime.agent import run_agent_episode

    episode = run_agent_episode(order_id)
    print(f"\nEpisode complete: {episode.metrics.valid_tool_calls} tool calls, "
          f"{episode.metrics.model_calls} model calls")
    if episode.final_answer:
        print(f"Final answer: {episode.final_answer}")


if __name__ == "__main__":
    main()
