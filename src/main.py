"""
Minimal entrypoint for running one episode outside the notebook.

Usage:
    python -m src.main [--order ORDER_ID] [--check-imports] [--episode] [--rollout]
                       [--nat] [--nemo-gym-export PATH]

This validates the package structure imports correctly and can run
a full agent episode against the local model endpoint. Supports both
direct tool dispatch and NAT FunctionGroup dispatch paths.
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
        "--nat",
        action="store_true",
        help="Use NAT FunctionGroup dispatch and NIMModelConfig for execution.",
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
    parser.add_argument(
        "--nemo-gym-export",
        default=None,
        metavar="PATH",
        help="Export the enriched episode as a NeMo Gym result row JSONL (requires --rollout).",
    )
    parser.add_argument(
        "--art-export",
        default=None,
        metavar="PATH",
        help="Export the episode as an art.Trajectory JSONL (requires --rollout).",
    )
    args = parser.parse_args(argv)

    if args.check_imports:
        _check_imports()
        print("All package imports OK.")
        return

    if args.rollout:
        _run_rollout(
            args.order,
            use_nat=args.nat,
            save_path=args.save_jsonl,
            nemo_gym_path=args.nemo_gym_export,
            art_path=args.art_export,
        )
        return

    if args.episode:
        _run_episode(args.order, use_nat=args.nat)
        return

    print(f"Order: {args.order}")
    print("Use --episode to run a full agent episode against the local model.")
    print("Use --episode --nat to run via NAT FunctionGroup dispatch.")
    print("Use --rollout to run via the rollout layer with env rewards.")
    print("Use --rollout --nemo-gym-export PATH to export NeMo Gym result rows.")
    print("Use --rollout --art-export PATH to export art.Trajectory JSONL.")
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
    import src.runtime.nat_tools
    import src.runtime.nat_llm
    import src.runtime.atif_adapter

    # Environment package
    import src.envs
    import src.envs.validators
    import src.envs.state
    import src.envs.transitions
    import src.envs.rewards
    import src.envs.late_order_env
    import src.envs.nemo_gym_adapter

    # Rollouts package
    import src.rollouts
    import src.rollouts.trace_types
    import src.rollouts.episode_runner
    import src.rollouts.serializers
    import src.rollouts.export_adapters
    import src.rollouts.scripted_traces
    import src.rollouts.nemo_gym_rollouts

    # Training package
    import src.training
    import src.training.curriculum
    import src.training.reward_views
    import src.training.datasets
    import src.training.experiments
    import src.training.openpipe_art_adapter

    # Evaluation package
    import src.eval
    import src.eval.metrics
    import src.eval.reports

    # Scenario data
    import src.scenario_data


def _run_rollout(
    order_id: str,
    use_nat: bool = False,
    save_path: str | None = None,
    nemo_gym_path: str | None = None,
    art_path: str | None = None,
) -> None:
    """Run one episode via the rollout layer with environment rewards."""
    from src.rollouts.serializers import save_episodes_jsonl
    from src.rollouts.export_adapters import episode_to_training_trajectory

    if use_nat:
        from src.rollouts.episode_runner import run_enriched_episode_nat
        result = run_enriched_episode_nat(order_id)
    else:
        from src.rollouts.episode_runner import run_enriched_episode
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

    if nemo_gym_path:
        from src.rollouts.nemo_gym_rollouts import save_enriched_as_nemo_gym
        save_enriched_as_nemo_gym([result], nemo_gym_path)
        print(f"Saved NeMo Gym result row to {nemo_gym_path}")

    if art_path:
        from src.training.openpipe_art_adapter import (
            episode_to_art_trajectory,
            save_art_trajectories_jsonl,
        )
        art_traj = episode_to_art_trajectory(result.episode)
        save_art_trajectories_jsonl([art_traj], art_path)
        print(f"Saved art.Trajectory to {art_path}")


def _run_episode(order_id: str, use_nat: bool = False) -> None:
    """Run one agent episode and print results."""
    if use_nat:
        from src.runtime.agent import run_agent_episode_nat
        episode = run_agent_episode_nat(order_id)
    else:
        from src.runtime.agent import run_agent_episode
        episode = run_agent_episode(order_id)

    print(f"\nEpisode complete: {episode.metrics.valid_tool_calls} tool calls, "
          f"{episode.metrics.model_calls} model calls")
    if episode.final_answer:
        print(f"Final answer: {episode.final_answer}")


if __name__ == "__main__":
    main()
