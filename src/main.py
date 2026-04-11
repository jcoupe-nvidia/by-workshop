"""
Minimal entrypoint for running one episode outside the notebook.

Usage:
    python -m src.main [--order ORDER_ID] [--check-imports] [--episode]

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
        "--structured",
        action="store_true",
        help="Use structured Episode output instead of legacy AgentTrace.",
    )
    args = parser.parse_args(argv)

    if args.check_imports:
        _check_imports()
        print("All package imports OK.")
        return

    if args.episode:
        _run_episode(args.order, structured=args.structured)
        return

    print(f"Order: {args.order}")
    print("Use --episode to run a full agent episode against the local model.")
    print("Use --check-imports to validate package structure.")


def _check_imports() -> None:
    """Import every package and key module to verify the skeleton is sound."""
    # New runtime package
    import src.runtime
    import src.runtime.schemas
    import src.runtime.tools
    import src.runtime.workflows
    import src.runtime.prompts
    import src.runtime.fallbacks
    import src.runtime.tracing
    import src.runtime.agent

    # Other packages from Phase 1
    import src.envs
    import src.envs.validators
    import src.rollouts
    import src.rollouts.trace_types
    import src.training
    import src.training.curriculum
    import src.systems
    import src.eval

    # Backward-compatible shims
    import src.schema
    import src.scenario_data
    import src.tools
    import src.skills
    import src.fallbacks
    import src.agent_loop
    import src.evaluation
    import src.training_export


def _run_episode(order_id: str, structured: bool = False) -> None:
    """Run one agent episode and print results."""
    if structured:
        from src.runtime.agent import run_agent_episode
        episode = run_agent_episode(order_id)
        print(f"\nEpisode complete: {episode.metrics.valid_tool_calls} tool calls, "
              f"{episode.metrics.model_calls} model calls")
        if episode.final_answer:
            print(f"Final answer: {episode.final_answer}")
    else:
        from src.runtime.agent import run_agent, print_trace_summary
        trace = run_agent(order_id)
        print()
        print_trace_summary(trace)


if __name__ == "__main__":
    main()
