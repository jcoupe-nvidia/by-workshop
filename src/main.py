"""
Minimal entrypoint for running one episode outside the notebook.

Usage:
    python -m src.main [--order ORDER_ID]

This validates the package structure imports correctly and will be
extended in later phases to run a full agent episode against the
local model endpoint.
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
    args = parser.parse_args(argv)

    if args.check_imports:
        _check_imports()
        print("All package imports OK.")
        return

    # Full episode execution will be wired up in Phase 3 (runtime refactor).
    print(f"Episode runner not yet wired. Order: {args.order}")
    print("Run with --check-imports to validate package structure.")


def _check_imports() -> None:
    """Import every package and key module to verify the skeleton is sound."""
    import src.runtime
    import src.runtime.schemas
    import src.envs
    import src.envs.validators
    import src.rollouts
    import src.rollouts.trace_types
    import src.training
    import src.training.curriculum
    import src.systems
    import src.eval

    # Existing modules still importable via original paths
    import src.schema
    import src.scenario_data
    import src.tools
    import src.skills
    import src.fallbacks
    import src.agent_loop
    import src.evaluation
    import src.training_export


if __name__ == "__main__":
    main()
