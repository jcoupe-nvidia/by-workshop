"""
NeMo Gym rollout collection adapter.

Bridges the rollout layer with NeMo Gym's training-time infrastructure:

    - Converts enriched Episodes to NemoGymResultRows for reward profiling
    - Builds RolloutCollectionConfig for NeMo Gym rollout collection runs
    - Runs reward profiling on collected rollouts via RewardProfiler

Owns:
    - Enriched episode -> NeMo Gym result row conversion (via envs.nemo_gym_adapter)
    - RolloutCollectionConfig construction for the late-order-recovery task
    - RewardProfiler integration for post-collection metrics

Does NOT own:
    - Environment state or reward computation (see envs/)
    - NeMo Gym resource server implementation (see envs.nemo_gym_adapter)
    - Episode types or enrichment (see rollouts.trace_types, rollouts.episode_runner)
    - Training dataset construction (see training/)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nemo_gym.rollout_collection import (
    RolloutCollectionConfig,
    RolloutCollectionHelper,
)
from nemo_gym.reward_profile import RewardProfiler

from src.envs.nemo_gym_adapter import (
    NemoGymResultRow,
    build_rollout_input_row,
    episode_to_nemo_gym_row,
    save_nemo_gym_inputs_jsonl,
    save_nemo_gym_rows_jsonl,
)
from src.rollouts.episode_runner import EnrichedEpisodeResult


def enriched_to_nemo_gym_row(
    result: EnrichedEpisodeResult,
    agent_name: str = "late-order-recovery-agent",
) -> NemoGymResultRow:
    """Convert an enriched episode result to a NeMo Gym result row.

    This is the bridge between the rollout layer (enriched episodes)
    and NeMo Gym's reward profiling infrastructure.
    """
    return episode_to_nemo_gym_row(
        episode=result.episode,
        reward_summary=result.reward_summary,
        agent_name=agent_name,
    )


def enriched_batch_to_nemo_gym_rows(
    results: list[EnrichedEpisodeResult],
    agent_name: str = "late-order-recovery-agent",
) -> list[NemoGymResultRow]:
    """Convert a batch of enriched results to NeMo Gym result rows."""
    return [enriched_to_nemo_gym_row(r, agent_name) for r in results]


def build_collection_config(
    input_jsonl_path: str,
    output_jsonl_path: str,
    agent_name: str = "late-order-recovery-agent",
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
    num_samples_in_parallel: int | None = None,
    num_repeats: int | None = None,
    limit: int | None = None,
) -> RolloutCollectionConfig:
    """Build a NeMo Gym RolloutCollectionConfig for the late-order task.

    This config is consumed by RolloutCollectionHelper.run_from_config()
    to orchestrate rollout collection against the resource server.

    Args:
        input_jsonl_path: Path to JSONL file with task input rows.
        output_jsonl_path: Path for collected rollout output.
        agent_name: Agent server name for routing.
        temperature: Sampling temperature.
        max_output_tokens: Max tokens per model response.
        num_samples_in_parallel: Concurrent rollout limit.
        num_repeats: Repeat each task N times for variance.
        limit: Max tasks to process.
    """
    return RolloutCollectionConfig(
        input_jsonl_fpath=input_jsonl_path,
        output_jsonl_fpath=output_jsonl_path,
        agent_name=agent_name,
        responses_create_params={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        },
        num_samples_in_parallel=num_samples_in_parallel,
        num_repeats=num_repeats,
        limit=limit,
    )


def prepare_rollout_inputs(
    order_ids: list[str],
    task_prompts: list[str],
    output_path: str,
    agent_name: str = "late-order-recovery-agent",
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
) -> str:
    """Prepare a NeMo Gym input JSONL file for rollout collection.

    Writes one row per order to the output path. Returns the path.

    Args:
        order_ids: List of order IDs to investigate.
        task_prompts: Corresponding task prompts.
        output_path: Path to write the input JSONL.
        agent_name: Agent server name.
        temperature: Sampling temperature.
        max_output_tokens: Max tokens per response.
    """
    rows = [
        build_rollout_input_row(
            order_id=oid,
            task_prompt=prompt,
            agent_name=agent_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        for oid, prompt in zip(order_ids, task_prompts)
    ]
    save_nemo_gym_inputs_jsonl(rows, output_path)
    return output_path


def profile_rollout_rewards(
    input_rows: list[dict[str, Any]],
    result_rows: list[NemoGymResultRow],
    output_dir: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run NeMo Gym RewardProfiler on collected rollout results.

    Computes group-level and agent-level reward statistics across
    the collected rollout batch.

    Args:
        input_rows: The original task input rows (list of dicts).
        result_rows: Collected result rows (NemoGymResultRow objects).
        output_dir: If provided, writes profiling results to disk.

    Returns:
        Tuple of (group_level_metrics, agent_level_metrics).
    """
    profiler = RewardProfiler()

    result_dicts = [row.to_dict() for row in result_rows]
    group_metrics, agent_metrics = profiler.profile_from_data(
        rows=input_rows,
        results=result_dicts,
    )

    if output_dir is not None:
        base_path = Path(output_dir)
        profiler.write_to_disk(group_metrics, agent_metrics, base_path)

    return group_metrics, agent_metrics


def save_enriched_as_nemo_gym(
    results: list[EnrichedEpisodeResult],
    output_path: str,
    agent_name: str = "late-order-recovery-agent",
) -> str:
    """Convert enriched episodes to NeMo Gym result rows and write to JSONL.

    This is the main export path from the rollout layer to NeMo Gym's
    reward profiling and training infrastructure.

    Args:
        results: Enriched episode results from run_enriched_episode().
        output_path: Path to write the output JSONL.
        agent_name: Agent name for result row metadata.

    Returns:
        The output path.
    """
    rows = enriched_batch_to_nemo_gym_rows(results, agent_name)
    save_nemo_gym_rows_jsonl(rows, output_path)
    return output_path
