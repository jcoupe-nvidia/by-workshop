"""
Reward visualization helpers for notebook GRPO training runs.

Provides matplotlib-based plots for:
    - Total reward distribution across trajectories
    - GRPO group-relative advantages
    - Per-step shaped reward heatmap
    - Reward component breakdown by episode

These functions consume the structured data dict returned by
``grpo_notebook.extract_reward_plot_data()``.

Owns:
    - Reward distribution plots
    - Advantage distribution plots
    - Per-step reward heatmaps
    - Summary table rendering

Does NOT own:
    - Reward computation (see envs.rewards)
    - Reward shaping (see training.reward_views)
    - Data extraction (see training.grpo_notebook)
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_grpo_rewards(
    plot_data: dict[str, Any],
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """Plot a 2x2 grid of GRPO reward visualizations.

    Panels:
        1. Total reward per trajectory (bar chart)
        2. Group-relative advantage per trajectory (bar chart)
        3. Per-step shaped rewards (heatmap)
        4. Combined vs trajectory reward comparison (grouped bar)

    Args:
        plot_data: Dict from ``extract_reward_plot_data()``.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure (displayed automatically in notebooks).
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"GRPO Reward Analysis — stage: {plot_data['stage']}",
        fontsize=13,
        fontweight="bold",
    )

    labels = plot_data["episode_labels"]
    n = len(labels)
    x = np.arange(n)

    # Panel 1: Total rewards
    ax1 = axes[0, 0]
    rewards = plot_data["total_rewards"]
    colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in rewards]
    ax1.bar(x, rewards, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_title("Total Reward per Trajectory")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    mean_r = sum(rewards) / len(rewards) if rewards else 0
    ax1.axhline(y=mean_r, color="blue", linewidth=1, linestyle=":", label=f"mean={mean_r:.3f}")
    ax1.legend(fontsize=8)

    # Panel 2: Group advantages
    ax2 = axes[0, 1]
    advantages = plot_data["advantages"]
    adv_colors = ["#3498db" if a >= 0 else "#e67e22" for a in advantages]
    ax2.bar(x, advantages, color=adv_colors, edgecolor="white", linewidth=0.5)
    ax2.set_title("Group-Relative Advantage")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Advantage (reward − group mean)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    # Panel 3: Per-step shaped rewards heatmap
    ax3 = axes[1, 0]
    shaped_data = plot_data["shaped_step_data"]
    max_steps = max(len(d["step_rewards"]) for d in shaped_data)
    # Include terminal column
    matrix = np.full((n, max_steps + 1), np.nan)
    for i, d in enumerate(shaped_data):
        for j, r in enumerate(d["step_rewards"]):
            matrix[i, j] = r
        if d["terminal_reward"] is not None:
            matrix[i, max_steps] = d["terminal_reward"]

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="white")
    im = ax3.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )
    ax3.set_title("Per-Step Shaped Rewards")
    ax3.set_xlabel("Step (T=terminal)")
    ax3.set_ylabel("Episode")
    step_labels = [str(s) for s in range(max_steps)] + ["T"]
    ax3.set_xticks(range(max_steps + 1))
    ax3.set_xticklabels(step_labels, fontsize=8)
    ax3.set_yticks(range(n))
    ax3.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax3, shrink=0.8, label="Shaped reward")

    # Panel 4: Combined vs trajectory reward
    ax4 = axes[1, 1]
    combined = [d["combined"] for d in shaped_data]
    traj_r = [d["trajectory_reward"] for d in shaped_data]
    width = 0.35
    ax4.bar(x - width / 2, combined, width, label="Combined", color="#9b59b6")
    ax4.bar(x + width / 2, traj_r, width, label="Trajectory", color="#1abc9c")
    ax4.set_title(
        f"Combined vs Trajectory Reward "
        f"(step_w={plot_data['step_weight']}, traj_w={plot_data['trajectory_weight']})"
    )
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Reward")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax4.legend(fontsize=8)
    ax4.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    return fig


def plot_reward_components(
    plot_data: dict[str, Any],
    enriched_results: list,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot reward component breakdown per episode.

    Shows a stacked bar chart of the reward signal components for
    each episode, making it easy to see which components drive the
    total reward.

    Args:
        plot_data: Dict from ``extract_reward_plot_data()``.
        enriched_results: List of EnrichedEpisodeResult.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    component_names = [
        "valid_call", "correct_tool", "correct_arguments",
        "dependency_satisfied", "non_redundant", "progress",
        "efficiency", "terminal_quality",
    ]
    colors = plt.cm.Set3(np.linspace(0, 1, len(component_names)))

    labels = plot_data["episode_labels"]
    n = len(labels)
    x = np.arange(n)

    # Aggregate component totals per episode from reward summaries
    component_totals: dict[str, list[float]] = {c: [] for c in component_names}

    for result in enriched_results:
        rs = result.reward_summary
        totals = {c: 0.0 for c in component_names}
        for sr in rs.step_rewards:
            for c in component_names:
                totals[c] += getattr(sr, c, 0.0)
        if rs.terminal_reward:
            for c in component_names:
                totals[c] += getattr(rs.terminal_reward, c, 0.0)
        for c in component_names:
            component_totals[c].append(totals[c])

    # Stacked bar
    bottom = np.zeros(n)
    for i, comp in enumerate(component_names):
        values = np.array(component_totals[comp])
        ax.bar(x, values, bottom=bottom, label=comp, color=colors[i],
               edgecolor="white", linewidth=0.3)
        bottom += values

    ax.set_title("Reward Component Breakdown per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Component Value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    fig.tight_layout()
    return fig


def print_grpo_summary_table(plot_data: dict[str, Any]) -> None:
    """Print a compact summary table of GRPO trajectory metrics.

    Args:
        plot_data: Dict from ``extract_reward_plot_data()``.
    """
    labels = plot_data["episode_labels"]
    rewards = plot_data["total_rewards"]
    advantages = plot_data["advantages"]
    shaped = plot_data["shaped_step_data"]

    print(f"{'Episode':<25} {'Reward':>8} {'Advntg':>8} {'Combined':>9} {'TrajRwd':>8} {'Steps':>6}")
    print("-" * 68)

    for i, label in enumerate(labels):
        n_steps = len(shaped[i]["step_rewards"])
        print(
            f"{label:<25} "
            f"{rewards[i]:>+8.4f} "
            f"{advantages[i]:>+8.4f} "
            f"{shaped[i]['combined']:>+9.4f} "
            f"{shaped[i]['trajectory_reward']:>+8.4f} "
            f"{n_steps:>6}"
        )

    print("-" * 68)
    mean_r = sum(rewards) / len(rewards) if rewards else 0
    mean_a = sum(advantages) / len(advantages) if advantages else 0
    mean_c = sum(d["combined"] for d in shaped) / len(shaped) if shaped else 0
    print(f"{'Mean':<25} {mean_r:>+8.4f} {mean_a:>+8.4f} {mean_c:>+9.4f}")
