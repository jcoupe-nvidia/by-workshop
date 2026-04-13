"""
Full GRPO training run for the late-order recovery supply chain scenario.

Uses NeMo RL's GRPO algorithm with:
    - Multi-step environment that validates tool-call sequences using the
      repo-owned LateOrderRecoveryEnv (dependency checking, subgoal tracking,
      dense decomposed rewards)
    - IterableDataset generating supply chain prompts with canonical system prompt
    - Nemotron-3-Nano model (FP8 weights via DTensor v2)
    - 7 GPUs (GPU 0 reserved for NIM inference server)

The environment scores multi-step tool-call sequences with the same
semantics as the interactive runtime: dependency satisfaction, argument
accuracy, subgoal progression, and terminal quality. This ensures
training-time rewards align with the evaluation metrics.

Modeled after NeMo RL's sliding puzzle example with adaptations for
the supply chain domain.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pprint
import random
import sys
from typing import Any, Iterator, Optional

import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import IterableDataset

sys.path.insert(0, "/workspace")

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

# ---------------------------------------------------------------------------
# Curriculum experiment plan
# ---------------------------------------------------------------------------
# See training.experiments for the illustrative 4-stage curriculum plan
# (ExperimentPlan / build_default_experiment_plan). That plan documents
# the intended progression but is not yet wired into this runner.

# ---------------------------------------------------------------------------
# Scenario prompts — use canonical system prompt from shared module
# ---------------------------------------------------------------------------

SCENARIO_PROMPTS = [
    # --- SO-10482: Transfer from alternate DC ---
    {
        "order_id": "SO-10482",
        "prompt": (
            "Customer order SO-10482 for 1,200 units of SKU-4090 is at risk of missing "
            "the committed delivery date of 2026-04-18. The order is currently assigned to "
            "DC-WEST-01. Determine whether the order can still be fulfilled on time. "
            "If not, recommend the best mitigation action."
        ),
    },
    {
        "order_id": "SO-10482",
        "prompt": (
            "Order SO-10482 (1,200 units, SKU-4090, DC-WEST-01) may not ship by the "
            "2026-04-18 commitment. Investigate fulfillment options and recommend a "
            "mitigation action."
        ),
    },
    {
        "order_id": "SO-10482",
        "prompt": (
            "SO-10482 is flagged at-risk: 1,200 units of SKU-4090 shipping from "
            "DC-WEST-01, committed delivery 2026-04-18. Diagnose the root cause, "
            "evaluate recovery options, and recommend the optimal action."
        ),
    },

    # --- SO-10483: Supplier expedite ---
    {
        "order_id": "SO-10483",
        "prompt": (
            "Customer order SO-10483 for 500 units of SKU-100 is at risk of missing "
            "the committed delivery date of 2026-04-21. The order ships from DC-EAST-02. "
            "Determine whether fulfillment is on track and recommend the best mitigation "
            "if needed."
        ),
    },
    {
        "order_id": "SO-10483",
        "prompt": (
            "Order SO-10483 (500 units, SKU-100, DC-EAST-02, due 2026-04-21) is flagged "
            "at-risk. Investigate available options and recommend a recovery action."
        ),
    },

    # --- SO-10484: Partial fulfillment ---
    {
        "order_id": "SO-10484",
        "prompt": (
            "Customer order SO-10484 for 2,000 units of SKU-200 is at risk of missing "
            "the committed date of 2026-04-15. The order ships from DC-CENTRAL-03. "
            "Assess whether full fulfillment is possible and recommend the best action."
        ),
    },
    {
        "order_id": "SO-10484",
        "prompt": (
            "SO-10484 (2,000 units, SKU-200, DC-CENTRAL-03, due 2026-04-15) has a "
            "tight deadline. Diagnose inventory and capacity, evaluate recovery paths, "
            "and recommend an action."
        ),
    },

    # --- SO-10485: Original DC works (false alarm) ---
    {
        "order_id": "SO-10485",
        "prompt": (
            "Order SO-10485 for 200 units of SKU-300 from DC-WEST-01, due 2026-04-22, "
            "was flagged at-risk. Investigate whether the order can ship on time and "
            "recommend any needed action."
        ),
    },
    {
        "order_id": "SO-10485",
        "prompt": (
            "SO-10485 (200 units, SKU-300, DC-WEST-01, committed 2026-04-22) needs "
            "a delivery risk assessment. Check shipment status, inventory, and capacity, "
            "then advise."
        ),
    },

    # --- SO-10486: Substitute SKU ---
    {
        "order_id": "SO-10486",
        "prompt": (
            "Customer order SO-10486 for 600 units of SKU-400 is at risk of missing "
            "the committed date of 2026-04-19. The order ships from DC-EAST-02. "
            "Evaluate all fulfillment options and recommend the best recovery path."
        ),
    },
    {
        "order_id": "SO-10486",
        "prompt": (
            "SO-10486 (600 units, SKU-400, DC-EAST-02, due 2026-04-19) may not ship "
            "on time. Diagnose the issue, explore alternatives including substitutes, "
            "and recommend an action."
        ),
    },

    # --- SO-10487: Escalate (no viable option) ---
    {
        "order_id": "SO-10487",
        "prompt": (
            "Critical order SO-10487 for 3,000 units of SKU-500 is at risk of missing "
            "the committed date of 2026-04-14. The order ships from DC-CENTRAL-03. "
            "This is critical priority. Assess all options and recommend the best action."
        ),
    },
    {
        "order_id": "SO-10487",
        "prompt": (
            "SO-10487 (3,000 units, SKU-500, DC-CENTRAL-03, due 2026-04-14, critical "
            "priority) is at immediate risk. Investigate inventory, transfers, supplier "
            "options, and recommend how to proceed."
        ),
    },

    # --- SO-10488: Transfer from DC-CENTRAL-03 ---
    {
        "order_id": "SO-10488",
        "prompt": (
            "Order SO-10488 for 800 units of SKU-600 from DC-EAST-02, due 2026-04-22, "
            "is flagged at-risk. Determine whether the order can be fulfilled on time "
            "and recommend the best mitigation."
        ),
    },
    {
        "order_id": "SO-10488",
        "prompt": (
            "SO-10488 (800 units, SKU-600, DC-EAST-02, committed 2026-04-22) needs "
            "a fulfillment risk assessment. Check all sourcing options and recommend "
            "an action."
        ),
    },

    # --- SO-10489: Supplier expedite (variant) ---
    {
        "order_id": "SO-10489",
        "prompt": (
            "Customer order SO-10489 for 350 units of SKU-700 is at risk of missing "
            "the committed date of 2026-04-19. The order ships from DC-WEST-01. "
            "Investigate options and recommend the best mitigation action."
        ),
    },
    {
        "order_id": "SO-10489",
        "prompt": (
            "SO-10489 (350 units, SKU-700, DC-WEST-01, due 2026-04-19) is flagged "
            "at-risk. Diagnose the issue, evaluate recovery options, and recommend "
            "the optimal action."
        ),
    },

    # --- SO-10490: Original DC works (false alarm, variant) ---
    {
        "order_id": "SO-10490",
        "prompt": (
            "Order SO-10490 for 100 units of SKU-800 from DC-CENTRAL-03, due "
            "2026-04-25, was flagged at-risk. Assess shipment status and inventory, "
            "then recommend any necessary action."
        ),
    },
    {
        "order_id": "SO-10490",
        "prompt": (
            "SO-10490 (100 units, SKU-800, DC-CENTRAL-03, committed 2026-04-25) "
            "needs a risk assessment. Check current fulfillment status and advise "
            "on next steps."
        ),
    },

    # --- SO-10491: Transfer from DC-EAST-02 ---
    {
        "order_id": "SO-10491",
        "prompt": (
            "Customer order SO-10491 for 1,500 units of SKU-900 is at risk of missing "
            "the committed date of 2026-04-24. The order ships from DC-WEST-01. "
            "Determine whether fulfillment is feasible and recommend the best recovery."
        ),
    },
    {
        "order_id": "SO-10491",
        "prompt": (
            "SO-10491 (1,500 units, SKU-900, DC-WEST-01, due 2026-04-24) is flagged "
            "at-risk. Investigate inventory across DCs, evaluate transfers and supplier "
            "options, and recommend the best action."
        ),
    },
]


def _get_system_prompt() -> str:
    """Load the canonical system prompt from shared.tool_schemas."""
    from src.shared.tool_schemas import build_default_system_prompt
    return build_default_system_prompt()


# ---------------------------------------------------------------------------
# Multi-step environment
# ---------------------------------------------------------------------------

LateOrderMetadata = dict[str, Any]

MAX_STEPS = 12


def _process_single_assistant_message(
    content: str,
    env: Any,
    tool_registry: dict[str, Any],
) -> tuple[float, bool]:
    """Process one assistant message through the environment, returning (step_reward, is_terminal).

    Uses the shared ``validate_tool_call`` from ``runtime.schemas`` to
    ensure parsing parity between training-time rewards and interactive
    evaluation metrics. This import is intentional: the validation
    pipeline is the single source of truth for structural correctness
    (see MEDIUM-1 in the code review).
    """
    from src.runtime.schemas import (
        ParsedToolCall,
        ParsedFinalAnswer,
        ValidationError as SchemaValidationError,
        validate_tool_call,
    )

    result = validate_tool_call(content, tool_registry)

    if isinstance(result, SchemaValidationError):
        env.record_invalid(result.error_type, result.message)
        step_reward = env.get_step_reward(env.get_step_count() - 1)
        return (step_reward.total if step_reward else 0.0), env.is_terminal

    if isinstance(result, ParsedFinalAnswer):
        env.terminate("final_answer", result.answer)
        terminal_reward = env.get_terminal_reward()
        return (terminal_reward.total if terminal_reward else 0.0), True

    assert isinstance(result, ParsedToolCall)
    tool_name = result.tool_name
    arguments = result.arguments

    # Check dependencies before executing — training-time enforcement must
    # match the interactive runtime (HIGH-3 train-eval parity fix).
    from src.envs.transitions import check_preconditions
    ok, err_type, err_msg = check_preconditions(env.state, tool_name)
    if not ok:
        env.record_invalid(err_type or "dependency_violation", err_msg or "Precondition failed")
        step_reward = env.get_step_reward(env.get_step_count() - 1)
        return (step_reward.total if step_reward else 0.0), env.is_terminal

    fn, _, _ = tool_registry[tool_name]
    try:
        tool_result = fn(**arguments)
    except Exception as exc:
        tool_result = {"error": f"{type(exc).__name__}: {exc}"}

    env.step(tool_name, arguments, tool_result)
    step_reward = env.get_step_reward(env.get_step_count() - 1)
    return (step_reward.total if step_reward else 0.0), env.is_terminal


@ray.remote
class LateOrderTrainingEnv(EnvironmentInterface[LateOrderMetadata]):
    """Multi-step environment that scores tool-call sequences using the repo's
    LateOrderRecoveryEnv for dependency checking, subgoal tracking, and
    dense decomposed rewards.

    Persists a repo-owned RepoEnv per episode across NeMo RL step calls,
    advancing only the new assistant message(s) on each step and returning
    marginal (per-step) rewards. This avoids O(N²) replay and ensures NeMo RL
    receives marginal rewards suitable for GRPO advantage computation.

    Each episode runs until the model emits a final_answer, reaches MAX_STEPS,
    or produces only invalid outputs. The environment provides intermediate
    observations that guide the model toward the next tool call.
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.max_steps = self.cfg.get("max_steps", MAX_STEPS)
        self._envs: dict[int, Any] = {}
        self._tool_registry: dict[str, Any] | None = None
        self._profiling_done = False
        self._profile_result: dict | None = None

    def _get_tool_registry(self) -> dict[str, Any]:
        # Lazy import of tool implementations from runtime/.  The training
        # layer normally avoids runtime imports, but this Ray actor needs
        # the actual tool functions to execute tool calls during scoring.
        # The import is deferred so it only loads on the Ray worker, not
        # at module-import time on the driver.
        if self._tool_registry is None:
            from src.runtime.tools import TOOL_REGISTRY
            self._tool_registry = dict(TOOL_REGISTRY)
        return self._tool_registry

    def _get_or_create_env(self, idx: int, order_id: str = "SO-10482") -> Any:
        """Get or create a persistent RepoEnv for the given episode index.

        Args:
            idx: Episode index used as the env cache key.
            order_id: Order ID to initialize the environment with.
                      Sourced from episode metadata so each prompt
                      scores against its own scenario data.
        """
        if idx not in self._envs:
            from src.envs.late_order_env import LateOrderRecoveryEnv as RepoEnv
            env = RepoEnv()
            env.reset(order_id)
            self._envs[idx] = env
        return self._envs[idx]

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[LateOrderMetadata],
    ) -> EnvironmentReturn[LateOrderMetadata]:
        observations = []
        rewards = []
        terminateds = []
        all_stop_strings: list[list[str] | None] = []
        all_next_metadata: list[LateOrderMetadata | None] = []
        all_answers: list[str | None] = []
        tool_registry = self._get_tool_registry()

        for message_log, meta in zip(message_log_batch, metadata):
            if meta is None:
                observations.append({"role": "environment", "content": "Episode ended."})
                rewards.append(0.0)
                terminateds.append(True)
                all_stop_strings.append(None)
                all_next_metadata.append(None)
                all_answers.append(None)
                continue

            step_num = meta.get("step_num", 0) + 1
            episode_idx = meta.get("episode_idx")
            if episode_idx is None:
                raise ValueError(
                    "episode_idx must be set in metadata — "
                    "falling back to id(meta) risks session collisions "
                    "under concurrent rollout collection"
                )
            messages_processed = meta.get("messages_processed", 0)
            cumulative_reward = meta.get("cumulative_reward", 0.0)
            order_id = meta.get("order_id", "SO-10482")

            env = self._get_or_create_env(episode_idx, order_id=order_id)

            # Returns raw marginal rewards (not shaped). See reward_views.py for
            # the offline shaped reward path and the documented reward semantics split.
            marginal_reward = 0.0
            new_messages_processed = messages_processed
            is_done = env.is_terminal

            for msg in message_log[messages_processed:]:
                if msg.get("role") != "assistant":
                    new_messages_processed += 1
                    continue
                content = str(msg.get("content", ""))
                if not content:
                    new_messages_processed += 1
                    continue

                step_reward, is_done = _process_single_assistant_message(
                    content, env, tool_registry,
                )
                marginal_reward += step_reward
                new_messages_processed += 1

                if is_done:
                    break

            new_cumulative = cumulative_reward + marginal_reward

            last_content = ""
            if message_log and message_log[-1].get("role") == "assistant":
                last_content = str(message_log[-1].get("content", ""))

            at_limit = step_num >= self.max_steps
            terminated = is_done or at_limit

            if not is_done and at_limit and not env.is_terminal:
                env.terminate("max_iterations")
                terminal_reward = env.get_terminal_reward()
                if terminal_reward:
                    marginal_reward += terminal_reward.total
                    new_cumulative += terminal_reward.total
                terminated = True

            if terminated:
                observations.append({
                    "role": "environment",
                    "content": f"Episode complete after {step_num} steps. Reward: {new_cumulative:.4f}",
                })
                all_next_metadata.append(None)
                all_answers.append(last_content[:200] if last_content else None)
                if episode_idx in self._envs:
                    del self._envs[episode_idx]
            else:
                obs_text = _build_step_observation(last_content, step_num, tool_registry)
                observations.append({"role": "environment", "content": obs_text})
                next_meta = dict(meta)
                next_meta["step_num"] = step_num
                next_meta["messages_processed"] = new_messages_processed
                next_meta["cumulative_reward"] = new_cumulative
                next_meta["episode_idx"] = episode_idx
                all_next_metadata.append(next_meta)
                all_answers.append(None)

            rewards.append(marginal_reward)
            terminateds.append(terminated)
            all_stop_strings.append(None)

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=all_answers,
        )

    def shutdown(self):
        self._envs.clear()

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        mean_reward = final_rewards.float().mean().item() if len(final_rewards) > 0 else 0.0
        success_rate = (
            (final_rewards >= 0.7).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )

        if not self._profiling_done:
            self._profiling_done = True
            if len(final_rewards) > 0:
                from src.training.grpo_notebook import profile_reward_distribution
                pilot_specs = [
                    {"extra_env_info": {"reward": float(r)}}
                    for r in final_rewards
                ]
                profile = profile_reward_distribution(pilot_specs, label="post-first-step")
                self._profile_result = profile
                if not profile["pass"]:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Degenerate reward distribution detected after first step. "
                        "GRPO training may not provide useful signal."
                    )

        metrics = {
            "mean_reward": round(mean_reward, 4),
            "high_reward_rate": round(success_rate, 4),
        }
        if self._profile_result is not None:
            metrics["reward_profile"] = self._profile_result
        return batch, metrics


def _build_step_observation(
    last_content: str,
    step_num: int,
    tool_registry: dict[str, Any],
) -> str:
    """Build a step observation to guide the model toward the next tool."""
    from src.runtime.schemas import (
        ParsedToolCall,
        ParsedFinalAnswer,
        validate_tool_call,
    )

    if not last_content:
        return (
            f"Step {step_num}: Could not parse your response as valid JSON. "
            f"Please respond with a JSON tool call or final answer."
        )

    result = validate_tool_call(last_content, tool_registry)

    if isinstance(result, ParsedToolCall):
        return f"Step {step_num}: Tool '{result.tool_name}' executed. Continue with the next step."

    if isinstance(result, ParsedFinalAnswer):
        return f"Step {step_num}: Final answer received."

    from src.envs.state import TOOL_DEPENDENCIES
    return (
        f"Step {step_num}: Could not parse your response. "
        f"Available tools: {', '.join(sorted(TOOL_DEPENDENCIES.keys()))}. "
        f"Please respond with a valid JSON tool call or final answer."
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LateOrderDataset(IterableDataset):
    """Generates supply chain prompts with canonical system prompt for GRPO training."""

    def __init__(self, tokenizer, length: int, add_system_prompt: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.length = length
        self.add_system_prompt = add_system_prompt
        self._system_prompt = _get_system_prompt()

    def __iter__(self) -> Iterator[DatumSpec]:
        for i in itertools.count():
            scenario = random.choice(SCENARIO_PROMPTS)
            prompt_text = scenario["prompt"]
            order_id = scenario["order_id"]

            messages = []
            if self.add_system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": prompt_text})

            prompt_content = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            tokenized = self.tokenizer(
                prompt_content, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]

            message_log: LLMMessageLogType = [
                {
                    "role": "user",
                    "content": prompt_content,
                    "token_ids": tokenized,
                }
            ]

            metadata: LateOrderMetadata = {
                "prompt_idx": i % len(SCENARIO_PROMPTS),
                "task_name": "late_order_recovery",
                "order_id": order_id,
                "step_num": 0,
                "episode_idx": i,
                "messages_processed": 0,
                "cumulative_reward": 0.0,
                "parallel_tool_calls": False,
                # Async GRPO metadata placeholders (RL_ARCHITECTURE.md § Async Metadata Contract).
                # Present with synchronous defaults so the contract is visible
                # in serialized output and ready for async collection.
                "gen_weight_version": 0,
                "train_weight_version": 0,
                "trajectory_age_ms": 0,
                "replay_status": "accepted",
            }

            datum: DatumSpec = {
                "message_log": message_log,
                "length": len(tokenized),
                "extra_env_info": metadata,
                "loss_multiplier": 1.0,
                "idx": i,
                "task_name": "late_order_recovery",
            }
            yield datum

    def __len__(self):
        return self.length


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run GRPO training for late-order recovery")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    return parser.parse_known_args()


def main() -> None:
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "grpo_config.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    init_ray()

    set_seed(config["grpo"]["seed"])

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # Build datasets
    ds_length = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        * config["grpo"]["max_num_steps"]
    )
    train_dataset = LateOrderDataset(tokenizer=tokenizer, length=ds_length)
    val_dataset = LateOrderDataset(tokenizer=tokenizer, length=config["grpo"]["max_val_samples"])

    # Setup policy, generation, cluster, etc.
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    # Create environment
    env = LateOrderTrainingEnv.options(num_gpus=0).remote(
        cfg=dict(config["env"]["late_order_recovery"])
    )
    task_to_env = {"late_order_recovery": env}
    val_task_to_env = task_to_env

    # Reward distribution profiling is handled inside
    # LateOrderTrainingEnv.global_post_process_and_metrics() as a one-shot
    # check after the first training step, where actual model-generated
    # rewards are available (RL_ARCHITECTURE.md § Verification and Reward Design).
    print("\nReward distribution profiling will run in the environment's post-process hook after the first step.")

    print("\nStarting GRPO training...")
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )

    print("\nGRPO training completed successfully!")


if __name__ == "__main__":
    main()
