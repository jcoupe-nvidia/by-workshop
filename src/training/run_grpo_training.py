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
# Scenario prompts — use canonical system prompt from shared module
# ---------------------------------------------------------------------------

SCENARIO_PROMPTS = [
    (
        "Customer order SO-10482 for 1,200 units of SKU-4090 is at risk of missing "
        "the committed delivery date of 2026-04-18. The order is currently assigned to "
        "DC-WEST-01. Determine whether the order can still be fulfilled on time. "
        "If not, recommend the best mitigation action."
    ),
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

    Uses the same validation pipeline as the interactive runtime to ensure
    parsing parity between training-time rewards and evaluation metrics.
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

    fn, _, _ = tool_registry[tool_name]
    try:
        tool_result = fn(**arguments)
    except Exception:
        tool_result = {"error": "execution_failed"}

    env.step(tool_name, arguments, tool_result)
    step_reward = env.get_step_reward(env.get_step_count() - 1)
    return (step_reward.total if step_reward else 0.0), env.is_terminal


@ray.remote
class LateOrderRecoveryEnv(EnvironmentInterface[LateOrderMetadata]):
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

    def _get_tool_registry(self) -> dict[str, Any]:
        if self._tool_registry is None:
            from src.runtime.tools import TOOL_REGISTRY
            self._tool_registry = dict(TOOL_REGISTRY)
        return self._tool_registry

    def _get_or_create_env(self, idx: int) -> Any:
        """Get or create a persistent RepoEnv for the given episode index."""
        if idx not in self._envs:
            from src.envs.late_order_env import LateOrderRecoveryEnv as RepoEnv
            env = RepoEnv()
            env.reset("SO-10482")
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
            episode_idx = meta.get("episode_idx", id(meta))
            messages_processed = meta.get("messages_processed", 0)
            cumulative_reward = meta.get("cumulative_reward", 0.0)

            env = self._get_or_create_env(episode_idx)

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
        return batch, {
            "mean_reward": round(mean_reward, 4),
            "tool_call_success_rate": round(success_rate, 4),
        }


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
            prompt_text = random.choice(SCENARIO_PROMPTS)

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
                "step_num": 0,
                "episode_idx": i,
                "messages_processed": 0,
                "cumulative_reward": 0.0,
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
    env = LateOrderRecoveryEnv.options(num_gpus=0).remote(
        cfg=dict(config["env"]["late_order_recovery"])
    )
    task_to_env = {"late_order_recovery": env}
    val_task_to_env = task_to_env

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
