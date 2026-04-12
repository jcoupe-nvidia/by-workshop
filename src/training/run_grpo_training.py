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
    (
        "Order SO-10483 for 800 units of SKU-3080 needs expedited fulfillment. "
        "The primary DC is DC-EAST-02 but stock is low. Check inventory and "
        "find alternate sources."
    ),
    (
        "Shipment for order SO-10484 (500 units of SKU-A100) from DC-CENTRAL-01 "
        "is delayed. Evaluate whether a transfer from another DC or a supplier "
        "expedite would resolve the delay."
    ),
    (
        "Order SO-10485 for 2,000 units of SKU-7090 is partially fulfilled. "
        "Only 1,200 units shipped from DC-WEST-01. Determine how to fulfill "
        "the remaining 800 units."
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


def _score_tool_call_sequence(message_log: list[dict[str, Any]]) -> tuple[float, bool]:
    """Score a multi-step tool-call sequence using the repo's environment.

    Replays all assistant messages through LateOrderRecoveryEnv, applying
    the same dependency checking, subgoal tracking, and dense reward
    computation as the interactive runtime.

    Returns (total_reward, is_terminal).
    """
    from src.envs.late_order_env import LateOrderRecoveryEnv as RepoEnv
    from src.envs.state import TOOL_DEPENDENCIES

    env = RepoEnv()
    env.reset("SO-10482")

    total_reward = 0.0
    step_count = 0

    for msg in message_log:
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", ""))
        if not content:
            continue

        parsed = _try_parse_tool_call(content)
        if parsed is None:
            result = env.record_invalid("malformed", "Could not parse tool call JSON")
            step_reward = env.get_step_reward(env.get_step_count() - 1)
            if step_reward:
                total_reward += step_reward.total
            step_count += 1
            continue

        if "final_answer" in parsed:
            env.terminate("final_answer", parsed["final_answer"])
            terminal_reward = env.get_terminal_reward()
            if terminal_reward:
                total_reward += terminal_reward.total
            return total_reward, True

        tool_call = parsed.get("tool_call", {})
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        if tool_name not in TOOL_DEPENDENCIES:
            result = env.record_invalid("unknown_tool", f"Unknown tool: {tool_name}")
            step_reward = env.get_step_reward(env.get_step_count() - 1)
            if step_reward:
                total_reward += step_reward.total
            step_count += 1
            continue

        from src.runtime.tools import TOOL_REGISTRY
        if tool_name in TOOL_REGISTRY:
            fn, _, _ = TOOL_REGISTRY[tool_name]
            try:
                tool_result = fn(**arguments)
            except Exception:
                tool_result = {"error": "execution_failed"}
        else:
            tool_result = {"error": f"Tool {tool_name} not in registry"}

        env.step(tool_name, arguments, tool_result)
        step_reward = env.get_step_reward(env.get_step_count() - 1)
        if step_reward:
            total_reward += step_reward.total
        step_count += 1

        if env.is_terminal:
            break

    if not env.is_terminal and step_count > 0:
        env.terminate("max_iterations")
        terminal_reward = env.get_terminal_reward()
        if terminal_reward:
            total_reward += terminal_reward.total

    summary = env.get_episode_reward_summary()
    return summary.total_reward, env.is_terminal


def _try_parse_tool_call(content: str) -> dict[str, Any] | None:
    """Attempt to parse a tool call or final answer from model output."""
    try:
        return json.loads(content.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    for line in content.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
    return None


@ray.remote
class LateOrderRecoveryEnv(EnvironmentInterface[LateOrderMetadata]):
    """Multi-step environment that scores tool-call sequences using the repo's
    LateOrderRecoveryEnv for dependency checking, subgoal tracking, and
    dense decomposed rewards.

    Each episode runs until the model emits a final_answer, reaches MAX_STEPS,
    or produces only invalid outputs. The environment provides intermediate
    observations that guide the model toward the next tool call.
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.max_steps = self.cfg.get("max_steps", MAX_STEPS)

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

            reward, is_done = _score_tool_call_sequence(message_log)

            last_content = ""
            if message_log and message_log[-1].get("role") == "assistant":
                last_content = str(message_log[-1].get("content", ""))

            parsed = _try_parse_tool_call(last_content) if last_content else None
            has_final = parsed is not None and "final_answer" in parsed
            at_limit = step_num >= self.max_steps

            terminated = is_done or has_final or at_limit

            if terminated:
                observations.append({
                    "role": "environment",
                    "content": f"Episode complete after {step_num} steps. Reward: {reward:.4f}",
                })
                all_next_metadata.append(None)
                all_answers.append(last_content[:200] if last_content else None)
            else:
                obs_text = _build_step_observation(parsed, step_num)
                observations.append({"role": "environment", "content": obs_text})
                next_meta = dict(meta)
                next_meta["step_num"] = step_num
                all_next_metadata.append(next_meta)
                all_answers.append(None)

            rewards.append(reward)
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
        pass

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


def _build_step_observation(parsed: dict | None, step_num: int) -> str:
    """Build a step observation to guide the model toward the next tool."""
    if parsed is None:
        return (
            f"Step {step_num}: Could not parse your response as valid JSON. "
            f"Please respond with a JSON tool call or final answer."
        )

    tool_call = parsed.get("tool_call", {})
    tool_name = tool_call.get("name", "unknown")

    from src.envs.state import TOOL_DEPENDENCIES
    if tool_name not in TOOL_DEPENDENCIES:
        return (
            f"Step {step_num}: Unknown tool '{tool_name}'. "
            f"Available tools: {', '.join(sorted(TOOL_DEPENDENCIES.keys()))}. "
            f"Please call a valid tool."
        )

    return f"Step {step_num}: Tool '{tool_name}' executed. Continue with the next step."


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
