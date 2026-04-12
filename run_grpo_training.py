"""
Full GRPO training run for the late-order recovery supply chain scenario.

Uses NeMo RL's GRPO algorithm with:
    - Custom single-turn environment that scores tool-call quality
    - IterableDataset generating supply chain prompts
    - Nemotron-3-Nano model (FP8 weights via DTensor v2)
    - 7 GPUs (GPU 0 reserved for NIM inference server)

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

# Add workspace to path for scenario data imports
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
# Scenario data (inline for self-containment)
# ---------------------------------------------------------------------------

SCENARIO_PROMPTS = [
    (
        "Customer order SO-10482 for 1,200 units of SKU-4090 is at risk of missing "
        "the committed delivery date of 2025-05-15. The order is currently assigned to "
        "DC-WEST-01. Determine whether the order can still be fulfilled on time. "
        "If not, recommend the best mitigation action.\n\n"
        "Available tools: get_order, get_shipment_status, get_inventory, "
        "find_alternate_inventory, get_transfer_eta, get_supplier_expedite_options, "
        "get_fulfillment_capacity, score_recovery_options.\n\n"
        "Respond with a JSON tool call in this format:\n"
        '{"tool_call": {"name": "tool_name", "arguments": {...}}}'
    ),
    (
        "Order SO-10483 for 800 units of SKU-3080 needs expedited fulfillment. "
        "The primary DC is DC-EAST-02 but stock is low. Check inventory and "
        "find alternate sources.\n\n"
        "Available tools: get_order, get_shipment_status, get_inventory, "
        "find_alternate_inventory, get_transfer_eta, get_supplier_expedite_options, "
        "get_fulfillment_capacity, score_recovery_options.\n\n"
        "Respond with a JSON tool call in this format:\n"
        '{"tool_call": {"name": "tool_name", "arguments": {...}}}'
    ),
    (
        "Shipment for order SO-10484 (500 units of SKU-A100) from DC-CENTRAL-01 "
        "is delayed. Evaluate whether a transfer from another DC or a supplier "
        "expedite would resolve the delay.\n\n"
        "Available tools: get_order, get_shipment_status, get_inventory, "
        "find_alternate_inventory, get_transfer_eta, get_supplier_expedite_options, "
        "get_fulfillment_capacity, score_recovery_options.\n\n"
        "Respond with a JSON tool call in this format:\n"
        '{"tool_call": {"name": "tool_name", "arguments": {...}}}'
    ),
    (
        "Order SO-10485 for 2,000 units of SKU-7090 is partially fulfilled. "
        "Only 1,200 units shipped from DC-WEST-01. Determine how to fulfill "
        "the remaining 800 units.\n\n"
        "Available tools: get_order, get_shipment_status, get_inventory, "
        "find_alternate_inventory, get_transfer_eta, get_supplier_expedite_options, "
        "get_fulfillment_capacity, score_recovery_options.\n\n"
        "Respond with a JSON tool call in this format:\n"
        '{"tool_call": {"name": "tool_name", "arguments": {...}}}'
    ),
]

VALID_TOOLS = {
    "get_order", "get_shipment_status", "get_inventory",
    "find_alternate_inventory", "get_transfer_eta",
    "get_supplier_expedite_options", "get_fulfillment_capacity",
    "score_recovery_options", "recommend_action",
}

SYSTEM_PROMPT = (
    "You are a supply chain operations agent. You help resolve late order "
    "delivery risks by analyzing orders, checking inventory, and recommending "
    "recovery actions. Always respond with structured tool calls."
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

LateOrderMetadata = dict[str, Any]


def score_response(content: str) -> float:
    """Score a model response for tool-call quality.

    Rewards:
        +1.0  valid JSON with correct tool_call structure and known tool
        +0.5  valid JSON with tool_call but unknown tool
        +0.2  valid JSON but missing tool_call structure
        -0.5  not valid JSON
    """
    if not content:
        return -0.5

    try:
        parsed = json.loads(content.strip())
    except (json.JSONDecodeError, ValueError):
        # Check for partial JSON in mixed text
        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    parsed = json.loads(line)
                    break
                except (json.JSONDecodeError, ValueError):
                    continue
            elif line.startswith("```"):
                continue
        else:
            return -0.5

    if not isinstance(parsed, dict):
        return -0.5

    tool_call = parsed.get("tool_call")
    if tool_call is None:
        return 0.2

    if not isinstance(tool_call, dict):
        return 0.2

    name = tool_call.get("name", "")
    has_args = "arguments" in tool_call

    if name in VALID_TOOLS and has_args:
        return 1.0
    elif name in VALID_TOOLS:
        return 0.7
    else:
        return 0.5


@ray.remote
class LateOrderRecoveryEnv(EnvironmentInterface[LateOrderMetadata]):
    """Single-turn environment that scores tool-call quality for supply chain tasks."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}

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

            last_content = ""
            if message_log and message_log[-1].get("role") == "assistant":
                last_content = str(message_log[-1].get("content", ""))

            reward = score_response(last_content)

            observations.append({
                "role": "environment",
                "content": f"Tool call scored. Reward: {reward:.1f}",
            })
            rewards.append(reward)
            terminateds.append(True)
            all_stop_strings.append(None)
            all_next_metadata.append(None)
            all_answers.append(last_content[:200] if last_content else None)

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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LateOrderDataset(IterableDataset):
    """Generates supply chain prompts indefinitely for GRPO training."""

    def __init__(self, tokenizer, length: int, add_system_prompt: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.length = length
        self.add_system_prompt = add_system_prompt

    def __iter__(self) -> Iterator[DatumSpec]:
        for i in itertools.count():
            prompt_text = random.choice(SCENARIO_PROMPTS)

            messages = []
            if self.add_system_prompt:
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
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
