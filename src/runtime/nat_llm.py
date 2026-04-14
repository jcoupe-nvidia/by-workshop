"""
NAT LLM configuration and model-call adapter for the local Nemotron endpoint.

Provides a NIMModelConfig pointing to the locally deployed NIM instance
described in documents/llm-access.md, plus a call_model_nim() function that
uses the config for HTTP calls instead of raw endpoint constants.

Owns:
    - NAT LLM provider configuration for the local Nemotron NIM
    - Factory function for building the config from llm-access defaults
    - NIM-backed model call adapter

Does NOT own:
    - Model serving or inference (the NIM container handles that)
    - Agent orchestration or prompt policy
    - Tool definitions or reward semantics
"""
from __future__ import annotations

import os

import requests

from nat.llm.nim_llm import NIMModelConfig


# Defaults from documents/llm-access.md (host-gateway address for container use)
_DEFAULT_BASE_URL = os.environ.get("MODEL_BASE_URL", "http://172.17.0.1:8000/v1")
_DEFAULT_MODEL_NAME = "nvidia/nemotron-3-nano"
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_TEMPERATURE = 0.1
_REQUEST_TIMEOUT = 60


def build_nim_config(
    base_url: str = _DEFAULT_BASE_URL,
    model_name: str = _DEFAULT_MODEL_NAME,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> NIMModelConfig:
    """Build a NIMModelConfig for the local Nemotron endpoint.

    Parameters match the defaults in documents/llm-access.md.
    Override any parameter for testing or alternate deployments.
    """
    return NIMModelConfig(
        base_url=base_url,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def call_model_nim(
    messages: list[dict[str, str]],
    config: NIMModelConfig | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Call the local Nemotron NIM using a NIMModelConfig for configuration.

    This replaces the raw-HTTP call_model() in agent.py. It derives the
    endpoint URL, model name, and generation parameters from the NAT config
    object, keeping the model adapter consistent with the NAT runtime surface.

    Args:
        messages: OpenAI-style message list.
        config: NIMModelConfig to use. If None, builds one from defaults.
        max_tokens: Override config.max_tokens for this call.
        temperature: Override config.temperature for this call.

    Returns:
        The assistant message content string.

    Raises:
        requests.RequestException: If the HTTP request fails.
        ValueError: If the response is missing expected fields.
    """
    if config is None:
        config = build_nim_config()

    effective_max_tokens = max_tokens if max_tokens is not None else config.max_tokens
    effective_temperature = temperature if temperature is not None else config.temperature

    url = f"{config.base_url}/chat/completions"
    payload = {
        "model": config.model_name,
        "messages": messages,
        "max_tokens": effective_max_tokens,
        "temperature": effective_temperature,
    }

    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"No choices in model response: {data}")

    msg = choices[0].get("message", {})
    content = msg.get("content") or msg.get("reasoning_content") or ""
    if not content:
        raise ValueError(f"Empty content in model response: {choices[0]}")

    return content
