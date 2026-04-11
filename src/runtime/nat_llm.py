"""
NAT LLM configuration for the local Nemotron endpoint.

Provides a NIMModelConfig pointing to the locally deployed NIM instance
described in documents/llm-access.md. This replaces the raw requests.post()
model adapter with a NAT-managed LLM provider.

Owns:
    - NAT LLM provider configuration for the local Nemotron NIM
    - Factory function for building the config from llm-access defaults

Does NOT own:
    - Model serving or inference (the NIM container handles that)
    - Agent orchestration or prompt policy
    - Tool definitions or reward semantics
"""
from __future__ import annotations

from nat.llm.nim_llm import NIMModelConfig


# Defaults from documents/llm-access.md
_DEFAULT_BASE_URL = "http://0.0.0.0:8000/v1"
_DEFAULT_MODEL_NAME = "nvidia/nemotron-3-nano"
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.1


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
