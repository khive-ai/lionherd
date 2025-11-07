# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from ..types.endpoint import EndpointConfig
from .openai_chat import OpenAIChatEndpoint

__all__ = (
    "GroqChatEndpoint",
    "NvidiaNimChatEndpoint",
    "OpenRouterChatEndpoint",
    "create_groq_config",
    "create_nvidia_nim_config",
    "create_openrouter_config",
)


# =============================================================================
# Config Factory Functions
# =============================================================================


def create_groq_config(
    api_key: str | None = None,
    base_url: str = "https://api.groq.com/openai/v1",
    endpoint: str = "chat/completions",
    default_model: str = "llama-3.3-70b-versatile",
    excluded_params: set[str] | None = None,
    **kwargs,
) -> EndpointConfig:
    """Create EndpointConfig for Groq API.

    Groq provides fast inference for open-source models (Llama, Mixtral, Gemma).
    Uses OpenAI-compatible API with some parameter restrictions.

    Args:
        api_key: API key or env var name (default: "GROQ_API_KEY")
        base_url: Groq API base URL
        endpoint: Endpoint path
        default_model: Default model (llama-3.3-70b-versatile, mixtral-8x7b, etc.)
        excluded_params: Additional params to exclude (reasoning_effort auto-added)
        **kwargs: Additional config passed to EndpointConfig.kwargs

    Returns:
        EndpointConfig instance configured for Groq

    Example:
        >>> config = create_groq_config(default_model="mixtral-8x7b-32768")
        >>> endpoint = GroqChatEndpoint(config=config)
    """
    # Groq doesn't support reasoning_effort (OpenAI o1-specific)
    if excluded_params is None:
        excluded_params = {"reasoning_effort"}
    else:
        excluded_params = excluded_params | {"reasoning_effort"}

    config_kwargs = {
        "default_model": default_model,
        "excluded_params": excluded_params,
        **kwargs,
    }

    return EndpointConfig(
        provider="groq",
        base_url=base_url,
        endpoint=endpoint,
        api_key=api_key or "GROQ_API_KEY",
        request_options=None,  # Use dict validation for MVP
        **config_kwargs,
    )


def create_openrouter_config(
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    endpoint: str = "chat/completions",
    default_model: str = "google/gemini-2.0-flash-exp:free",
    excluded_params: set[str] | None = None,
    **kwargs,
) -> EndpointConfig:
    """Create EndpointConfig for OpenRouter API.

    OpenRouter provides unified access to multiple model providers (OpenAI, Anthropic,
    Google, Meta, etc.) with consistent pricing and rate limits.

    Args:
        api_key: API key or env var name (default: "OPENROUTER_API_KEY")
        base_url: OpenRouter API base URL
        endpoint: Endpoint path
        default_model: Default model (use provider/model-name format)
        excluded_params: Params to exclude from payload
        **kwargs: Additional config (e.g., HTTP-Referer, X-Title for rankings)

    Returns:
        EndpointConfig instance configured for OpenRouter

    Example:
        >>> config = create_openrouter_config(
        ...     default_model="anthropic/claude-sonnet-4",
        ...     **{"HTTP-Referer": "https://yourapp.com", "X-Title": "YourApp"},
        ... )
        >>> endpoint = OpenRouterChatEndpoint(config=config)

    Note:
        OpenRouter supports model routing and fallbacks. See:
        https://openrouter.ai/docs
    """
    config_kwargs = {
        "default_model": default_model,
        "excluded_params": excluded_params or set(),
        **kwargs,
    }

    return EndpointConfig(
        provider="openrouter",
        base_url=base_url,
        endpoint=endpoint,
        api_key=api_key or "OPENROUTER_API_KEY",
        request_options=None,
        **config_kwargs,
    )


def create_nvidia_nim_config(
    api_key: str | None = None,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    endpoint: str = "chat/completions",
    default_model: str = "meta/llama-3.1-8b-instruct",
    excluded_params: set[str] | None = None,
    **kwargs,
) -> EndpointConfig:
    """Create EndpointConfig for NVIDIA NIM API.

    NVIDIA Inference Microservices (NIM) provides optimized inference for
    popular models (Llama, Mistral, etc.) on NVIDIA infrastructure.

    Args:
        api_key: API key or env var name (default: "NVIDIA_NIM_API_KEY")
        base_url: NVIDIA NIM API base URL
        endpoint: Endpoint path
        default_model: Default model (meta/llama-3.1-8b-instruct, etc.)
        excluded_params: Params to exclude from payload
        **kwargs: Additional config

    Returns:
        EndpointConfig instance configured for NVIDIA NIM

    Example:
        >>> config = create_nvidia_nim_config(default_model="mistralai/mixtral-8x7b-instruct-v0.1")
        >>> endpoint = NvidiaNimChatEndpoint(config=config)

    Note:
        Get API keys from: https://build.nvidia.com/
        Available models: https://build.nvidia.com/explore/discover
    """
    config_kwargs = {
        "default_model": default_model,
        "excluded_params": excluded_params or set(),
        **kwargs,
    }

    return EndpointConfig(
        provider="nvidia_nim",
        base_url=base_url,
        endpoint=endpoint,
        api_key=api_key or "NVIDIA_NIM_API_KEY",
        request_options=None,
        **config_kwargs,
    )


# =============================================================================
# Provider-Specific Endpoint Classes (Convenience Wrappers)
# =============================================================================


class GroqChatEndpoint(OpenAIChatEndpoint):
    """Groq chat completions endpoint (OpenAI-compatible).

    Fast inference for open-source models with automatic parameter filtering.

    Defaults:
        - provider: groq
        - base_url: https://api.groq.com/openai/v1
        - endpoint: chat/completions
        - default_model: llama-3.3-70b-versatile
        - API key: Loaded from GROQ_API_KEY env var
        - Excludes: reasoning_effort (not supported)

    Popular models:
        - llama-3.3-70b-versatile (recommended)
        - llama-3.1-70b-versatile
        - mixtral-8x7b-32768
        - gemma2-9b-it

    Example:
        >>> endpoint = GroqChatEndpoint(default_model="mixtral-8x7b-32768")
        >>> response = await endpoint.call({"messages": [{"role": "user", "content": "Hello"}]})
    """

    def __init__(
        self,
        config: dict | EndpointConfig | None = None,
        circuit_breaker: Any | None = None,
        **kwargs,
    ):
        """Initialize with Groq config."""
        if config is None:
            config = create_groq_config(**kwargs)
        super().__init__(config=config, circuit_breaker=circuit_breaker)


class OpenRouterChatEndpoint(OpenAIChatEndpoint):
    """OpenRouter chat completions endpoint (OpenAI-compatible).

    Multi-provider gateway with unified API and model fallbacks.

    Defaults:
        - provider: openrouter
        - base_url: https://openrouter.ai/api/v1
        - endpoint: chat/completions
        - default_model: google/gemini-2.0-flash-exp:free
        - API key: Loaded from OPENROUTER_API_KEY env var

    Popular models:
        - anthropic/claude-sonnet-4
        - openai/gpt-4o
        - google/gemini-2.0-flash-exp:free (free tier)
        - meta-llama/llama-3.3-70b-instruct

    Example:
        >>> endpoint = OpenRouterChatEndpoint(
        ...     default_model="anthropic/claude-sonnet-4", **{"HTTP-Referer": "https://yourapp.com"}
        ... )
        >>> response = await endpoint.call({"messages": [{"role": "user", "content": "Hello"}]})

    Note:
        For better rankings on OpenRouter, pass HTTP-Referer and X-Title:
        ```python
        endpoint = OpenRouterChatEndpoint(
            **{"HTTP-Referer": "https://yourapp.com", "X-Title": "YourApp"}
        )
        ```
    """

    def __init__(
        self,
        config: dict | EndpointConfig | None = None,
        circuit_breaker: Any | None = None,
        **kwargs,
    ):
        """Initialize with OpenRouter config."""
        if config is None:
            config = create_openrouter_config(**kwargs)
        super().__init__(config=config, circuit_breaker=circuit_breaker)


class NvidiaNimChatEndpoint(OpenAIChatEndpoint):
    """NVIDIA NIM chat completions endpoint (OpenAI-compatible).

    Optimized inference for popular models on NVIDIA infrastructure.

    Defaults:
        - provider: nvidia_nim
        - base_url: https://integrate.api.nvidia.com/v1
        - endpoint: chat/completions
        - default_model: meta/llama-3.1-8b-instruct
        - API key: Loaded from NVIDIA_NIM_API_KEY env var

    Popular models:
        - meta/llama-3.1-70b-instruct
        - meta/llama-3.1-8b-instruct
        - mistralai/mixtral-8x7b-instruct-v0.1
        - microsoft/phi-3-medium-4k-instruct

    Example:
        >>> endpoint = NvidiaNimChatEndpoint(default_model="meta/llama-3.1-70b-instruct")
        >>> response = await endpoint.call({"messages": [{"role": "user", "content": "Hello"}]})

    Note:
        - Get API keys from: https://build.nvidia.com/
        - Browse models: https://build.nvidia.com/explore/discover
        - Free tier available for testing
    """

    def __init__(
        self,
        config: dict | EndpointConfig | None = None,
        circuit_breaker: Any | None = None,
        **kwargs,
    ):
        """Initialize with NVIDIA NIM config."""
        if config is None:
            config = create_nvidia_nim_config(**kwargs)
        super().__init__(config=config, circuit_breaker=circuit_breaker)
