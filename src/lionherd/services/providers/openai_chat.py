# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from lionherd.services.types import NormalizedResponse

from ..types.endpoint import Endpoint, EndpointConfig

__all__ = (
    "OpenAIChatEndpoint",
    "create_openai_config",
)


def create_openai_config(
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    endpoint: str = "chat/completions",
    default_model: str = "gpt-4o-mini",
    excluded_params: set[str] | None = None,
    **kwargs,
) -> EndpointConfig:
    """Factory for OpenAI Chat API config.

    Args:
        api_key: API key or env var name (default: "OPENAI_API_KEY")
        base_url: Base API URL
        endpoint: Endpoint path
        default_model: Default model
        excluded_params: Parameters to exclude from payload
        **kwargs: Additional config parameters

    Returns:
        EndpointConfig instance
    """
    config_kwargs = {
        "default_model": default_model,
        "excluded_params": excluded_params or set(),
        **kwargs,
    }

    return EndpointConfig(
        provider="openai",
        base_url=base_url,
        endpoint=endpoint,
        api_key=api_key or "OPENAI_API_KEY",
        request_options=None,
        **config_kwargs,
    )


class OpenAIChatEndpoint(Endpoint):
    """OpenAI Chat Completions API endpoint.

    Supports OpenAI and OpenAI-compatible providers.

    Usage:
        endpoint = OpenAIChatEndpoint()
        response = await endpoint.call({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}]
        })
    """

    def __init__(
        self,
        config: dict | EndpointConfig | None = None,
        circuit_breaker: Any | None = None,
        **kwargs,
    ):
        """Initialize with OpenAI config."""
        if config is None:
            config = create_openai_config(**kwargs)

        if config.request_options is None:
            from ..third_party.openai_models import OpenAIChatCompletionsRequest

            config.request_options = OpenAIChatCompletionsRequest

        super().__init__(config=config, circuit_breaker=circuit_breaker)

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ) -> tuple[dict, dict]:
        """Create payload with parameter filtering.

        Filters out excluded parameters (e.g., reasoning_effort for
        providers that don't support it).
        """
        # Convert request to dict
        request_dict = (
            request if isinstance(request, dict) else request.model_dump(exclude_none=True)
        )

        # Add default model if not present
        default_model = self.config.kwargs.get("default_model")
        if default_model and "model" not in request_dict:
            request_dict["model"] = default_model

        # Merge config.kwargs + request + kwargs
        payload = {**self.config.kwargs, **request_dict, **kwargs}

        # Remove excluded params (provider-specific filtering)
        excluded_params = self.config.kwargs.get("excluded_params", set())
        for param in excluded_params:
            payload.pop(param, None)

        # Remove config-specific fields from payload
        payload.pop("default_model", None)
        payload.pop("excluded_params", None)

        # Build headers (standard Authorization: Bearer)
        headers = {"Content-Type": "application/json"}
        api_key = self.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if extra_headers:
            headers.update(extra_headers)

        return (payload, headers)

    def normalize_response(self, response: dict[str, Any]) -> NormalizedResponse:
        """Normalize OpenAI response to standard format.

        Extracts:
        - Text from choices[0].message.content
        - Usage stats
        - Model info
        - Finish reason
        - Tool calls (if present)
        """
        # Extract text from first choice
        text = ""
        choices = response.get("choices")
        if choices and len(choices) > 0:
            choice = choices[0]
            message = choice.get("message", {})
            text = message.get("content") or ""

        # Extract metadata
        metadata: dict[str, Any] = {
            k: response[k] for k in ("model", "usage", "id") if k in response
        }

        if choices and len(choices) > 0:
            choice = choices[0]
            metadata.update({k: choice[k] for k in ("finish_reason",) if k in choice})

            # Extract tool calls if present
            message = choice.get("message", {})
            metadata.update({k: message[k] for k in ("tool_calls",) if k in message})

        return NormalizedResponse(
            text=text,
            raw=response,
            provider=self.config.provider,
            metadata=metadata,
        )
