# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from lionherd.services.types import NormalizedResponse

from ..types.endpoint import Endpoint, EndpointConfig

__all__ = (
    "AnthropicMessagesEndpoint",
    "create_anthropic_config",
)


def create_anthropic_config(
    api_key: str | None = None,
    base_url: str = "https://api.anthropic.com",
    endpoint: str = "v1/messages",
    default_model: str = "claude-sonnet-4-5-20250929",
    anthropic_version: str = "2023-06-01",
    beta_headers: list[str] | None = None,
    **kwargs,
) -> EndpointConfig:
    """Factory for Anthropic Messages API config.

    Args:
        api_key: API key or env var name (default: "ANTHROPIC_API_KEY")
        base_url: Base API URL
        endpoint: Endpoint path
        default_model: Default Claude model
        anthropic_version: API version header
        beta_headers: Beta feature flags (e.g., ["extended-thinking-2025-01-17"])
        **kwargs: Additional config parameters

    Returns:
        EndpointConfig instance
    """
    config_kwargs = {
        "default_model": default_model,
        "anthropic_version": anthropic_version,
        "beta_headers": beta_headers or [],
        **kwargs,
    }

    from ..third_party.anthropic_models import CreateMessageRequest

    return EndpointConfig(
        provider="anthropic",
        base_url=base_url,
        endpoint=endpoint,
        api_key=api_key or "ANTHROPIC_API_KEY",
        request_options=CreateMessageRequest,
        **config_kwargs,
    )


class AnthropicMessagesEndpoint(Endpoint):
    """Anthropic Messages API endpoint.

    Adds Anthropic-specific headers and response normalization.

    Usage:
        endpoint = AnthropicMessagesEndpoint()
        response = await endpoint.call({
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024
        })
    """

    def __init__(
        self,
        config: dict | EndpointConfig | None = None,
        circuit_breaker: Any | None = None,
        **kwargs,
    ):
        """Initialize with Anthropic config."""
        if config is None:
            config = create_anthropic_config(**kwargs)

        super().__init__(config=config, circuit_breaker=circuit_breaker)

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ) -> tuple[dict, dict]:
        """Create payload with Anthropic-specific headers.

        Adds:
        - anthropic-version header (required)
        - x-api-key header (Anthropic uses this instead of Authorization)
        - anthropic-beta header (if beta features enabled)
        - Default model and max_tokens if not provided
        """
        # Convert request to dict
        request_dict = (
            request if isinstance(request, dict) else request.model_dump(exclude_none=True)
        )

        # Add default model if not present
        default_model = self.config.kwargs.get("default_model")
        if default_model and "model" not in request_dict:
            request_dict["model"] = default_model

        # Ensure max_tokens is present (required by Anthropic)
        if "max_tokens" not in request_dict:
            request_dict["max_tokens"] = 4096

        # Merge config.kwargs + request + kwargs
        payload = {**self.config.kwargs, **request_dict, **kwargs}

        # Remove headers-related fields from payload
        payload.pop("anthropic_version", None)
        payload.pop("beta_headers", None)
        payload.pop("default_model", None)

        # Build headers with Anthropic-specific requirements
        headers = {"Content-Type": "application/json"}

        # Add anthropic-version header (required)
        anthropic_version = self.config.kwargs.get("anthropic_version", "2023-06-01")
        headers["anthropic-version"] = anthropic_version

        # Add API key with x-api-key header (Anthropic-specific)
        api_key = self.get_api_key()
        if api_key:
            headers["x-api-key"] = api_key

        # Add beta headers if specified
        beta_headers = self.config.kwargs.get("beta_headers", [])
        if beta_headers:
            headers["anthropic-beta"] = ",".join(beta_headers)

        # Merge extra headers
        if extra_headers:
            headers.update(extra_headers)

        return (payload, headers)

    def normalize_response(self, response: dict[str, Any]) -> NormalizedResponse:
        """Normalize Anthropic response to standard format.

        Extracts:
        - Text from content blocks
        - Thinking blocks (if extended thinking enabled)
        - Usage stats
        - Stop reason
        - Model info
        - Tool uses
        """
        # Extract text and thinking from content blocks
        text_parts = []
        thinking_parts = []

        content = response.get("content")
        if content:
            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "thinking":
                    thinking_parts.append(block.get("thinking", ""))

        # Combine text
        text = "".join(text_parts)

        # Extract metadata
        metadata: dict[str, Any] = {
            k: response[k]
            for k in ("model", "usage", "stop_reason", "stop_sequence", "id")
            if k in response
        }

        # Add thinking if present
        if thinking_parts:
            metadata["thinking"] = "\n\n".join(thinking_parts)

        # Add tool use blocks if present
        tool_uses = [
            block for block in response.get("content", []) if block.get("type") == "tool_use"
        ]
        if tool_uses:
            metadata["tool_uses"] = tool_uses

        return NormalizedResponse(
            text=text,
            raw=response,
            provider=self.config.provider,
            metadata=metadata,
        )
