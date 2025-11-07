# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for Anthropic Messages provider."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lionherd.services.providers.anthropic_messages import (
    AnthropicMessagesEndpoint,
    create_anthropic_config,
)
from lionherd.services.types import NormalizedResponse
from lionherd.services.types.endpoint import EndpointConfig


class TestCreateAnthropicConfig:
    """Test create_anthropic_config factory function."""

    def test_create_anthropic_config_defaults(self):
        """Test factory with default values."""
        config = create_anthropic_config(name="test-anthropic")

        assert config.provider == "anthropic"
        assert config.base_url == "https://api.anthropic.com"
        assert config.endpoint == "v1/messages"
        assert config.api_key == "ANTHROPIC_API_KEY"
        assert config.kwargs["default_model"] == "claude-sonnet-4-5-20250929"
        assert config.kwargs["anthropic_version"] == "2023-06-01"
        assert config.kwargs["beta_headers"] == []

    def test_create_anthropic_config_custom_values(self):
        """Test factory with custom values."""
        config = create_anthropic_config(
            name="test-anthropic",
            api_key="custom_key",
            base_url="https://custom.api.com",
            endpoint="custom/endpoint",
            default_model="claude-opus-4",
            anthropic_version="2024-01-01",
            beta_headers=["extended-thinking-2025-01-17"],
        )

        assert config.provider == "anthropic"
        assert config.base_url == "https://custom.api.com"
        assert config.endpoint == "custom/endpoint"
        assert config.api_key == "custom_key"
        assert config.kwargs["default_model"] == "claude-opus-4"
        assert config.kwargs["anthropic_version"] == "2024-01-01"
        assert config.kwargs["beta_headers"] == ["extended-thinking-2025-01-17"]

    def test_create_anthropic_config_api_key_none(self):
        """Test factory with api_key=None uses env var name."""
        config = create_anthropic_config(name="test-default", api_key=None)
        assert config.api_key == "ANTHROPIC_API_KEY"

    def test_create_anthropic_config_beta_headers_none(self):
        """Test factory with beta_headers=None uses empty list."""
        config = create_anthropic_config(name="test-default", beta_headers=None)
        assert config.kwargs["beta_headers"] == []

    def test_create_anthropic_config_extra_kwargs(self):
        """Test factory passes extra kwargs to config."""
        config = create_anthropic_config(
            name="test-anthropic",
            custom_field="custom_value",
            another_field=123,
        )
        assert config.kwargs["custom_field"] == "custom_value"
        assert config.kwargs["another_field"] == 123

    def test_create_anthropic_config_request_options(self):
        """Test factory sets request_options from third_party."""
        config = create_anthropic_config(name="test-anthropic")
        # Should import and set CreateMessageRequest
        assert config.request_options is not None
        assert config.request_options.__name__ == "CreateMessageRequest"


class TestAnthropicMessagesEndpointInit:
    """Test AnthropicMessagesEndpoint initialization."""

    def test_init_with_none_config(self):
        """Test initialization with config=None uses factory defaults."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")

        assert endpoint.config.provider == "anthropic"
        assert endpoint.config.base_url == "https://api.anthropic.com"
        assert endpoint.config.endpoint == "v1/messages"
        assert endpoint.config.kwargs["default_model"] == "claude-sonnet-4-5-20250929"

    def test_init_with_dict_config(self):
        """Test initialization with dict config."""
        config_dict = {
            "name": "test-anthropic",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "endpoint": "v1/messages",
            "api_key": "test_key",
        }
        endpoint = AnthropicMessagesEndpoint(config=config_dict)

        assert endpoint.config.provider == "anthropic"
        assert endpoint.config.api_key == "test_key"

    def test_init_with_endpoint_config(self):
        """Test initialization with EndpointConfig instance."""
        config = create_anthropic_config(name="test-default", api_key="test_key")
        endpoint = AnthropicMessagesEndpoint(config=config)

        assert endpoint.config.provider == "anthropic"
        assert endpoint.config.api_key == "test_key"

    def test_init_with_kwargs_override(self):
        """Test initialization with kwargs overrides."""
        endpoint = AnthropicMessagesEndpoint(
            config=None,
            name="test-anthropic",
            default_model="claude-opus-4",
            base_url="https://custom.api.com",
        )

        assert endpoint.config.kwargs["default_model"] == "claude-opus-4"
        assert endpoint.config.base_url == "https://custom.api.com"

    def test_init_with_circuit_breaker(self):
        """Test initialization with circuit breaker."""
        from lionherd.services.utilities.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_time=1.0)
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-default", circuit_breaker=cb)

        assert endpoint.circuit_breaker is not None
        assert endpoint.circuit_breaker.failure_threshold == 3


class TestAnthropicMessagesEndpointCreatePayload:
    """Test AnthropicMessagesEndpoint.create_payload method."""

    def test_create_payload_default_model(self):
        """Test create_payload adds default model if not present."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        payload, _headers = endpoint.create_payload(request)

        assert payload["model"] == "claude-sonnet-4-5-20250929"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_create_payload_custom_model(self):
        """Test create_payload uses custom model from request."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        request = {
            "model": "claude-opus-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        payload, _headers = endpoint.create_payload(request)

        assert payload["model"] == "claude-opus-4"

    def test_create_payload_default_max_tokens(self):
        """Test create_payload adds default max_tokens if not present."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        payload, _headers = endpoint.create_payload(request)

        assert payload["max_tokens"] == 4096

    def test_create_payload_custom_max_tokens(self):
        """Test create_payload uses custom max_tokens from request."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
        }

        payload, _headers = endpoint.create_payload(request)

        assert payload["max_tokens"] == 1024

    def test_create_payload_anthropic_version_header(self):
        """Test create_payload adds anthropic-version header."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        _payload, headers = endpoint.create_payload(request)

        assert headers["anthropic-version"] == "2023-06-01"

    def test_create_payload_custom_anthropic_version(self):
        """Test create_payload uses custom anthropic_version."""
        endpoint = AnthropicMessagesEndpoint(
            config=None,
            name="test-anthropic",
            anthropic_version="2024-01-01",
        )
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        _payload, headers = endpoint.create_payload(request)

        assert headers["anthropic-version"] == "2024-01-01"

    def test_create_payload_api_key_header(self):
        """Test create_payload adds x-api-key header."""
        endpoint = AnthropicMessagesEndpoint(
            config=None, name="test-default", api_key="test_key_123"
        )
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        _payload, headers = endpoint.create_payload(request)

        assert headers["x-api-key"] == "test_key_123"

    def test_create_payload_beta_headers_single(self):
        """Test create_payload adds beta headers."""
        endpoint = AnthropicMessagesEndpoint(
            config=None,
            name="test-anthropic",
            beta_headers=["extended-thinking-2025-01-17"],
        )
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        _payload, headers = endpoint.create_payload(request)

        assert headers["anthropic-beta"] == "extended-thinking-2025-01-17"

    def test_create_payload_beta_headers_multiple(self):
        """Test create_payload joins multiple beta headers with comma."""
        endpoint = AnthropicMessagesEndpoint(
            config=None,
            name="test-anthropic",
            beta_headers=["feature1", "feature2", "feature3"],
        )
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        _payload, headers = endpoint.create_payload(request)

        assert headers["anthropic-beta"] == "feature1,feature2,feature3"

    def test_create_payload_no_beta_headers(self):
        """Test create_payload without beta headers."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        _payload, headers = endpoint.create_payload(request)

        assert "anthropic-beta" not in headers

    def test_create_payload_content_type_header(self):
        """Test create_payload adds Content-Type header."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        _payload, headers = endpoint.create_payload(request)

        assert headers["Content-Type"] == "application/json"

    def test_create_payload_extra_headers(self):
        """Test create_payload merges extra headers."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        request = {"messages": [{"role": "user", "content": "Hello"}]}
        extra_headers = {"X-Custom": "custom_value"}

        _payload, headers = endpoint.create_payload(request, extra_headers=extra_headers)

        assert headers["X-Custom"] == "custom_value"
        assert headers["Content-Type"] == "application/json"

    def test_create_payload_removes_non_payload_fields(self):
        """Test create_payload removes header-related fields from payload."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "anthropic_version": "should_be_removed",
            "beta_headers": ["should_be_removed"],
            "default_model": "should_be_removed",
        }

        payload, _headers = endpoint.create_payload(request)

        assert "anthropic_version" not in payload
        assert "beta_headers" not in payload
        assert "default_model" not in payload

    def test_create_payload_merges_kwargs(self):
        """Test create_payload merges config.kwargs, request, and kwargs."""
        config = create_anthropic_config(name="test-default", temperature=0.5, top_p=0.9)
        endpoint = AnthropicMessagesEndpoint(config=config)
        request = {"messages": [{"role": "user", "content": "Hello"}], "temperature": 0.7}

        payload, _headers = endpoint.create_payload(request, top_k=40)

        # Request overrides config.kwargs, kwargs overrides request
        assert payload["temperature"] == 0.7  # From request
        assert payload["top_p"] == 0.9  # From config.kwargs
        assert payload["top_k"] == 40  # From kwargs


class TestAnthropicMessagesEndpointNormalizeResponse:
    """Test AnthropicMessagesEndpoint.normalize_response method."""

    def test_normalize_response_text_only(self):
        """Test normalize_response with text content only."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        response = {
            "id": "msg_123",
            "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": "Hello, world!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }

        normalized = endpoint.normalize_response(response)

        assert isinstance(normalized, NormalizedResponse)
        assert normalized.data == "Hello, world!"
        assert normalized.metadata["model"] == "claude-sonnet-4"
        assert normalized.metadata["usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert normalized.metadata["stop_reason"] == "end_turn"
        assert normalized.metadata["id"] == "msg_123"
        assert normalized.raw_response == response

    def test_normalize_response_multiple_text_blocks(self):
        """Test normalize_response with multiple text blocks."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        response = {
            "content": [
                {"type": "text", "text": "First block. "},
                {"type": "text", "text": "Second block."},
            ]
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.data == "First block. Second block."

    def test_normalize_response_with_thinking(self):
        """Test normalize_response with thinking blocks."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        response = {
            "content": [
                {"type": "thinking", "thinking": "Let me think about this..."},
                {"type": "text", "text": "The answer is 42."},
            ]
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.data == "The answer is 42."
        assert normalized.metadata["thinking"] == "Let me think about this..."

    def test_normalize_response_multiple_thinking_blocks(self):
        """Test normalize_response with multiple thinking blocks."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        response = {
            "content": [
                {"type": "thinking", "thinking": "First thought."},
                {"type": "thinking", "thinking": "Second thought."},
                {"type": "text", "text": "Final answer."},
            ]
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.data == "Final answer."
        assert normalized.metadata["thinking"] == "First thought.\n\nSecond thought."

    def test_normalize_response_with_tool_use(self):
        """Test normalize_response with tool_use blocks."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        response = {
            "content": [
                {"type": "text", "text": "I'll use a tool."},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "calculator",
                    "input": {"operation": "add", "a": 1, "b": 2},
                },
            ]
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.data == "I'll use a tool."
        assert "tool_uses" in normalized.metadata
        assert len(normalized.metadata["tool_uses"]) == 1
        assert normalized.metadata["tool_uses"][0]["id"] == "tool_123"
        assert normalized.metadata["tool_uses"][0]["name"] == "calculator"

    def test_normalize_response_multiple_tool_uses(self):
        """Test normalize_response with multiple tool_use blocks."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        response = {
            "content": [
                {"type": "tool_use", "id": "tool_1", "name": "tool1"},
                {"type": "text", "text": "Between tools."},
                {"type": "tool_use", "id": "tool_2", "name": "tool2"},
            ]
        }

        normalized = endpoint.normalize_response(response)

        assert len(normalized.metadata["tool_uses"]) == 2
        assert normalized.metadata["tool_uses"][0]["id"] == "tool_1"
        assert normalized.metadata["tool_uses"][1]["id"] == "tool_2"

    def test_normalize_response_empty_content(self):
        """Test normalize_response with empty content."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        response = {"content": []}

        normalized = endpoint.normalize_response(response)

        assert normalized.data == ""
        assert normalized.metadata == {}

    def test_normalize_response_no_content(self):
        """Test normalize_response with missing content field."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        response = {"id": "msg_123"}

        normalized = endpoint.normalize_response(response)

        assert normalized.data == ""
        assert normalized.metadata == {"id": "msg_123"}

    def test_normalize_response_stop_sequence(self):
        """Test normalize_response extracts stop_sequence if present."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        response = {
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "stop_sequence",
            "stop_sequence": "###",
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.metadata["stop_reason"] == "stop_sequence"
        assert normalized.metadata["stop_sequence"] == "###"

    def test_normalize_response_all_metadata_fields(self):
        """Test normalize_response extracts all metadata fields."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-anthropic")
        response = {
            "id": "msg_123",
            "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
            "stop_sequence": None,
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.metadata["id"] == "msg_123"
        assert normalized.metadata["model"] == "claude-sonnet-4"
        assert normalized.metadata["usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert normalized.metadata["stop_reason"] == "end_turn"
        assert normalized.metadata["stop_sequence"] is None


class TestAnthropicMessagesEndpointIntegration:
    """Integration tests with mocked httpx."""

    @pytest.mark.asyncio
    async def test_call_success(self):
        """Test successful API call with mocked httpx."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-default", api_key="test_key")

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_123",
            "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            result = await endpoint.call(
                request={"messages": [{"role": "user", "content": "Hello"}]}
            )

        assert result.data == "Hello from Claude!"
        assert result.metadata["model"] == "claude-sonnet-4"
        assert result.metadata["usage"]["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_call_with_extended_thinking(self):
        """Test API call with extended thinking response."""
        endpoint = AnthropicMessagesEndpoint(
            config=None,
            name="test-anthropic",
            api_key="test_key",
            beta_headers=["extended-thinking-2025-01-17"],
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [
                {"type": "thinking", "thinking": "Deep analysis here..."},
                {"type": "text", "text": "Final answer."},
            ],
            "stop_reason": "end_turn",
        }

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            result = await endpoint.call(
                request={"messages": [{"role": "user", "content": "Solve this"}]}
            )

        assert result.data == "Final answer."
        assert "thinking" in result.metadata
        assert result.metadata["thinking"] == "Deep analysis here..."

    @pytest.mark.asyncio
    async def test_call_http_error(self):
        """Test API call with HTTP error."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test-default", api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "Invalid request"}}
        mock_response.request = MagicMock()

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(endpoint, "_create_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError, match="400"),
        ):
            await endpoint.call(request={"messages": [{"role": "user", "content": "Hello"}]})
