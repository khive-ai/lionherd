# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for OpenAI Chat provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lionherd.services.providers.openai_chat import (
    OpenAIChatEndpoint,
    create_openai_config,
)
from lionherd.services.types import NormalizedResponse
from lionherd.services.types.endpoint import EndpointConfig


class TestCreateOpenAIConfig:
    """Test create_openai_config factory function."""

    def test_create_openai_config_defaults(self):
        """Test factory with default values."""
        config = create_openai_config(name="test-openai")

        assert config.provider == "openai"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.endpoint == "chat/completions"
        assert config.api_key == "OPENAI_API_KEY"
        assert config.kwargs["default_model"] == "gpt-4o-mini"
        assert config.kwargs["excluded_params"] == set()
        assert config.request_options is None

    def test_create_openai_config_custom_values(self):
        """Test factory with custom values."""
        config = create_openai_config(
            name="test-openai",
            api_key="custom_key",
            base_url="https://custom.api.com",
            endpoint="custom/endpoint",
            default_model="gpt-4o",
            excluded_params={"reasoning_effort"},
        )

        assert config.provider == "openai"
        assert config.base_url == "https://custom.api.com"
        assert config.endpoint == "custom/endpoint"
        assert config.api_key == "custom_key"
        assert config.kwargs["default_model"] == "gpt-4o"
        assert config.kwargs["excluded_params"] == {"reasoning_effort"}

    def test_create_openai_config_api_key_none(self):
        """Test factory with api_key=None uses env var name."""
        config = create_openai_config(name="test-default", api_key=None)
        assert config.api_key == "OPENAI_API_KEY"

    def test_create_openai_config_excluded_params_none(self):
        """Test factory with excluded_params=None uses empty set."""
        config = create_openai_config(name="test-default", excluded_params=None)
        assert config.kwargs["excluded_params"] == set()

    def test_create_openai_config_extra_kwargs(self):
        """Test factory passes extra kwargs to config."""
        config = create_openai_config(
            name="test-openai",
            custom_field="custom_value",
            another_field=123,
        )
        assert config.kwargs["custom_field"] == "custom_value"
        assert config.kwargs["another_field"] == 123


class TestOpenAIChatEndpointInit:
    """Test OpenAIChatEndpoint initialization."""

    def test_init_with_none_config(self):
        """Test initialization with config=None uses factory defaults."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")

        assert endpoint.config.provider == "openai"
        assert endpoint.config.base_url == "https://api.openai.com/v1"
        assert endpoint.config.endpoint == "chat/completions"
        assert endpoint.config.kwargs["default_model"] == "gpt-4o-mini"
        # Should set request_options to OpenAIChatCompletionsRequest
        assert endpoint.config.request_options is not None
        assert endpoint.config.request_options.__name__ == "OpenAIChatCompletionsRequest"

    def test_init_with_dict_config(self):
        """Test initialization with dict config."""
        config_dict = {
            "name": "test-openai",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "endpoint": "chat/completions",
            "api_key": "test_key",
        }
        endpoint = OpenAIChatEndpoint(config=config_dict)

        assert endpoint.config.provider == "openai"
        assert endpoint.config.api_key == "test_key"

    def test_init_with_endpoint_config(self):
        """Test initialization with EndpointConfig instance."""
        config = create_openai_config(name="test-default", api_key="test_key")
        endpoint = OpenAIChatEndpoint(config=config)

        assert endpoint.config.provider == "openai"
        assert endpoint.config.api_key == "test_key"

    def test_init_sets_request_options(self):
        """Test initialization sets request_options if None."""
        config = create_openai_config(name="test-openai")
        # config.request_options is None initially
        assert config.request_options is None

        endpoint = OpenAIChatEndpoint(config=config)

        # After init, should be set to OpenAIChatCompletionsRequest
        assert endpoint.config.request_options is not None
        assert endpoint.config.request_options.__name__ == "OpenAIChatCompletionsRequest"

    def test_init_with_kwargs_override(self):
        """Test initialization with kwargs overrides."""
        endpoint = OpenAIChatEndpoint(
            config=None,
            name="test-openai",
            default_model="gpt-4o",
            base_url="https://custom.api.com",
        )

        assert endpoint.config.kwargs["default_model"] == "gpt-4o"
        assert endpoint.config.base_url == "https://custom.api.com"

    def test_init_with_circuit_breaker(self):
        """Test initialization with circuit breaker."""
        from lionherd.services.utilities.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_time=1.0)
        endpoint = OpenAIChatEndpoint(config=None, name="test-default", circuit_breaker=cb)

        assert endpoint.circuit_breaker is not None
        assert endpoint.circuit_breaker.failure_threshold == 3


class TestOpenAIChatEndpointCreatePayload:
    """Test OpenAIChatEndpoint.create_payload method."""

    def test_create_payload_default_model(self):
        """Test create_payload adds default model if not present."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        payload, headers = endpoint.create_payload(request)

        assert payload["model"] == "gpt-4o-mini"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_create_payload_custom_model(self):
        """Test create_payload uses custom model from request."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        payload, headers = endpoint.create_payload(request)

        assert payload["model"] == "gpt-4o"

    def test_create_payload_authorization_header(self):
        """Test create_payload adds Authorization Bearer header."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-default", api_key="test_key_123")
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        _payload, headers = endpoint.create_payload(request)

        assert headers["Authorization"] == "Bearer test_key_123"

    def test_create_payload_content_type_header(self):
        """Test create_payload adds Content-Type header."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        _payload, headers = endpoint.create_payload(request)

        assert headers["Content-Type"] == "application/json"

    def test_create_payload_extra_headers(self):
        """Test create_payload merges extra headers."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        request = {"messages": [{"role": "user", "content": "Hello"}]}
        extra_headers = {"X-Custom": "custom_value"}

        _payload, headers = endpoint.create_payload(request, extra_headers=extra_headers)

        assert headers["X-Custom"] == "custom_value"
        assert headers["Content-Type"] == "application/json"

    def test_create_payload_excluded_params_single(self):
        """Test create_payload filters out excluded params."""
        endpoint = OpenAIChatEndpoint(
            config=None,
            name="test-openai",
            excluded_params={"reasoning_effort"},
        )
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "high",
            "temperature": 0.7,
        }

        payload, _headers = endpoint.create_payload(request)

        assert "reasoning_effort" not in payload
        assert payload["temperature"] == 0.7

    def test_create_payload_excluded_params_multiple(self):
        """Test create_payload filters out multiple excluded params."""
        endpoint = OpenAIChatEndpoint(
            config=None,
            name="test-openai",
            excluded_params={"reasoning_effort", "top_logprobs"},
        )
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "high",
            "top_logprobs": 5,
            "temperature": 0.7,
        }

        payload, _headers = endpoint.create_payload(request)

        assert "reasoning_effort" not in payload
        assert "top_logprobs" not in payload
        assert payload["temperature"] == 0.7

    def test_create_payload_removes_config_fields(self):
        """Test create_payload removes config-specific fields from payload."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "default_model": "should_be_removed",
            "excluded_params": {"should_be_removed"},
        }

        payload, _headers = endpoint.create_payload(request)

        assert "default_model" not in payload
        assert "excluded_params" not in payload

    def test_create_payload_merges_kwargs(self):
        """Test create_payload merges config.kwargs, request, and kwargs."""
        config = create_openai_config(name="test-default", temperature=0.5, top_p=0.9)
        endpoint = OpenAIChatEndpoint(config=config)
        request = {"messages": [{"role": "user", "content": "Hello"}], "temperature": 0.7}

        payload, _headers = endpoint.create_payload(request, max_tokens=100)

        # Request overrides config.kwargs, kwargs overrides request
        assert payload["temperature"] == 0.7  # From request
        assert payload["top_p"] == 0.9  # From config.kwargs
        assert payload["max_tokens"] == 100  # From kwargs


class TestOpenAIChatEndpointNormalizeResponse:
    """Test OpenAIChatEndpoint.normalize_response method."""

    def test_normalize_response_text_only(self):
        """Test normalize_response with text content only."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello, world!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        normalized = endpoint.normalize_response(response)

        assert isinstance(normalized, NormalizedResponse)
        assert normalized.data == "Hello, world!"
        assert normalized.metadata["model"] == "gpt-4o-mini"
        assert normalized.metadata["usage"]["prompt_tokens"] == 10
        assert normalized.metadata["finish_reason"] == "stop"
        assert normalized.metadata["id"] == "chatcmpl-123"
        assert normalized.raw_response == response

    def test_normalize_response_empty_content(self):
        """Test normalize_response with empty content."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        response = {"choices": [{"index": 0, "message": {"role": "assistant", "content": None}}]}

        normalized = endpoint.normalize_response(response)

        assert normalized.data == ""

    def test_normalize_response_no_choices(self):
        """Test normalize_response with no choices."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        response = {"id": "chatcmpl-123", "choices": []}

        normalized = endpoint.normalize_response(response)

        assert normalized.data == ""
        assert normalized.metadata == {"id": "chatcmpl-123"}

    def test_normalize_response_missing_choices(self):
        """Test normalize_response with missing choices field."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        response = {"id": "chatcmpl-123"}

        normalized = endpoint.normalize_response(response)

        assert normalized.data == ""
        assert normalized.metadata == {"id": "chatcmpl-123"}

    def test_normalize_response_with_tool_calls(self):
        """Test normalize_response with tool_calls."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me call a tool.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "calculator", "arguments": '{"a":1,"b":2}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.data == "Let me call a tool."
        assert "tool_calls" in normalized.metadata
        assert len(normalized.metadata["tool_calls"]) == 1
        assert normalized.metadata["tool_calls"][0]["id"] == "call_123"
        assert normalized.metadata["finish_reason"] == "tool_calls"

    def test_normalize_response_multiple_tool_calls(self):
        """Test normalize_response with multiple tool_calls."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "tool1"}},
                            {"id": "call_2", "type": "function", "function": {"name": "tool2"}},
                        ],
                    }
                }
            ]
        }

        normalized = endpoint.normalize_response(response)

        assert len(normalized.metadata["tool_calls"]) == 2
        assert normalized.metadata["tool_calls"][0]["id"] == "call_1"
        assert normalized.metadata["tool_calls"][1]["id"] == "call_2"

    def test_normalize_response_all_metadata_fields(self):
        """Test normalize_response extracts all metadata fields."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.metadata["id"] == "chatcmpl-123"
        assert normalized.metadata["model"] == "gpt-4o"
        assert normalized.metadata["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
        assert normalized.metadata["finish_reason"] == "stop"

    def test_normalize_response_extracts_first_choice_only(self):
        """Test normalize_response uses first choice only."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-openai")
        response = {
            "choices": [
                {"message": {"content": "First choice"}, "finish_reason": "stop"},
                {"message": {"content": "Second choice"}, "finish_reason": "length"},
            ]
        }

        normalized = endpoint.normalize_response(response)

        # Should only extract from first choice
        assert normalized.data == "First choice"
        assert normalized.metadata["finish_reason"] == "stop"


class TestOpenAIChatEndpointIntegration:
    """Integration tests with mocked httpx."""

    @pytest.mark.asyncio
    async def test_call_success(self):
        """Test successful API call with mocked httpx."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-default", api_key="test_key")

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "Hello from GPT!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            result = await endpoint.call(
                request={"messages": [{"role": "user", "content": "Hello"}]}
            )

        assert result.data == "Hello from GPT!"
        assert result.metadata["model"] == "gpt-4o-mini"
        assert result.metadata["usage"]["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_call_with_tool_calls(self):
        """Test API call with tool_calls response."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-default", api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            result = await endpoint.call(
                request={"messages": [{"role": "user", "content": "What's the weather?"}]}
            )

        assert "tool_calls" in result.metadata
        assert result.metadata["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result.metadata["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_call_http_error(self):
        """Test API call with HTTP error."""
        endpoint = OpenAIChatEndpoint(config=None, name="test-default", api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.request = MagicMock()

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(endpoint, "_create_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError, match="401"),
        ):
            await endpoint.call(request={"messages": [{"role": "user", "content": "Hello"}]})

    @pytest.mark.asyncio
    async def test_call_with_excluded_params(self):
        """Test that excluded params are filtered in actual call."""
        endpoint = OpenAIChatEndpoint(
            config=None,
            name="test-openai",
            api_key="test_key",
            excluded_params={"reasoning_effort"},
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Success"}}]}

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            await endpoint.call(
                request={
                    "messages": [{"role": "user", "content": "Test"}],
                    "reasoning_effort": "high",  # Should be excluded
                    "temperature": 0.7,  # Should be included
                }
            )

        # Verify request was called without reasoning_effort
        call_args = mock_client.request.call_args
        payload = call_args[1]["json"]
        assert "reasoning_effort" not in payload
        assert payload["temperature"] == 0.7
