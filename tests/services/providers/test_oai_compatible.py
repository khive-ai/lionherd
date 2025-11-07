# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for OAI-compatible providers (Groq, OpenRouter, NVIDIA NIM)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lionherd.services.providers.oai_compatible import (
    GroqChatEndpoint,
    NvidiaNimChatEndpoint,
    OpenRouterChatEndpoint,
    create_groq_config,
    create_nvidia_nim_config,
    create_openrouter_config,
)
from lionherd.services.types.endpoint import EndpointConfig

# =============================================================================
# Parametrized Provider Configurations
# =============================================================================


@pytest.fixture(
    params=[
        pytest.param(
            {
                "name": "groq",
                "factory": create_groq_config,
                "endpoint_class": GroqChatEndpoint,
                "provider": "groq",
                "base_url": "https://api.groq.com/openai/v1",
                "endpoint": "chat/completions",
                "api_key_env": "GROQ_API_KEY",
                "default_model": "llama-3.3-70b-versatile",
                "excluded_params": {"reasoning_effort"},
                "example_response": {
                    "id": "chatcmpl-123",
                    "model": "llama-3.3-70b-versatile",
                    "choices": [
                        {"message": {"content": "Hello from Groq!"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                "response_text": "Hello from Groq!",
            },
            id="groq",
        ),
        pytest.param(
            {
                "name": "openrouter",
                "factory": create_openrouter_config,
                "endpoint_class": OpenRouterChatEndpoint,
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "endpoint": "chat/completions",
                "api_key_env": "OPENROUTER_API_KEY",
                "default_model": "google/gemini-2.0-flash-exp:free",
                "excluded_params": set(),
                "example_response": {
                    "id": "gen-123",
                    "model": "google/gemini-2.0-flash-exp:free",
                    "choices": [
                        {"message": {"content": "Hello from OpenRouter!"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                "response_text": "Hello from OpenRouter!",
            },
            id="openrouter",
        ),
        pytest.param(
            {
                "name": "nvidia_nim",
                "factory": create_nvidia_nim_config,
                "endpoint_class": NvidiaNimChatEndpoint,
                "provider": "nvidia_nim",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "endpoint": "chat/completions",
                "api_key_env": "NVIDIA_NIM_API_KEY",
                "default_model": "meta/llama-3.1-8b-instruct",
                "excluded_params": set(),
                "example_response": {
                    "id": "cmpl-123",
                    "model": "meta/llama-3.1-8b-instruct",
                    "choices": [
                        {"message": {"content": "Hello from NVIDIA!"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                "response_text": "Hello from NVIDIA!",
            },
            id="nvidia_nim",
        ),
    ]
)
def provider_config(request):
    """Parametrized fixture providing all OAI-compatible provider configurations."""
    return request.param


# =============================================================================
# Parametrized Factory Tests
# =============================================================================


class TestCreateConfigFactories:
    """Test create_*_config factory functions for all OAI-compatible providers."""

    def test_factory_defaults(self, provider_config):
        """Test factory with default values."""
        config = provider_config["factory"](name=f"test-{provider_config['name']}")

        assert config.provider == provider_config["provider"]
        assert config.base_url == provider_config["base_url"]
        assert config.endpoint == provider_config["endpoint"]
        assert config.api_key == provider_config["api_key_env"]
        assert config.kwargs["default_model"] == provider_config["default_model"]
        assert config.kwargs["excluded_params"] == provider_config["excluded_params"]

    def test_factory_custom_values(self, provider_config):
        """Test factory with custom values."""
        config = provider_config["factory"](
            name=f"test-{provider_config['name']}",
            api_key="custom_key",
            base_url="https://custom.api.com",
            endpoint="custom/endpoint",
            default_model="custom-model",
        )

        assert config.provider == provider_config["provider"]
        assert config.base_url == "https://custom.api.com"
        assert config.endpoint == "custom/endpoint"
        assert config.api_key == "custom_key"
        assert config.kwargs["default_model"] == "custom-model"

    def test_factory_extra_kwargs(self, provider_config):
        """Test factory passes extra kwargs to config."""
        config = provider_config["factory"](
            name=f"test-{provider_config['name']}",
            custom_field="custom_value",
            another_field=123,
        )
        assert config.kwargs["custom_field"] == "custom_value"
        assert config.kwargs["another_field"] == 123


# =============================================================================
# Parametrized Endpoint Tests
# =============================================================================


class TestOAICompatibleEndpoints:
    """Test OAI-compatible endpoint classes (Groq, OpenRouter, NVIDIA NIM)."""

    def test_init_with_none_config(self, provider_config):
        """Test initialization with config=None uses provider defaults."""
        endpoint = provider_config["endpoint_class"](
            config=None, name=f"test-{provider_config['name']}"
        )

        assert endpoint.config.provider == provider_config["provider"]
        assert endpoint.config.base_url == provider_config["base_url"]
        assert endpoint.config.kwargs["default_model"] == provider_config["default_model"]

    def test_init_with_dict_config(self, provider_config):
        """Test initialization with dict config."""
        config_dict = {
            "name": f"test-{provider_config['name']}",
            "provider": provider_config["provider"],
            "base_url": provider_config["base_url"],
            "endpoint": provider_config["endpoint"],
            "api_key": "test_key",
        }
        endpoint = provider_config["endpoint_class"](config=config_dict)

        assert endpoint.config.provider == provider_config["provider"]
        assert endpoint.config.api_key == "test_key"

    def test_init_with_endpoint_config(self, provider_config):
        """Test initialization with EndpointConfig instance."""
        config = provider_config["factory"](name="test-default", api_key="test_key")
        endpoint = provider_config["endpoint_class"](config=config)

        assert endpoint.config.provider == provider_config["provider"]
        assert endpoint.config.api_key == "test_key"

    def test_init_with_kwargs_override(self, provider_config):
        """Test initialization with kwargs overrides."""
        endpoint = provider_config["endpoint_class"](
            config=None,
            name=f"test-{provider_config['name']}",
            default_model="custom-model",
        )

        assert endpoint.config.kwargs["default_model"] == "custom-model"

    def test_init_with_circuit_breaker(self, provider_config):
        """Test initialization with circuit breaker."""
        from lionherd.services.utilities.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_time=1.0)
        endpoint = provider_config["endpoint_class"](
            config=None, name="test-default", circuit_breaker=cb
        )

        assert endpoint.circuit_breaker is not None

    def test_inherits_from_openai_chat(self, provider_config):
        """Test endpoint inherits from OpenAIChatEndpoint."""
        from lionherd.services.providers.openai_chat import OpenAIChatEndpoint

        endpoint = provider_config["endpoint_class"](
            config=None, name=f"test-{provider_config['name']}"
        )
        assert isinstance(endpoint, OpenAIChatEndpoint)

    @pytest.mark.asyncio
    async def test_call_success(self, provider_config):
        """Test successful API call."""
        endpoint = provider_config["endpoint_class"](
            config=None, name="test-default", api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = provider_config["example_response"]

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            result = await endpoint.call(
                request={"messages": [{"role": "user", "content": "Hello"}]}
            )

        assert result.data == provider_config["response_text"]
        assert result.metadata["model"] == provider_config["example_response"]["model"]


# =============================================================================
# Provider-Specific Tests
# =============================================================================


class TestGroqSpecificFeatures:
    """Test Groq-specific features (reasoning_effort exclusion)."""

    def test_create_groq_config_auto_excludes_reasoning_effort(self):
        """Test factory automatically excludes reasoning_effort."""
        config = create_groq_config(name="test-default", excluded_params=None)
        assert config.kwargs["excluded_params"] == {"reasoning_effort"}

    def test_create_groq_config_merges_excluded_params(self):
        """Test factory merges excluded_params with reasoning_effort."""
        config = create_groq_config(name="test-default", excluded_params={"custom_param"})
        assert config.kwargs["excluded_params"] == {"reasoning_effort", "custom_param"}

    @pytest.mark.asyncio
    async def test_call_excludes_reasoning_effort(self):
        """Test that reasoning_effort is excluded from payload."""
        endpoint = GroqChatEndpoint(config=None, name="test-default", api_key="test_key")

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
                }
            )

        # Verify reasoning_effort was excluded
        call_args = mock_client.request.call_args
        payload = call_args[1]["json"]
        assert "reasoning_effort" not in payload


class TestOpenRouterSpecificFeatures:
    """Test OpenRouter-specific features (HTTP-Referer, X-Title)."""

    def test_create_openrouter_config_http_referer(self):
        """Test factory with HTTP-Referer in kwargs."""
        config = create_openrouter_config(
            name="test-openrouter", **{"HTTP-Referer": "https://myapp.com"}
        )
        assert config.kwargs["HTTP-Referer"] == "https://myapp.com"

    def test_create_openrouter_config_x_title(self):
        """Test factory with X-Title in kwargs."""
        config = create_openrouter_config(name="test-openrouter", **{"X-Title": "MyApp"})
        assert config.kwargs["X-Title"] == "MyApp"

    @pytest.mark.asyncio
    async def test_call_with_http_referer(self):
        """Test API call with HTTP-Referer header."""
        endpoint = OpenRouterChatEndpoint(
            config=None,
            name="test-openrouter",
            api_key="test_key",
            **{"HTTP-Referer": "https://myapp.com"},
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Success"}}]}

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            await endpoint.call(request={"messages": [{"role": "user", "content": "Test"}]})

        # Verify HTTP-Referer is in the request
        # Note: It's in kwargs, not headers, so it should be in payload
        call_args = mock_client.request.call_args
        payload = call_args[1]["json"]
        assert payload.get("HTTP-Referer") == "https://myapp.com"


class TestNvidaNimSpecificFeatures:
    """Test NVIDIA NIM-specific features (error handling)."""

    @pytest.mark.asyncio
    async def test_call_http_error(self):
        """Test API call with HTTP error."""
        endpoint = NvidiaNimChatEndpoint(config=None, name="test-default", api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"error": {"message": "Forbidden"}}
        mock_response.request = MagicMock()

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(endpoint, "_create_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError, match="403"),
        ):
            await endpoint.call(request={"messages": [{"role": "user", "content": "Hello"}]})


# =============================================================================
# Cross-Provider Integration Tests
# =============================================================================


class TestCrossProviderCompatibility:
    """Test that all OAI-compatible providers work consistently."""

    @pytest.mark.asyncio
    async def test_all_providers_normalize_responses_consistently(self):
        """Test all providers normalize responses in the same way."""
        # Same response structure for all providers
        response = {
            "id": "test-123",
            "model": "test-model",
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        groq = GroqChatEndpoint(config=None, name="test-groq")
        openrouter = OpenRouterChatEndpoint(config=None, name="test-openrouter")
        nvidia = NvidiaNimChatEndpoint(config=None, name="test-nvidia")

        groq_normalized = groq.normalize_response(response)
        openrouter_normalized = openrouter.normalize_response(response)
        nvidia_normalized = nvidia.normalize_response(response)

        # All should extract text the same way
        assert groq_normalized.data == "Hello"
        assert openrouter_normalized.data == "Hello"
        assert nvidia_normalized.data == "Hello"

        # All should extract metadata consistently
        assert groq_normalized.metadata["model"] == "test-model"
        assert openrouter_normalized.metadata["model"] == "test-model"
        assert nvidia_normalized.metadata["model"] == "test-model"

    def test_all_providers_use_bearer_auth(self):
        """Test all providers use Bearer authentication."""
        groq = GroqChatEndpoint(config=None, name="test-default", api_key="test_key")
        openrouter = OpenRouterChatEndpoint(config=None, name="test-default", api_key="test_key")
        nvidia = NvidiaNimChatEndpoint(config=None, name="test-default", api_key="test_key")

        request = {"messages": [{"role": "user", "content": "Test"}]}

        _g_payload, g_headers = groq.create_payload(request)
        _or_payload, or_headers = openrouter.create_payload(request)
        _n_payload, n_headers = nvidia.create_payload(request)

        assert g_headers["Authorization"] == "Bearer test_key"
        assert or_headers["Authorization"] == "Bearer test_key"
        assert n_headers["Authorization"] == "Bearer test_key"

    def test_all_providers_have_different_defaults(self):
        """Test all providers have different default configurations."""
        groq = GroqChatEndpoint(config=None, name="test-groq")
        openrouter = OpenRouterChatEndpoint(config=None, name="test-openrouter")
        nvidia = NvidiaNimChatEndpoint(config=None, name="test-nvidia")

        # Different providers
        assert groq.config.provider == "groq"
        assert openrouter.config.provider == "openrouter"
        assert nvidia.config.provider == "nvidia_nim"

        # Different base URLs
        assert "groq.com" in groq.config.base_url
        assert "openrouter.ai" in openrouter.config.base_url
        assert "nvidia.com" in nvidia.config.base_url

        # Different default models
        assert "llama" in groq.config.kwargs["default_model"]
        assert "gemini" in openrouter.config.kwargs["default_model"]
        assert "llama" in nvidia.config.kwargs["default_model"]
