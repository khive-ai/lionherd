# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive coverage tests for Endpoint module."""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from lionherd_core.errors import ConnectionError
from lionherd.services.types.endpoint import APICalling, Endpoint, EndpointConfig
from lionherd.services.utilities.resilience import CircuitBreaker, RetryConfig


# Test Request Models
class SimpleRequest(BaseModel):
    """Simple request model for testing."""

    message: str
    temperature: float = 0.7


class TestEndpointConfig:
    """Test EndpointConfig validation and properties."""

    def test_validate_kwargs_moves_extra_fields(self):
        """Test that extra fields are moved to kwargs dict."""
        data = {
            "name": "test",
            "provider": "test_provider",
            "endpoint": "/test",
            "extra_field": "extra_value",
            "another_field": 123,
        }
        config = EndpointConfig(**data)
        assert config.kwargs["extra_field"] == "extra_value"
        assert config.kwargs["another_field"] == 123

    def test_validate_api_key_from_secret_str(self):
        """Test API key validation from SecretStr."""
        secret = SecretStr("secret_key_123")
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            api_key=secret,
        )
        assert config._api_key == "secret_key_123"

    def test_validate_api_key_from_string_literal(self):
        """Test API key validation from string (not env var)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            api_key="literal_key_456",
        )
        assert config._api_key == "literal_key_456"

    def test_validate_api_key_from_env_var(self):
        """Test API key validation from environment variable."""
        os.environ["TEST_API_KEY"] = "env_key_789"
        try:
            config = EndpointConfig(
                name="test",
                provider="test",
                endpoint="/test",
                api_key="TEST_API_KEY",
            )
            assert config._api_key == "env_key_789"
        finally:
            del os.environ["TEST_API_KEY"]

    def test_validate_provider_empty_raises(self):
        """Test that empty provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider must be specified"):
            EndpointConfig(
                name="test",
                provider="",
                endpoint="/test",
            )

    def test_full_url_with_endpoint_params(self):
        """Test full_url property with endpoint params."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="api/{version}/test",
            base_url="https://api.test.com",
            endpoint_params=["version"],
            params={"version": "v1"},
            request_options=SimpleRequest,
        )
        # Test on both config and endpoint
        assert config.full_url == "https://api.test.com/api/v1/test"

        endpoint = Endpoint(config=config)
        assert endpoint.full_url == "https://api.test.com/api/v1/test"

    def test_validate_request_options_none(self):
        """Test request_options validator with None."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=None,
        )
        assert config.request_options is None

    def test_validate_request_options_pydantic_class(self):
        """Test request_options validator with Pydantic class."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        assert config.request_options == SimpleRequest

    def test_validate_request_options_pydantic_instance(self):
        """Test request_options validator with Pydantic instance."""
        instance = SimpleRequest(message="test")
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=instance,
        )
        assert config.request_options == SimpleRequest

    def test_validate_request_options_dict_no_schema_gen(self):
        """Test request_options with dict when schema-gen not installed."""
        with patch("lionherd.services.types.endpoint.logger") as mock_logger:
            config = EndpointConfig(
                name="test",
                provider="test",
                endpoint="/test",
                request_options={"type": "object", "properties": {}},
            )
            # Should log warning and return None
            assert config.request_options is None
            mock_logger.warning.assert_called_once()

    def test_full_url_without_endpoint_params(self):
        """Test full_url property without endpoint params."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
        )
        # Test when endpoint_params is None (line 80)
        assert config.endpoint_params is None
        assert config.full_url == "https://api.test.com/test"

    def test_validate_request_options_generic_exception(self):
        """Test request_options with exception during validation."""

        # Create a mock that raises a non-ImportError exception
        class BadModel:
            """A class that will cause issues during validation."""

            pass

        with pytest.raises(ValueError, match="Invalid request options"):
            EndpointConfig(
                name="test",
                provider="test",
                endpoint="/test",
                request_options=BadModel,  # Invalid - not a BaseModel
            )

    def test_validate_request_options_invalid_type(self):
        """Test request_options with invalid type raises."""
        with pytest.raises(ValueError, match="Invalid request options"):
            EndpointConfig(
                name="test",
                provider="test",
                endpoint="/test",
                request_options=12345,  # Invalid type
            )

    def test_serialize_request_options_none(self):
        """Test serializing None request_options."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=None,
        )
        serialized = config.model_dump()
        # request_options should be None in serialization
        assert serialized.get("request_options") is None

    def test_serialize_request_options_with_model(self):
        """Test serializing request_options with model."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        serialized = config.model_dump()
        assert serialized["request_options"] is not None
        assert "properties" in serialized["request_options"]

    def test_update_merges_kwargs(self):
        """Test update method merges kwargs dict."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            kwargs={"existing": "value"},
        )
        config.update(kwargs={"new": "data"})
        assert config.kwargs["existing"] == "value"
        assert config.kwargs["new"] == "data"

    def test_update_sets_attributes(self):
        """Test update method sets attributes."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
        )
        config.update(timeout=600, method="GET")
        assert config.timeout == 600
        assert config.method == "GET"

    def test_update_adds_unknown_to_kwargs(self):
        """Test update adds unknown fields to kwargs."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
        )
        config.update(custom_field="custom_value")
        assert config.kwargs["custom_field"] == "custom_value"

    def test_validate_payload_no_request_options(self):
        """Test validate_payload returns data when no request_options."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=None,
        )
        data = {"test": "data"}
        result = config.validate_payload(data)
        assert result == data

    def test_validate_payload_with_validation(self):
        """Test validate_payload validates against request_options."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        data = {"message": "hello", "temperature": 0.5}
        result = config.validate_payload(data)
        assert result == data

    def test_validate_payload_invalid_raises(self):
        """Test validate_payload raises on invalid data."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        data = {"temperature": "not_a_float"}  # Invalid
        with pytest.raises(ValueError, match="Invalid payload"):
            config.validate_payload(data)


class TestEndpoint:
    """Test Endpoint class."""

    def test_init_with_dict_config(self):
        """Test Endpoint initialization with dict config."""
        config_dict = {
            "name": "test",
            "provider": "test",
            "endpoint": "/test",
            "base_url": "https://api.test.com",
            "request_options": SimpleRequest,
        }
        endpoint = Endpoint(config=config_dict)
        assert endpoint.config.name == "test"
        assert endpoint.config.provider == "test"

    def test_init_with_endpoint_config(self):
        """Test Endpoint initialization with EndpointConfig instance."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        endpoint = Endpoint(config=config)
        assert endpoint.config.name == "test"

    def test_init_with_invalid_config_type(self):
        """Test Endpoint initialization with invalid config type."""
        with pytest.raises(ValueError, match="Config must be a dict or EndpointConfig"):
            Endpoint(config="invalid")

    def test_event_type_property(self):
        """Test event_type property returns APICalling."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        endpoint = Endpoint(config=config)
        assert endpoint.event_type == APICalling

    def test_request_options_property(self):
        """Test request_options property."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        endpoint = Endpoint(config=config)
        assert endpoint.request_options == SimpleRequest

    def test_request_options_setter(self):
        """Test request_options setter."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        endpoint = Endpoint(config=config)

        # Define another model
        class AnotherRequest(BaseModel):
            data: str

        endpoint.request_options = AnotherRequest
        assert endpoint.request_options == AnotherRequest

    def test_normalize_response_default(self):
        """Test default normalize_response passes through."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        endpoint = Endpoint(config=config)
        raw = {"result": "success"}
        normalized = endpoint.normalize_response(raw)
        assert normalized.status == "success"
        assert normalized.data == raw
        assert normalized.raw_response == raw

    def test_create_payload_with_extra_headers(self):
        """Test create_payload with extra headers."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
            api_key="test_key",
            auth_type="bearer",
        )
        endpoint = Endpoint(config=config)
        request = {"message": "test", "temperature": 0.8}
        extra_headers = {"X-Custom": "header"}

        payload, headers = endpoint.create_payload(request, extra_headers=extra_headers)

        assert "X-Custom" in headers
        assert headers["X-Custom"] == "header"
        assert "Authorization" in headers

    def test_create_payload_with_kwargs(self):
        """Test create_payload merges kwargs."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)
        request = {"message": "test"}

        payload, headers = endpoint.create_payload(
            request, extra_param="extra_value", temperature=0.9
        )

        assert payload["message"] == "test"
        assert payload["temperature"] == 0.9  # From kwargs

    def test_create_payload_no_request_options_raises(self):
        """Test create_payload raises if no request_options."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=None,  # No schema
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        with pytest.raises(ValueError, match="must define request_options"):
            endpoint.create_payload({"data": "test"})

    @pytest.mark.asyncio
    async def test_call_skip_payload_creation_dict(self):
        """Test call with skip_payload_creation and dict request."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # Mock _call
        async def mock_call(payload, headers, **kwargs):
            return {"result": "success"}

        endpoint._call = mock_call

        result = await endpoint.call(
            request={"message": "test", "temperature": 0.5},
            skip_payload_creation=True,
        )

        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_call_skip_payload_creation_basemodel(self):
        """Test call with skip_payload_creation and BaseModel request."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # Mock _call
        async def mock_call(payload, headers, **kwargs):
            return {"result": "success"}

        endpoint._call = mock_call

        request_model = SimpleRequest(message="test", temperature=0.5)
        result = await endpoint.call(
            request=request_model,
            skip_payload_creation=True,
        )

        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_call_with_retry_only(self):
        """Test call with retry_config only (no circuit breaker)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        retry_config = RetryConfig(max_retries=2, initial_delay=0.01)
        endpoint = Endpoint(config=config, retry_config=retry_config)

        call_count = 0

        async def mock_call(payload, headers, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Retry me")
            return {"result": "success"}

        endpoint._call = mock_call

        result = await endpoint.call(request={"message": "test", "temperature": 0.7})

        assert result.status == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_call_with_circuit_breaker_only(self):
        """Test call with circuit_breaker only (no retry)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_time=1.0)
        endpoint = Endpoint(config=config, circuit_breaker=circuit_breaker)

        async def mock_call(payload, headers, **kwargs):
            return {"result": "success"}

        endpoint._call = mock_call

        result = await endpoint.call(request={"message": "test", "temperature": 0.7})

        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_call_with_both_retry_and_circuit_breaker(self):
        """Test call with both retry_config and circuit_breaker."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        retry_config = RetryConfig(max_retries=2, initial_delay=0.01)
        circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_time=1.0)
        endpoint = Endpoint(
            config=config, retry_config=retry_config, circuit_breaker=circuit_breaker
        )

        call_count = 0

        async def mock_call(payload, headers, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Retry error")
            return {"result": "success"}

        endpoint._call = mock_call

        result = await endpoint.call(request={"message": "test", "temperature": 0.7})

        assert result.status == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_call_http_success(self):
        """Test _call_http with successful response."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # Mock httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            result = await endpoint._call_http(
                payload={"message": "test", "temperature": 0.7},
                headers={"Authorization": "Bearer test"},
            )

        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_call_http_429_raises(self):
        """Test _call_http with 429 rate limit raises."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limited", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                await endpoint._call_http(
                    payload={"message": "test", "temperature": 0.7},
                    headers={"Authorization": "Bearer test"},
                )

    @pytest.mark.asyncio
    async def test_call_http_500_raises(self):
        """Test _call_http with 500 server error raises."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                await endpoint._call_http(
                    payload={"message": "test", "temperature": 0.7},
                    headers={"Authorization": "Bearer test"},
                )

    @pytest.mark.asyncio
    async def test_call_http_non_200_with_json_error(self):
        """Test _call_http with non-200 status and JSON error body."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request", "code": "INVALID"}
        mock_response.request = MagicMock()

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError, match="400"):
                await endpoint._call_http(
                    payload={"message": "test", "temperature": 0.7},
                    headers={"Authorization": "Bearer test"},
                )

    @pytest.mark.asyncio
    async def test_call_http_non_200_without_json_error(self):
        """Test _call_http with non-200 status and no JSON error body."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.side_effect = Exception("Not JSON")
        mock_response.request = MagicMock()

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError, match="403"):
                await endpoint._call_http(
                    payload={"message": "test", "temperature": 0.7},
                    headers={"Authorization": "Bearer test"},
                )

    @pytest.mark.asyncio
    async def test_stream(self):
        """Test stream method."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        async def mock_stream_http(payload, headers, **kwargs):
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        endpoint._stream_http = mock_stream_http

        chunks = []
        async for chunk in endpoint.stream(request={"message": "test", "temperature": 0.7}):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2", "chunk3"]

    @pytest.mark.asyncio
    async def test_stream_http(self):
        """Test _stream_http method."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # Mock streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_aiter_lines():
            yield "line1"
            yield ""  # Empty line (should be skipped)
            yield "line2"
            yield "line3"

        mock_response.aiter_lines = mock_aiter_lines
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            chunks = []
            async for chunk in endpoint._stream_http(
                payload={"message": "test", "temperature": 0.7},
                headers={"Authorization": "Bearer test"},
            ):
                chunks.append(chunk)

        # Empty line should be filtered out
        assert chunks == ["line1", "line2", "line3"]

    @pytest.mark.asyncio
    async def test_stream_http_error(self):
        """Test _stream_http with error response."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.request = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError, match="500"):
                async for _ in endpoint._stream_http(
                    payload={"message": "test", "temperature": 0.7},
                    headers={"Authorization": "Bearer test"},
                ):
                    pass

    def test_to_dict_with_retry_config(self):
        """Test to_dict with retry_config."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        retry_config = RetryConfig(max_retries=3, initial_delay=0.1)
        endpoint = Endpoint(config=config, retry_config=retry_config)

        result = endpoint.to_dict()

        assert result["retry_config"] is not None
        assert result["retry_config"]["max_retries"] == 3

    def test_to_dict_with_circuit_breaker(self):
        """Test to_dict with circuit_breaker."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_time=2.0)
        endpoint = Endpoint(config=config, circuit_breaker=circuit_breaker)

        result = endpoint.to_dict()

        assert result["circuit_breaker"] is not None
        assert result["circuit_breaker"]["failure_threshold"] == 5

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "config": {
                "name": "test",
                "provider": "test",
                "endpoint": "/test",
                "request_options": SimpleRequest.model_json_schema(),
            },
            "retry_config": {
                "max_retries": 3,
                "initial_delay": 0.1,
            },
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_time": 2.0,
            },
        }

        endpoint = Endpoint.from_dict(data)

        assert endpoint.config.name == "test"
        assert endpoint.retry_config is not None
        assert endpoint.retry_config.max_retries == 3
        assert endpoint.circuit_breaker is not None
        assert endpoint.circuit_breaker.failure_threshold == 5

    def test_from_dict_invalid_type(self):
        """Test from_dict with invalid type raises."""
        with pytest.raises(TypeError, match="Expected dict"):
            Endpoint.from_dict("not a dict")

    def test_create_http_client(self):
        """Test _create_http_client creates httpx.AsyncClient."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            timeout=120,
            client_kwargs={"follow_redirects": True},
        )
        endpoint = Endpoint(config=config)

        # Create client
        client = endpoint._create_http_client()

        # Verify it's an httpx.AsyncClient
        assert client.__class__.__name__ == "AsyncClient"
        # Timeout should be configured
        assert client.timeout.read == 120

    def test_from_dict_with_none_values(self):
        """Test from_dict with None retry_config and circuit_breaker."""
        data = {
            "config": {
                "name": "test",
                "provider": "test",
                "endpoint": "test",
                "request_options": SimpleRequest.model_json_schema(),
            },
            "retry_config": None,
            "circuit_breaker": None,
        }

        endpoint = Endpoint.from_dict(data)

        assert endpoint.config.name == "test"
        assert endpoint.retry_config is None
        assert endpoint.circuit_breaker is None


class TestAPICalling:
    """Test APICalling class."""

    @pytest.mark.asyncio
    async def test_invoke(self):
        """Test _invoke method calls backend.call."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # Mock _call_http instead since call is an async method we can't easily patch on Pydantic
        async def mock_call_http(payload, headers, **kwargs):
            return {"result": "success"}

        endpoint._call_http = mock_call_http

        calling = APICalling(
            backend=endpoint,
            payload={"message": "test", "temperature": 0.7},
            headers={"X-Custom": "header"},
        )

        result = await calling._invoke()

        # Should get the normalized response back
        assert result.status == "success"
        assert result.data == {"result": "success"}

    def test_request_property(self):
        """Test request property returns sanitized request info."""
        config = EndpointConfig(
            name="test",
            provider="test_provider",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"message": "test", "temperature": 0.7},
            headers={"Authorization": "Bearer secret"},
        )

        request_info = calling.request

        assert request_info["provider"] == "test_provider"
        assert request_info["endpoint_url"] == "https://api.test.com/test"
        assert request_info["payload"] == {"message": "test", "temperature": 0.7}
        assert "headers" not in request_info  # Should be excluded for security
