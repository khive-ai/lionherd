# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from typing import Any, TypeVar

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    SecretStr,
    field_serializer,
    field_validator,
    model_validator,
)

from ..utilities.header_factory import AUTH_TYPES, HeaderFactory
from ..utilities.resilience import CircuitBreaker, RetryConfig, retry_with_backoff
from .backend import Calling, NormalizedResponse, ServiceBackend, ServiceConfig

logger = logging.getLogger(__name__)


B = TypeVar("B", bound=type[BaseModel])


class EndpointConfig(ServiceConfig):
    base_url: str | None = None
    endpoint: str
    endpoint_params: list[str] | None = None
    method: str = "POST"
    params: dict[str, str] = Field(default_factory=dict)
    content_type: str | None = "application/json"
    auth_type: AUTH_TYPES = "bearer"
    default_headers: dict = {}
    api_key: str | SecretStr | None = Field(None, exclude=True)
    timeout: int = 300
    max_retries: int = 3
    openai_compatible: bool = False
    requires_tokens: bool = False
    version: str | None = None
    tags: list[str] = Field(default_factory=list)
    kwargs: dict = Field(default_factory=dict)
    client_kwargs: dict = Field(default_factory=dict)
    _api_key: str | None = PrivateAttr(None)

    @model_validator(mode="before")
    def _validate_kwargs(cls, data: dict):  # noqa: N805
        kwargs = data.pop("kwargs", {})
        field_keys = list(cls.model_json_schema().get("properties", {}).keys())
        for k in list(data.keys()):
            if k not in field_keys:
                kwargs[k] = data.pop(k)
        data["kwargs"] = kwargs
        return data

    @model_validator(mode="after")
    def _validate_api_key(self):
        if self.api_key is not None:
            if isinstance(self.api_key, SecretStr):
                self._api_key = self.api_key.get_secret_value()
            elif isinstance(self.api_key, str):
                # Try environment variable first, then use as-is
                self._api_key = os.getenv(self.api_key, self.api_key)

        return self

    @field_validator("provider", mode="before")
    def _validate_provider(cls, v: str):  # noqa: N805
        if not v:
            raise ValueError("Provider must be specified")
        return v.strip().lower()

    @property
    def full_url(self):
        if not self.endpoint_params:
            return f"{self.base_url}/{self.endpoint}"
        return f"{self.base_url}/{self.endpoint.format(**self.params)}"

    @field_validator("request_options", mode="before")
    def _validate_request_options(cls, v):  # noqa: N805
        # Create a simple empty model if None is provided
        if v is None:
            return None

        try:
            if isinstance(v, type) and issubclass(v, BaseModel):
                return v
            if isinstance(v, BaseModel):
                return v.__class__
            if isinstance(v, dict | str):
                try:
                    from lionherd_core import schema_handlers

                    return schema_handlers.load_pydantic_model_from_schema(v)
                except ImportError:
                    logger.warning(
                        "datamodel-code-generator not installed. "
                        "Install with: pip install 'lionherd[schema-gen]' "
                        "request_options will not be validated"
                    )
                    return None
        except Exception as e:
            raise ValueError("Invalid request options") from e
        raise ValueError("Invalid request options: must be a Pydantic model or a schema dict")

    @field_serializer("request_options")
    def _serialize_request_options(self, v: type[BaseModel] | None):
        if v is None:
            return None
        return v.model_json_schema()

    def update(self, **kwargs):
        """Update the config with new values."""
        # Handle the special case of kwargs dict
        if "kwargs" in kwargs:
            # Merge the kwargs dicts
            self.kwargs.update(kwargs.pop("kwargs"))

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Add to kwargs dict if not a direct attribute
                self.kwargs[key] = value

    def validate_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.request_options:
            return data

        try:
            self.request_options.model_validate(data)
            return data
        except Exception as e:
            raise ValueError("Invalid payload") from e


class Endpoint(ServiceBackend):
    circuit_breaker: CircuitBreaker | None = Field(None, exclude=True)
    retry_config: RetryConfig | None = Field(None, exclude=True)
    config: EndpointConfig

    def __init__(
        self,
        config: dict | EndpointConfig,
        circuit_breaker: CircuitBreaker | None = None,
        retry_config: RetryConfig | None = None,
        **kwargs,
    ):
        if isinstance(config, dict):
            _config = EndpointConfig(**config, **kwargs)
        elif isinstance(config, EndpointConfig):
            _config = config.model_copy(deep=True)
            _config.update(**kwargs)
        else:
            raise ValueError("Config must be a dict or EndpointConfig instance")

        # Initialize ServiceBackend with config and resilience components
        super().__init__(
            config=_config,
            circuit_breaker=circuit_breaker,
            retry_config=retry_config,
        )

        logger.debug(
            f"Initialized Endpoint with provider={self.config.provider}, "
            f"endpoint={self.config.endpoint}, circuit_breaker={circuit_breaker is not None}, "
            f"retry_config={retry_config is not None}"
        )

    def _create_http_client(self):
        """Create a new HTTP client for requests."""
        import httpx

        return httpx.AsyncClient(
            timeout=self.config.timeout,
            **self.config.client_kwargs,
        )

    @property
    def event_type(self) -> type:
        """Return Event/Calling type for this backend."""
        # APICalling defined later in this file
        return APICalling

    @property
    def full_url(self) -> str:
        """Return full URL from config."""
        return self.config.full_url

    @property
    def request_options(self):
        return self.config.request_options

    @request_options.setter
    def request_options(self, value):
        self.config.request_options = EndpointConfig._validate_request_options(value)

    def normalize_response(self, raw_response: dict) -> NormalizedResponse:
        """Normalize raw API response. Override in provider-specific endpoints.

        Args:
            raw_response: Raw dict from API

        Returns:
            NormalizedResponse with status, data, error fields

        Example override in OpenAIChat:
            def normalize_response(self, raw_response: dict) -> NormalizedResponse:
                choice = raw_response.get("choices", [{}])[0]
                return NormalizedResponse(
                    status="success",
                    data=choice.get("message", {}).get("content"),
                    raw_response=raw_response,
                    metadata={"usage": raw_response.get("usage")},
                )
        """
        # Default: pass through as-is
        return NormalizedResponse(
            status="success",
            data=raw_response,
            raw_response=raw_response,
        )

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        # First, create headers
        headers = HeaderFactory.get_header(
            auth_type=self.config.auth_type,
            content_type=self.config.content_type,
            api_key=self.config._api_key,
            default_headers=self.config.default_headers,
        )
        if extra_headers:
            headers.update(extra_headers)

        # Convert request to dict if it's a BaseModel
        request = request if isinstance(request, dict) else request.model_dump(exclude_none=True)

        # Start with config defaults
        payload = self.config.kwargs.copy()

        # Update with request data
        payload.update(request)

        # Update with additional kwargs
        if kwargs:
            payload.update(kwargs)

        # Validate payload using request_options schema (required)
        if self.config.request_options is None:
            raise ValueError(
                f"Endpoint {self.config.name} must define request_options schema. "
                "All endpoint backends must use proper request validation."
            )

        # Get valid field names from the model
        valid_fields = set(self.config.request_options.model_fields.keys())

        # Filter payload to only include valid fields
        filtered_payload = {k: v for k, v in payload.items() if k in valid_fields}

        # Validate the filtered payload
        payload = self.config.validate_payload(filtered_payload)

        return (payload, headers)

    async def _call(self, payload: dict, headers: dict, **kwargs):
        return await self._call_http(payload=payload, headers=headers, **kwargs)

    async def call(
        self,
        request: dict | BaseModel,
        skip_payload_creation: bool = False,
        **kwargs,
    ):
        """
        Make a call to the endpoint.

        Args:
            request: The request parameters or model.
            skip_payload_creation: Whether to skip create_payload and treat request as ready payload.
            **kwargs: Additional keyword arguments for the request.

        Returns:
            NormalizedResponse from the endpoint.
        """
        # Extract extra_headers before passing to create_payload
        extra_headers = kwargs.pop("extra_headers", None)

        payload, headers = None, None
        if skip_payload_creation:
            # Treat request as the ready payload
            payload = request if isinstance(request, dict) else request.model_dump()
            headers = extra_headers or {}
        else:
            payload, headers = self.create_payload(request, extra_headers=extra_headers, **kwargs)

        # Apply resilience patterns if configured
        call_func = self._call

        # Apply retry if configured
        if self.retry_config:

            async def call_func(p, h, **kw):
                return await retry_with_backoff(
                    self._call, p, h, **kw, **self.retry_config.as_kwargs()
                )

        # Apply circuit breaker if configured
        if self.circuit_breaker:
            if self.retry_config:
                # If both are configured, apply circuit breaker to the retry-wrapped function
                raw_response = await self.circuit_breaker.execute(
                    call_func, payload, headers, **kwargs
                )
                return self.normalize_response(raw_response)
            else:
                # If only circuit breaker is configured, apply it directly
                raw_response = await self.circuit_breaker.execute(
                    self._call, payload, headers, **kwargs
                )
                return self.normalize_response(raw_response)

        # Apply resilience patterns directly
        if self.retry_config:
            raw_response = await call_func(payload, headers, **kwargs)
        else:
            raw_response = await self._call(payload, headers, **kwargs)

        # Normalize response before returning
        return self.normalize_response(raw_response)

    async def _call_http(self, payload: dict, headers: dict, **kwargs):
        import httpx

        # Create a new client for this request
        async with self._create_http_client() as client:
            response = await client.request(
                method=self.config.method,
                url=self.config.full_url,
                headers=headers,
                json=payload,
                **kwargs,
            )

            # Check for rate limit or server errors that should be retried
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            elif response.status_code != 200:
                # Try to get error details from response body
                try:
                    error_body = response.json()
                    error_message = (
                        f"Request failed with status {response.status_code}: {error_body}"
                    )
                except:
                    error_message = f"Request failed with status {response.status_code}"

                raise httpx.HTTPStatusError(
                    message=error_message,
                    request=response.request,
                    response=response,
                )

            # Extract and return the JSON response
            return response.json()

    async def stream(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        """
        Stream responses from the endpoint.

        Args:
            request: The request parameters or model.
            extra_headers: Additional headers for the request.
            **kwargs: Additional keyword arguments for the request.

        Yields:
            Streaming chunks from the API.
        """
        payload, headers = self.create_payload(request, extra_headers, **kwargs)

        # Direct streaming without context manager
        async for chunk in self._stream_http(payload=payload, headers=headers, **kwargs):
            yield chunk

    async def _stream_http(self, payload: dict, headers: dict, **kwargs):
        """
        Stream responses using httpx with a fresh client.

        Args:
            payload: The request payload.
            headers: The request headers.
            **kwargs: Additional keyword arguments for the request.

        Yields:
            Streaming chunks from the API.
        """
        import httpx

        # Ensure stream is enabled
        payload["stream"] = True

        # Create a new client for streaming
        async with (
            self._create_http_client() as client,
            client.stream(
                method=self.config.method,
                url=self.config.full_url,
                headers=headers,
                json=payload,
                **kwargs,
            ) as response,
        ):
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    message=f"Request failed with status {response.status_code}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if line:
                    yield line

    def to_dict(self):
        return {
            "retry_config": (self.retry_config.to_dict() if self.retry_config else None),
            "circuit_breaker": (self.circuit_breaker.to_dict() if self.circuit_breaker else None),
            "config": self.config.model_dump(exclude_none=True),
        }

    @classmethod
    def from_dict(cls, data: dict):
        # Ensure data is dict (already validated by type hint)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data)}")

        retry_config = data.get("retry_config")
        circuit_breaker = data.get("circuit_breaker")
        config = data.get("config")

        if retry_config:
            retry_config = RetryConfig(**retry_config)
        if circuit_breaker:
            circuit_breaker = CircuitBreaker(**circuit_breaker)
        if config:
            config = EndpointConfig(**config)

        return cls(
            config=config,
            circuit_breaker=circuit_breaker,
            retry_config=retry_config,
        )


class APICalling(Calling):
    """API call via Endpoint.

    Extends Calling (Event-based) to execute API requests through Endpoint.
    Supports timeout, response normalization, and state-based error handling.

    Attributes:
        backend: Endpoint backend instance
        payload: Request payload dict
        headers: Request headers dict

    Usage:
        endpoint = Endpoint(config=config)
        calling = APICalling(
            backend=endpoint,
            payload={"model": "gpt-4", "messages": [...]},
            headers={},
            timeout=30.0
        )
        await calling.invoke()
        response = calling.execution.response  # NormalizedResponse
    """

    backend: Endpoint = Field(  # type: ignore[annotation-unchecked]
        ...,
        description="Endpoint backend instance",
        exclude=True,  # Not serializable
    )

    payload: dict[str, Any] = Field(
        ...,
        description="Request payload dict",
    )

    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Request headers dict",
    )

    async def _invoke(self) -> Any:
        """Execute via endpoint.call().

        Called by parent Calling.invoke() with timeout and normalization.

        Returns:
            Raw response from endpoint (will be normalized by parent)
        """
        return await self.backend.call(
            request=self.payload,
            skip_payload_creation=True,
            extra_headers=self.headers,
        )

    @property
    def request(self) -> dict:
        """Request info (excludes API keys for security).

        Returns:
            Dict with provider, endpoint_url, and payload
        """
        return {
            "provider": self.backend.config.provider,
            "endpoint_url": self.backend.full_url,
            "payload": self.payload,
            # headers excluded - may contain Authorization
        }
