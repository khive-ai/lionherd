# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive coverage tests for iModel wrapper.

Target: 100% branch coverage for src/lionherd/services/types/imodel.py

Coverage areas:
- iModel initialization with backend, rate_limiter, hook_registry
- Property delegation (name, version, tags)
- create_calling() with Endpoint backend (payload/headers path)
- create_calling() with non-Endpoint backend
- Hook attachment logic
- invoke() with rate limiting
- __repr__()

Note: The current Tool implementation has issues (ToolCalling doesn't accept 'request' parameter),
so we use mocks for non-Endpoint backend tests.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from lionherd.services import Calling, ServiceBackend, iModel
from lionherd.services.types.endpoint import Endpoint
from lionherd.services.utilities.rate_limiter import TokenBucket

# Rebuild iModel to resolve forward references
iModel.model_rebuild()


# =============================================================================
# Mock Components
# =============================================================================


class MockCalling(Calling):
    """Mock Calling for testing."""

    backend: Any  # Override to allow any backend type

    async def _invoke(self) -> Any:
        """Execute mock invocation."""
        return "mock_response"


class MockBackend(ServiceBackend):
    """Mock ServiceBackend implementation."""

    @property
    def event_type(self) -> type[Calling]:
        """Return MockCalling type."""
        return MockCalling

    async def call(self, *args, **kwargs) -> Any:
        """Mock call implementation."""
        return {"status": "success"}


# Rebuild MockCalling to resolve forward references
MockCalling.model_rebuild()


class MockEndpoint(Endpoint):
    """Mock Endpoint for testing Endpoint-specific path."""

    def __init__(self, name: str, version: str = "1.0.0", tags: set[str] | None = None):
        from pydantic import BaseModel

        # Create minimal config
        class RequestOptions(BaseModel):
            data: str | None = None

        config = {
            "provider": "mock",
            "name": name,
            "endpoint": "/test",
            "base_url": "http://localhost",
            "request_options": RequestOptions,
            "version": version,
            "tags": list(tags or set()),
        }
        super().__init__(config=config)

    def create_payload(self, request: dict, **kwargs):
        """Mock create_payload."""
        return ({"data": "payload"}, {"Authorization": "Bearer mock"})

    async def call(self, request: dict, **kwargs):
        """Override call() to return mock response instead of making real HTTP request."""
        return {"result": "mock_response"}


class MockTokenBucket(TokenBucket):
    """Mock TokenBucket for rate limiting tests."""

    def __init__(self, should_timeout: bool = False):
        # Skip parent __init__ to avoid real initialization
        self.should_timeout = should_timeout
        # Set required parent attributes manually
        self.capacity = 100
        self.tokens = 100
        self.rate = 10
        self._lock = None  # Won't be used in tests

    async def acquire(self, timeout: float) -> bool:
        """Mock acquire - returns False if should_timeout."""
        return not self.should_timeout


# =============================================================================
# iModel Tests - Property Delegation
# =============================================================================


class TestiModelProperties:
    """Test property delegation to backend."""

    def test_name_delegation(self):
        """Test name property delegates to backend."""
        backend = MockBackend(config={"provider": "test", "name": "my_service"})
        model = iModel(backend=backend)
        assert model.name == "my_service"

    def test_version_delegation(self):
        """Test version property delegates to backend."""
        backend = MockBackend(config={"provider": "test", "name": "service"})
        backend.version = "2.0.0"
        model = iModel(backend=backend)
        assert model.version == "2.0.0"

    def test_tags_delegation(self):
        """Test tags property delegates to backend."""
        backend = MockBackend(config={"provider": "test", "name": "service"})
        backend.tags = {"production", "api"}
        model = iModel(backend=backend)
        assert model.tags == {"production", "api"}


# =============================================================================
# iModel Tests - create_calling
# =============================================================================


class TestiModelCreateCalling:
    """Test create_calling() for both Endpoint and non-Endpoint paths."""

    async def test_create_calling_endpoint_path(self):
        """Test create_calling with Endpoint backend (uses create_payload)."""
        endpoint = MockEndpoint(name="api_service", version="1.0.0")
        model = iModel(backend=endpoint)

        calling = await model.create_calling(data="test_data")

        # Verify APICalling was created with payload and headers
        assert calling.backend is endpoint
        assert calling.payload == {"data": "payload"}
        assert calling.headers == {"Authorization": "Bearer mock"}

    async def test_create_calling_non_endpoint_path(self):
        """Test create_calling with non-Endpoint backend."""
        backend = MockBackend(config={"provider": "test", "name": "tool_service"})
        model = iModel(backend=backend)

        # Create real calling instance - no patching needed
        calling = await model.create_calling(param1="value1", param2="value2")

        # Verify calling was created correctly
        assert calling.backend is backend
        assert calling.metadata["arguments"] == {"param1": "value1", "param2": "value2"}

    async def test_create_calling_with_hooks_attached(self):
        """Test hook_registry is passed to iModel when configured."""
        from lionherd.services.types.hook import HookRegistry

        backend = MockBackend(config={"provider": "test", "name": "service"})
        registry = HookRegistry()

        model = iModel(backend=backend, hook_registry=registry)

        # Verify hook_registry is set
        assert model.hook_registry is registry

        # Create calling - just verify it works with hook_registry set
        result = await model.create_calling(data="test")
        assert result.backend is backend
        assert isinstance(result, MockCalling)

    async def test_create_calling_without_hooks(self):
        """Test no hook attachment when hook_registry is None."""
        backend = MockBackend(config={"provider": "test", "name": "service"})
        model = iModel(backend=backend, hook_registry=None)

        # Create calling without hooks
        result = await model.create_calling(data="test")
        assert result.backend is backend
        assert isinstance(result, MockCalling)


# =============================================================================
# iModel Tests - invoke with rate limiting
# =============================================================================


class TestiModelInvoke:
    """Test invoke() with rate limiting."""

    async def test_invoke_without_rate_limiter(self):
        """Test invoke when no rate_limiter configured."""
        backend = MockBackend(config={"provider": "test", "name": "service"})
        model = iModel(backend=backend, rate_limiter=None)

        calling = await model.invoke(data="test")

        assert calling.execution.response == "mock_response"

    async def test_invoke_with_rate_limiter_success(self):
        """Test invoke with rate_limiter that successfully acquires."""
        backend = MockBackend(config={"provider": "test", "name": "service"})
        rate_limiter = MockTokenBucket(should_timeout=False)
        model = iModel(backend=backend, rate_limiter=rate_limiter)

        calling = await model.invoke(data="test")

        assert calling.execution.response == "mock_response"

    async def test_invoke_with_rate_limiter_timeout(self):
        """Test invoke raises TimeoutError when rate_limiter.acquire times out."""
        backend = MockBackend(config={"provider": "test", "name": "service"})
        rate_limiter = MockTokenBucket(should_timeout=True)
        model = iModel(backend=backend, rate_limiter=rate_limiter)

        with pytest.raises(TimeoutError, match="Rate limit acquisition timeout"):
            await model.invoke(data="test")


# =============================================================================
# iModel Tests - repr
# =============================================================================


class TestiModelRepr:
    """Test __repr__() string representation."""

    def test_repr(self):
        """Test string representation includes backend name and version."""
        backend = MockBackend(config={"provider": "test", "name": "my_service"})
        backend.version = "3.5.2"
        model = iModel(backend=backend)

        repr_str = repr(model)

        assert "iModel" in repr_str
        assert "my_service" in repr_str
        assert "3.5.2" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================


class TestiModelIntegration:
    """Integration tests for full workflows."""

    async def test_full_workflow_endpoint(self):
        """Test complete workflow with Endpoint backend."""
        endpoint = MockEndpoint(name="api", version="1.0.0", tags={"production"})
        model = iModel(backend=endpoint)

        # Verify properties
        assert model.name == "api"
        assert model.version == "1.0.0"
        assert "production" in model.tags

        # Create and invoke calling
        calling = await model.invoke(data="test")

        assert calling.execution.response == {"result": "mock_response"}

    async def test_full_workflow_non_endpoint(self):
        """Test complete workflow with non-Endpoint backend."""
        backend = MockBackend(config={"provider": "test", "name": "tool"})
        backend.version = "2.0.0"
        backend.tags = {"development"}

        model = iModel(backend=backend)

        # Verify properties
        assert model.name == "tool"
        assert model.version == "2.0.0"
        assert "development" in model.tags

        # Create and invoke calling
        calling = await model.invoke(param="value")

        assert calling.execution.response == "mock_response"
