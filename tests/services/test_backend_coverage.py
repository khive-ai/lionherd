# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for src/lionherd/services/types/backend.py

Targets 100% coverage of:
- NormalizedResponse class
- normalize_response() function
- Calling class (Event-based async invoke/stream)
- ServiceBackend abstract class
- ServiceConfig
"""

from __future__ import annotations

import pytest
from lionherd_core import EventStatus
from pydantic import BaseModel

from lionherd.services.types.backend import (
    Calling,
    NormalizedResponse,
    ServiceBackend,
    ServiceConfig,
    normalize_response,
)
from lionherd.services.types.hook import HookEvent, HookPhase, HookRegistry


# =============================================================================
# Test Fixtures
# =============================================================================


class PydanticTestModel(BaseModel):
    """Test Pydantic model for normalize_response tests."""

    name: str
    value: int


class BrokenPydanticModel(BaseModel):
    """Model that raises exception in model_dump()."""

    name: str

    def model_dump(self, **kwargs):
        raise RuntimeError("model_dump failed")


class MockCalling(Calling):
    """Mock Calling implementation for testing."""

    result_value: str = "test_result"
    should_fail: bool = False
    should_cancel: bool = False

    async def _invoke(self):
        """Mock invoke implementation."""
        if self.should_cancel:
            import asyncio

            raise asyncio.CancelledError("Test cancellation")
        if self.should_fail:
            raise RuntimeError("Test failure")
        return self.result_value

    async def _stream(self):
        """Mock stream implementation."""
        raise NotImplementedError("Stream not implemented")


class MockServiceBackend(ServiceBackend):
    """Mock ServiceBackend for testing properties."""

    @property
    def event_type(self) -> type[Calling]:
        """Return MockCalling type."""
        return MockCalling

    async def call(self, *args, **kw) -> NormalizedResponse:
        """Mock call implementation."""
        return NormalizedResponse(
            status="success",
            data={"result": "test"},
            raw_response={"result": "test"},
        )


@pytest.fixture
def hook_registry():
    """Create empty hook registry."""
    return HookRegistry()


@pytest.fixture
def mock_calling():
    """Create mock calling instance."""
    return MockCalling()


# =============================================================================
# NormalizedResponse Tests
# =============================================================================


class TestNormalizedResponse:
    """Test NormalizedResponse class."""

    def test_normalized_response_basic(self):
        """Test basic NormalizedResponse creation."""
        response = NormalizedResponse(
            status="success",
            data={"key": "value"},
            raw_response={"original": "data"},
        )

        assert response.status == "success"
        assert response.data == {"key": "value"}
        assert response.error is None
        assert response.raw_response == {"original": "data"}
        assert response.metadata is None

    def test_normalized_response_with_error(self):
        """Test NormalizedResponse with error."""
        response = NormalizedResponse(
            status="error",
            error="Something went wrong",
            raw_response={"error": "details"},
        )

        assert response.status == "error"
        assert response.error == "Something went wrong"
        assert response.data is None

    def test_to_dict_excludes_none(self):
        """Test to_dict() excludes None values."""
        response = NormalizedResponse(
            status="success",
            data="result",
            raw_response={"result": "test"},
        )

        result = response.to_dict()
        assert "error" not in result
        assert "metadata" not in result
        assert result["status"] == "success"
        assert result["data"] == "result"

    def test_to_dict_includes_metadata(self):
        """Test to_dict() includes metadata when present."""
        response = NormalizedResponse(
            status="success",
            data="result",
            raw_response={"result": "test"},
            metadata={"usage": {"tokens": 100}},
        )

        result = response.to_dict()
        assert result["metadata"] == {"usage": {"tokens": 100}}


# =============================================================================
# normalize_response() Function Tests
# =============================================================================


class TestNormalizeResponse:
    """Test normalize_response() function."""

    def test_normalize_string(self):
        """Test normalize_response with string input."""
        text, raw, metadata = normalize_response("Hello")

        assert text == "Hello"
        assert raw == "Hello"
        assert metadata == {}

    def test_normalize_pydantic_model_success(self):
        """Test normalize_response with Pydantic model (success)."""
        model = PydanticTestModel(name="test", value=42)
        text, raw, metadata = normalize_response(model)

        assert "test" in text
        assert raw == {"name": "test", "value": 42}
        assert metadata == {}

    def test_normalize_pydantic_model_exception(self):
        """Test normalize_response with Pydantic model that raises exception."""
        model = BrokenPydanticModel(name="test")
        text, raw, metadata = normalize_response(model)

        # Should fallback to str() and return model object
        assert "test" in text
        assert raw == model
        assert metadata == {}

    def test_normalize_dict_with_result(self):
        """Test normalize_response with dict containing 'result' key."""
        response = {"result": "Done"}
        text, raw, metadata = normalize_response(response)

        assert text == "Done"
        assert raw == {"result": "Done"}
        assert metadata == {}

    def test_normalize_dict_with_text(self):
        """Test normalize_response with dict containing 'text' key."""
        response = {"text": "Message"}
        text, raw, metadata = normalize_response(response)

        assert text == "Message"
        assert raw == {"text": "Message"}
        assert metadata == {}

    def test_normalize_dict_with_message(self):
        """Test normalize_response with dict containing 'message' key."""
        response = {"message": "Info"}
        text, raw, metadata = normalize_response(response)

        assert text == "Info"
        assert raw == {"message": "Info"}
        assert metadata == {}

    def test_normalize_dict_with_content(self):
        """Test normalize_response with dict containing 'content' key."""
        response = {"content": "Data"}
        text, raw, metadata = normalize_response(response)

        assert text == "Data"
        assert raw == {"content": "Data"}
        assert metadata == {}

    def test_normalize_dict_with_non_string_value(self):
        """Test normalize_response with dict containing non-string value."""
        response = {"result": 123}
        text, raw, metadata = normalize_response(response)

        assert text == "123"
        assert raw == {"result": 123}
        assert metadata == {}

    def test_normalize_dict_without_common_keys(self):
        """Test normalize_response with dict without common keys."""
        response = {"custom": "value", "other": "data"}
        text, raw, metadata = normalize_response(response)

        # Should fallback to str(response)
        assert "custom" in text
        assert raw == response
        assert metadata == {}

    def test_normalize_other_types(self):
        """Test normalize_response with other types (list, int, etc)."""
        # List
        text, raw, metadata = normalize_response([1, 2, 3])
        assert text == "[1, 2, 3]"
        assert raw == [1, 2, 3]
        assert metadata == {}

        # Int
        text, raw, metadata = normalize_response(42)
        assert text == "42"
        assert raw == 42
        assert metadata == {}

        # None
        text, raw, metadata = normalize_response(None)
        assert text == "None"
        assert raw is None
        assert metadata == {}


# =============================================================================
# Calling Class Tests
# =============================================================================


class TestCalling:
    """Test Calling class invoke() and hook integration."""

    @pytest.mark.asyncio
    async def test_invoke_success_no_hooks(self, mock_calling):
        """Test successful invoke without hooks."""
        from lionherd_core import Unset

        await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.COMPLETED
        assert mock_calling.execution.response == "test_result"
        # Error is Unset (sentinel value) or None after successful completion
        assert mock_calling.execution.error in (None, Unset)
        assert mock_calling.execution.duration > 0

    @pytest.mark.asyncio
    async def test_invoke_failure(self, mock_calling):
        """Test invoke with failure."""
        mock_calling.should_fail = True
        await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.FAILED
        assert mock_calling.execution.error == "Test failure"

    @pytest.mark.asyncio
    async def test_invoke_cancellation(self, mock_calling):
        """Test invoke with cancellation."""
        import asyncio

        mock_calling.should_cancel = True

        with pytest.raises(asyncio.CancelledError):
            await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.CANCELLED
        assert "cancelled" in mock_calling.execution.error.lower()

    @pytest.mark.asyncio
    async def test_invoke_with_pre_hook_success(self, mock_calling):
        """Test invoke with pre-invoke hook that succeeds."""
        # Create a mock hook event
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.COMPLETED
        mock_hook.execution.error = None
        mock_hook._should_exit = False
        mock_hook.invoke = AsyncMock()

        mock_calling._pre_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent actual broadcast calls
        with patch("lionherd.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.COMPLETED
        mock_hook.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_with_pre_hook_failed(self, mock_calling):
        """Test invoke when pre-invoke hook fails."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.PENDING
        mock_hook._should_exit = False

        async def failing_hook():
            mock_hook.execution.status = EventStatus.FAILED
            mock_hook.execution.error = "Hook failed"

        mock_hook.invoke = failing_hook

        mock_calling._pre_invoke_hook_event = mock_hook

        await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.FAILED
        assert "Pre-invoke hook failed:" in mock_calling.execution.error

    @pytest.mark.asyncio
    async def test_invoke_with_pre_hook_cancelled(self, mock_calling):
        """Test invoke when pre-invoke hook is cancelled."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.PENDING
        mock_hook._should_exit = False

        async def cancelled_hook():
            mock_hook.execution.status = EventStatus.CANCELLED
            mock_hook.execution.error = "Hook cancelled"

        mock_hook.invoke = cancelled_hook

        mock_calling._pre_invoke_hook_event = mock_hook

        await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.CANCELLED
        assert "Pre-invoke hook cancelled:" in mock_calling.execution.error

    @pytest.mark.asyncio
    async def test_invoke_with_pre_hook_exit_with_cause(self, mock_calling):
        """Test invoke when pre-invoke hook requests exit with cause."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.COMPLETED
        mock_hook._should_exit = True
        mock_hook._exit_cause = RuntimeError("Exit requested")
        mock_hook.invoke = AsyncMock()

        mock_calling._pre_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent actual broadcast calls
        with patch("lionherd.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        # Exception caught and converted to FAILED status
        assert mock_calling.execution.status == EventStatus.FAILED
        assert "Exit requested" in mock_calling.execution.error

    @pytest.mark.asyncio
    async def test_invoke_with_pre_hook_exit_without_cause(self, mock_calling):
        """Test invoke when pre-invoke hook requests exit without cause."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.COMPLETED
        mock_hook._should_exit = True
        mock_hook._exit_cause = None
        mock_hook.invoke = AsyncMock()

        mock_calling._pre_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent actual broadcast calls
        with patch("lionherd.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        # Exception caught and converted to FAILED status
        assert mock_calling.execution.status == EventStatus.FAILED
        assert "requested exit without a cause" in mock_calling.execution.error

    @pytest.mark.asyncio
    async def test_invoke_with_post_hook_success(self, mock_calling):
        """Test invoke with post-invoke hook that succeeds."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.COMPLETED
        mock_hook.execution.error = None
        mock_hook._should_exit = False
        mock_hook.invoke = AsyncMock()

        mock_calling._post_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent actual broadcast calls
        with patch("lionherd.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.COMPLETED
        assert mock_calling.execution.response == "test_result"
        mock_hook.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_with_post_hook_failed(self, mock_calling):
        """Test invoke when post-invoke hook fails."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.PENDING
        mock_hook._should_exit = False

        async def failing_hook():
            mock_hook.execution.status = EventStatus.FAILED
            mock_hook.execution.error = "Hook failed"

        mock_hook.invoke = failing_hook

        mock_calling._post_invoke_hook_event = mock_hook

        await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.FAILED
        assert "Post-invoke hook failed:" in mock_calling.execution.error
        assert mock_calling.execution.response == "test_result"  # Response preserved

    @pytest.mark.asyncio
    async def test_invoke_with_post_hook_cancelled(self, mock_calling):
        """Test invoke when post-invoke hook is cancelled."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.PENDING
        mock_hook._should_exit = False

        async def cancelled_hook():
            mock_hook.execution.status = EventStatus.CANCELLED
            mock_hook.execution.error = "Hook cancelled"

        mock_hook.invoke = cancelled_hook

        mock_calling._post_invoke_hook_event = mock_hook

        await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.CANCELLED
        assert "Post-invoke hook cancelled:" in mock_calling.execution.error
        assert mock_calling.execution.response == "test_result"  # Response preserved

    @pytest.mark.asyncio
    async def test_invoke_with_post_hook_exit_with_cause(self, mock_calling):
        """Test invoke when post-invoke hook requests exit with cause."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.COMPLETED
        mock_hook._should_exit = True
        mock_hook._exit_cause = RuntimeError("Exit requested")
        mock_hook.invoke = AsyncMock()

        mock_calling._post_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent actual broadcast calls
        with patch("lionherd.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        # Exception caught and converted to FAILED status
        assert mock_calling.execution.status == EventStatus.FAILED
        assert "Exit requested" in mock_calling.execution.error

    @pytest.mark.asyncio
    async def test_invoke_with_post_hook_exit_without_cause(self, mock_calling):
        """Test invoke when post-invoke hook requests exit without cause."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.COMPLETED
        mock_hook._should_exit = True
        mock_hook._exit_cause = None
        mock_hook.invoke = AsyncMock()

        mock_calling._post_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent actual broadcast calls
        with patch("lionherd.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        # Exception caught and converted to FAILED status
        assert mock_calling.execution.status == EventStatus.FAILED
        assert "requested exit without a cause" in mock_calling.execution.error

    @pytest.mark.asyncio
    async def test_stream_not_implemented(self, mock_calling):
        """Test that _stream raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await mock_calling._stream()


# =============================================================================
# ServiceConfig Tests
# =============================================================================


class TestServiceConfig:
    """Test ServiceConfig class."""

    def test_service_config_basic(self):
        """Test basic ServiceConfig creation."""
        config = ServiceConfig(provider="test", name="test_service")

        assert config.provider == "test"
        assert config.name == "test_service"
        assert config.request_options is None

    def test_service_config_with_request_options(self):
        """Test ServiceConfig with request_options."""

        class TestOptions(BaseModel):
            timeout: int = 30

        config = ServiceConfig(
            provider="test", name="test_service", request_options=TestOptions
        )

        assert config.request_options == TestOptions


# =============================================================================
# ServiceBackend Tests
# =============================================================================


class TestServiceBackend:
    """Test ServiceBackend abstract class properties."""

    def test_provider_property(self):
        """Test provider property."""
        config = ServiceConfig(provider="test_provider", name="test")
        backend = MockServiceBackend(config=config)

        assert backend.provider == "test_provider"

    def test_name_property(self):
        """Test name property."""
        config = ServiceConfig(provider="test", name="test_name")
        backend = MockServiceBackend(config=config)

        assert backend.name == "test_name"

    def test_version_property_from_config(self):
        """Test version property when config has version attribute."""
        # Create a config class that supports version
        class VersionedConfig(ServiceConfig):
            version: str | None = None

        config = VersionedConfig(provider="test", name="test", version="1.0.0")
        backend = MockServiceBackend(config=config)

        assert backend.version == "1.0.0"

    def test_version_property_from_metadata(self):
        """Test version property from metadata when config doesn't have version."""
        config = ServiceConfig(provider="test", name="test")
        backend = MockServiceBackend(config=config)
        backend.metadata["version"] = "2.0.0"

        assert backend.version == "2.0.0"

    def test_version_setter_to_config(self):
        """Test version setter when config has version attribute."""
        # Create a config class that supports version
        class VersionedConfig(ServiceConfig):
            version: str | None = None

        config = VersionedConfig(provider="test", name="test", version="1.0.0")
        backend = MockServiceBackend(config=config)
        backend.version = "1.5.0"

        assert backend.version == "1.5.0"
        assert config.version == "1.5.0"

    def test_version_setter_to_metadata(self):
        """Test version setter when config doesn't have version attribute."""
        config = ServiceConfig(provider="test", name="test")
        backend = MockServiceBackend(config=config)

        backend.version = "3.0.0"

        assert backend.version == "3.0.0"
        assert backend.metadata["version"] == "3.0.0"

    def test_tags_property_from_config_list(self):
        """Test tags property when config has tags as list."""
        # Create a config class that supports tags
        class TaggedConfig(ServiceConfig):
            tags: list[str] | None = None

        config = TaggedConfig(provider="test", name="test", tags=["tag1", "tag2"])
        backend = MockServiceBackend(config=config)

        assert backend.tags == {"tag1", "tag2"}

    def test_tags_property_from_config_tuple(self):
        """Test tags property when config has tags as tuple."""
        # Create a config class that supports tags
        class TaggedConfig(ServiceConfig):
            tags: tuple[str, ...] | None = None

        config = TaggedConfig(provider="test", name="test", tags=("tag1", "tag2"))
        backend = MockServiceBackend(config=config)

        assert backend.tags == {"tag1", "tag2"}

    def test_tags_property_from_config_set(self):
        """Test tags property when config has tags as set."""
        # Create a config class that supports tags
        class TaggedConfig(ServiceConfig):
            tags: set[str] | None = None

        config = TaggedConfig(provider="test", name="test", tags={"tag1", "tag2"})
        backend = MockServiceBackend(config=config)

        assert backend.tags == {"tag1", "tag2"}

    def test_tags_property_from_config_none(self):
        """Test tags property when config.tags is None."""
        # Create a config class that supports tags
        class TaggedConfig(ServiceConfig):
            tags: list[str] | None = None

        config = TaggedConfig(provider="test", name="test", tags=None)
        backend = MockServiceBackend(config=config)

        assert backend.tags == set()

    def test_tags_property_from_metadata_set(self):
        """Test tags property from metadata as set."""
        config = ServiceConfig(provider="test", name="test")
        backend = MockServiceBackend(config=config)
        backend.metadata["tags"] = {"meta1", "meta2"}

        assert backend.tags == {"meta1", "meta2"}

    def test_tags_property_from_metadata_list(self):
        """Test tags property from metadata as list."""
        config = ServiceConfig(provider="test", name="test")
        backend = MockServiceBackend(config=config)
        backend.metadata["tags"] = ["meta1", "meta2"]

        assert backend.tags == {"meta1", "meta2"}

    def test_tags_property_from_metadata_empty(self):
        """Test tags property from metadata when empty."""
        config = ServiceConfig(provider="test", name="test")
        backend = MockServiceBackend(config=config)

        assert backend.tags == set()

    def test_tags_setter_to_config_from_list(self):
        """Test tags setter when config has tags attribute (from list)."""
        # Create a config class that supports tags
        class TaggedConfig(ServiceConfig):
            tags: list[str] | None = None

        config = TaggedConfig(provider="test", name="test", tags=[])
        backend = MockServiceBackend(config=config)
        backend.tags = ["new1", "new2"]

        assert backend.tags == {"new1", "new2"}
        assert set(config.tags) == {"new1", "new2"}

    def test_tags_setter_to_config_from_set(self):
        """Test tags setter when config has tags attribute (from set)."""
        # Create a config class that supports tags
        class TaggedConfig(ServiceConfig):
            tags: list[str] | None = None

        config = TaggedConfig(provider="test", name="test", tags=[])
        backend = MockServiceBackend(config=config)
        backend.tags = {"new1", "new2"}

        assert backend.tags == {"new1", "new2"}
        assert set(config.tags) == {"new1", "new2"}

    def test_tags_setter_to_metadata(self):
        """Test tags setter when config doesn't have tags attribute."""
        config = ServiceConfig(provider="test", name="test")
        backend = MockServiceBackend(config=config)

        backend.tags = ["meta1", "meta2"]

        assert backend.tags == {"meta1", "meta2"}
        assert backend.metadata["tags"] == {"meta1", "meta2"}

    def test_request_options_property_getter(self):
        """Test request_options property getter."""

        class TestOptions(BaseModel):
            timeout: int = 30

        config = ServiceConfig(
            provider="test", name="test", request_options=TestOptions
        )
        backend = MockServiceBackend(config=config)

        assert backend.request_options == TestOptions

    def test_request_options_property_getter_none(self):
        """Test request_options property getter when None."""
        config = ServiceConfig(provider="test", name="test")
        backend = MockServiceBackend(config=config)

        assert backend.request_options is None

    def test_request_options_property_setter(self):
        """Test request_options property setter."""

        class TestOptions(BaseModel):
            timeout: int = 30

        config = ServiceConfig(provider="test", name="test")
        backend = MockServiceBackend(config=config)

        backend.request_options = TestOptions

        assert backend.request_options == TestOptions
        assert config.request_options == TestOptions

    @pytest.mark.asyncio
    async def test_stream_not_implemented(self):
        """Test that stream() raises NotImplementedError."""
        config = ServiceConfig(provider="test", name="test")
        backend = MockServiceBackend(config=config)

        with pytest.raises(
            NotImplementedError, match="does not support streaming calls"
        ):
            await backend.stream()
