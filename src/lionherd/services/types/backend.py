from __future__ import annotations

from abc import abstractmethod
from typing import Any

from lionherd_core import Element, Event, EventStatus, concurrency
from pydantic import BaseModel, Field, PrivateAttr

from .hook import HookBroadcaster, HookEvent, HookPhase


class NormalizedResponse(BaseModel):
    """Generic normalized response for all service backends.

    Works for any backend type: HTTP endpoints, tools, LLM APIs, etc.
    Provides consistent interface regardless of underlying service.
    """

    status: str = Field(..., description="Response status: 'success' or 'error'")
    data: Any = Field(None, description="Response payload (any type)")
    error: str | None = Field(None, description="Error message if status='error'")
    raw_response: dict = Field(..., description="Original unmodified response")
    metadata: dict | None = Field(None, description="Provider-specific metadata")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict (alias for model_dump)."""
        return self.model_dump(exclude_none=True)


def normalize_response(response: Any) -> tuple[str, Any, dict]:
    """Basic response normalization utility.

    Provides simple text extraction from common response formats.
    Provider-specific normalization (MCP, Anthropic, OpenAI) should
    happen in their respective backend classes (Tool, Endpoint).

    Args:
        response: Raw response

    Returns:
        Tuple of (text_content, raw_response, metadata_dict)

    Examples:
        >>> normalize_response("Hello")
        ('Hello', 'Hello', {})

        >>> normalize_response({"result": "Done"})
        ('Done', {'result': 'Done'}, {})
    """
    # Handle string responses
    if isinstance(response, str):
        return response, response, {}

    # Handle Pydantic models
    if isinstance(response, BaseModel):
        try:
            raw_dict = response.model_dump()
            text = str(response)
            return text, raw_dict, {}
        except Exception:
            text = str(response)
            return text, response, {}

    # Handle dict-like responses - simple extraction only
    if isinstance(response, dict):
        # Try common text fields
        for key in ["result", "text", "message", "content"]:
            if key in response:
                value = response[key]
                text = str(value) if not isinstance(value, str) else value
                return text, response, {}

        # Fallback: stringify
        return str(response), response, {}

    # Fallback for all other types
    return str(response), response, {}


class Calling(Event):
    """Handles asynchronous API calls with automatic token usage tracking.

    This class manages API calls through endpoints, handling both regular
    and streaming responses with optional token usage tracking.
    """

    _pre_invoke_hook_event: HookEvent | None = PrivateAttr(None)
    _post_invoke_hook_event: HookEvent | None = PrivateAttr(None)

    async def _stream(self):
        raise NotImplementedError

    async def _invoke(self):
        raise NotImplementedError

    async def invoke(self) -> None:
        """Execute the API call through the endpoint.

        Updates execution status and stores the response or error.
        """
        start = concurrency.current_time()

        try:
            self.execution.status = EventStatus.PROCESSING
            if h_ev := self._pre_invoke_hook_event:
                await h_ev.invoke()

                # Check if hook failed or was cancelled - propagate to main event
                if h_ev.execution.status in (
                    EventStatus.FAILED,
                    EventStatus.CANCELLED,
                ):
                    self.execution.status = h_ev.execution.status
                    self.execution.error = (
                        f"Pre-invoke hook {h_ev.execution.status.value}: {h_ev.execution.error}"
                    )
                    return

                if h_ev._should_exit:
                    raise h_ev._exit_cause or RuntimeError(
                        "Pre-invocation hook requested exit without a cause"
                    )
                await HookBroadcaster.broadcast(h_ev)

            response = await self._invoke()

            if h_ev := self._post_invoke_hook_event:
                await h_ev.invoke()

                # Check if hook failed or was cancelled - propagate to main event
                if h_ev.execution.status in (
                    EventStatus.FAILED,
                    EventStatus.CANCELLED,
                ):
                    self.execution.status = h_ev.execution.status
                    self.execution.error = (
                        f"Post-invoke hook {h_ev.execution.status.value}: {h_ev.execution.error}"
                    )
                    # Keep response even if hook failed
                    self.execution.response = response
                    return

                if h_ev._should_exit:
                    raise h_ev._exit_cause or RuntimeError(
                        "Post-invocation hook requested exit without a cause"
                    )
                await HookBroadcaster.broadcast(h_ev)

            self.execution.response = response
            self.execution.status = EventStatus.COMPLETED

        except concurrency.get_cancelled_exc_class():
            self.execution.error = "Invocation cancelled"
            self.execution.status = EventStatus.CANCELLED
            raise

        except Exception as e:
            self.execution.error = str(e)
            self.execution.status = EventStatus.FAILED

        finally:
            self.execution.duration = concurrency.current_time() - start

    def create_pre_invoke_hook(
        self,
        hook_registry,
        exit_hook: bool | None = None,
        hook_timeout: float = 30.0,
        hook_params: dict | None = None,
    ):
        h_ev = HookEvent(
            hook_phase=HookPhase.PreInvocation,
            event_like=self,
            registry=hook_registry,
            exit=exit_hook,
            timeout=hook_timeout,
            params=hook_params or {},
        )
        self._pre_invoke_hook_event = h_ev

    def create_post_invoke_hook(
        self,
        hook_registry,
        exit_hook: bool | None = None,
        hook_timeout: float = 30.0,
        hook_params: dict | None = None,
    ):
        h_ev = HookEvent(
            hook_phase=HookPhase.PostInvocation,
            event_like=self,
            registry=hook_registry,
            exit=exit_hook,
            timeout=hook_timeout,
            params=hook_params or {},
        )
        self._post_invoke_hook_event = h_ev


class ServiceConfig(BaseModel):
    provider: str
    name: str
    request_options: type[BaseModel] | None = None


class ServiceBackend(Element):
    config: ServiceConfig

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str | None:
        """Backend version for compatibility tracking."""
        # Try config first, then fall back to metadata
        if hasattr(self.config, "version"):
            return getattr(self.config, "version", None)
        return self.metadata.get("version")

    @version.setter
    def version(self, value: str | None):
        """Set backend version."""
        if hasattr(self.config, "version"):
            self.config.version = value
        else:
            self.metadata["version"] = value

    @property
    def tags(self) -> set[str]:
        """Tags for categorization and filtering."""
        # Try config first, then fall back to metadata
        if hasattr(self.config, "tags"):
            config_tags = getattr(self.config, "tags", None)
            if config_tags is not None:
                return set(config_tags) if isinstance(config_tags, (list, tuple, set)) else set()
        metadata_tags = self.metadata.get("tags", set())
        return (
            metadata_tags
            if isinstance(metadata_tags, set)
            else set(metadata_tags)
            if metadata_tags
            else set()
        )

    @tags.setter
    def tags(self, value: list[str] | set[str]):
        """Set backend tags."""
        tag_set = set(value) if isinstance(value, (list, tuple)) else value
        if hasattr(self.config, "tags"):
            self.config.tags = list(tag_set)
        else:
            self.metadata["tags"] = tag_set

    @property
    def request_options(self):
        """Request options schema (Pydantic model type)."""
        return getattr(self.config, "request_options", None)

    @request_options.setter
    def request_options(self, value):
        """Set request options schema."""
        if hasattr(self.config, "request_options"):
            self.config.request_options = value

    @property
    @abstractmethod
    def event_type(self) -> type[Calling]:
        """Return Calling type for this backend (e.g., ToolCalling, APICalling)."""
        ...

    @abstractmethod
    async def call(self, *args, **kw) -> NormalizedResponse: ...

    async def stream(self, *args, **kw):
        raise NotImplementedError("This backend does not support streaming calls.")
