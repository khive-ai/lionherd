# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionherd_core import Element
from lionherd_core.protocols import Deserializable, Hashable, Observable, Serializable, implements
from pydantic import Field

if TYPE_CHECKING:
    from ..utilities.rate_limiter import TokenBucket
    from .backend import Calling

# Import at runtime for Pydantic
from .backend import ServiceBackend
from .endpoint import Endpoint

__all__ = ("iModel",)


@implements(Observable, Serializable, Deserializable, Hashable)
class iModel(Element):  # noqa: N801
    """Interface wrapper for ServiceBackend with unified invocation API.

    Supports optional rate limiting via TokenBucket and hook lifecycle management.
    """

    backend: ServiceBackend = Field(
        ...,
        description="ServiceBackend instance (Tool, Endpoint, Action)",
        exclude=True,  # Don't serialize (has callables)
    )

    rate_limiter: TokenBucket | None = Field(
        None,
        description="Optional TokenBucket rate limiter",
        exclude=True,  # Don't serialize (has Lock)
    )

    hook_registry: Any | None = Field(
        None,
        description="Optional HookRegistry for invocation lifecycle hooks",
        exclude=True,  # Don't serialize (has callables)
    )

    def __init__(
        self,
        backend: ServiceBackend,
        rate_limiter: TokenBucket | None = None,
        hook_registry: Any | None = None,
        **kwargs: Any,
    ):
        """Initialize with ServiceBackend, optional rate limiter, and optional hooks."""
        super().__init__(
            backend=backend,
            rate_limiter=rate_limiter,
            hook_registry=hook_registry,
            **kwargs,
        )

    @property
    def name(self) -> str:
        """Service name from backend."""
        return self.backend.name

    @property
    def version(self) -> str:
        """Service version from backend."""
        return self.backend.version

    @property
    def tags(self) -> set[str]:
        """Service tags from backend."""
        return self.backend.tags

    async def create_calling(self, **arguments: Any) -> Calling:
        """Create Calling instance via backend.

        For API backends (Endpoint): calls create_payload to get (payload, headers)
        For Tool backends: passes request arguments directly
        Attaches hook_registry to Calling if configured.
        """
        calling_cls = self.backend.event_type

        # For API backends (Endpoint, Action)
        if isinstance(self.backend, Endpoint):
            payload, headers = self.backend.create_payload(request=arguments)
            calling = calling_cls(
                backend=self.backend,
                payload=payload,
                headers=headers,
            )
        # For Tool backends - pass arguments via metadata
        else:
            calling = calling_cls(
                backend=self.backend,
                metadata={"arguments": arguments},
            )

        # Attach hooks if registry is configured and calling supports it
        if self.hook_registry and hasattr(calling, "create_pre_invoke_hook"):
            calling.create_pre_invoke_hook(
                hook_registry=self.hook_registry,
                exit_hook=False,
                hook_timeout=30.0,
            )
            calling.create_post_invoke_hook(
                hook_registry=self.hook_registry,
                exit_hook=False,
                hook_timeout=30.0,
            )

        return calling

    async def invoke(self, **arguments: Any) -> Calling:
        """Create and invoke calling (convenience).

        Applies rate limiting if configured. Hooks are handled by Calling itself.
        """
        # Rate limiting
        if self.rate_limiter:
            acquired = await self.rate_limiter.acquire(timeout=30.0)
            if not acquired:
                raise TimeoutError("Rate limit acquisition timeout (30s)")

        calling = await self.create_calling(**arguments)
        await calling.invoke()
        return calling

    def __repr__(self) -> str:
        """String representation."""
        return f"iModel(backend={self.backend.name}, version={self.backend.version})"


# Import TokenBucket at runtime for Pydantic model rebuild
from ..utilities.rate_limiter import TokenBucket  # noqa: E402, F401

# Rebuild model now that TokenBucket is available
iModel.model_rebuild()
