# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from lionherd_core.errors import ConnectionError
from lionherd_core.libs.concurrency import Lock, current_time, sleep

__all__ = (
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
    "RetryConfig",
    "retry_with_backoff",
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class CircuitBreakerOpenError(ConnectionError):
    """Exception raised when a circuit breaker is open.

    Inherits from ConnectionError as circuit breaker prevents connections
    to failing services, conceptually similar to connection unavailability.
    """

    default_message = "Circuit breaker is open"
    default_retryable = True  # Circuit breaker errors are inherently retryable

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize with message and optional retry_after.

        Args:
            message: Error message (uses default_message if None)
            retry_after: Seconds until retry should be attempted
            details: Additional context dict
        """
        # Add retry_after to details if provided
        if retry_after is not None:
            details = details or {}
            details["retry_after"] = retry_after

        super().__init__(message=message, details=details, retryable=True)
        self.retry_after = retry_after


class CircuitState(Enum):
    """Circuit breaker states.

    Values:
        CLOSED: Normal operation, requests pass through
        OPEN: Service failing, rejecting requests immediately
        HALF_OPEN: Testing recovery, limited requests allowed
    """

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Fail-fast circuit breaker for service resilience.
    States: CLOSED → OPEN → HALF_OPEN → CLOSED."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_time: float = 30.0,
        half_open_max_calls: int = 1,
        excluded_exceptions: set[type[Exception]] | None = None,
        name: str = "default",
    ):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions or set()
        self.name = name

        # State variables
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = Lock()  # libs.concurrency.Lock

        # Metrics
        self._metrics = {
            "success_count": 0,
            "failure_count": 0,
            "rejected_count": 0,
            "state_changes": [],
        }

        logger.debug(
            f"Initialized CircuitBreaker '{self.name}' with failure_threshold={failure_threshold}, "
            f"recovery_time={recovery_time}, half_open_max_calls={half_open_max_calls}"
        )

    @property
    def metrics(self) -> dict[str, Any]:
        """Get circuit breaker metrics."""
        return self._metrics.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize circuit breaker configuration."""
        return {
            "failure_threshold": self.failure_threshold,
            "recovery_time": self.recovery_time,
            "half_open_max_calls": self.half_open_max_calls,
            "name": self.name,
        }

    async def _change_state(self, new_state: CircuitState) -> None:
        """Change state with logging."""
        old_state = self.state
        if new_state != old_state:
            self.state = new_state
            self._metrics["state_changes"].append(
                {
                    "time": current_time(),
                    "from": old_state.value,
                    "to": new_state.value,
                }
            )

            logger.info(
                f"Circuit '{self.name}' state changed from {old_state.value} to {new_state.value}"
            )

            # Reset counters on state change
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
            elif new_state == CircuitState.CLOSED:
                self.failure_count = 0

    async def _check_state(self) -> tuple[bool, float]:
        """Check if request can proceed.

        Returns:
            Tuple of (can_proceed, retry_after_seconds)
        """
        async with self._lock:
            now = current_time()

            if self.state == CircuitState.OPEN:
                # Check if recovery time has elapsed
                if now - self.last_failure_time >= self.recovery_time:
                    await self._change_state(CircuitState.HALF_OPEN)
                else:
                    recovery_remaining = self.recovery_time - (now - self.last_failure_time)
                    self._metrics["rejected_count"] += 1

                    logger.warning(
                        f"Circuit '{self.name}' is OPEN, rejecting request. "
                        f"Try again in {recovery_remaining:.2f}s"
                    )

                    return False, recovery_remaining

            if self.state == CircuitState.HALF_OPEN:
                # Only allow a limited number of calls in half-open state
                if self._half_open_calls >= self.half_open_max_calls:
                    self._metrics["rejected_count"] += 1

                    logger.warning(
                        f"Circuit '{self.name}' is HALF_OPEN and at capacity. Try again later."
                    )

                    return False, self.recovery_time

                self._half_open_calls += 1

            return True, 0.0

    async def execute(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute with circuit breaker protection."""
        # Check if circuit allows this call (atomically returns retry_after to avoid TOCTOU)
        can_proceed, retry_after = await self._check_state()
        if not can_proceed:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open. Retry after {retry_after:.2f} seconds",
                retry_after=retry_after,
            )

        try:
            logger.debug(
                f"Executing {func.__name__} with circuit '{self.name}' state: {self.state.value}"
            )
            result = await func(*args, **kwargs)

            # Handle success
            async with self._lock:
                self._metrics["success_count"] += 1

                # On success in half-open state, close the circuit
                if self.state == CircuitState.HALF_OPEN:
                    await self._change_state(CircuitState.CLOSED)

            return result

        except Exception as e:
            # Determine if this exception should count as a circuit failure
            is_excluded = any(isinstance(e, exc_type) for exc_type in self.excluded_exceptions)

            if not is_excluded:
                async with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = current_time()
                    self._metrics["failure_count"] += 1

                    # Log failure
                    logger.warning(
                        f"Circuit '{self.name}' failure: {e}. "
                        f"Count: {self.failure_count}/{self.failure_threshold}"
                    )

                    # Check if we need to open the circuit
                    if (
                        self.state == CircuitState.CLOSED
                        and self.failure_count >= self.failure_threshold
                    ) or self.state == CircuitState.HALF_OPEN:
                        await self._change_state(CircuitState.OPEN)

            logger.exception(f"Circuit breaker '{self.name}' caught exception")
            raise


@dataclass
class RetryConfig:
    """Retry configuration with exponential backoff + jitter.

    Uses lionherd-core primitives only (no external dependencies).

    Attributes:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to prevent thundering herd
        retry_on: Tuple of exception types that should trigger retries.
            Defaults to transient errors only (ConnectionError, CircuitBreakerOpenError,
            TimeoutError, OSError). Avoids retrying programming errors (TypeError,
            AttributeError, ValueError, etc.).

            BREAKING CHANGE (v1.0.0): Default changed from (Exception,) to narrow
            transient-only errors. If you need broad exception retry, explicitly pass:
            retry_on=(Exception,)
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] = field(
        default_factory=lambda: (ConnectionError, CircuitBreakerOpenError, TimeoutError, OSError)
    )

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff + optional jitter.

        Args:
            attempt: Current retry attempt number (0-indexed)

        Returns:
            Delay in seconds before next retry
        """
        delay = min(self.initial_delay * (self.exponential_base**attempt), self.max_delay)
        if self.jitter:
            # Add jitter: multiply by random value in [0.5, 1.0]
            delay = delay * (0.5 + random.random() * 0.5)
        return delay

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict."""
        return {
            "max_retries": self.max_retries,
            "initial_delay": self.initial_delay,
            "max_delay": self.max_delay,
            "exponential_base": self.exponential_base,
            "jitter": self.jitter,
            "retry_on": self.retry_on,
        }

    def as_kwargs(self) -> dict[str, Any]:
        """Convert config to kwargs for retry_with_backoff."""
        return self.to_dict()


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: tuple[type[Exception], ...] = (
        ConnectionError,
        CircuitBreakerOpenError,
        TimeoutError,
        OSError,
    ),
    **kwargs,
) -> T:
    """Retry async function with exponential backoff using lionherd-core.

    Uses only lionherd_core.libs.concurrency.sleep - no external dependencies.
    Implements exponential backoff with optional jitter to prevent thundering herd.

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        exponential_base: Base for exponential calculation (default: 2.0)
        jitter: Add random jitter to prevent thundering herd (default: True)
        retry_on: Tuple of exception types that should trigger retries.
            Defaults to transient errors (ConnectionError, CircuitBreakerOpenError,
            TimeoutError, OSError). Does NOT retry programming errors (TypeError,
            ValueError, etc.) by default.

            BREAKING CHANGE (v1.0.0): Default changed from (Exception,) to narrow
            transient-only errors. Migration:
            - Old: retry_with_backoff(func)  # Retried ALL exceptions
            - New: retry_with_backoff(func, retry_on=(Exception,))  # Explicit broad retry
        **kwargs: Keyword arguments for func

    Returns:
        Result from successful func execution

    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retry_on as e:
            last_exception = e

            # If this was the last attempt, raise the exception
            if attempt >= max_retries:
                logger.error(f"All {max_retries} retry attempts exhausted for {func.__name__}: {e}")
                raise

            # Calculate delay with exponential backoff
            delay = min(initial_delay * (exponential_base**attempt), max_delay)

            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)

            logger.debug(
                f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__} "
                f"after {delay:.2f}s: {e}"
            )

            await sleep(delay)

    # This should never be reached, but for type safety
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected retry loop exit")
