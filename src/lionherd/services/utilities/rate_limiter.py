# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass

from lionherd_core.libs.concurrency import Lock, current_time, sleep

__all__ = ("RateLimitConfig", "TokenBucket")

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RateLimitConfig:
    """Token bucket rate limiting configuration.

    Uses lionherd-core primitives only.
    """

    capacity: int  # Maximum tokens
    refill_rate: float  # Tokens per second
    initial_tokens: int | None = None  # Start with N tokens (default: capacity)

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if self.refill_rate <= 0:
            raise ValueError("refill_rate must be > 0")

        if self.initial_tokens is None:
            self.initial_tokens = self.capacity


class TokenBucket:
    """Token bucket rate limiter using lionherd-core primitives.

    Thread-safe async implementation for API rate limiting.
    """

    def __init__(self, config: RateLimitConfig):
        self.capacity = config.capacity
        self.refill_rate = config.refill_rate
        self.tokens = float(config.initial_tokens)
        self.last_refill = current_time()
        self._lock = Lock()

    async def acquire(self, tokens: int = 1, *, timeout: float | None = None) -> bool:
        """Acquire N tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire
            timeout: Max wait time in seconds (None = wait forever)

        Returns:
            True if acquired, False if timeout
        """
        start_time = current_time()

        while True:
            async with self._lock:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    logger.debug(f"Acquired {tokens} tokens, {self.tokens:.2f} remaining")
                    return True

                # Calculate wait time for next refill
                deficit = tokens - self.tokens
                wait_time = deficit / self.refill_rate

            # Check timeout
            if timeout is not None:
                elapsed = current_time() - start_time
                if elapsed + wait_time > timeout:
                    logger.warning(f"Rate limit timeout after {elapsed:.2f}s")
                    return False
                wait_time = min(wait_time, timeout - elapsed)

            logger.debug(f"Waiting {wait_time:.2f}s for {deficit:.2f} tokens")
            await sleep(wait_time)

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = current_time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting.

        Returns:
            True if acquired immediately, False if insufficient tokens
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "capacity": self.capacity,
            "refill_rate": self.refill_rate,
            "tokens": self.tokens,
            "last_refill": self.last_refill,
        }
