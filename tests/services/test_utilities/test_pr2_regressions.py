# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for bugs found in PR #2 review.

These tests ensure the specific issues identified in the multi-agent PR review
don't regress in future changes.

Bugs covered:
1. BLOCK-1: Missing parameters in token_calculator.py:89 (dict with 'text' field)
2. BLOCK-2: CircuitBreaker TOCTOU race condition
3. CRIT-1: Silent failure masking (verify logging on exceptions)
4. CRIT-2: Overly broad retry default (programming errors not retried)
"""

import asyncio
import logging
from unittest.mock import patch

import pytest

from lionherd.services.utilities.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    retry_with_backoff,
)
from lionherd.services.utilities.token_calculator import TokenCalculator


class TestPR2Regression:
    """Regression tests for PR #2 bugs."""

    def test_token_calculator_dict_with_text_field(self):
        """BLOCK-1: Verify dict content with 'text' field passes parameters correctly.

        Bug: line 89 was missing tokenizer and model_name parameters in recursive call.
        Symptom: TypeError when processing messages with dict content containing 'text'.
        """
        # This would have raised TypeError before the fix
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world"},
                    {"type": "text", "text": "How are you?"},
                ],
            }
        ]

        result = TokenCalculator.calculate_message_tokens(messages, model="gpt-4o")

        # Should return non-zero token count without raising TypeError
        assert result > 0
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_circuit_breaker_toctou_race_atomicity(self):
        """BLOCK-2: Verify CircuitBreaker doesn't have TOCTOU race on retry_after.

        Bug: _check_state() returned bool, then retry_after calculated outside lock.
        Symptom: Concurrent requests could get stale retry_after values.
        Fix: _check_state() now returns (bool, float) tuple atomically.
        """
        cb = CircuitBreaker(failure_threshold=1, recovery_time=5.0, name="toctou_test")

        async def failing_func():
            raise ValueError("Intentional failure")

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.execute(failing_func)

        assert cb.state == CircuitState.OPEN

        # Capture retry_after values from concurrent requests
        retry_afters = []

        async def attempt_call():
            try:
                await cb.execute(failing_func)
            except CircuitBreakerOpenError as e:
                retry_afters.append(e.retry_after)

        # Launch multiple concurrent requests
        await asyncio.gather(*(attempt_call() for _ in range(10)))

        # All retry_after values should be consistent (within small timing variance)
        assert len(retry_afters) == 10
        max_variance = max(retry_afters) - min(retry_afters)
        assert max_variance < 0.1  # Should be nearly identical (< 100ms variance)

    def test_token_calculator_logs_and_raises_on_error(self, caplog):
        """CRIT-1: Verify errors are logged and raise TokenCalculationError.

        Bug: Silent `except Exception: return 0` hid all errors (couldn't distinguish
        empty input from actual failure).
        Fix: Now logs with exc_info=True and raises TokenCalculationError for real errors.
        """
        from lionherd.services.utilities.token_calculator import TokenCalculationError

        with caplog.at_level(logging.ERROR):
            # Force an error by passing invalid tokenizer that will fail
            def failing_tokenizer(text):
                raise RuntimeError("Simulated tokenizer failure")

            # Should raise TokenCalculationError (not return 0)
            with pytest.raises(TokenCalculationError) as exc_info:
                TokenCalculator.tokenize("test", tokenizer=failing_tokenizer)

            # Verify error was logged before raising
            assert len(caplog.records) > 0
            assert any("Tokenization failed" in record.message for record in caplog.records)
            # Verify exc_info was logged (traceback present)
            assert any(record.exc_info is not None for record in caplog.records)
            # Verify exception chain preserved
            assert "Simulated tokenizer failure" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    async def test_retry_default_does_not_retry_programming_errors(self):
        """CRIT-2: Verify default retry_on excludes programming errors.

        Bug: Default was `retry_on=(Exception,)` which retried TypeError, AttributeError, etc.
        Fix: Default is now `(ConnectionError, CircuitBreakerOpenError)` - transient errors only.
        """
        call_count = 0

        async def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Programming error - should not retry")

        # Should NOT retry TypeError with default retry_on
        with pytest.raises(TypeError):
            await retry_with_backoff(raises_type_error, max_retries=3, initial_delay=0.01)

        # Should have been called exactly once (no retries)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_default_does_retry_transient_errors(self):
        """CRIT-2 complement: Verify default DOES retry transient errors."""
        from lionherd_core.errors import ConnectionError

        call_count = 0

        async def raises_connection_error():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient network error")
            return "success"

        # Should retry ConnectionError with default retry_on
        result = await retry_with_backoff(
            raises_connection_error, max_retries=3, initial_delay=0.01
        )

        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third attempt
