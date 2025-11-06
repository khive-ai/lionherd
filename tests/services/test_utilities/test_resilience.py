# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for resilience patterns (CircuitBreaker, Retry, TokenBucket)."""

import asyncio

import pytest

from lionherd.services.utilities import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    RateLimitConfig,
    RetryConfig,
    TokenBucket,
    retry_with_backoff,
)

# Skip integration tests in PR 1 (Endpoint comes in PR 3)
try:
    from lionherd.services.types import Endpoint, EndpointConfig

    ENDPOINT_AVAILABLE = True
except ImportError:
    ENDPOINT_AVAILABLE = False


class TestCircuitBreaker:
    """Test CircuitBreaker 3-state pattern."""

    @pytest.mark.asyncio
    async def test_closed_state_allows_calls(self):
        """Circuit starts CLOSED and allows calls."""
        cb = CircuitBreaker(failure_threshold=3, name="test")

        async def success_func():
            return "success"

        result = await cb.execute(success_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.metrics["success_count"] == 1

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        """Circuit opens after failure_threshold failures."""
        cb = CircuitBreaker(failure_threshold=3, name="test")

        async def failing_func():
            raise ValueError("Simulated failure")

        # Fail 3 times to hit threshold
        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.execute(failing_func)

        # Circuit should be OPEN now
        assert cb.state == CircuitState.OPEN
        assert cb.metrics["failure_count"] == 3

    @pytest.mark.asyncio
    async def test_rejects_when_open(self):
        """Circuit rejects calls when OPEN."""
        cb = CircuitBreaker(failure_threshold=2, recovery_time=10.0, name="test")

        async def failing_func():
            raise ValueError("Fail")

        # Trigger circuit open
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.execute(failing_func)

        assert cb.state == CircuitState.OPEN

        # Next call should be rejected immediately
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await cb.execute(failing_func)

        assert "is open" in str(exc_info.value).lower()
        assert cb.metrics["rejected_count"] == 1

    @pytest.mark.asyncio
    async def test_transitions_to_half_open(self):
        """Circuit transitions OPEN → HALF_OPEN after recovery_time."""
        cb = CircuitBreaker(
            failure_threshold=2, recovery_time=0.1, half_open_max_calls=1, name="test"
        )

        async def failing_func():
            raise ValueError("Fail")

        # Trigger circuit open
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.execute(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery time
        await asyncio.sleep(0.15)

        # Next call should enter HALF_OPEN (allowed through for testing)
        async def success_func():
            return "recovered"

        result = await cb.execute(success_func)
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED  # Successful call closes circuit

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self):
        """Successful call in HALF_OPEN closes circuit."""
        cb = CircuitBreaker(
            failure_threshold=1, recovery_time=0.05, half_open_max_calls=1, name="test"
        )

        async def failing_func():
            raise ValueError("Fail")

        async def success_func():
            return "success"

        # Open circuit
        with pytest.raises(ValueError):
            await cb.execute(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.1)

        # Successful call in HALF_OPEN should close circuit
        result = await cb.execute(success_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self):
        """Failure in HALF_OPEN reopens circuit immediately."""
        cb = CircuitBreaker(
            failure_threshold=1, recovery_time=0.05, half_open_max_calls=1, name="test"
        )

        async def failing_func():
            raise ValueError("Fail")

        # Open circuit
        with pytest.raises(ValueError):
            await cb.execute(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.1)

        # Failure in HALF_OPEN should reopen immediately
        with pytest.raises(ValueError):
            await cb.execute(failing_func)

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_excluded_exceptions_dont_count(self):
        """Excluded exceptions don't increment failure count."""
        cb = CircuitBreaker(failure_threshold=2, excluded_exceptions={KeyError}, name="test")

        async def excluded_error_func():
            raise KeyError("Excluded")

        async def normal_error_func():
            raise ValueError("Normal")

        # Excluded exception shouldn't count
        with pytest.raises(KeyError):
            await cb.execute(excluded_error_func)

        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

        # Normal exceptions should count
        with pytest.raises(ValueError):
            await cb.execute(normal_error_func)

        assert cb.failure_count == 1

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Circuit breaker tracks metrics correctly."""
        cb = CircuitBreaker(failure_threshold=3, name="test")

        async def success_func():
            return "ok"

        async def fail_func():
            raise ValueError("fail")

        # Track successes and failures
        await cb.execute(success_func)
        await cb.execute(success_func)

        with pytest.raises(ValueError):
            await cb.execute(fail_func)

        metrics = cb.metrics
        assert metrics["success_count"] == 2
        assert metrics["failure_count"] == 1
        assert len(metrics["state_changes"]) == 0  # Still CLOSED


class TestRetryWithBackoff:
    """Test retry_with_backoff with exponential backoff + jitter."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        """Function succeeds on first attempt - no retries."""

        async def success_func():
            return "success"

        result = await retry_with_backoff(success_func, max_retries=3, initial_delay=0.01)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Retries up to max_retries on failures."""
        call_count = 0

        async def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = await retry_with_backoff(
            eventually_succeeds,
            max_retries=3,
            initial_delay=0.01,
            jitter=False,
            retry_on=(Exception,),  # Test with broad exception catching
        )
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        """Raises exception after max_retries exhausted."""

        async def always_fails():
            raise ValueError("Always fail")

        with pytest.raises(ValueError, match="Always fail"):
            await retry_with_backoff(always_fails, max_retries=2, initial_delay=0.01)

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Delay increases exponentially with exponential_base."""
        call_times = []

        async def track_timing():
            import time

            call_times.append(time.monotonic())
            if len(call_times) < 3:
                raise ValueError("Retry")
            return "done"

        await retry_with_backoff(
            track_timing,
            max_retries=3,
            initial_delay=0.01,
            exponential_base=2.0,
            jitter=False,
            retry_on=(Exception,),  # Test with broad exception catching
        )

        # Check delays are increasing (roughly exponential)
        # Delay 1: ~0.01s, Delay 2: ~0.02s
        # Allow tolerance for timing variance
        assert len(call_times) == 3
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        assert delay2 > delay1  # Exponential increase

    @pytest.mark.asyncio
    async def test_jitter_adds_randomness(self):
        """Jitter adds randomness to prevent thundering herd."""
        delays = []

        for _ in range(5):
            call_times = []

            async def track_timing(times=call_times):
                import time

                times.append(time.monotonic())
                if len(times) < 2:
                    raise ValueError("Retry")
                return "done"

            await retry_with_backoff(
                track_timing,
                max_retries=2,
                initial_delay=0.1,
                jitter=True,
                retry_on=(Exception,),  # Test with broad exception catching
            )
            if len(call_times) >= 2:
                delays.append(call_times[1] - call_times[0])

        # With jitter, delays should vary (not all identical)
        assert len(set(delays)) > 1  # At least some variation

    @pytest.mark.asyncio
    async def test_excluded_exceptions_not_retried(self):
        """Excluded exceptions are not retried (only retry ValueError)."""
        call_count = 0

        async def raises_excluded():
            nonlocal call_count
            call_count += 1
            raise KeyError("Excluded")

        # Only retry ValueError, so KeyError should not be retried
        with pytest.raises(KeyError):
            await retry_with_backoff(
                raises_excluded, max_retries=3, initial_delay=0.01, retry_on=(ValueError,)
            )

        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Delay is capped at max_delay."""
        call_times = []

        async def track_timing():
            import time

            call_times.append(time.monotonic())
            if len(call_times) < 4:
                raise ValueError("Retry")
            return "done"

        await retry_with_backoff(
            track_timing,
            max_retries=5,
            initial_delay=1.0,
            exponential_base=10.0,
            max_delay=0.1,
            jitter=False,
            retry_on=(Exception,),  # Test with broad exception catching
        )

        # All delays should be capped at max_delay (0.1s)
        for i in range(1, len(call_times)):
            delay = call_times[i] - call_times[i - 1]
            assert delay <= 0.15  # Allow small tolerance


class TestTokenBucket:
    """Test TokenBucket rate limiting."""

    @pytest.mark.asyncio
    async def test_initial_tokens_available(self):
        """Bucket starts with capacity available."""
        config = RateLimitConfig(capacity=100, refill_rate=10.0)
        bucket = TokenBucket(config)

        # Initial tokens should equal capacity
        assert bucket.tokens == 100.0

    @pytest.mark.asyncio
    async def test_acquire_tokens_success(self):
        """Can acquire tokens when available."""
        config = RateLimitConfig(capacity=10, refill_rate=5.0)
        bucket = TokenBucket(config)

        success = await bucket.acquire(tokens=5)
        assert success is True

        # Allow small tolerance for time-based refill
        assert 4.9 <= bucket.tokens <= 5.1

    @pytest.mark.asyncio
    async def test_acquire_tokens_failure(self):
        """Cannot acquire more tokens than available (with timeout)."""
        config = RateLimitConfig(capacity=10, refill_rate=5.0)
        bucket = TokenBucket(config)

        # Consume all tokens
        await bucket.acquire(tokens=10)

        # Try to acquire more - should timeout and fail
        success = await bucket.acquire(tokens=1, timeout=0.05)
        assert success is False

    @pytest.mark.asyncio
    async def test_token_refill_over_time(self):
        """Tokens refill based on elapsed time."""
        config = RateLimitConfig(capacity=100, refill_rate=100.0)  # 100 tokens/sec
        bucket = TokenBucket(config)

        # Consume all tokens
        await bucket.acquire(tokens=100)
        # Allow small tolerance for continuous time-based refill
        assert bucket.tokens <= 0.5

        # Wait for refill (0.1s = 10 tokens at 100/sec rate)
        await asyncio.sleep(0.1)

        # Should have refilled ~10 tokens (may be slightly more due to timing)
        async with bucket._lock:
            bucket._refill()
        assert bucket.tokens >= 9.0
        assert bucket.tokens <= 11.0  # Allow some timing variance

    @pytest.mark.asyncio
    async def test_refill_capped_at_max(self):
        """Refill doesn't exceed capacity."""
        config = RateLimitConfig(capacity=10, refill_rate=1000.0)  # Very high refill rate
        bucket = TokenBucket(config)

        # Wait for multiple refill intervals
        await asyncio.sleep(0.3)

        # Force refill calculation
        async with bucket._lock:
            bucket._refill()
        assert bucket.tokens == 10.0  # Capped at capacity

    @pytest.mark.asyncio
    async def test_validation_errors(self):
        """RateLimitConfig validates parameters."""
        with pytest.raises(ValueError, match="capacity must be > 0"):
            RateLimitConfig(capacity=0, refill_rate=10.0)

        with pytest.raises(ValueError, match="refill_rate must be > 0"):
            RateLimitConfig(capacity=10, refill_rate=0.0)

        with pytest.raises(ValueError, match="refill_rate must be > 0"):
            RateLimitConfig(capacity=10, refill_rate=-5.0)

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self):
        """TokenBucket handles concurrent access safely."""
        config = RateLimitConfig(
            capacity=100, refill_rate=1.0
        )  # Low refill to minimize timing effects
        bucket = TokenBucket(config)

        # Simulate concurrent consumers
        async def consume_tokens():
            for _ in range(10):
                await bucket.acquire(tokens=1)
                await asyncio.sleep(0.001)

        # Run multiple consumers concurrently
        await asyncio.gather(*[consume_tokens() for _ in range(5)])

        # Should have consumed 50 tokens total (allow small tolerance for time-based refill)
        assert 48.0 <= bucket.tokens <= 52.0


@pytest.mark.skipif(not ENDPOINT_AVAILABLE, reason="Endpoint not available (added in PR 3)")
class TestEndpointIntegration:
    """Test Endpoint integration with resilience patterns."""

    @pytest.mark.asyncio
    async def test_endpoint_with_circuit_breaker(self):
        """Endpoint integrates CircuitBreaker correctly."""
        from lionherd.services.types import Endpoint, EndpointConfig

        config = EndpointConfig(
            name="test-endpoint",
            provider="test",
            base_url="https://example.com",
            endpoint="api/test",
            method="POST",
        )

        cb = CircuitBreaker(failure_threshold=2, recovery_time=1.0, name="test")
        endpoint = Endpoint(config=config, circuit_breaker=cb)

        assert endpoint.circuit_breaker is cb

    @pytest.mark.asyncio
    async def test_endpoint_with_retry_config(self):
        """Endpoint integrates RetryConfig correctly."""
        from lionherd.services.types import Endpoint, EndpointConfig

        config = EndpointConfig(
            name="test-endpoint",
            provider="test",
            base_url="https://example.com",
            endpoint="api/test",
            method="POST",
        )

        retry = RetryConfig(max_retries=3, initial_delay=0.1)
        endpoint = Endpoint(config=config, retry_config=retry)

        assert endpoint.retry_config is retry

    @pytest.mark.asyncio
    async def test_endpoint_resilience_layering(self):
        """Endpoint layers retry → circuit breaker → HTTP correctly."""
        from pydantic import BaseModel, SecretStr

        from lionherd.services.types import Endpoint, EndpointConfig

        # Define minimal request schema
        class TestRequest(BaseModel):
            pass

        config = EndpointConfig(
            name="test-endpoint",
            provider="test",
            base_url="https://httpbin.org",
            endpoint="status/500",  # Returns 500 error
            method="GET",
            api_key=SecretStr("dummy_key"),  # Required for authentication
            request_options=TestRequest,  # Required schema validation
        )

        cb = CircuitBreaker(
            failure_threshold=1, recovery_time=10.0, name="test"
        )  # Open after 1 failure
        retry = RetryConfig(max_retries=1, initial_delay=0.01)

        endpoint = Endpoint(config=config, circuit_breaker=cb, retry_config=retry)

        # This should fail after retries and open circuit
        import httpx

        with pytest.raises(httpx.HTTPStatusError):
            await endpoint.call(request={})

        # Circuit should be OPEN after the failure (threshold=1)
        assert cb.state == CircuitState.OPEN


# =============================================================================
# Coverage Push: Missing Lines 127, 184-190, 307, 318, 360
# =============================================================================


class TestCircuitBreakerCoveragePush:
    """Tests targeting uncovered CircuitBreaker lines."""

    def test_circuit_breaker_to_dict(self):
        """Test CircuitBreaker.to_dict() serialization (line 127)."""
        # COVERAGE TARGET: Line 127 (to_dict method)
        cb = CircuitBreaker(
            failure_threshold=5,
            recovery_time=30.0,
            half_open_max_calls=2,
            name="test_breaker",
        )

        config = cb.to_dict()

        assert config["failure_threshold"] == 5
        assert config["recovery_time"] == 30.0
        assert config["half_open_max_calls"] == 2
        assert config["name"] == "test_breaker"

    @pytest.mark.asyncio
    async def test_half_open_rejects_when_at_capacity(self):
        """Test HALF_OPEN state rejects calls when at capacity (lines 184-190)."""
        # COVERAGE TARGET: Lines 184-190 (HALF_OPEN capacity rejection)
        # NOTE: Circuit transitions HALF_OPEN → CLOSED on first success,
        # so we need concurrent calls to hit the capacity limit
        cb = CircuitBreaker(
            failure_threshold=1, recovery_time=0.05, half_open_max_calls=1, name="capacity_test"
        )

        async def failing_func():
            raise ValueError("Fail")

        async def slow_success():
            await asyncio.sleep(0.1)  # Slow enough for concurrent call to check state
            return "success"

        # Open circuit
        with pytest.raises(ValueError):
            await cb.execute(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for HALF_OPEN transition
        await asyncio.sleep(0.1)

        import anyio

        # Try to execute 2 calls concurrently while in HALF_OPEN (max_calls=1)
        # First call should be allowed, second should be rejected
        results = []
        errors = []

        async def call_with_tracking(call_id):
            try:
                result = await cb.execute(slow_success)
                results.append((call_id, result))
            except CircuitBreakerOpenError as e:
                errors.append((call_id, e))

        # Launch concurrent calls
        async with anyio.create_task_group() as tg:
            tg.start_soon(call_with_tracking, 1)
            await asyncio.sleep(0.01)  # Small delay so first call enters HALF_OPEN
            tg.start_soon(call_with_tracking, 2)  # Second call should be rejected

        # One call should succeed, one should be rejected at capacity
        assert len(errors) >= 1, f"Expected rejection. Results: {results}, Errors: {errors}"
        assert cb.metrics["rejected_count"] >= 1


class TestRetryConfigCoveragePush:
    """Tests targeting uncovered RetryConfig lines."""

    def test_retry_config_to_dict(self):
        """Test RetryConfig.to_dict() serialization."""
        config = RetryConfig(
            max_retries=5, initial_delay=2.0, max_delay=120.0, exponential_base=3.0, jitter=True
        )

        config_dict = config.to_dict()

        assert config_dict["max_retries"] == 5
        assert config_dict["initial_delay"] == 2.0
        assert config_dict["max_delay"] == 120.0
        assert config_dict["exponential_base"] == 3.0
        assert config_dict["jitter"] is True

    def test_retry_config_as_kwargs(self):
        """Test RetryConfig.as_kwargs() conversion."""
        config = RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            retry_on=(KeyError, ValueError),
        )

        kwargs = config.as_kwargs()

        assert kwargs["max_retries"] == 3
        assert kwargs["initial_delay"] == 1.0
        assert kwargs["retry_on"] == (KeyError, ValueError)
        # Verify all expected keys present
        assert "exponential_base" in kwargs
        assert "jitter" in kwargs

    @pytest.mark.asyncio
    async def test_retry_defaults(self):
        """Test retry_with_backoff uses defaults when not specified."""
        from lionherd_core.errors import ConnectionError as LionConnectionError

        call_count = 0

        async def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LionConnectionError("Not yet")
            return "success"

        # Call with minimal parameters (uses defaults - retries ConnectionError)
        result = await retry_with_backoff(eventually_succeeds)

        assert result == "success"
        assert call_count == 2  # Failed once, succeeded on retry
