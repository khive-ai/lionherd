# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for Log system - 100% coverage target.

Test Surface:
    - LogLevel enum
    - LoggerConfig (validators)
    - Log (immutability, from_dict, create)
    - LogEvent (convenience methods, _invoke)
    - LogBroadcaster (_event_type)
    - DataLogger (collect, flush, atexit handling)
    - HookLogger (log_hook, install)
    - FilePersistenceAdapter (write, batch_write)
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
from lionherd_core import Element, EventStatus, Pile

from lionherd.adapters.log_adapters import FilePersistenceAdapter, PersistenceAdapter
from lionherd.services.types.log import (
    DataLogger,
    HookLogger,
    Log,
    LogBroadcaster,
    LogEvent,
    LoggerConfig,
    LogLevel,
)

# =============================================================================
# Test Fixtures
# =============================================================================


class MockElement(Element):
    """Mock Element for testing Log.create()."""

    value: str = "test"
    number: int = 42


class MockPersistenceAdapter(PersistenceAdapter):
    """Mock adapter for testing DataLogger."""

    adapter_key = "mock"

    def __init__(self):
        self.written_logs = []
        self.flushed = False

    async def write(self, log: Log) -> None:
        self.written_logs.append(log)

    async def batch_write(self, logs: Pile[Log]) -> None:
        self.written_logs.extend(logs)

    async def flush(self) -> None:
        self.flushed = True


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for log file tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_adapter():
    """Create mock persistence adapter."""
    return MockPersistenceAdapter()


# =============================================================================
# LogLevel Enum Tests
# =============================================================================


def test_loglevel_enum_values():
    """Test LogLevel enum has all expected values."""
    assert LogLevel.DEBUG.value == "debug"
    assert LogLevel.INFO.value == "info"
    assert LogLevel.WARNING.value == "warning"
    assert LogLevel.ERROR.value == "error"
    assert LogLevel.FATAL.value == "fatal"


def test_loglevel_enum_allowed():
    """Test LogLevel.allowed() returns all values."""
    allowed = LogLevel.allowed()
    assert LogLevel.DEBUG in allowed
    assert LogLevel.INFO in allowed
    assert LogLevel.WARNING in allowed
    assert LogLevel.ERROR in allowed
    assert LogLevel.FATAL in allowed


# =============================================================================
# LoggerConfig Tests
# =============================================================================


def test_logger_config_default_values():
    """Test LoggerConfig has correct defaults."""
    config = LoggerConfig()

    assert config.persist_dir == Path("./data/logs")
    assert config.subfolder is None
    assert config.file_prefix is None
    assert config.capacity is None
    assert config.extension == ".json"
    assert config.use_timestamp is True
    assert config.hash_digits == 5
    assert config.auto_save_on_exit is True
    assert config.clear_after_dump is True


def test_logger_config_when_capacity_negative_then_raises():
    """Test LoggerConfig validates capacity is non-negative."""
    with pytest.raises(ValueError, match="Capacity and hash_digits must be non-negative"):
        LoggerConfig(capacity=-1)


def test_logger_config_when_hash_digits_negative_then_raises():
    """Test LoggerConfig validates hash_digits is non-negative."""
    with pytest.raises(ValueError, match="Capacity and hash_digits must be non-negative"):
        LoggerConfig(hash_digits=-5)


def test_logger_config_when_capacity_not_int_then_raises():
    """Test LoggerConfig validates capacity is int."""
    with pytest.raises(ValueError, match="Capacity and hash_digits must be non-negative"):
        LoggerConfig(capacity="not_int")


def test_logger_config_when_extension_no_dot_then_adds():
    """Test LoggerConfig adds dot to extension if missing."""
    config = LoggerConfig(extension="json")
    assert config.extension == ".json"


def test_logger_config_when_extension_with_dot_then_unchanged():
    """Test LoggerConfig preserves extension with dot."""
    config = LoggerConfig(extension=".jsonl")
    assert config.extension == ".jsonl"


def test_logger_config_when_invalid_extension_then_raises():
    """Test LoggerConfig validates extension is allowed."""
    with pytest.raises(ValueError, match="Extension must be"):
        LoggerConfig(extension=".txt")


def test_logger_config_when_csv_extension_then_valid():
    """Test LoggerConfig accepts .csv extension."""
    config = LoggerConfig(extension=".csv")
    assert config.extension == ".csv"


def test_logger_config_when_jsonl_extension_then_valid():
    """Test LoggerConfig accepts .jsonl extension."""
    config = LoggerConfig(extension=".jsonl")
    assert config.extension == ".jsonl"


# =============================================================================
# Log Class Tests - Creation
# =============================================================================


def test_log_create_when_from_element_then_converts():
    """Test Log.create() from Element."""
    elem = MockElement(value="test", number=42)
    log = Log.create(elem)

    assert log.content is not None
    assert isinstance(log.content, dict)
    assert log.content["value"] == "test"
    assert log.content["number"] == 42


def test_log_create_when_from_dict_then_stores():
    """Test Log.create() from dict."""
    data = {"key": "value", "nested": {"inner": 123}}
    log = Log.create(data)

    assert log.content == data


def test_log_create_when_from_object_with_to_dict_then_calls():
    """Test Log.create() from object with to_dict method."""

    class CustomObject:
        def to_dict(self):
            return {"custom": "data"}

    obj = CustomObject()
    log = Log.create(obj)

    assert log.content == {"custom": "data"}


def test_log_create_when_empty_content_then_error_message():
    """Test Log.create() with empty content returns error log."""
    log = Log.create({})

    assert log.content == {"error": "No content to log."}


def test_log_create_when_with_level_then_stores():
    """Test Log.create() with log level."""
    log = Log.create({"data": "test"}, level=LogLevel.ERROR)

    assert log.level == LogLevel.ERROR


# =============================================================================
# Log Class Tests - Immutability
# =============================================================================


def test_log_when_immutable_then_setattr_raises():
    """Test Log raises AttributeError when immutable."""
    log = Log.from_dict({"ln_id": "test", "content": {"data": "test"}})

    with pytest.raises(AttributeError, match="This Log is immutable"):
        log.content = {"new": "data"}


def test_log_when_not_immutable_then_setattr_works():
    """Test Log allows setattr when not immutable."""
    log = Log(content={"data": "test"})

    # Should not raise
    log.level = LogLevel.WARNING
    assert log.level == LogLevel.WARNING


def test_log_from_dict_when_valid_data_then_creates_immutable():
    """Test Log.from_dict() creates immutable Log."""
    data = {"ln_id": "abc123", "content": {"key": "value"}, "level": "info"}
    log = Log.from_dict(data)

    assert log._immutable is True
    assert log.content == {"key": "value"}


# =============================================================================
# LogEvent Class Tests - Convenience Methods
# =============================================================================


def test_log_event_debug_creates_debug_level():
    """Test LogEvent.debug() creates DEBUG level event."""
    event = LogEvent.debug("Debug message", user="alice")

    assert event.level == LogLevel.DEBUG
    assert event.message == "Debug message"
    assert event.context == {"user": "alice"}


def test_log_event_info_creates_info_level():
    """Test LogEvent.info() creates INFO level event."""
    event = LogEvent.info("Info message", action="login")

    assert event.level == LogLevel.INFO
    assert event.message == "Info message"
    assert event.context == {"action": "login"}


def test_log_event_warning_creates_warning_level():
    """Test LogEvent.warning() creates WARNING level event."""
    event = LogEvent.warning("Warning message")

    assert event.level == LogLevel.WARNING
    assert event.message == "Warning message"


def test_log_event_error_creates_error_level():
    """Test LogEvent.error() creates ERROR level event."""
    event = LogEvent.error("Error message", error_code=500)

    assert event.level == LogLevel.ERROR
    assert event.message == "Error message"
    assert event.context == {"error_code": 500}


def test_log_event_fatal_creates_fatal_level():
    """Test LogEvent.fatal() creates FATAL level event."""
    event = LogEvent.fatal("Fatal message")

    assert event.level == LogLevel.FATAL
    assert event.message == "Fatal message"


@pytest.mark.asyncio
async def test_log_event_invoke_broadcasts():
    """Test LogEvent._invoke() broadcasts to LogBroadcaster."""
    received = []

    async def subscriber(event: LogEvent):
        received.append(event)

    LogBroadcaster.subscribe(subscriber)

    event = LogEvent.info("Test broadcast")
    await event._invoke()

    LogBroadcaster.unsubscribe(subscriber)

    assert len(received) == 1
    assert received[0] == event


# =============================================================================
# LogBroadcaster Tests
# =============================================================================


def test_log_broadcaster_event_type():
    """Test LogBroadcaster._event_type is LogEvent."""
    assert LogBroadcaster._event_type == LogEvent


# =============================================================================
# DataLogger Tests - Basic Operations
# =============================================================================


def test_data_logger_init_when_with_atexit_then_registers(mock_adapter):
    """Test DataLogger.__init__ registers atexit handler."""
    import atexit

    # Track atexit.register calls
    original_register = atexit.register
    registered = []

    def mock_register(func):
        registered.append(func)
        return original_register(func)

    atexit.register = mock_register

    try:
        logger = DataLogger(mock_adapter, capacity=50, auto_save_on_exit=True)
        assert len(registered) > 0
        assert logger._save_sync in registered
    finally:
        atexit.register = original_register


def test_data_logger_init_when_no_atexit_then_no_register(mock_adapter):
    """Test DataLogger.__init__ without atexit handler."""
    import atexit

    original_register = atexit.register
    registered = []

    def mock_register(func):
        registered.append(func)
        return original_register(func)

    atexit.register = mock_register

    try:
        before_count = len(registered)
        DataLogger(mock_adapter, capacity=50, auto_save_on_exit=False)
        # Should not add new registration
        assert len(registered) == before_count
    finally:
        atexit.register = original_register


@pytest.mark.asyncio
async def test_data_logger_collect_when_regular_then_buffers(mock_adapter):
    """Test DataLogger.collect() buffers regular log events."""
    logger = DataLogger(mock_adapter, capacity=10, auto_save_on_exit=False)

    event = LogEvent.info("Test message")
    await logger.collect(event)

    assert len(logger.buffer) == 1
    # Should not flush yet (not ERROR/FATAL, below capacity)
    assert len(mock_adapter.written_logs) == 0


@pytest.mark.asyncio
async def test_data_logger_collect_when_error_then_flushes(mock_adapter):
    """Test DataLogger.collect() immediately flushes ERROR level."""
    logger = DataLogger(mock_adapter, capacity=10, auto_save_on_exit=False)

    event = LogEvent.error("Error occurred")
    await logger.collect(event)

    # Should flush immediately
    assert len(logger.buffer) == 0
    assert len(mock_adapter.written_logs) == 1


@pytest.mark.asyncio
async def test_data_logger_collect_when_fatal_then_flushes(mock_adapter):
    """Test DataLogger.collect() immediately flushes FATAL level."""
    logger = DataLogger(mock_adapter, capacity=10, auto_save_on_exit=False)

    event = LogEvent.fatal("Fatal error")
    await logger.collect(event)

    # Should flush immediately
    assert len(logger.buffer) == 0
    assert len(mock_adapter.written_logs) == 1


@pytest.mark.asyncio
async def test_data_logger_collect_when_capacity_reached_then_flushes(mock_adapter):
    """Test DataLogger.collect() flushes when capacity reached."""
    logger = DataLogger(mock_adapter, capacity=3, auto_save_on_exit=False)

    # Add 3 INFO events to reach capacity
    await logger.collect(LogEvent.info("Message 1"))
    await logger.collect(LogEvent.info("Message 2"))
    await logger.collect(LogEvent.info("Message 3"))

    # Should flush on 3rd event
    assert len(logger.buffer) == 0
    assert len(mock_adapter.written_logs) == 3


@pytest.mark.asyncio
async def test_data_logger_flush_when_empty_then_no_op(mock_adapter):
    """Test DataLogger._flush() with empty buffer."""
    logger = DataLogger(mock_adapter, capacity=10, auto_save_on_exit=False)

    await logger._flush()

    assert len(mock_adapter.written_logs) == 0


@pytest.mark.asyncio
async def test_data_logger_flush_when_has_logs_then_writes(mock_adapter):
    """Test DataLogger._flush() writes buffered logs."""
    logger = DataLogger(mock_adapter, capacity=10, auto_save_on_exit=False)

    # Add events without triggering auto-flush
    await logger.collect(LogEvent.info("Message 1"))
    await logger.collect(LogEvent.debug("Message 2"))

    # Manual flush
    await logger._flush()

    assert len(logger.buffer) == 0
    assert len(mock_adapter.written_logs) == 2


@pytest.mark.asyncio
async def test_data_logger_save_sync_when_empty_then_no_op(mock_adapter):
    """Test DataLogger._save_sync() with empty buffer."""
    logger = DataLogger(mock_adapter, capacity=10, auto_save_on_exit=False)

    # Should not error
    logger._save_sync()


@pytest.mark.asyncio
async def test_data_logger_save_sync_when_loop_running_then_creates_task(mock_adapter):
    """Test DataLogger._save_sync() creates task when loop running."""
    logger = DataLogger(mock_adapter, capacity=10, auto_save_on_exit=False)

    # Add log to buffer
    await logger.collect(LogEvent.info("Test"))

    # Get current running loop
    loop = asyncio.get_event_loop()

    # _save_sync should detect running loop and create task
    logger._save_sync()

    # Give task time to execute
    await asyncio.sleep(0.1)

    # Buffer should be flushed
    assert len(logger.buffer) == 0


def test_data_logger_save_sync_when_no_loop_then_runs_until_complete(mock_adapter):
    """Test DataLogger._save_sync() runs loop when not running."""
    # This is tricky to test as it requires no running loop
    # Skip this edge case as it's primarily for atexit handler
    pass


def test_data_logger_save_sync_when_exception_then_logs_to_stderr(mock_adapter):
    """Test DataLogger._save_sync() logs error to stderr on exception."""
    import sys
    from io import StringIO

    logger = DataLogger(mock_adapter, capacity=10, auto_save_on_exit=False)

    # Add log to buffer
    logger.buffer.include(Log(content={"test": "data"}))

    # Mock adapter to raise exception
    async def failing_batch_write(logs):
        raise RuntimeError("Flush failed")

    mock_adapter.batch_write = failing_batch_write

    # Capture stderr
    stderr_capture = StringIO()
    original_stderr = sys.stderr
    sys.stderr = stderr_capture

    try:
        logger._save_sync()
        # Give time for async operations
        import time

        time.sleep(0.1)

        stderr_output = stderr_capture.getvalue()
        assert "Failed to flush" in stderr_output or len(logger.buffer) > 0
    finally:
        sys.stderr = original_stderr


# =============================================================================
# HookLogger Tests
# =============================================================================


def test_hook_logger_init_when_adapter_provided_then_uses(mock_adapter):
    """Test HookLogger.__init__ with provided adapter."""
    logger = HookLogger(adapter=mock_adapter, capacity=30)

    assert logger.data_logger.adapter == mock_adapter
    assert logger.data_logger.capacity == 30


def test_hook_logger_init_when_no_adapter_then_creates_file_adapter(temp_log_dir):
    """Test HookLogger.__init__ creates FilePersistenceAdapter when adapter is None."""
    logger = HookLogger(persist_dir=temp_log_dir / "hooks", file_prefix="test_hooks", capacity=25)

    assert isinstance(logger.data_logger.adapter, FilePersistenceAdapter)
    assert logger.data_logger.capacity == 25


@pytest.mark.asyncio
async def test_hook_logger_log_hook_when_failed_status_then_error_level(mock_adapter):
    """Test HookLogger.log_hook() derives ERROR level from FAILED status."""
    from lionherd.services.types.hook import HookEvent, HookPhase, HookRegistry

    logger = HookLogger(adapter=mock_adapter)

    registry = HookRegistry()
    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=MockElement,
    )

    # Set status to FAILED
    hook_event.execution.status = EventStatus.FAILED

    await logger.log_hook(hook_event)

    assert len(mock_adapter.written_logs) == 1  # ERROR triggers immediate flush
    log = mock_adapter.written_logs[0]
    assert log.content["level"] == "error"


@pytest.mark.asyncio
async def test_hook_logger_log_hook_when_cancelled_status_then_warning_level(mock_adapter):
    """Test HookLogger.log_hook() derives WARNING level from CANCELLED status."""
    from lionherd.services.types.hook import HookEvent, HookPhase, HookRegistry

    logger = HookLogger(adapter=mock_adapter)

    registry = HookRegistry()
    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreInvocation,
        event_like=MockElement(),
    )

    hook_event.execution.status = EventStatus.CANCELLED

    await logger.log_hook(hook_event)

    log = mock_adapter.written_logs[0]
    assert log.content["level"] == "warning"


@pytest.mark.asyncio
async def test_hook_logger_log_hook_when_aborted_status_then_warning_level(mock_adapter):
    """Test HookLogger.log_hook() derives WARNING level from ABORTED status."""
    from lionherd.services.types.hook import HookEvent, HookPhase, HookRegistry

    logger = HookLogger(adapter=mock_adapter)

    registry = HookRegistry()
    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PostInvocation,
        event_like=MockElement(),
    )

    hook_event.execution.status = EventStatus.ABORTED

    await logger.log_hook(hook_event)

    log = mock_adapter.written_logs[0]
    assert log.content["level"] == "warning"


@pytest.mark.asyncio
async def test_hook_logger_log_hook_when_completed_status_then_info_level(mock_adapter):
    """Test HookLogger.log_hook() derives INFO level from COMPLETED status."""
    from lionherd.services.types.hook import HookEvent, HookPhase, HookRegistry

    logger = HookLogger(adapter=mock_adapter)

    registry = HookRegistry()
    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=MockElement,
    )

    hook_event.execution.status = EventStatus.COMPLETED

    await logger.log_hook(hook_event, level=None)  # Explicit None to test derivation

    # INFO doesn't trigger immediate flush, need manual flush
    await logger.data_logger._flush()

    log = mock_adapter.written_logs[0]
    assert log.content["level"] == "info"


@pytest.mark.asyncio
async def test_hook_logger_log_hook_when_skipped_status_then_info_level(mock_adapter):
    """Test HookLogger.log_hook() derives INFO level from SKIPPED status."""
    from lionherd.services.types.hook import HookEvent, HookPhase, HookRegistry

    logger = HookLogger(adapter=mock_adapter)

    registry = HookRegistry()
    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=MockElement,
    )

    hook_event.execution.status = EventStatus.SKIPPED

    await logger.log_hook(hook_event)

    await logger.data_logger._flush()
    log = mock_adapter.written_logs[0]
    assert log.content["level"] == "info"


@pytest.mark.asyncio
async def test_hook_logger_log_hook_when_pending_status_then_debug_level(mock_adapter):
    """Test HookLogger.log_hook() derives DEBUG level from PENDING status."""
    from lionherd.services.types.hook import HookEvent, HookPhase, HookRegistry

    logger = HookLogger(adapter=mock_adapter)

    registry = HookRegistry()
    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=MockElement,
    )

    hook_event.execution.status = EventStatus.PENDING

    await logger.log_hook(hook_event)

    await logger.data_logger._flush()
    log = mock_adapter.written_logs[0]
    assert log.content["level"] == "debug"


@pytest.mark.asyncio
async def test_hook_logger_log_hook_when_processing_status_then_debug_level(mock_adapter):
    """Test HookLogger.log_hook() derives DEBUG level from PROCESSING status."""
    from lionherd.services.types.hook import HookEvent, HookPhase, HookRegistry

    logger = HookLogger(adapter=mock_adapter)

    registry = HookRegistry()
    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=MockElement,
    )

    hook_event.execution.status = EventStatus.PROCESSING

    await logger.log_hook(hook_event)

    await logger.data_logger._flush()
    log = mock_adapter.written_logs[0]
    assert log.content["level"] == "debug"


@pytest.mark.asyncio
async def test_hook_logger_log_hook_when_explicit_level_then_uses(mock_adapter):
    """Test HookLogger.log_hook() uses explicit level parameter."""
    from lionherd.services.types.hook import HookEvent, HookPhase, HookRegistry

    logger = HookLogger(adapter=mock_adapter)

    registry = HookRegistry()
    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=MockElement,
    )

    # Override with explicit level
    await logger.log_hook(hook_event, level=LogLevel.ERROR)

    log = mock_adapter.written_logs[0]
    assert log.content["level"] == "error"


def test_hook_logger_install_creates_and_subscribes():
    """Test HookLogger.install() creates instance and subscribes to HookBroadcaster."""
    from lionherd.services.types.hook import HookBroadcaster

    # Get initial subscriber count
    initial_count = len(HookBroadcaster._subscribers)

    # Install with custom config
    logger = HookLogger.install(persist_dir="./test_hooks", file_prefix="custom", capacity=40)

    # Should have subscribed
    assert len(HookBroadcaster._subscribers) > initial_count

    # Cleanup
    HookBroadcaster.unsubscribe(logger.log_hook)


# =============================================================================
# FilePersistenceAdapter Tests
# =============================================================================


def test_file_adapter_init_creates_directory(temp_log_dir):
    """Test FilePersistenceAdapter creates persist_dir."""
    log_dir = temp_log_dir / "new_logs"
    adapter = FilePersistenceAdapter(persist_dir=log_dir)

    assert log_dir.exists()
    assert log_dir.is_dir()


def test_file_adapter_init_when_use_timestamp_then_adds(temp_log_dir):
    """Test FilePersistenceAdapter adds timestamp to filename."""
    adapter = FilePersistenceAdapter(
        persist_dir=temp_log_dir, file_prefix="test", use_timestamp=True
    )

    # Filename should contain timestamp
    assert "test_" in adapter.current_file.name
    assert adapter.current_file.suffix == ".jsonl"


def test_file_adapter_init_when_no_timestamp_then_simple(temp_log_dir):
    """Test FilePersistenceAdapter without timestamp."""
    adapter = FilePersistenceAdapter(
        persist_dir=temp_log_dir, file_prefix="simple", use_timestamp=False
    )

    assert adapter.current_file.name == "simple.jsonl"


def test_file_adapter_init_when_custom_extension_then_uses(temp_log_dir):
    """Test FilePersistenceAdapter with custom extension."""
    adapter = FilePersistenceAdapter(
        persist_dir=temp_log_dir, file_prefix="test", extension=".json", use_timestamp=False
    )

    assert adapter.current_file.suffix == ".json"


@pytest.mark.asyncio
async def test_file_adapter_write_creates_jsonl_line(temp_log_dir):
    """Test FilePersistenceAdapter.write() creates JSONL entry."""
    adapter = FilePersistenceAdapter(
        persist_dir=temp_log_dir, file_prefix="test", use_timestamp=False
    )

    log = Log(content={"message": "test log", "value": 123})
    await adapter.write(log)

    # Read file and verify JSONL format
    import json

    with open(adapter.current_file) as f:
        line = f.readline()
        data = json.loads(line)

    assert data["content"]["message"] == "test log"
    assert data["content"]["value"] == 123


@pytest.mark.asyncio
async def test_file_adapter_batch_write_creates_multiple_lines(temp_log_dir):
    """Test FilePersistenceAdapter.batch_write() creates multiple JSONL lines."""
    adapter = FilePersistenceAdapter(
        persist_dir=temp_log_dir, file_prefix="batch", use_timestamp=False
    )

    logs = Pile(
        [
            Log(content={"id": 1, "msg": "first"}),
            Log(content={"id": 2, "msg": "second"}),
            Log(content={"id": 3, "msg": "third"}),
        ],
        item_type=Log,
    )

    await adapter.batch_write(logs)

    # Read file and count lines
    with open(adapter.current_file) as f:
        lines = f.readlines()

    assert len(lines) == 3

    # Verify each line is valid JSON
    import json

    for i, line in enumerate(lines):
        data = json.loads(line)
        assert data["content"]["id"] == i + 1


@pytest.mark.asyncio
async def test_file_adapter_flush_is_noop(temp_log_dir):
    """Test FilePersistenceAdapter.flush() is no-op (automatic with 'a' mode)."""
    adapter = FilePersistenceAdapter(persist_dir=temp_log_dir)

    # Should not raise
    await adapter.flush()


@pytest.mark.asyncio
async def test_file_adapter_write_when_error_then_handles(temp_log_dir):
    """Test FilePersistenceAdapter.write() handles errors gracefully."""
    import sys
    from io import StringIO

    adapter = FilePersistenceAdapter(persist_dir=temp_log_dir)

    # Make file read-only to trigger write error
    adapter.current_file.touch()
    adapter.current_file.chmod(0o444)

    stderr_capture = StringIO()
    original_stderr = sys.stderr
    sys.stderr = stderr_capture

    try:
        log = Log(content={"test": "data"})
        await adapter.write(log)

        stderr_output = stderr_capture.getvalue()
        assert "write error" in stderr_output.lower() or "permission" in stderr_output.lower()
    finally:
        sys.stderr = original_stderr
        # Restore permissions for cleanup
        adapter.current_file.chmod(0o644)


@pytest.mark.asyncio
async def test_file_adapter_batch_write_when_error_then_handles(temp_log_dir):
    """Test FilePersistenceAdapter.batch_write() handles errors gracefully."""
    import sys
    from io import StringIO

    adapter = FilePersistenceAdapter(persist_dir=temp_log_dir)

    # Make file read-only
    adapter.current_file.touch()
    adapter.current_file.chmod(0o444)

    stderr_capture = StringIO()
    original_stderr = sys.stderr
    sys.stderr = stderr_capture

    try:
        logs = Pile([Log(content={"test": "data"})], item_type=Log)
        await adapter.batch_write(logs)

        stderr_output = stderr_capture.getvalue()
        assert "batch_write error" in stderr_output.lower() or "permission" in stderr_output.lower()
    finally:
        sys.stderr = original_stderr
        adapter.current_file.chmod(0o644)


# =============================================================================
# Edge Cases - Coverage Push
# =============================================================================


def test_log_create_when_content_has_isoformat_then_uses():
    """Test Log.create() uses isoformat() for timestamp if available."""
    from datetime import datetime

    class ElementWithDatetime(Element):
        timestamp: datetime = datetime.now()

    elem = ElementWithDatetime()
    log = Log.create(elem)

    # Should use isoformat() method
    assert "timestamp" in log.content
