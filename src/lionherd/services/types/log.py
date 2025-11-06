# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from lionherd_core import Broadcaster, Element, Enum, Event, Pile, ln
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from ...adapters.log_adapters import FilePersistenceAdapter, PersistenceAdapter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log severity levels aligned with standard logging."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class LoggerConfig(BaseModel):
    persist_dir: str | Path = "./data/logs"
    subfolder: str | None = None
    file_prefix: str | None = None
    capacity: int | None = None
    extension: str = ".json"
    use_timestamp: bool = True
    hash_digits: int | None = Field(5, ge=0, le=10)
    auto_save_on_exit: bool = True
    clear_after_dump: bool = True

    @field_validator("capacity", "hash_digits", mode="before")
    def _validate_non_negative(cls, value):  # noqa: N805
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError("Capacity and hash_digits must be non-negative.")
        return value

    @field_validator("extension")
    def _ensure_dot_extension(cls, value):  # noqa: N805
        if not value.startswith("."):
            return "." + value
        if value not in {".csv", ".json", ".jsonl"}:
            raise ValueError("Extension must be '.csv', '.json' or '.jsonl'.")
        return value


class Log(Element):
    """
    Immutable log data snapshot for persistence.

    Inherits from Element (not Event) - this is a data container,
    not an executable action.

    Once created or restored from a dictionary, the log is marked
    as read-only.
    """

    content: dict[str, Any]
    level: LogLevel | None = None
    _immutable: bool = PrivateAttr(False)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutation if log is immutable."""
        if getattr(self, "_immutable", False):
            raise AttributeError("This Log is immutable.")
        super().__setattr__(name, value)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Log:
        """
        Create a Log from a dictionary previously produced by `to_dict`.

        The dictionary must contain keys in `serialized_keys`.
        """
        self = cls.model_validate(data)
        self._immutable = True
        return self

    @classmethod
    def create(cls, content: Element | dict, level: LogLevel | None = None) -> Log:
        """
        Create a new Log from an Element or dict, storing a dict snapshot
        of the element's data.
        """
        if isinstance(content, Element):
            content_dict = content.to_dict(mode="json")
        elif hasattr(content, "to_dict"):
            content_dict = content.to_dict()
        else:
            content_dict = ln.to_dict(content, recursive=True)

        if content_dict == {}:
            return cls(content={"error": "No content to log."}, level=level)

        return cls(content=content_dict, level=level)


class LogEvent(Event):
    """
    Executable log event that broadcasts to subscribers.

    Inherits from Event - this is an actionable trigger that
    invokes broadcasting when executed.
    """

    level: LogLevel
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    async def _invoke(self) -> None:
        """Broadcast to LogBroadcaster subscribers."""
        await LogBroadcaster.broadcast(self)

    @classmethod
    def debug(cls, message: str, **context) -> LogEvent:
        """Create DEBUG level log event."""
        return cls(level=LogLevel.DEBUG, message=message, context=context)

    @classmethod
    def info(cls, message: str, **context) -> LogEvent:
        """Create INFO level log event."""
        return cls(level=LogLevel.INFO, message=message, context=context)

    @classmethod
    def warning(cls, message: str, **context) -> LogEvent:
        """Create WARNING level log event."""
        return cls(level=LogLevel.WARNING, message=message, context=context)

    @classmethod
    def error(cls, message: str, **context) -> LogEvent:
        """Create ERROR level log event."""
        return cls(level=LogLevel.ERROR, message=message, context=context)

    @classmethod
    def fatal(cls, message: str, **context) -> LogEvent:
        """Create FATAL level log event."""
        return cls(level=LogLevel.FATAL, message=message, context=context)


class LogBroadcaster(Broadcaster):
    """
    Singleton broadcaster for LogEvent distribution.

    Inherits singleton pattern and pub/sub infrastructure from
    lionherd_core.Broadcaster. Each subclass maintains separate
    instance and subscriber list for isolation.

    Usage:
        # Subscribe to log events
        async def my_handler(log_event: LogEvent):
            print(f"Received: {log_event.message}")

        LogBroadcaster.subscribe(my_handler)

        # Broadcast (usually called by LogEvent._invoke())
        await LogBroadcaster.broadcast(log_event)

        # Unsubscribe
        LogBroadcaster.unsubscribe(my_handler)
    """

    _event_type: ClassVar[type[LogEvent]] = LogEvent


class DataLogger:
    """Log collector with capacity/severity-based flushing.

    Buffers LogEvents converted to Log snapshots and flushes to
    PersistenceAdapter based on capacity or severity thresholds.

    Architecture:
        LogEvent (executable) → DataLogger (buffer) → PersistenceAdapter (storage)

    Flush triggers:
        - Immediate: ERROR or FATAL severity
        - Capacity: buffer size >= capacity threshold
        - Exit: atexit handler (if enabled)
    """

    def __init__(
        self,
        adapter: PersistenceAdapter,
        capacity: int = 100,
        auto_save_on_exit: bool = True,
    ):
        """Initialize DataLogger with persistence adapter.

        Args:
            adapter: PersistenceAdapter for backend storage
            capacity: Max buffer size before auto-flush (default: 100)
            auto_save_on_exit: Register atexit handler for cleanup (default: True)
        """
        self.adapter = adapter
        self.capacity = capacity
        self.buffer: Pile[Log] = Pile(item_type=Log, strict_type=True)

        if auto_save_on_exit:
            atexit.register(self._save_sync)

    async def collect(self, log_event: LogEvent) -> None:
        """Collect LogEvent, convert to Log, buffer and flush if needed.

        Args:
            log_event: LogEvent to collect

        Conversion:
            LogEvent (executable) → Log (immutable snapshot)
            - level, message, context, extra → content dict
            - created_at → timestamp field

        Flush triggers:
            - Immediate: ERROR or FATAL severity
            - Capacity: buffer size >= capacity threshold
        """
        # Convert LogEvent to Log snapshot
        log = Log(
            content={
                "level": log_event.level.value,
                "message": log_event.message,
                "context": log_event.context,
                "extra": log_event.extra,
                "timestamp": log_event.created_at.isoformat()
                if hasattr(log_event.created_at, "isoformat")
                else str(log_event.created_at),
            },
            level=log_event.level,
        )

        self.buffer.include(log)

        # Immediate flush for ERROR/FATAL
        if log_event.level in (LogLevel.ERROR, LogLevel.FATAL) or len(self.buffer) >= self.capacity:
            await self._flush()

    async def _flush(self) -> None:
        """Flush buffer to adapter via batch_write.

        Clears buffer after successful write.
        """
        if not self.buffer or len(self.buffer) == 0:
            return

        await self.adapter.batch_write(self.buffer)
        self.buffer.clear()

    def _save_sync(self) -> None:
        """Synchronous flush for atexit handler.

        Attempts async flush if event loop available, logs error otherwise.
        Logging system shouldn't crash application shutdown.
        """
        import asyncio
        import sys

        if not self.buffer or len(self.buffer) == 0:
            return

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create task if loop running
                _task = loop.create_task(self._flush())  # noqa: RUF006
            else:
                # Run synchronously if loop stopped
                loop.run_until_complete(self._flush())
        except Exception as exc:
            # Fallback: log error but don't crash
            print(
                f"[DataLogger] Failed to flush {len(self.buffer)} logs on exit: {exc}",
                file=sys.stderr,
            )


class HookLogger:
    def __init__(
        self,
        adapter: PersistenceAdapter | None = None,
        persist_dir: str | Path = "./data/logs/hooks",
        file_prefix: str = "hook_events",
        capacity: int = 50,
    ):
        """Initialize HookLogger with configurable adapter.

        Args:
            adapter: PersistenceAdapter instance (creates FilePersistenceAdapter if None)
            persist_dir: Directory for hook log files (default: "./data/logs/hooks")
            file_prefix: Prefix for log filenames (default: "hook_events")
            capacity: DataLogger capacity (default: 50)

        Note:
            If adapter is None, creates FilePersistenceAdapter with JSONL format and timestamp
            using persist_dir and file_prefix parameters.
        """
        if adapter is None:
            adapter = FilePersistenceAdapter(
                persist_dir=persist_dir,
                file_prefix=file_prefix,
                extension=".jsonl",
                use_timestamp=True,
            )
        self.data_logger = DataLogger(adapter=adapter, capacity=capacity)

    async def log_hook(self, hook_event, level: LogLevel | None = None) -> None:
        """Convert HookEvent to LogEvent and collect via DataLogger.

        Args:
            hook_event: HookEvent from HookBroadcaster
            level: Log level (if None, derives from hook execution status)

        Log level derivation (when level=None):
            - FAILED → ERROR
            - CANCELLED | ABORTED → WARNING
            - COMPLETED | SKIPPED → INFO
            - PENDING | PROCESSING → DEBUG
            - Unknown → DEBUG

        Conversion:
            HookEvent → LogEvent
            - message: "Hook {phase}"
            - context: hook_phase, event_id, status, duration
            - extra: full hook_event.to_dict() for complete record
        """
        if level is None:
            # Derive level from execution status
            from lionherd_core import EventStatus

            status = hook_event.execution.status
            match status:
                case EventStatus.FAILED:
                    level = LogLevel.ERROR
                case EventStatus.CANCELLED | EventStatus.ABORTED:
                    level = LogLevel.WARNING
                case EventStatus.COMPLETED | EventStatus.SKIPPED:
                    level = LogLevel.INFO
                case EventStatus.PENDING | EventStatus.PROCESSING:
                    level = LogLevel.DEBUG
                case _:
                    level = LogLevel.DEBUG

        log_event = LogEvent(
            level=level,
            message=f"Hook {hook_event.hook_phase.value}",
            context={
                "hook_phase": hook_event.hook_phase.value,
                "event_id": str(hook_event.ln_id),
                "status": hook_event.execution.status.value,
                "duration": hook_event.execution.duration,
            },
            extra=hook_event.to_dict(),
        )
        await self.data_logger.collect(log_event)

    @classmethod
    def install(cls, **kwargs) -> HookLogger:
        """Create HookLogger and subscribe to HookBroadcaster.

        Args:
            **kwargs: Passed to HookLogger.__init__
                (persist_dir, file_prefix)

        Returns:
            HookLogger instance subscribed to HookBroadcaster

        Usage:
            # Default configuration
            hook_logger = HookLogger.install()

            # Custom configuration
            hook_logger = HookLogger.install(
                persist_dir="./custom/hooks",
                file_prefix="my_hooks"
            )

        The returned HookLogger will automatically receive and log
        all HookEvent instances broadcast by HookBroadcaster.
        """
        from .hook import HookBroadcaster

        logger = cls(**kwargs)
        HookBroadcaster.subscribe(logger.log_hook)
        return logger
