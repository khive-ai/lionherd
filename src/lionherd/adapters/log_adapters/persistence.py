# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Log persistence adapters for various storage backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from lionherd_core import Pile

    from lionherd.services.types.log import Log


class PersistenceAdapter(ABC):
    """Abstract adapter for log persistence backends."""

    adapter_key: ClassVar[str] = "base"

    @abstractmethod
    async def write(self, log: Log) -> None:
        """Write single log."""
        pass

    @abstractmethod
    async def batch_write(self, logs: Pile[Log]) -> None:
        """Write multiple logs (optimized)."""
        pass

    @abstractmethod
    async def flush(self) -> None:
        """Force immediate flush (for ERROR/FATAL)."""
        pass

    def _handle_error(self, exc: Exception, category: str) -> None:
        """Categorized error handling."""
        # Log to stderr, don't raise (logging system shouldn't crash app)
        import sys

        print(f"[PersistenceAdapter] {category} error: {exc}", file=sys.stderr)


class FilePersistenceAdapter(PersistenceAdapter):
    """File-based log persistence (JSONL format)."""

    adapter_key: ClassVar[str] = "file"

    def __init__(
        self,
        persist_dir: str | Path = "./data/logs",
        file_prefix: str = "logs",
        extension: str = ".jsonl",
        use_timestamp: bool = True,
    ):
        self.persist_dir = Path(persist_dir)
        self.file_prefix = file_prefix
        self.extension = extension
        self.use_timestamp = use_timestamp

        # Create directory if it doesn't exist
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Current file path
        if use_timestamp:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_file = self.persist_dir / f"{file_prefix}_{timestamp}{extension}"
        else:
            self.current_file = self.persist_dir / f"{file_prefix}{extension}"

    async def write(self, log: Log) -> None:
        """Write single log as JSONL line."""
        try:
            with open(self.current_file, "a") as f:
                json.dump(log.to_dict(mode="json"), f)
                f.write("\n")
        except Exception as exc:
            self._handle_error(exc, "write")

    async def batch_write(self, logs: Pile[Log]) -> None:
        """Write multiple logs."""
        try:
            with open(self.current_file, "a") as f:
                for log in logs:
                    json.dump(log.to_dict(mode="json"), f)
                    f.write("\n")
        except Exception as exc:
            self._handle_error(exc, "batch_write")

    async def flush(self) -> None:
        """Flush is automatic with 'a' mode."""
        pass
