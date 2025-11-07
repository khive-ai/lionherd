# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Service types and abstractions."""

from .backend import (
    Calling,
    NormalizedResponse,
    ServiceBackend,
    ServiceConfig,
    normalize_response,
)
from .endpoint import Endpoint, EndpointConfig
from .hook import HookBroadcaster, HookEvent, HookPhase, HookRegistry
from .imodel import iModel
from .log import (
    DataLogger,
    FilePersistenceAdapter,
    HookLogger,
    Log,
    LogBroadcaster,
    LogEvent,
    LogLevel,
    PersistenceAdapter,
)
from .tool import Tool, ToolCalling, ToolConfig

__all__ = (
    "Calling",
    "DataLogger",
    "Endpoint",
    "EndpointConfig",
    "FilePersistenceAdapter",
    "HookBroadcaster",
    "HookEvent",
    "HookLogger",
    "HookPhase",
    "HookRegistry",
    "Log",
    "LogBroadcaster",
    "LogEvent",
    "LogLevel",
    "NormalizedResponse",
    "PersistenceAdapter",
    "ServiceBackend",
    "ServiceConfig",
    "Tool",
    "ToolCalling",
    "ToolConfig",
    "iModel",
    "normalize_response",
)
