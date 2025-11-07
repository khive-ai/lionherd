# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Self, get_origin, get_type_hints

from lionherd_core import concurrency, schema_handlers
from pydantic import BaseModel, Field, field_validator, model_validator

from .backend import Calling, ServiceBackend, ServiceConfig

__all__ = ("Tool", "ToolCalling", "ToolConfig")


def _extract_json_schema_from_callable(
    func: Callable[..., Any],
    request_options: type | None = None,
) -> dict[str, Any]:
    """Generate JSON Schema from function signature or Pydantic model.

    Args:
        func: Callable to extract schema from
        request_options: Optional Pydantic model for parameters

    Returns:
        JSON Schema dict with type, properties, required fields
    """
    if request_options is not None:
        # Use Pydantic model schema
        if hasattr(request_options, "model_json_schema"):
            return request_options.model_json_schema()
        raise ValueError(f"request_options must be Pydantic model, got {type(request_options)}")

    # Build schema from function signature
    sig = inspect.signature(func)

    # Use get_type_hints to resolve string annotations
    try:
        type_hints = get_type_hints(func)
    except Exception:
        # Fallback to raw annotations if get_type_hints fails
        type_hints = {}

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        # Skip variadic args
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # Determine type - prefer type_hints over param.annotation
        if name in type_hints:
            param_type = _python_type_to_json_type(type_hints[name])
        elif param.annotation is not inspect.Parameter.empty:
            param_type = _python_type_to_json_type(param.annotation)
        else:
            param_type = "string"  # Default

        properties[name] = {"type": param_type}

        # Add to required if no default
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _python_type_to_json_type(annotation: Any) -> str:
    """Convert Python type annotation to JSON Schema type string.

    Args:
        annotation: Python type annotation

    Returns:
        JSON Schema type string (string, number, boolean, array, object, null)
    """
    # Handle None
    if annotation is type(None):
        return "null"

    # Get origin for generic types
    origin = get_origin(annotation)

    # Handle Optional/Union
    if origin is type(None):
        return "null"

    # Handle List/list
    if origin in (list, tuple):
        return "array"

    # Handle Dict/dict
    if origin is dict:
        return "object"

    # Handle simple types
    type_map = {
        str: "string",
        int: "number",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }

    if annotation in type_map:
        return type_map[annotation]

    # Default
    return "string"


class ToolConfig(ServiceConfig):
    """Configuration for Tool backend.

    Extends ServiceConfig with tool-specific defaults.
    """
    provider: str = "tool"


class Tool(ServiceBackend):
    config: ToolConfig
    func_callable: Callable[..., Any] = Field(
        ...,
        frozen=True,
        exclude=True,
    )
    tool_schema: dict[str, Any] | None = None

    @field_validator("func_callable", mode="before")
    def _validate_func_callable(cls, value: Any) -> Callable[..., Any]:  # noqa: N805
        if not callable(value):
            raise ValueError(f"func_callable must be callable, got {type(value)}")
        if not hasattr(value, "__name__"):
            raise ValueError("func_callable must have __name__ attribute")
        return value

    @model_validator(mode="before")
    @classmethod
    def _set_defaults_from_function(cls, data: Any) -> Any:
        """Set config defaults from function if not provided."""
        if isinstance(data, dict) and "func_callable" in data:
            func = data["func_callable"]

            # Ensure func_callable is valid before using it
            # Field validator will run after this and raise if invalid
            if not (callable(func) and hasattr(func, "__name__")):
                return data

            # Initialize config if not present
            if "config" not in data:
                data["config"] = ToolConfig(
                    provider="tool",
                    name=func.__name__
                )
            elif isinstance(data["config"], dict):
                # Convert dict to ToolConfig, set name from function if not present
                config_dict = data["config"].copy()
                if "name" not in config_dict:
                    config_dict["name"] = func.__name__
                if "provider" not in config_dict:
                    config_dict["provider"] = "tool"
                data["config"] = ToolConfig(**config_dict)
            # If config is already ToolConfig, leave it as is

        return data

    @model_validator(mode="after")
    def _generate_schema(self) -> Self:
        """Generate internal JSON Schema dict from tool_schema source.

        Converts tool_schema (None/Pydantic/dict) into tool_schema dict for
        consistent internal access. Auto-generates from signature if None.
        """

        if self.request_options is None and self.tool_schema is None:
            json_schema = _extract_json_schema_from_callable(
                self.func_callable, self.request_options
            )
            self.tool_schema = json_schema
            return self
        if self.request_options is not None and self.tool_schema is None:
            self.tool_schema = self.request_options.model_json_schema()
            return self
        if isinstance(self.tool_schema, dict):
            return self
        raise ValueError("must provide tool_schema as dict or request_options as Pydantic model")

    @property
    def function_name(self) -> str:
        """Get function name."""
        return self.func_callable.__name__

    @property
    def rendered(self) -> str:
        """Render tool schema as TypeScript for LLM consumption.

        Format:
            # Tool description (if present)
            TypeScript parameter definitions

        Returns:
            TypeScript-formatted schema string
        """
        if isinstance(self.tool_schema, type) and issubclass(self.tool_schema, BaseModel):
            desc = self.tool_schema.get("description", "")
            params_ts = schema_handlers.typescript_schema(self.tool_schema)
            return f"# {desc}\n{params_ts}" if desc else params_ts

        params = self.tool_schema.get("parameters", {})
        desc = self.tool_schema.get("description", "")

        if params and params.get("properties"):
            params_ts = schema_handlers.typescript_schema(params)
            return f"# {desc}\n{params_ts}" if desc else params_ts

        return f"# {desc}" if desc else ""

    @property
    def event_type(self) -> type[ToolCalling]:
        """Get Event/Calling type for this backend."""
        return ToolCalling

    @property
    def required_fields(self) -> frozenset[str]:
        """Get required parameter fields from function signature or schema.

        Used by Pattern B validation when request_options is None.
        """
        # Prefer schema as SSOT (respects Pydantic Field annotations)
        if self.tool_schema and "required" in self.tool_schema:
            return frozenset(self.tool_schema["required"])

        # Fallback: inspect signature for auto-generated schemas
        try:
            sig = inspect.signature(self.func_callable)
            return frozenset(
                {
                    name
                    for name, param in sig.parameters.items()
                    if param.default == inspect.Parameter.empty
                    and param.kind
                    not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                }
            )
        except Exception:
            return frozenset()

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs):
        """Not implemented - Tools are created from callables, not dicts."""
        raise NotImplementedError("Tool.from_dict is not supported - create from callable")

    async def call(self, arguments: dict[str, Any]) -> Any:
        """Execute tool callable with sync/async detection.

        Args:
            arguments: Dict of validated parameters to pass to callable

        Returns:
            Result from callable execution
        """
        if concurrency.is_coro_func(self.func_callable):
            return await self.func_callable(**arguments)
        else:
            return await concurrency.run_sync(lambda: self.func_callable(**arguments))


class ToolCalling(Calling):
    """Tool execution - delegates to Tool.call() which handles sync/async.

    Attributes:
        backend: Tool instance (contains func_callable)
        arguments: Dict of validated parameters to pass to tool (stored in metadata)
    """

    backend: Tool

    async def _invoke(self) -> Any:
        """Execute tool callable with validated parameters."""
        arguments = self.metadata.get("arguments", {})
        return await self.backend.call(arguments)

    async def _stream(self):
        """Tools don't support streaming."""
        raise NotImplementedError("Tool backends do not support streaming")
