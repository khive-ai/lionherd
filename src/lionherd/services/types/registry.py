# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from lionherd_core import Pile

if TYPE_CHECKING:
    from .imodel import iModel

__all__ = ("ServiceRegistry",)


class ServiceRegistry:
    """Resource boundary managing iModel instances with Pile-based storage and O(1) name lookup."""

    def __init__(self):
        """Initialize empty registry."""
        from .imodel import iModel

        self._pile: Pile[iModel] = Pile(item_type=iModel)
        self._name_index: dict[str, UUID] = {}

    def register(self, model: iModel, update: bool = False) -> UUID:  # type: ignore[name-defined]
        """Register iModel by name, returns UUID. Set update=True to replace existing."""
        if model.name in self._name_index:
            if not update:
                raise ValueError(f"Service '{model.name}' already registered")
            # Update: remove old, add new
            old_uid = self._name_index[model.name]
            self._pile.remove(old_uid)

        self._pile.add(model)
        self._name_index[model.name] = model.id

        return model.id

    def unregister(self, name: str) -> iModel:  # type: ignore[name-defined]
        """Remove and return iModel by name."""
        if name not in self._name_index:
            raise KeyError(f"Service '{name}' not found")

        uid = self._name_index.pop(name)
        return self._pile.remove(uid)

    def get(self, name: str) -> iModel:  # type: ignore[name-defined]
        """Get iModel by name."""
        if name not in self._name_index:
            raise KeyError(f"Service '{name}' not found")

        uid = self._name_index[name]
        return self._pile[uid]

    def has(self, name: str) -> bool:
        """Check if service exists."""
        return name in self._name_index

    def list_names(self) -> list[str]:
        """List all registered service names."""
        return list(self._name_index.keys())

    def list_by_tag(self, tag: str) -> list[str]:
        """List services with specific tag."""
        return [model.name for model in self._pile.items.values() if tag in model.tags]

    def count(self) -> int:
        """Count registered services."""
        return len(self._pile)

    def clear(self) -> None:
        """Remove all registered services."""
        self._pile.clear()
        self._name_index.clear()

    def __len__(self) -> int:
        """Return number of registered services."""
        return len(self._pile)

    def __contains__(self, name: str) -> bool:
        """Check if service exists (supports `name in registry`)."""
        return name in self._name_index

    def __repr__(self) -> str:
        """String representation."""
        return f"ServiceRegistry(count={len(self)})"

    # =========================================================================
    # MCP Integration Methods
    # =========================================================================

    async def register_mcp_server(
        self,
        server_config: dict,
        tool_names: list[str] | None = None,
        request_options: dict[str, type] | None = None,
        update: bool = False,
    ) -> list[str]:
        """Register tools from an MCP server.

        Args:
            server_config: MCP server configuration (command, args, etc.)
                          Can be {"server": "name"} to reference loaded config
            tool_names: Optional list of specific tool names to register.
                       If None, will discover and register all available tools.
            request_options: Optional dict mapping tool names to Pydantic models
            update: If True, allow updating existing tools.

        Returns:
            List of registered tool names
        """
        import logging

        from lionherd.services.mcps import MCPConnectionPool

        from .imodel import iModel
        from .tool import Tool

        logger = logging.getLogger(__name__)
        registered_tools = []

        # Extract server name for qualified naming
        server_name = None
        if isinstance(server_config, dict) and "server" in server_config:
            server_name = server_config["server"]

        if tool_names:
            # Register specific tools
            for tool_name in tool_names:
                qualified_name = f"{server_name}_{tool_name}" if server_name else tool_name

                if self.has(qualified_name) and not update:
                    raise ValueError(
                        f"Tool '{qualified_name}' already registered. Use update=True to replace."
                    )

                tool_request_options = None
                if request_options and tool_name in request_options:
                    tool_request_options = request_options[tool_name]

                mcp_callable = self._create_mcp_callable(server_config, tool_name)

                try:
                    tool = Tool(
                        name=qualified_name,
                        func_callable=mcp_callable,
                        request_options=tool_request_options,
                    )
                    model = iModel(backend=tool)
                    self.register(model, update=update)
                    registered_tools.append(qualified_name)
                except Exception as e:
                    logger.warning(f"Failed to register tool {tool_name}: {e}")
        else:
            # Auto-discover tools from the server
            client = await MCPConnectionPool.get_client(server_config)
            tools = await client.list_tools()

            for tool in tools:
                qualified_name = f"{server_name}_{tool.name}" if server_name else tool.name

                tool_request_options = None
                if request_options and tool.name in request_options:
                    tool_request_options = request_options[tool.name]

                tool_schema = None
                try:
                    if (
                        hasattr(tool, "inputSchema")
                        and tool.inputSchema is not None
                        and isinstance(tool.inputSchema, dict)
                    ):
                        from lionherd_core import schema_handlers

                        tool_schema = schema_handlers.typescript_schema(tool.inputSchema)
                except Exception as schema_error:
                    logger.warning(f"Could not extract schema for {tool.name}: {schema_error}")
                    tool_schema = None

                try:
                    mcp_callable = self._create_mcp_callable(server_config, tool.name)

                    if self.has(qualified_name) and not update:
                        logger.warning(f"Tool '{qualified_name}' already registered. Skipping.")
                        continue

                    tool_obj = Tool(
                        name=qualified_name,
                        func_callable=mcp_callable,
                        tool_schema=tool_schema,
                        request_options=tool_request_options,
                    )
                    model = iModel(backend=tool_obj)
                    self.register(model, update=update)
                    registered_tools.append(qualified_name)
                except Exception as e:
                    logger.warning(f"Failed to register tool {tool.name}: {e}")

        return registered_tools

    @staticmethod
    def _create_mcp_callable(server_config: dict, tool_name: str):
        """Create async callable that wraps MCP tool execution."""
        from typing import Any

        from lionherd.services.mcps import MCPConnectionPool

        async def mcp_wrapper(**kwargs: Any) -> Any:
            client = await MCPConnectionPool.get_client(server_config)
            result = await client.call_tool(tool_name, kwargs)

            # Extract content from FastMCP response
            if hasattr(result, "content"):
                content = result.content
                if isinstance(content, list) and len(content) == 1:
                    item = content[0]
                    if hasattr(item, "text"):
                        return item.text
                    elif isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
                return content
            elif isinstance(result, list) and len(result) == 1:
                item = result[0]
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")

            return result

        mcp_wrapper.__name__ = tool_name
        return mcp_wrapper

    async def load_mcp_config(
        self,
        config_path: str,
        server_names: list[str] | None = None,
        update: bool = False,
    ) -> dict[str, list[str]]:
        """Load MCP configurations from a .mcp.json file.

        Args:
            config_path: Path to .mcp.json configuration file
            server_names: Optional list of server names to load.
                         If None, loads all servers.
            update: If True, allow updating existing tools.

        Returns:
            Dict mapping server names to lists of registered tool names
        """
        import logging

        from lionherd.services.mcps import MCPConnectionPool

        logger = logging.getLogger(__name__)

        MCPConnectionPool.load_config(config_path)

        if server_names is None:
            server_names = list(MCPConnectionPool._configs.keys())

        all_tools = {}
        for server_name in server_names:
            try:
                tools = await self.register_mcp_server({"server": server_name}, update=update)
                all_tools[server_name] = tools
                logger.info(f"✅ Registered {len(tools)} tools from server '{server_name}'")
            except Exception as e:
                logger.error(f"⚠️  Failed to register server '{server_name}': {e}")
                all_tools[server_name] = []

        return all_tools
