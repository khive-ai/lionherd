# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
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

        Delegates to load_mcp_tools from the loader module.

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
        from lionherd.services.mcps.loader import load_mcp_tools

        return await load_mcp_tools(
            registry=self,
            server_config=server_config,
            tool_names=tool_names,
            request_options=request_options,
            update=update,
        )

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
