# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""MCP (Model Context Protocol) service integration.

This module provides MCP connection pooling and tool loading for integrating
MCP tools into ServiceRegistry. MCP tools are loaded as regular Tool instances
wrapped in iModel - no special MCP types needed.

Key components:
- MCPConnectionPool: Client pooling and lifecycle management
- load_mcp_tools(): Load MCP tools into ServiceRegistry
- load_mcp_config(): Load from .mcp.json file

Example:
    >>> from lionherd.sessions import Session
    >>> from lionherd.services.mcps import load_mcp_tools, load_mcp_config
    >>>
    >>> session = Session()
    >>>
    >>> # Load from config file
    >>> tools = await load_mcp_config(session.registry, ".mcp.json")
    >>> print(f"Registered {sum(len(t) for t in tools.values())} tools")
    >>>
    >>> # Or load specific server
    >>> tools = await load_mcp_tools(session.registry, {"server": "search"})
    >>>
    >>> # Use like any other tool
    >>> tool = session.registry.get(tools[0])
    >>> calling = await tool.create_calling(query="AI")
    >>> await calling.invoke()
"""

from .loader import create_mcp_callable, load_mcp_config, load_mcp_tools
from .wrapper import MCPConnectionPool

__all__ = (
    "MCPConnectionPool",
    "create_mcp_callable",
    "load_mcp_config",
    "load_mcp_tools",
)
