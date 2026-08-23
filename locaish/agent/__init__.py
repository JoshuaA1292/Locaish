"""The location scout agent: Gemini, on Google's Agent Development Kit.

The model never touches geometry. It is given a tool surface -- the scout
report, a tape measure, a shot renderer, a dolly simulator, and SQL over the
shot table through the official ClickHouse MCP server -- and it decides what
to call. Every number in its answers traces to a tool result.
"""

from __future__ import annotations

from .core import AgentService, AgentUnavailable, register_location

__all__ = ["AgentService", "AgentUnavailable", "register_location"]
