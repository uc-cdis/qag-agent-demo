from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Any

from cachetools import TTLCache
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent


class MCPBackendError(RuntimeError):
    """Base error raised by the MCP execution backend"""


class MCPToolError(MCPBackendError):
    """Raised when an MCP tool reports an execution error."""


class MCPResultError(MCPBackendError):
    """Raised when an MCP result cannot be normalized safely."""


@dataclass(frozen=True)
class MCPToolDefinition:
    name: str
    description: str | None
    input_schema: dict[str, Any]


class MCPExecutionBackend:
    """
    Minimal backend to support execution of tool calls with QAG MCP
    for ReWOO workflow
    """

    def __init__(
        self,
        mcp_url: str,
        cache: TTLCache,
    ):
        super().__init__()
        self.mcp_url = mcp_url
        self.cache = cache
        self._client = Client(mcp_url)
        self._connected = False
        self._tool_definitions: dict[str, MCPToolDefinition] = {}

    async def __aenter__(self) -> MCPExecutionBackend:
        await self._client.__aenter__()
        self._connected = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        try:
            await self._client.__aexit__(
                exc_type,
                exc,
                traceback,
            )
        finally:
            self._connected = False
            self._tool_definitions.clear()

    async def load_tools(self) -> dict[str, MCPToolDefinition]:
        """gather MCP tools"""
        if not self._connected:
            raise MCPBackendError(
                "MCPExecutionBackend must be used inside "
                "'async with MCPExecutionBackend(...)'."
            )
        tools = await self._client.list_tools()
        defs: dict[str, MCPToolDefinition] = {}

        for tool in tools:
            defs[tool.name] = MCPToolDefinition(
                name=tool.name,
                description=getattr(tool, "description", None),
                input_schema=getattr(tool, "inputSchema", None) or {},
            )
        self._tool_definitions = defs
        return dict(defs)

    async def call_tool(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ) -> Any:
        """execute tool call with MCP"""
        if not self._connected:
            raise MCPBackendError(
                "MCPExecutionBackend must be used inside "
                "'async with MCPExecutionBackend(...)'."
            )

        if self._tool_definitions and tool_name not in self._tool_definitions:
            raise MCPBackendError(
                f"MCP tool {tool_name!r} is not exposed by the server."
            )

        try:
            result = await self._client.call_tool(
                tool_name,
                tool_arguments,
            )
            if tool_name == "get_survival_data":
                assert len(result.content) == 1
                content = result.content[0]
                assert isinstance(content, TextContent)
                payload = json.loads(content.text)

                # cache survival payload server side, return ID to retrieve
                cache_id = secrets.token_urlsafe(8)
                agent_payload = {
                    "payload_id": cache_id,
                    "summary_stats": [
                        x.pop("summary")  # extract summary for agent
                        for x in payload["results"]
                    ],
                }
                self.cache[cache_id] = payload
                content.text = json.dumps(agent_payload)  # replace payload for agent
        except Exception as exc:
            raise MCPBackendError(
                f"MCP call failed for tool {tool_name!r}: {type(exc).__name__}: {exc}"
            ) from exc

        return await self._normalize_result(
            tool_name=tool_name,
            tool_result=result,
        )

    async def _normalize_result(
        self, *, tool_name: str, tool_result: CallToolResult
    ) -> Any:
        """extract tool call result and handle neatly-ish"""
        if tool_result.is_error:
            try:
                message = [
                    block.text  # type: ignore
                    for block in tool_result.content
                    if getattr(block, "type", None) == "text"
                ]
                raise MCPToolError(f"MCP tool {tool_name!r} failed: {message}")
            except Exception as e:
                pass

        structured = tool_result.structured_content
        if structured is not None:
            if set(structured.keys()) == {"result"}:
                return structured["result"]

            return structured

        data = tool_result.data
        # output of the tool call (for MCP tools this is case_set_id, cohort_filter_id etc)
        if data is not None:
            if hasattr(data, "model_dump"):
                dumped = data.model_dump()
                if set(dumped.keys()) == {"result"}:
                    return dumped["result"]
                return dumped

        content = [block for block in tool_result.content]
        if content is not None:
            return content

        raise MCPResultError(f"MCP tool {tool_name!r} returned no usable result.")
