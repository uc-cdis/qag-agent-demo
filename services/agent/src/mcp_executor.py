from __future__ import annotations

import argparse
import asyncio


from pprint import pprint
from typing import Any
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from dataclasses import dataclass
from collections.abc import Awaitable, Callable, Mapping


class MCPBackendError(RuntimeError):
    """Base error raised by the MCP execution backend"""


class MCPToolError(MCPBackendError):
    """Raised when an MCP tool reports an execution error."""


class MCPResultError(MCPBackendError):
    """Raised when an MCP result cannot be normalized safely."""


def resolve_tool(tools: list[str], tool_name: str, tool_args: Mapping[str, Any]):
    if tool_name is None:
        return

    available_tools = {tool.name: tool for tool in tools}
    if tool_name not in available_tools.keys():
        raise ValueError(
            f"Tool {tool_name!r} is unavailable. "
            f"Available tools: {sorted(available_tools)}"
        )

    tool = available_tools[tool_name]
    properties = tool.inputSchema.get("properties", {})
    required_args = tool.inputSchema.get("required", [])

    for arg_name, arg_value in tool_args:
        if arg_name not in properties.keys():
            raise ValueError(
                f"arg {arg_name!r} is unavailable. "
                f"allowed arguements for tool {tool_name} are: {properties.keys()}"
            )
        else:
            pass

    return


async def execute_with_mcp(
    mcp_url: str,
    tool_name: str | None = None,
    tool_args: dict[str, Any] | None = None,
) -> None:
    client = Client(mcp_url)

    async with client:
        result = await client.call_tool(
            tool_name,
            tool_args or {},
        )

        return getattr(result, "structured_content", None), getattr()


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
    ):
        super().__init__()
        self.mcp_url = mcp_url
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
        tool_args: dict[str, Any],
    ) -> Any:
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
                tool_args,
            )
        # except Exception as exc:
        #     raise MCPBackendError(f"MCP call failed for tool {tool_name!r}.") from exc
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
        if tool_result.is_error:
            try:
                message = [
                    block.text
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

        data = (
            tool_result.data
        )  # output of the tool call (for MCP tools this is case_set_id, cohort_filter_id etc)
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


async def execute_run_manual(mcp_url: str) -> None:
    async with MCPExecutionBackend(mcp_url) as backend:
        await backend.load_tools()
        # step 1
        result_cohort_filter = await backend.call_tool(
            tool_name="construct_cohort_filter",
            tool_args={"project": ["TCGA-BRCA"]},
        )
        print(result_cohort_filter)
        # step 2
        result_get_cases = await backend.call_tool(
            tool_name="get_cases_by_cohort_filter",
            tool_args={
                "cohort_filter": "Filter-(['TCGA-BRCA'])-(None)-(None)-(None)-(None)-(None)-(None)"
            },
        )
        print(result_get_cases)
        # step 3
        result_case_size = await backend.call_tool(
            tool_name="get_case_set_size",
            tool_args={
                "case_set_id": "Cases-Filter-(Filter-(['TCGA-BRCA'])-(None)-(None)-(None)-(None)-(None)-(None))"
            },
        )
        print(result_case_size)
        return


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mcp-url",
        required=True,
    )
    args = parser.parse_args()
    asyncio.run(execute_run_manual(args.mcp_url))

    return


if __name__ == "__main__":
    main()
