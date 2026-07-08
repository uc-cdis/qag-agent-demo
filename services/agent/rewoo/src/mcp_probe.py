from __future__ import annotations

import argparse
import asyncio
from pprint import pprint
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def probe_mcp_server(
    mcp_url: str,
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> None:
    async with streamable_http_client(mcp_url) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:
            await session.initialize()

            tools_response = await session.list_tools()

            print("\nAvailable MCP tools:")
            for tool in tools_response.tools:
                print(f"  - {tool.name}")

            if tool_name is None:
                return

            available_names = {tool.name for tool in tools_response.tools}

            if tool_name not in available_names:
                raise ValueError(
                    f"Tool {tool_name!r} is not exposed by the MCP server."
                )

            result = await session.call_tool(
                tool_name,
                arguments=arguments or {},
            )

            print("\nRaw CallToolResult:")
            pprint(result)

            print("\nResult type:")
            print(type(result))

            print("\nisError:")
            print(result.isError)

            print("\nstructuredContent:")
            pprint(result.structuredContent)

            print("\ncontent:")
            for index, content_item in enumerate(result.content):
                print(
                    f"[{index}] "
                    f"{type(content_item).__module__}."
                    f"{type(content_item).__name__}"
                )
                pprint(content_item)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mcp-url",
        required=True,
    )
    parser.add_argument(
        "--tool-name",
    )

    args = parser.parse_args()

    asyncio.run(
        probe_mcp_server(
            mcp_url=args.mcp_url,
            tool_name=args.tool_name,
            arguments={},
        )
    )


if __name__ == "__main__":
    main()
