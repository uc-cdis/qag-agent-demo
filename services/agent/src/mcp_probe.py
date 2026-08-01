from __future__ import annotations

import argparse
import asyncio
from pprint import pprint
from typing import Any

from fastmcp import Client


async def probe_mcp(
    mcp_url: str,
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> None:
    client = Client(mcp_url)

    async with client:
        tools = await client.list_tools()

        print("\nAvailable tools:")
        for i, tool in enumerate(tools, 1):
            print(f"{i}")
            print(f"  - name: {tool.name}")
            # print(f"  - properties: {tool.inputSchema.get("properties", {})}")
            # print(f"  - required: {tool.inputSchema.get("required", {})}")
            print("\n")
            # print(f"  - repr args: {tool.__repr_args__}")

        if tool_name is None:
            return

        available_names = {tool.name for tool in tools}
        if tool_name not in available_names:
            raise ValueError(
                f"Tool {tool_name!r} is unavailable. "
                f"Available tools: {sorted(available_names)}"
            )

        result = await client.call_tool(
            tool_name,
            arguments or {},
        )

        print("\nRaw result:")
        pprint(result)

        print("\nResult type:")
        print(type(result))

        print("\nNormalized data:")
        pprint(result.data)

        print("\nContent:")
        pprint(result.content)

        print("\nStructured content:")
        pprint(getattr(result, "structured_content", None))

        print("\nError state:")
        pprint(getattr(result, "is_error", None))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mcp-url",
        required=True,
    )
    parser.add_argument(
        "--tool-name",
        default=None,
    )

    args = parser.parse_args()

    # asyncio.run(
    #     probe_mcp(
    #         mcp_url=args.mcp_url,
    #         tool_name=args.tool_name,
    #         arguments={"project": ["TCGA-BRCA"]},
    #     )
    # )

    # asyncio.run(
    #     probe_mcp(
    #         mcp_url=args.mcp_url,
    #         tool_name=args.tool_name,
    #         arguments={
    #             "cohort_filter": "Filter-(['TCGA-BRCA'])-(None)-(None)-(None)-(None)-(None)-(None)"
    #         },
    #     )
    # )

    asyncio.run(
        probe_mcp(
            mcp_url=args.mcp_url,
            tool_name=args.tool_name,
            arguments={
                "case_set_id": "Cases-Filter-(Filter-(['TCGA-BRCA'])-(None)-(None)-(None)-(None)-(None)-(None))"
            },
        )
    )


if __name__ == "__main__":
    main()
