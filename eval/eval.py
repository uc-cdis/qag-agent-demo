import argparse
import asyncio
import json
import os

import pandas as pd
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import OpenAI
from pqdm.threads import pqdm

SYSTEM_PROMPT = (
    "You are a helpful bioinformatics assistant with access to specialized "
    "tools to query information about genetic mutations using the Genomic Data Commons (GDC). "
    "Use the tools to answer questions. Multiple chained tool calls may be required to answer a question. "
    "If a user query is ambiguous, stop and ask them to clarify before trying to find the answer. "
    "Also math is hard for LLMs to do in natural language. If you need to do arithmetic, just use the tools."
)
MAX_DEPTH = 100


async def load_mcp_tools(session: ClientSession) -> list[dict]:
    """Fetch tools from MCP server and convert to OpenAI Responses API format."""
    tools_result = await session.list_tools()

    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        }
        for tool in tools_result.tools
    ]


async def call_mcp_tool(session: ClientSession, name: str, arguments: dict) -> str:
    """Execute a tool on the MCP server and return its output as a string."""
    result = await session.call_tool(name, arguments=arguments)

    content_parts = []
    for block in result.content:
        if hasattr(block, "text"):
            content_parts.append(block.text)
        else:
            content_parts.append(str(block))

    return "\n".join(content_parts)


async def agent_loop(
    llm_url: str,
    mcp_url: str,
    user_message: str,
):
    client = OpenAI(base_url=llm_url, api_key="EMPTY")
    async with sse_client(mcp_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            # print(f"Loaded {len(tools)} tools: {[t['name'] for t in tools]}\n")

            # The Responses API is stateful via `previous_response_id`,
            # so we only need to track the input for the current turn.
            input_items = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
            previous_response_id = None

            # ── Agent loop ──────────────────────────────────────────────────
            for i in range(MAX_DEPTH):
                response = client.responses.create(
                    model="openai/gpt-oss-120b",
                    input=input_items,
                    tools=tools,
                    temperature=0,
                    **(
                        {"previous_response_id": previous_response_id}
                        if previous_response_id
                        else {}
                    ),
                )  # type: ignore

                previous_response_id = response.id

                # Collect all function_call items from this response
                tool_calls = [
                    item for item in response.output if item.type == "function_call"
                ]

                # ── No tool calls → final answer ─────────────────────────────
                if not tool_calls:
                    # The text response is in the output_text convenience field
                    # print("Final answer:", response.output_text)
                    return response.output_text

                # ── Execute tool calls and build function_call_output items ───
                # print(f"Model requested {len(tool_calls)} tool call(s)...")

                # Next input = the model's output items + our tool results
                input_items = list(response.output)

                for tool_call in tool_calls:
                    tool_name = tool_call.name
                    tool_args = json.loads(tool_call.arguments)

                    # print(f"  → Calling '{tool_name}' with args: {tool_args}")
                    result = await call_mcp_tool(session, tool_name, tool_args)
                    # print(f"  ← Result: {result[:200]}{'...' if len(result) > 200 else ''}\n")

                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": result,
                        }
                    )

                # Loop — next call picks up from previous_response_id
            return "MAX_DEPTH REACHED"


MAX_RETRIES = 10


def worker_fn(llm_url: str, mcp_url: str, user_message: str, temp_file: str):
    if os.path.exists(temp_file):
        return
    answer = ""
    for i in range(MAX_RETRIES):
        try:
            answer = asyncio.run(agent_loop(llm_url, mcp_url, user_message))
        except:
            continue
        if answer.strip():
            break
    if not answer:
        answer = "MAX_RETRIES REACHED"
    with open(temp_file, "w") as f:
        f.write(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-url", required=True)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--query-csv", required=True)
    parser.add_argument("--query-col", default="questions")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    queries = pd.read_csv(args.query_csv)[args.query_col].to_list()

    temp_dir = os.path.join(os.path.dirname(args.query_csv), "temp-answers")
    n_queries = len(queries)
    n_digits = len(str(n_queries))
    os.makedirs(temp_dir, exist_ok=True)

    worker_kwargs = [
        {
            "llm_url": args.llm_url,
            "mcp_url": args.mcp_url,
            "user_message": q,
            "temp_file": os.path.join(temp_dir, f"{i:0{n_digits}}.txt"),
        }
        for i, q in enumerate(queries)
    ]
    pqdm(
        worker_kwargs,
        worker_fn,  # write only
        n_jobs=args.num_workers,
        argument_type="kwargs",
        exception_behaviour="immediate",
    )

    answers = []
    for kwargs in worker_kwargs:
        with open(kwargs["temp_file"], "r") as f:
            answers.append(f.read())
    out = pd.Series(answers, name="answers")
    out.to_csv(args.query_csv.replace(".csv", "-answers.csv"), index=False)
