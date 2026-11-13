import os
import secrets
from pathlib import Path

from cachetools import TTLCache
from pydantic_ai import Agent, ConcurrencyLimit, RunContext
from pydantic_ai.mcp import (
    CallToolFunc,
    MCPToolset,
    StreamableHttpTransport,
    ToolResult,
)
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from .._agui import make_ag_ui_app
from .._cache import make_cache_app
from ._endpoint import make_endpoint_app

cache = TTLCache(maxsize=1000, ttl=10800)


async def process_tool_call(
    ctx: RunContext,
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict,
) -> ToolResult:
    result = await call_tool(name, tool_args)
    if name != "get_survival_data":
        return result

    # cache survival payload server side, return ID to retrieve
    assert isinstance(result, dict)
    agent_result = [x.pop("summary") for x in result["results"]]
    cache_id = secrets.token_urlsafe(8)
    cache[cache_id] = result
    return {
        "payload_id": cache_id,
        "summary_stats": agent_result,
    }


def make_agent(
    mcp_url: str,
    retries: int = 1,
    max_running: int = 4,
    max_queued: int = 8,
) -> Agent:
    mcp_server = MCPToolset(
        StreamableHttpTransport(mcp_url),
        include_instructions=True,
        process_tool_call=process_tool_call,
    )

    agent = Agent(
        toolsets=[mcp_server],
        instructions=(
            "You are a helpful bioinformatics assistant with access to specialized "
            "tools to query information about genetic mutations using the Genomic Data Commons (GDC). "
            "Use the tools to answer questions. Multiple chained tool calls may be required to answer a question. "
            "If a user query is ambiguous, stop and ask them to clarify before trying to find the answer. "
            "Pay particular attention to the instructions provided by the tool providers, as they may further clarify what ambiguities exist. "
            "Also math is hard for LLMs to do in natural language. If you need to do arithmetic, just use the tools."
        ),
        max_concurrency=ConcurrencyLimit(
            max_running=max_running, max_queued=max_queued
        ),
        retries=retries,
    )

    @agent.tool_plain
    def add(a: float, b: float) -> float:
        """
        compute a + b
        """
        return a + b

    @agent.tool_plain
    def subtract(a: float, b: float) -> float:
        """
        compute a - b
        """
        return a - b

    @agent.tool_plain
    def multiply(a: float, b: float) -> float:
        """
        compute a * b
        """
        return a * b

    @agent.tool_plain
    def divide(a: float, b: float) -> float:
        """
        compute a / b
        """
        return a / b

    return agent


def make_agent_app(models, mcp_url, retries, root_path):
    agent = make_agent(mcp_url=mcp_url, retries=retries)

    # replace /ROOT_PATH/ in static index.html
    rel_dir = Path(os.path.dirname(__file__))
    _root_path = root_path.rstrip("/") + "/"
    with (
        open(rel_dir / "../../static/index_base.html", "r") as f_in,
        open(rel_dir / "../../static/index_v2.html", "w") as f_out,
    ):
        temp = f_in.read()
        temp = temp.replace("/ROOT_PATH/", _root_path)
        temp = temp.replace("GDC QAG", "GDC QAG v2")
        f_out.write(temp)

    # serve the agent with the built in dev UI
    ui_app = agent.to_web(
        models=models,
        html_source=rel_dir / "../../static/index_v2.html",
    )
    ag_ui_app = make_ag_ui_app(agent, models=models)
    endpoint_app = make_endpoint_app(agent=agent, models=models)
    cache_app = make_cache_app(cache)

    app = Starlette(
        routes=[
            Mount("/endpoint", app=endpoint_app),
            Mount(
                "/assets", app=StaticFiles(directory=rel_dir / "../../static/assets_v2")
            ),
            Mount("/agui", app=ag_ui_app),
            Mount("/cache", app=cache_app),
            Mount("/", app=ui_app),
        ],
    )
    return app
