import os
import sys
from pathlib import Path

from pydantic_ai import Agent, ConcurrencyLimit
from pydantic_ai.mcp import MCPToolset, StreamableHttpTransport
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from .._agui import make_ag_ui_app
from ._endpoint import make_endpoint_app


def make_agent(
    mcp_url: str,
    retries: int = 1,
    max_running: int = 4,
    max_queued: int = 8,
) -> Agent:
    mcp_server = MCPToolset(
        StreamableHttpTransport(mcp_url),
        include_instructions=True,
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
    print("v2 rel_dir:", rel_dir, file=sys.stderr)
    _root_path = root_path.rstrip("/") + "/"
    with (
        open(rel_dir / "../../static/index_base.html", "r") as f_in,
        open(rel_dir / "../../static/index_v2.html", "w") as f_out,
    ):
        f_out.write(f_in.read().replace("/ROOT_PATH/", _root_path))

    # serve the agent with the built in dev UI
    ui_app = agent.to_web(
        models=models,
        html_source=rel_dir / "../../static/index_v2.html",
    )
    ag_ui_app = make_ag_ui_app(agent, models=models)
    endpoint_app = make_endpoint_app(agent=agent, models=models)

    app = Starlette(
        routes=[
            Mount("/endpoint", app=endpoint_app),
            Mount(
                "/assets", app=StaticFiles(directory=rel_dir / "../../static/assets_v2")
            ),
            Mount("/agui", app=ag_ui_app),
            Mount("/", app=ui_app),
        ],
    )
    return app
