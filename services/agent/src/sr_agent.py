import os
from pathlib import Path

from cachetools import TTLCache
from pydantic_ai import Agent, ConcurrencyLimit
from pydantic_ai.mcp import MCPToolset, StreamableHttpTransport
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

cache = TTLCache(maxsize=1000, ttl=10800)


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
            "tools to do search and retrieval of cases and files from the Genomic Data Commons (GDC). "
            "Use the tools to answer questions. Multiple chained tool calls may be required to answer a question. "
            "If a user query is ambiguous, stop and ask them to clarify before trying to find the answer. "
            # "Pay particular attention to the instructions provided by the tool providers, as they may further clarify what ambiguities exist. "
        ),
        max_concurrency=ConcurrencyLimit(
            max_running=max_running, max_queued=max_queued
        ),
        retries=retries,
    )

    return agent


def make_agent_app(models, mcp_url, retries, root_path):
    agent = make_agent(mcp_url=mcp_url, retries=retries)

    # replace /ROOT_PATH/ in static index.html
    rel_dir = Path(os.path.dirname(__file__))
    _root_path = root_path.rstrip("/") + "/"
    with (
        open(rel_dir / "static/index_sr_base.html", "r") as f_in,
        open(rel_dir / "static/index_sr.html", "w") as f_out,
    ):
        temp = f_in.read()
        temp = temp.replace("/ROOT_PATH/", _root_path)
        f_out.write(temp)

    # serve the agent with the built in dev UI
    ui_app = agent.to_web(
        models=models,
        html_source=rel_dir / "static/index_sr.html",
    )

    app = Starlette(
        routes=[
            Mount("/assets", app=StaticFiles(directory=rel_dir / "static/assets_sr")),
            Mount("/", app=ui_app),
        ],
    )
    return app
