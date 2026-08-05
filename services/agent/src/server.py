import argparse
import os
import sys
from pathlib import Path

import uvicorn
import yaml
from pydantic_ai import Agent, ConcurrencyLimit
from pydantic_ai.mcp import MCPToolset, StreamableHttpTransport
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from _endpoint import make_endpoint_app  # isort: skip

PROVIDER_MAP: dict[
    str,
    tuple[
        type[AnthropicModel | GoogleModel | OpenAIChatModel],
        type[AnthropicProvider | GoogleProvider | OpenAIProvider],
    ],
] = {
    "anthropic": (AnthropicModel, AnthropicProvider),
    "google": (GoogleModel, GoogleProvider),
    "openai": (OpenAIChatModel, OpenAIProvider),
    "local": (
        OpenAIChatModel,
        OpenAIProvider,
    ),  # assume local is using OpenAI compatible API
}


def make_ag_ui_app(agent: Agent, models: dict) -> Starlette:
    async def run_ag_ui(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            body = {}

        requested_model_alias = body.get("model")
        selected_model = models.get(requested_model_alias)

        if not selected_model and models:
            default_alias = next(iter(models.keys()))
            selected_model = models[default_alias]
            print(
                f"Warning: Model '{requested_model_alias}' not found. Falling back to '{default_alias}'."
            )

        return await AGUIAdapter.dispatch_request(
            request, agent=agent, model=selected_model
        )

    return Starlette(routes=[Route("/", run_ag_ui, methods=["POST"])])


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--root-path", default="/qag")
    args = parser.parse_args()

    agent = make_agent(mcp_url=args.mcp_url)

    models = dict()
    with open(args.model_config) as f:
        backends = yaml.safe_load(f)["backends"]
    for backend in backends:
        provider_type = backend["provider"]
        env_prefix = backend["env_prefix"]
        model_cls, provider_cls = PROVIDER_MAP[provider_type]
        api_key = os.environ.get(f"{env_prefix}_KEY", None)
        base_url = os.environ.get(f"{env_prefix}_URL", None)

        if provider_type == "local":
            assert (
                base_url is not None and base_url != ""
            ), f"Must set {env_prefix}_URL env var for {provider_type} backend"
        else:
            assert (
                api_key is not None and api_key != ""
            ), f"Must set {env_prefix}_KEY env var for {provider_type} backend"

        provider_inst = provider_cls(
            base_url=base_url,
            api_key=api_key,
        )
        models[backend["alias"]] = model_cls(
            model_name=backend["name"],
            provider=provider_inst,  # type: ignore
        )

    # replace /ROOT_PATH/ in static index.html
    rel_dir = Path(os.path.dirname(__file__))
    _root_path = args.root_path.rstrip("/") + "/"
    with (
        open(rel_dir / "static/index_base.html", "r") as f_in,
        open(rel_dir / "static/index.html", "w") as f_out,
    ):
        f_out.write(f_in.read().replace("/ROOT_PATH/", _root_path))

    # serve the agent with the built in dev UI
    ui_app = agent.to_web(
        models=models,
        html_source=rel_dir / "static/index.html",
    )
    ag_ui_app = make_ag_ui_app(agent, models=models)
    endpoint_app = make_endpoint_app(agent=agent, models=models)

    inner = Starlette(
        routes=[
            Mount("/endpoint", app=endpoint_app),
            Mount("/assets", app=StaticFiles(directory=rel_dir / "static/assets")),
            Mount("/agui", app=ag_ui_app),
            Mount("/", app=ui_app),
        ],
    )
    app = Starlette(routes=[Mount(args.root_path, app=inner)])
    uvicorn.run(app, host="0.0.0.0", port=args.port)
