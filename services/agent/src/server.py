import argparse
import os

import uvicorn
import yaml
from pydantic_ai import Agent, ConcurrencyLimit
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from starlette.applications import Starlette
from starlette.routing import Mount

from _endpoint import make_endpoint_app  # isort: skip


PROVIDER_MAP: dict[
    str,
    tuple[
        str,  # env var prefix, suffix with _KEY or _URL
        type[AnthropicModel | GoogleModel | OpenAIChatModel],
        type[AnthropicProvider | GoogleProvider | OpenAIProvider],
    ],
] = {
    "anthropic": ("ANTHROPIC_API", AnthropicModel, AnthropicProvider),
    "google": ("GOOGLE_API", GoogleModel, GoogleProvider),
    "openai": ("OPENAI_API", OpenAIChatModel, OpenAIProvider),
    "local": (
        "LOCAL_API",
        OpenAIChatModel,
        OpenAIProvider,
    ),  # assume local is using OpenAI compatible API
}

# NOTE: if you want to switch to the newer OpenAI responses API (with OpenAIResponsesModel),
# you may need to do some patching of the response stream for the pydantic AI model client implementation
# see this prior commit as a reference: https://github.com/StevenSong/qag-agent-demo/blob/79ff198623fd2fc0025e383c323cd526a75684a0/agent/agent.py


def make_agent(
    mcp_url: str,
    retries: int = 1,
    max_running: int = 4,
    max_queued: int = 8,
) -> Agent:
    mcp_server = MCPServerStreamableHTTP(mcp_url)

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

    @agent.instructions
    async def mcp_server_instructions():
        return mcp_server.instructions

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()

    agent = make_agent(mcp_url=args.mcp_url)

    models = dict()
    with open(args.model_config) as f:
        backends = yaml.safe_load(f)["backends"]
    for backend in backends:
        provider = backend["provider"]
        env_key, model_cls, provider_cls = PROVIDER_MAP[provider]
        api_key = os.environ.get(f"{env_key}_KEY", None)
        base_url = os.environ.get(f"{env_key}_URL", None)

        if provider == "local":
            assert (
                base_url is not None and base_url != ""
            ), f"Must set {env_key}_URL env var for {provider} backend"
        else:
            assert (
                api_key is not None and api_key != ""
            ), f"Must set {env_key}_KEY env var for {provider} backend"

        provider = provider_cls(
            base_url=base_url,
            api_key=api_key,
        )
        models[backend["alias"]] = model_cls(
            model_name=backend["name"],
            provider=provider,  # type: ignore
        )

    # serve the agent with the built in dev UI
    # this app cannot be mounted under a base path as the frontend logic hard codes some paths
    # see: https://github.com/pydantic/pydantic-ai/issues/4156
    # however, maybe you could instead use a subdomain?
    rel_dir = os.path.dirname(__file__)
    ui_app = agent.to_web(
        models=models,
        html_source=os.path.join(rel_dir, "qag.html"),
    )
    endpoint_app = make_endpoint_app(agent=agent, models=models)
    app = Starlette(
        routes=[
            Mount("/endpoint", app=endpoint_app),
            Mount("/", app=ui_app),
        ],
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)
