"""
vllm serve openai/gpt-oss-120b --port 8001 --structured-outputs-config '{"backend": "xgrammar"}'

python services/qag_rewoo/app.py --openai-api-url "http://localhost:8000/v1" --openai-model-name "openai/gpt-oss-120b" --openai-model-alias "(On-Prem) GPT OSS 120B" -p 8001
"""

import argparse
import asyncio
import json
import os
import traceback
from collections import defaultdict

import uvicorn
import yaml
from _defines import *
from _tool_desc import TOOLS_DESCRIPTION
from factories import make_default_orchestrator
from openai import OpenAI
from orchestrator import Orchestrator
from pydantic_ai import Agent, AgentRunResult, ConcurrencyLimit, RunContext, ToolOutput
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from starlette.applications import Starlette
from starlette.routing import Mount

INSTRUCTIONS = """
You are the conversational interface for the Query Augmented Generation tool 
for accurate variant statistics from the GDC.

For substantive questions involving GDC data, cohorts, genomic availability,
somatic mutations, copy-number variants, microsatellite instability, gene
expression, or cancer statistics, call `run_gdc_qag` exactly once.

Pass the user's complete request to the workflow without rewriting away
important constraints.

After the workflow returns:
- Preserve its factual answer.
- Do not invent additional statistics or evidence.
- Do not independently perform the underlying analysis.
- Clearly preserve any limitations or warnings returned by the workflow.

For otherwise general conversation, respond normally.
"""

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


def make_agent(
    *, orchestrator: Orchestrator, for_endpoint: bool = False
) -> Agent[Any, Any]:
    output_type = EndpointResponse if for_endpoint else str
    agent = Agent(
        instructions=INSTRUCTIONS,
        output_type=output_type,
        max_concurrency=ConcurrencyLimit(
            max_running=2,
            max_queued=16,
        ),
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

    @agent.tool_plain(
        name="run_gdc_qag",
        description=(
            "Run the complete GDC QAG workflow for a user question. "
            "This plans the analysis, executes the required GDC tools, "
            "and produces a grounded final answer."
        ),
        sequential=True,
    )
    async def run_gdc_qag(
        user_query: str,
    ) -> str | EndpointResponse:
        try:
            result = await orchestrator.run(user_query)
            if for_endpoint:
                return EndpointResponse(
                    answer=result.answer,
                    warnings=result.warnings,
                    metadata=result.metadata,
                )
        except Exception as exc:
            traceback.print_exception(exc)
            raise

        return result.answer

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query")
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("--openai-model-name")
    parser.add_argument("--openai-model-alias")
    parser.add_argument("--openai-api-url")
    parser.add_argument("--openai-api-key-file", default=None)
    # parser.add_argument("--model-config", required=True)
    args = parser.parse_args()

    models = dict()
    if args.openai_model_name is not None:
        alias = args.openai_model_alias or args.openai_model_name
        api_key = None
        if args.openai_api_key_file is not None:
            with open(args.openai_api_key_file, "r") as f:
                api_key = f.read().strip()
        models[alias] = OpenAIChatModel(
            model_name=args.openai_model_name,
            provider=OpenAIProvider(
                base_url=args.openai_api_url,  # can be None
                api_key=api_key,  # can be None
            ),
        )

    # models = dict()
    # with open(args.model_config) as f:
    #     backends = yaml.safe_load(f)["backends"]
    # for backend in backends:
    #     provider = backend["provider"]
    #     env_key, model_cls, provider_cls = PROVIDER_MAP[provider]
    #     api_key = os.environ.get(f"{env_key}_KEY", None)
    #     base_url = os.environ.get(f"{env_key}_URL", None)

    #     if provider == "local":
    #         assert (
    #             base_url is not None and base_url != ""
    #         ), f"Must set {env_key}_URL env var for {provider} backend"
    #     else:
    #         assert (
    #             api_key is not None and api_key != ""
    #         ), f"Must set {env_key}_KEY env var for {provider} backend"

    #     provider = provider_cls(
    #         base_url=base_url,
    #         api_key=api_key,
    #     )
    #     models[backend["alias"]] = model_cls(
    #         model_name=backend["name"],
    #         provider=provider,  # type: ignore
    #     )

    model_name = "openai/gpt-oss-120b"
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    orchestrator = make_default_orchestrator(
        client=client,
        model=model_name,
        tools_description=TOOLS_DESCRIPTION,
    )

    chat_agent = make_agent(
        orchestrator=orchestrator,
        for_endpoint=False,
    )
    endpoint_agent = make_agent(
        orchestrator=orchestrator,
        for_endpoint=True,
    )

    if args.query is None:
        # serve the agent with a UI
        rel_dir = os.path.dirname(__file__)
        ui_app = chat_agent.to_web(
            # NOTE: With pydantic AI's builtin dev UI, can't provide fresh deps for
            # each convo, so we don't track the subagent runs (via OuterDeps).
            # Can revisit in the future with dedicated UI and lower level adapters.
            # For now, we only track subagent runs via the endpoint agent.
            models=models,
            html_source=os.path.join(rel_dir, "static/index.html"),
        )
        # endpoint app
        # TODO

        app = Starlette(
            routes=[
                # Mount("/endpoint", app=endpoint_app), #TODO
                Mount("/", app=ui_app),
            ],
        )
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        # run single query
        result = chat_agent.run_stream_sync(args.query)
        for message in result.all_messages():
            print("====================================================")
            print(message)
            print()
