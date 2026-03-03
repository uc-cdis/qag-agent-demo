import argparse
import os

import uvicorn
from pydantic_ai import Agent, ConcurrencyLimit
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider


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
    parser.add_argument("-q", "--query")
    parser.add_argument("-p", "--port", type=int)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--use-dev-ui", action="store_true", default=False)
    parser.add_argument("--openai-model-name")
    parser.add_argument("--openai-model-alias")
    parser.add_argument("--openai-api-url")
    parser.add_argument("--openai-api-key-file")
    parser.add_argument("--anthropic-model-name")
    parser.add_argument("--anthropic-model-alias")
    parser.add_argument("--anthropic-api-url")
    parser.add_argument("--anthropic-api-key-file")
    args = parser.parse_args()

    # p: None, q: None - run as server w/ default port
    # p: int,  q: None - run as server w/ given port
    # p: None, q: str - run query
    # p: int,  q: str - error
    if args.port is not None and args.query is not None:
        raise ValueError(
            f"Cannot use both --query ({args.query}) and --port ({args.port}), setting port implies serving the agent UI while setting query implies running the agent over provided input"
        )

    agent = make_agent(mcp_url=args.mcp_url)

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
        # NOTE: if you want to switch to the newer OpenAI responses API (with OpenAIResponsesModel),
        # you may need to do some patching of the response stream for the pydantic AI model client implementation
        # see this prior commit as a reference: https://github.com/StevenSong/qag-agent-demo/blob/79ff198623fd2fc0025e383c323cd526a75684a0/agent/agent.py

    if args.anthropic_model_name is not None:
        alias = args.anthropic_model_alias or args.anthropic_model_name
        api_key = None
        if args.anthropic_api_key_file is not None:
            with open(args.anthropic_api_key_file, "r") as f:
                api_key = f.read().strip()
        temp = AnthropicModel(
            model_name=args.anthropic_model_name,
            provider=AnthropicProvider(
                base_url=args.anthropic_api_url,  # can be None
                api_key=api_key,  # can be None
            ),
        )
        models[alias] = temp

    if args.query is None:
        if args.use_dev_ui:
            # serve the agent with the built in dev UI
            # this app cannot be mounted under a base path as the frontend logic hard codes some paths
            # see: https://github.com/pydantic/pydantic-ai/issues/4156
            # however, maybe you could instead use a subdomain?
            rel_dir = os.path.dirname(__file__)
            app = agent.to_web(
                models=models, html_source=os.path.join(rel_dir, "index.html")
            )
        else:
            if len(models) > 1:
                raise ValueError(f"Can only use a single model with AG-UI server")
            # start a server for the agent, exposing an AG-UI interface
            # requires a separate app server (CopilotKit) that talks to the agent via AG-UI
            app = agent.to_ag_ui(model=list(models.values())[0])
        port = 8000  # default
        if args.port is not None:
            port = args.port
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        if len(models) > 1:
            raise ValueError(f"Can only use a single model with CLI query")
        # run single query
        result = agent.run_sync(args.query, model=list(models.values())[0])
        for message in result.all_messages():
            print("====================================================")
            print(message)
            print()
