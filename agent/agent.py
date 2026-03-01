import argparse

import uvicorn
from pydantic_ai import Agent, ConcurrencyLimit
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def make_agent(
    mcp_url: str,
    llm_url: str,
    retries: int = 1,
    max_running: int = 4,
    max_queued: int = 8,
) -> Agent:
    mcp_server = MCPServerStreamableHTTP(mcp_url)
    model = OpenAIChatModel(
        model_name="model",
        provider=OpenAIProvider(base_url=llm_url, api_key="NONE"),
    )

    # NOTE: if you want to switch to the newer responses API, you may need to do some patching of the response stream for the pydantic AI model client implementation
    # see this prior commit as a reference: https://github.com/StevenSong/qag-agent-demo/blob/79ff198623fd2fc0025e383c323cd526a75684a0/agent/agent.py

    agent = Agent(
        model,
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
    parser.add_argument("--llm-url", required=True)
    args = parser.parse_args()

    # p: None, q: None - run as server w/ default port
    # p: int,  q: None - run as server w/ given port
    # p: None, q: str - run query
    # p: int,  q: str - error
    if args.port is not None and args.query is not None:
        raise ValueError(
            f"Cannot use both --query ({args.query}) and --port ({args.port}), setting port implies serving the agent UI while setting query implies running the agent over provided input"
        )

    agent = make_agent(mcp_url=args.mcp_url, llm_url=args.llm_url)

    if args.query is None:
        # serve the agent with a UI - this is just a dev UI! cannot be mounted under a root_path
        # see: https://github.com/pydantic/pydantic-ai/issues/4156
        # rel_dir = os.path.dirname(__file__)
        # app = agent.to_web()

        # start a server for the agent, exposing an AG-UI interface
        # requires a separate app server (CopilotKit) that talks to the agent via AG-UI
        app = agent.to_ag_ui()
        port = 8000  # default
        if args.port is not None:
            port = args.port
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # run single query
        result = agent.run_sync(args.query)
        for message in result.all_messages():
            print("====================================================")
            print(message)
            print()
