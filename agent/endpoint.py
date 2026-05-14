"""
python agent/endpoint.py \
--mcp-url "http://localhost:8011/mcp" \
--openai-api-url "http://localhost:8080/v1" \
--openai-model-name "openai/gpt-oss-120b" \
--openai-model-alias "(On-Prem) GPT OSS 120B" \
-p 8110 \
--use-dev-ui "qag.html"

curl -X POST http://localhost:8110/invoke \
-H "Content-Type: application/json" \
-d '{"query": "what is the prevalence of braf mutations in tcga-luad, use ssms", "backend": "openai"}'
"""

import argparse
import json
import os
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from pydantic_ai import Agent, ConcurrencyLimit, RunContext
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import ToolDefinition
from pydantic_core import to_json


async def sync_cohort_copilot_tool_backend(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition] | None:
    # NOTE beware this is very very hardcoded logic that isn't even written in this codebase!
    # it's determined at runtime! hah! have fun debugging whoever is reading this >:)
    if ctx.model.system == "openai":
        # for the demo, openai is the local backend, so exclude the copilot v2 tool that uses claude backend
        exclude = "get_cohort_filter_by_cohort_description"
    else:
        # use claude backend, exclude the copilot v2 tool that uses local backend
        exclude = "generate_cohort_filter_using_cohort_description"
    return [tool_def for tool_def in tool_defs if tool_def.name != exclude]


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
        prepare_tools=sync_cohort_copilot_tool_backend,
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


# FastAPI

app = FastAPI(title="QAG Agent Endpoint API")


class InvokeRequest(BaseModel):
    query: str
    backend: str


class InvokeResponse(BaseModel):
    response: str
    backed: str
    steps: list[dict]


def extract_steps(messages) -> list[dict]:
    steps = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    steps.append(
                        {
                            "type": "tool_call",
                            "tool": part.tool_name,
                            "args": part.args,
                            "id": part.tool_call_id,
                        }
                    )
                elif isinstance(part, TextPart):
                    steps.append(
                        {
                            "type": "model_text",
                            "content": part.content,
                        }
                    )
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    steps.append(
                        {
                            "type": "tool_return",
                            "tool": part.tool_name,
                            "id": part.tool_call_id,
                            "content": part.content,
                        }
                    )
    return steps


@app.post("/invoke")
async def invoke(req: InvokeRequest):
    if not req.backend:
        raise HTTPException(
            400,
            f"Multiple backends configured {list(models)}; specify 'backend'",
        )
    model = models.get(req.backend)
    if model is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: {req.backend}. Available: {list(models.keys())}",
        )
    result = await agent.run(req.query, model=model)
    payload = {
        "response": result.output,
        "backend": model,
        "steps": extract_steps(result.new_messages()),
    }
    body = to_json(payload, fallback=str)
    return Response(content=body, media_type="application/json")


@app.get("/health")
async def health():
    return {"status": "ok", "backends": list(models.keys())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query")
    parser.add_argument("-p", "--port", type=int)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--use-dev-ui")
    parser.add_argument("--openai-model-name")
    parser.add_argument("--openai-model-alias")
    parser.add_argument("--openai-api-url")
    parser.add_argument("--openai-api-key-file")
    parser.add_argument("--anthropic-model-name")
    parser.add_argument("--anthropic-model-alias")
    parser.add_argument("--anthropic-api-url")
    parser.add_argument("--anthropic-api-key-file")
    parser.add_argument("--google-model-name")
    parser.add_argument("--google-model-alias")
    parser.add_argument("--google-api-url")
    parser.add_argument("--google-api-key-file")
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

    if args.google_model_name is not None:
        alias = args.google_model_alias or args.google_model_name
        api_key = os.environ["GEMINI_API_KEY"]
        if args.google_api_key_file is not None:
            with open(args.google_api_key_file, "r") as f:
                api_key = f.read().strip()
        temp = GoogleModel(
            model_name=args.google_model_name,
            provider=GoogleProvider(
                vertexai=False,  # can be None
                api_key=api_key,  # can be None
            ),
        )
        models[alias] = temp

    if not models:
        raise ValueError(
            "No backends configured. Provide at least --google-api-key or --anthropic-api-key or --openai-model."
        )

    print(f"Starting with backends: {list(models.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
