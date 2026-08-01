import argparse
import os
import uvicorn

from pathlib import Path
from typing import Any, TypedDict, cast

from pydantic_ai import (
    Agent,
    ApprovalRequired,
    ConcurrencyLimit,
    DeferredToolRequests,
    RunContext,
)
from openai import OpenAI
from orchestrator import make_default_rewoo_orchestrator
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

class ApprovalMetadata(TypedDict, total=False):
    run_id: str
    user_feedback: str



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


# NOTE: if you want to switch to the newer OpenAI responses API (with OpenAIResponsesModel),
# you may need to do some patching of the response stream for the pydantic AI model client implementation
# see this prior commit as a reference: https://github.com/StevenSong/qag-agent-demo/blob/79ff198623fd2fc0025e383c323cd526a75684a0/agent/agent.py


def make_agent(
    *,
    model: str | Any,
    orchestrator: Any,
    retries: int = 1,
    max_running: int = 4,
    max_queued: int = 8,
) -> Agent:
    agent = Agent(
        model=model,
        output_type=[str, DeferredToolRequests],
        instructions=(
            "You are a light user-facing controller for a ReWOO bioinformatics "
            "workflow over Genomic Data Commons data.\n\n"
            "Never write plans, cohort definitions, or GDC queries as prose."
            "To help the user, your only option is to call start_rewoo(user_query=...)." 
            "That tool will draft the plan and return it for approval — the UI will show the plan and approve/deny buttons." 
            "Do not narrate what you're about to do; just call the tool."
        ),
        max_concurrency=ConcurrencyLimit(
            max_running=max_running, max_queued=max_queued
        ),
        retries=retries,
    )

    @agent.tool
    async def start_rewoo(
        ctx: RunContext,
        user_query: str,
    ) -> dict[str, Any]:
        if not ctx.tool_call_approved:
            pending = await orchestrator.create_plan(user_query)
            orchestrator._register_pending(ctx.tool_call_id, pending.run_id)
            
            raise ApprovalRequired(
                metadata={
                    "approval_kind": "rewoo_plan",
                    "run_id": pending.run_id,
                    "status": str(pending.run_status),
                    "query": user_query,
                    "plan": pending.plan_summary,
                    "message": (
                        "Approve this ReWOO plan to execute it, or deny and "
                        "provide feedback for a revised plan."
                    ),
                }
            )

        run_id = orchestrator._pop_pending(ctx.tool_call_id)
        if not run_id:
            raise RuntimeError("Approved ReWOO tool call is missing run_id metadata")
        result = await orchestrator.approve_and_execute(run_id=run_id)
        return {
            "run_id": result.run_id,
            "status": str(result.run_status),
            "answer": result.answer,
            "execution_summary": getattr(result, "execution_summary", None),
            "final_evidence": getattr(result, "final_evidence", None),
        }

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()

    provider = OpenAIProvider(
        base_url=args.vllm_url,
        api_key="EMPTY",
    )
    agent_model = OpenAIChatModel(
        model_name=args.model,
        provider=provider,
    )
    client = OpenAI(base_url=args.vllm_url, api_key="EMPTY")
    orchestrator = make_default_rewoo_orchestrator(
        mcp_url=args.mcp_url, model=args.model, client=client
    )
    agent = make_agent(model=agent_model, orchestrator=orchestrator)
    models = dict()
    models["local"] = agent_model
    rel_dir = Path(os.path.dirname(__file__))
    ui_app = agent.to_web(
        models=models,
    )
    endpoint_app = make_endpoint_app(orchestrator=orchestrator)
    app = Starlette(
        routes=[
            Mount("/rewoo", app=endpoint_app),
            Mount("/", app=ui_app),
        ],
    )

    uvicorn.run(app, host="0.0.0.0", port=args.port)
