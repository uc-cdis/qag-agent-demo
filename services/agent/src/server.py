import argparse
import os
import sys
from pathlib import Path
from typing import Any, TypedDict, cast

import uvicorn
import yaml
from openai import OpenAI
from orchestrator import make_default_rewoo_orchestrator
from pydantic_ai import (
    Agent,
    ApprovalRequired,
    ConcurrencyLimit,
    DeferredToolRequests,
    RunContext,
)
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


def make_agent(
    *,
    models: dict[str, Any],
    orchestrator: Any,
    retries: int = 0,
    max_running: int = 4,
    max_queued: int = 8,
) -> Agent:
    agent = Agent(
        model=models,
        output_type=[str, DeferredToolRequests],
        instructions=(
            "You are a light user-facing controller for a plan then execute bioinformatics workflow over data from the Genomic Data Commons.\n\n"
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
        """single rewoo entry tool with approval gate baked in"""
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

        run_id = orchestrator._get_pending(ctx.tool_call_id)
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
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()

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

    # replace /ROOT_PATH/ in static index.html
    rel_dir = Path(os.path.dirname(__file__))
    _root_path = args.root_path.rstrip("/") + "/"
    with (
        open(rel_dir / "static/index_base.html", "r") as f_in,
        open(rel_dir / "static/index.html", "w") as f_out,
    ):
        f_out.write(f_in.read().replace("/ROOT_PATH/", _root_path))

    # Pick the first model as the default for the orchestrator
    default_alias = backends[0]["alias"]
    agent_model = models[default_alias]

    client = OpenAI(
        base_url=os.environ.get(f"{backends[0]['env_prefix']}_URL"),
        api_key=os.environ.get(f"{backends[0]['env_prefix']}_KEY", "EMPTY"),
    )
    orchestrator = make_default_rewoo_orchestrator(
        mcp_url=args.mcp_url, model=backends[0]["name"], client=client
    )
    agent = make_agent(model=agent_model, orchestrator=orchestrator)

    ui_app = agent.to_web(
        models=models,
        html_source=rel_dir / "static/index.html",
    )
    ag_ui_app = make_ag_ui_app(agent, models=models)
    endpoint_app = make_endpoint_app(orchestrator=orchestrator)

    app = Starlette(
        routes=[
            Mount("/qag/endpoint", app=endpoint_app),
            Mount("/qag/assets", app=StaticFiles(directory=rel_dir / "static/assets")),
            Mount("/qag/agui", app=ag_ui_app),
            Mount("/qag", app=ui_app),
        ],
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)
