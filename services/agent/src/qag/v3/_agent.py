import json
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import TypedDict

from pydantic_ai import (
    Agent,
    ApprovalRequired,
    ConcurrencyLimit,
    DeferredToolRequests,
    RunContext,
)
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from .._agui import make_ag_ui_app
from ._endpoint import make_endpoint_app
from ._orchestrator import RewooOrchestrator


class ApprovalMetadata(TypedDict, total=False):
    run_id: str
    user_feedback: str


def make_agent(
    *,
    orchestrator: RewooOrchestrator,
    retries: int = 0,
    max_running: int = 4,
    max_queued: int = 8,
) -> Agent[None, str | DeferredToolRequests]:
    agent = Agent(
        instructions=dedent(
            """\
            You are a light user-facing controller for a plan then execute bioinformatics workflow over data from the Genomic Data Commons.

            To help the user, your only option is to call run_query.
            Call the tool with the exact user query, never modify the user's query.
            Never write plans, cohort definitions, or GDC queries as prose.
            Do not narrate what you're about to do; just call the tool."""
        ),
        max_concurrency=ConcurrencyLimit(
            max_running=max_running, max_queued=max_queued
        ),
        retries=retries,
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool
    async def run_query(ctx: RunContext, user_query: str) -> dict:
        if not ctx.tool_call_approved:
            pending = await orchestrator.create_plan(user_query, model=ctx.model)
            orchestrator._register_pending(ctx.tool_call_id, pending.run_id)
            print(
                f"Waiting for approval for this plan: {json.dumps(pending.plan_summary, indent=2)}",
                file=sys.stderr,
            )

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
        result = await orchestrator.approve_and_execute(run_id=run_id, model=ctx.model)
        return {
            "run_id": result.run_id,
            "status": str(result.run_status),
            "answer": result.answer,
        }

    return agent


def make_agent_app(models, mcp_url, retries, root_path):
    orchestrator = RewooOrchestrator(mcp_url=mcp_url)
    agent = make_agent(orchestrator=orchestrator, retries=retries)

    # replace /ROOT_PATH/ in static index.html
    rel_dir = Path(os.path.dirname(__file__))
    _root_path = root_path.rstrip("/") + "/"
    with (
        open(rel_dir / "../../static/index_base.html", "r") as f_in,
        open(rel_dir / "../../static/index_v3.html", "w") as f_out,
    ):
        temp = f_in.read()
        temp = temp.replace("/ROOT_PATH/", _root_path)
        temp = temp.replace("GDC QAG", "GDC QAG v3")
        f_out.write(temp)

    ui_app = agent.to_web(
        models=models,
        html_source=rel_dir / "../../static/index_v3.html",
    )
    ag_ui_app = make_ag_ui_app(agent, models=models)
    endpoint_app = make_endpoint_app(models=models, orchestrator=orchestrator)

    app = Starlette(
        routes=[
            Mount("/endpoint", app=endpoint_app),
            Mount(
                "/assets", app=StaticFiles(directory=rel_dir / "../../static/assets_v3")
            ),
            Mount("/agui", app=ag_ui_app),
            Mount("/", app=ui_app),
        ],
    )
    return app
