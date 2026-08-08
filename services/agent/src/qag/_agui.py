from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route


async def select_model(request: Request, models: dict):
    """pick the model named in the AG-UI body, falling back to the first configured"""
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

    return selected_model


def make_ag_ui_app(
    agent: Agent | Agent[None, str | DeferredToolRequests],
    models: dict,
) -> Starlette:
    async def run_ag_ui(request: Request) -> Response:
        selected_model = await select_model(request, models)

        return await AGUIAdapter.dispatch_request(
            request, agent=agent, model=selected_model
        )

    return Starlette(routes=[Route("/", run_ag_ui, methods=["POST"])])
