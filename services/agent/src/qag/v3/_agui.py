import asyncio

from ag_ui.core import BaseEvent
from pydantic_ai import Agent, DeferredToolRequests
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from .._agui import select_model
from ._events import RewooAGUIAdapter, RewooDeps, RewooEventSink, merge_event_stream


def make_rewoo_ag_ui_app(
    agent: Agent[RewooDeps, str | DeferredToolRequests],
    models: dict,
) -> Starlette:
    """AG-UI app for v3.

    Unlike v2 this does not use `AGUIAdapter.dispatch_request`, because the ReWOO
    executor needs to push tool call events into the response while `run_query` is
    still running. `dispatch_request` owns the whole stream; here we build the same
    pieces by hand so the executor's queue can be merged in. Once pydantic-ai ships
    `RunContext.emit_event` (pydantic/pydantic-ai#6258) the queue plumbing can go away
    and this can collapse back to `dispatch_request`.
    """

    async def run_ag_ui(request: Request) -> Response:
        selected_model = await select_model(request, models)

        queue: asyncio.Queue[BaseEvent] = asyncio.Queue()
        adapter = await RewooAGUIAdapter.from_request(request, agent=agent)  # type: ignore

        return adapter.streaming_response(
            merge_event_stream(
                adapter.run_stream(
                    model=selected_model,
                    deps=RewooDeps(events=RewooEventSink(queue=queue)),
                ),
                queue,
            )
        )

    return Starlette(routes=[Route("/", run_ag_ui, methods=["POST"])])
