from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ag_ui.core import (
    AssistantMessage,
    BaseEvent,
    EventType,
    Message,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    ToolMessage,
)
from pydantic_ai.ui.ag_ui import AGUIAdapter
from pydantic_ai.ui.ag_ui._utils import dump_tool_return_content

REWOO_TOOL_CALL_PREFIX = "qag_rewoo|"
"""Marks tool call ids this backend synthesizes rather than receives from the model.

Kept stable: `RewooAGUIAdapter` strips messages carrying these ids back out of any
history a frontend echoes to us. Mirrors how pydantic-ai namespaces provider-side
tool calls with `BUILTIN_TOOL_CALL_ID_PREFIX`.
"""


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str)


@dataclass
class RewooRunEvents:
    """Emits AG-UI tool call events for the steps of a single ReWOO run.

    The DAG executes inside one `run_query` tool call, so its MCP calls are invisible
    to the AG-UI stream. This synthesizes a tool call per plan step, using the same
    `TOOL_CALL_*` events the model's own tool calls produce, so a frontend renders
    them with no protocol changes.

    Events are pushed onto a per-request queue that the v3 AG-UI route interleaves
    into the agent's event stream. `evidence_id` is unique within a run, so the
    lifecycle bookkeeping here needs no locking beyond the queue itself.
    """

    run_id: str
    queue: asyncio.Queue[BaseEvent]
    _started: set[str] = field(default_factory=set)
    _closed: set[str] = field(default_factory=set)

    def tool_call_id(self, evidence_id: str) -> str:
        return f"{REWOO_TOOL_CALL_PREFIX}{self.run_id}|{evidence_id}"

    def _put(self, event: BaseEvent) -> None:
        """Stamp and enqueue.

        `AGUIEventStream.handle_event` stamps the events it generates, but merged
        events never pass through it, so they would otherwise go out unstamped.
        Milliseconds, matching the adapter's convention.
        """
        if event.timestamp is None:
            event.timestamp = int(time.time() * 1000)
        self.queue.put_nowait(event)

    def call_started(self, evidence_id: str, tool_name: str) -> None:
        if evidence_id in self._started:
            return
        self._started.add(evidence_id)
        self._put(
            ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=self.tool_call_id(evidence_id),
                tool_call_name=tool_name,
            )
        )

    def call_args(self, evidence_id: str, tool_arguments: Any) -> None:
        """Emit the arguments once they are final.

        Deliberately separate from `call_started`: a resolver step's real arguments
        only exist after its own LLM hop, and AG-UI lets args stream in after the
        start event. Args are sent as a single delta, since concatenating two
        complete JSON blobs would not parse.
        """
        if evidence_id in self._closed:
            return
        self._put(
            ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=self.tool_call_id(evidence_id),
                delta=_dumps(tool_arguments),
            )
        )

    def call_ended(self, evidence_id: str) -> None:
        if evidence_id in self._closed:
            return
        self._closed.add(evidence_id)
        self._put(
            ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=self.tool_call_id(evidence_id),
            )
        )

    def call_result(self, evidence_id: str, output: Any) -> None:
        self._emit_result(evidence_id, dump_tool_return_content(output))

    def call_failed(self, evidence_id: str, error: BaseException) -> None:
        self._emit_result(
            evidence_id,
            _dumps({"error": f"{type(error).__name__}: {error}"}),
        )

    def _emit_result(self, evidence_id: str, content: str) -> None:
        # A step can fail before its arguments resolve, which would otherwise leave the
        # call open on the frontend forever.
        self.call_ended(evidence_id)
        tool_call_id = self.tool_call_id(evidence_id)
        self._put(
            ToolCallResultEvent(
                type=EventType.TOOL_CALL_RESULT,
                message_id=f"{tool_call_id}|result",
                role="tool",
                tool_call_id=tool_call_id,
                content=content,
            )
        )


@dataclass
class RewooEventSink:
    """Per-request handle passed to the agent as deps; bound to a run id once known."""

    queue: asyncio.Queue[BaseEvent]

    def bind(self, run_id: str) -> RewooRunEvents:
        return RewooRunEvents(run_id=run_id, queue=self.queue)


@dataclass
class RewooDeps:
    """Agent deps for the v3 controller. `events` is None outside the AG-UI route."""

    events: RewooEventSink | None = None


class RewooAGUIAdapter(AGUIAdapter):
    """Drops synthesized tool calls a frontend echoes back to us.

    Frontends fold the event stream into message state and replay it on the next
    request. Our synthetic calls are indistinguishable from real ones there, so left
    alone they either replay into the controller's history as tool calls it never
    made, or -- when the reducer records the result without a matching assistant tool
    call -- make `load_messages` raise `Tool call with ID ... not found in the
    history`. Filtering at the AG-UI message level rather than on the resulting
    `ModelMessage`s matters, because that error is raised inside `load_messages`.
    """

    @classmethod
    def load_messages(
        cls, messages: Sequence[Message], *, preserve_file_data: bool = False
    ):
        return super().load_messages(
            [msg for msg in map(_strip_synthetic, messages) if msg is not None],
            preserve_file_data=preserve_file_data,
        )


def _strip_synthetic(message: Message) -> Message | None:
    if isinstance(message, ToolMessage):
        if message.tool_call_id.startswith(REWOO_TOOL_CALL_PREFIX):
            return None
        return message

    if isinstance(message, AssistantMessage) and message.tool_calls:
        kept = [
            call
            for call in message.tool_calls
            if not call.id.startswith(REWOO_TOOL_CALL_PREFIX)
        ]
        if len(kept) == len(message.tool_calls):
            return message
        if not kept and not message.content:
            # Reducers that see no `parent_message_id` invent an assistant message per
            # synthetic call; nothing of the model's survives in it.
            return None
        return message.model_copy(update={"tool_calls": kept or None})

    return message


_STREAM_DONE = object()


async def _next_event(stream: AsyncIterator[BaseEvent]) -> Any:
    try:
        return await stream.__anext__()
    except StopAsyncIteration:
        return _STREAM_DONE


async def merge_event_stream(
    stream: AsyncIterator[BaseEvent],
    queue: asyncio.Queue[BaseEvent],
) -> AsyncIterator[BaseEvent]:
    """Interleave queued execution events into the agent's AG-UI event stream.

    Queued events are flushed before each agent event is forwarded, so anything
    enqueued while the agent was busy is ordered ahead of it -- in particular ahead of
    `run_query`'s own `TOOL_CALL_RESULT` and the terminal `RunFinishedEvent`.
    """
    stream_task = asyncio.ensure_future(_next_event(stream))
    queue_task = asyncio.ensure_future(queue.get())
    try:
        while True:
            done, _ = await asyncio.wait(
                {stream_task, queue_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if queue_task in done:
                yield queue_task.result()
                queue_task = asyncio.ensure_future(queue.get())

            if stream_task in done:
                event = stream_task.result()
                if event is _STREAM_DONE:
                    return
                while not queue.empty():
                    yield queue.get_nowait()
                yield event
                stream_task = asyncio.ensure_future(_next_event(stream))
    finally:
        # Cancelling the agent side is what we want if the client disconnects.
        for task in (stream_task, queue_task):
            task.cancel()
