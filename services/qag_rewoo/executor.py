import asyncio
import inspect
import re
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests
from planner import PlanDag
from pydantic import ConfigDict, ValidationError, validate_call
from schema import GDCCohortSchema


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class ExecutionResult:
    evidence: dict[str, Any] = field(default_factory=dict)
    statuses: dict[str, NodeStatus] = field(default_factory=dict)
    errors: dict[str, BaseException] = field(default_factory=dict)


ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]
EVIDENCE_REF_PATTERN = re.compile(r"^#E[1-9]\d*$")


class ToolExecutionError(Exception):
    pass


class UnknownToolError(ToolExecutionError):
    pass


class UnknownEvidenceReferenceError(ToolExecutionError):
    pass


class InvalidToolArgumentsError(Exception):
    def __init__(
        self,
        tool_name: str,
        validation_error: ValidationError,
    ) -> None:
        self.tool_name = tool_name
        self.validation_error = validation_error

        super().__init__(
            f"Invalid arguments for tool {tool_name!r}: " f"{validation_error}"
        )


def resolve_evidence_refs(
    value: Any,
    evidence_store: Mapping[str, Any],
) -> Any:
    if isinstance(value, str) and EVIDENCE_REF_PATTERN.fullmatch(value):
        try:
            return evidence_store[value]
        except KeyError as exc:
            raise UnknownEvidenceReferenceError(
                f"Evidence reference {value!r} has not been produced"
            ) from exc

    if isinstance(value, Mapping):
        return {
            key: resolve_evidence_refs(item, evidence_store)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [resolve_evidence_refs(item, evidence_store) for item in value]

    if isinstance(value, tuple):
        return tuple(resolve_evidence_refs(item, evidence_store) for item in value)

    return value


class ToolExecutor:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        tool: Callable[..., Any],
        *,
        name: str | None = None,
    ) -> None:
        tool_name = name or tool.__name__

        if tool_name in self._tools:
            raise ValueError(f"Tool {tool_name!r} is already registered")

        validated_tool = validate_call(
            config=ConfigDict(
                extra="forbid",
                arbitrary_types_allowed=True,
            )
        )(tool)

        self._tools[tool_name] = validated_tool

    async def __call__(
        self,
        tool_name: str,
        tool_arguments: Mapping[str, Any],
        *,
        evidence_store: Mapping[str, Any],
    ) -> Any:
        try:
            tool = self._tools[tool_name]
        except KeyError as exc:
            raise UnknownToolError(
                f"Unknown tool {tool_name!r}. "
                f"Available tools: {sorted(self._tools)}"
            ) from exc

        if not isinstance(tool_arguments, Mapping):
            raise InvalidToolArgumentsError(
                tool_name,
                TypeError(
                    "Tool arguments must be a mapping, "
                    f"received {type(tool_arguments).__name__}"
                ),
            )

        resolved_arguments = resolve_evidence_refs(
            tool_arguments,
            evidence_store,
        )

        try:
            if inspect.iscoroutinefunction(tool):
                return await tool(**resolved_arguments)

            # Run synchronous tools outside the event loop.
            return await asyncio.to_thread(
                tool,
                **resolved_arguments,
            )

        except ValidationError as exc:
            raise InvalidToolArgumentsError(
                tool_name,
                exc,
            ) from exc


class Executor:
    def __init__(self, exec_tool: ToolExecutor) -> None:
        self.exec_tool = exec_tool

    async def execute_plan(
        self,
        plan: PlanDag,
        *,
        max_concurrency: int = 8,
    ) -> ExecutionResult:
        dag = plan
        exec_tool = self.exec_tool
        result = ExecutionResult(
            statuses={node: NodeStatus.PENDING for node in dag.steps}
        )

        remaining = dict(dag.in_degree)
        exec_queue: asyncio.Queue[str | None] = asyncio.Queue()

        for node, degree in remaining.items():
            if degree == 0:
                exec_queue.put_nowait(node)

        lock = asyncio.Lock()

        async def block_descendants(failed_node: str) -> None:
            stack = list(dag.successors[failed_node])

            async with lock:
                while stack:
                    current = stack.pop()

                    if result.statuses[current] != NodeStatus.PENDING:
                        continue

                    result.statuses[current] = NodeStatus.BLOCKED
                    stack.extend(dag.successors[current])

        async def worker() -> None:
            while True:
                node = await exec_queue.get()

                try:
                    if node is None:
                        return

                    async with lock:
                        if result.statuses[node] != NodeStatus.PENDING:
                            continue

                        result.statuses[node] = NodeStatus.RUNNING

                    step = dag.steps[node]

                    try:
                        output = await exec_tool(
                            step.tool_name,
                            step.tool_arguments,
                            evidence_store=result.evidence,
                        )
                        # print(output)
                    except Exception as error:
                        async with lock:
                            result.statuses[node] = NodeStatus.FAILED
                            result.errors[node] = error

                        await block_descendants(node)
                        continue

                    async with lock:
                        result.evidence[node] = output
                        result.statuses[node] = NodeStatus.SUCCEEDED

                        for child in dag.successors[node]:
                            if result.statuses[child] != NodeStatus.PENDING:
                                continue

                            remaining[child] -= 1

                            if remaining[child] == 0:
                                exec_queue.put_nowait(child)

                finally:
                    exec_queue.task_done()

        if len(dag.steps) == 0:
            raise ValueError(f"empty plan generated, cannot execute")
        worker_count = min(
            max_concurrency,
            len(dag.steps),
        )

        tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]

        await exec_queue.join()

        for task in tasks:
            task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        return result
