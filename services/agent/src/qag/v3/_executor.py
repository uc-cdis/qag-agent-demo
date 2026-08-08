from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_ai.models import Model

from ._defines import SIMPLE_TOOLS
from ._mcp_executor import MCPExecutionBackend
from ._planner import ExecutionKind, PlanDag
from ._resolver import ConstructCohortFilterArgs, Resolver, StringList


class SimpleToolArgs(BaseModel):
    a: float
    b: float


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
    resolved_arguments: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # added this to track argument resolution from initial LLM scope to actual fields and values


# ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]
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


def construct_cohort_filter_args(
    project: StringList | None = None,
    primary_site: StringList | None = None,
    disease_type: StringList | None = None,
    project_id: StringList | None = None,
    sex: StringList | None = None,
    smoking_status: StringList | None = None,
    min_age: int | None = None,
    max_age: int | None = None,
) -> ConstructCohortFilterArgs:
    args = ConstructCohortFilterArgs(
        project=project,
        primary_site=primary_site,
        disease_type=disease_type,
        project_id=project_id,
        sex=sex,
        min_age=min_age,
        max_age=max_age,
        smoking_status=smoking_status,
    )

    return args


def resolve_evidence_refs(
    value: Any,
    evidence_store: Mapping[str, Any],
) -> Any:
    """function to retrieve computed evidences"""
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


class Executor:
    """
    ReWOO worker wrapper class
    Orchestrates a single ReWOO run and composes different execution kinds
    per the plan. Uses dag in-degree for parallel execution
    """

    def __init__(self) -> None:
        self.resolver = Resolver()

    async def execute_rewoo_run(
        self,
        dag: PlanDag,
        mcp_backend: MCPExecutionBackend,
        token_usage: dict[str, int],
        *,
        max_concurrency: int = 8,
        model: Model,
    ):
        """executes the rewoo plan using the compiled dag by topological sort execution using an async queue"""

        # init a exec results object to capture execution state at any time
        # set all nodes to PENDING status at the start of the run
        result = ExecutionResult(
            statuses={node: NodeStatus.PENDING for node in dag.steps}
        )
        remaining = dict(dag.in_degree)
        exec_queue: asyncio.Queue[str | None] = asyncio.Queue()

        # add 0 degree nodes immediately
        for node, degree in remaining.items():
            if degree == 0:
                exec_queue.put_nowait(node)

        lock = asyncio.Lock()

        async def block_descendants(failed_node: str) -> None:
            """runs under lock to ensure that workers don't pickup children of failed nodes and execute"""
            stack = list(dag.successors[failed_node])

            while stack:
                current = stack.pop()
                if result.statuses[current] != NodeStatus.PENDING:
                    continue
                result.statuses[current] = NodeStatus.BLOCKED
                stack.extend(dag.successors[current])

        async def worker(mcp_backend: MCPExecutionBackend) -> None:
            """async worker to execute plan step with appropriate backend, generate evidence object and store result"""
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
                        resolved_tool_arguments = resolve_evidence_refs(
                            step.tool_arguments, result.evidence
                        )
                        result.resolved_arguments[step.evidence_id] = (
                            resolved_tool_arguments
                        )
                        # execute with pre-existing MCP backend
                        # or execute simple arithmetic tool
                        if step.execution_kind == ExecutionKind.MCP:
                            if step.executor_name in SIMPLE_TOOLS:
                                tool_ = SIMPLE_TOOLS[step.executor_name]
                                args_ = SimpleToolArgs(**resolved_tool_arguments)
                                output = tool_(a=args_.a, b=args_.b)
                            else:
                                output = await mcp_backend.call_tool(
                                    tool_name=step.executor_name,
                                    tool_arguments=resolved_tool_arguments,
                                )
                        # execute using custom resolver; separate scoped LLM call for cohort field and value resolution (currently sequential) and then execute final GDC query against existing MCP
                        elif step.execution_kind == ExecutionKind.RESOLVER:
                            resolved_tool_arguments = dict(resolved_tool_arguments)
                            resolver_args = construct_cohort_filter_args(
                                **resolved_tool_arguments
                            )
                            resolver_output = await self.resolver.resolve(
                                args=resolver_args,
                                token_usage=token_usage,
                                model=model,
                            )
                            result.resolved_arguments[step.evidence_id] = (
                                resolver_output
                            )
                            output = await mcp_backend.call_tool(
                                tool_name=step.tool_name,
                                tool_arguments=resolver_output,
                            )
                    # if execution  of a step FAILS, make sure to block all descendants with lock
                    except Exception as error:
                        async with lock:
                            result.statuses[node] = NodeStatus.FAILED
                            result.errors[node] = error
                            await block_descendants(node)
                        continue

                    # on successful execution and resolution, update the result and reduce in-degree and add new 0 in-degree nodes to exec queue for the next round
                    async with lock:
                        result.evidence[node] = output  # type: ignore
                        result.statuses[node] = NodeStatus.SUCCEEDED
                        for child in dag.successors[node]:
                            if result.statuses[child] != NodeStatus.PENDING:
                                continue

                            remaining[child] -= 1

                            if remaining[child] == 0:
                                exec_queue.put_nowait(child)

                finally:
                    exec_queue.task_done()

        worker_count = min(
            max_concurrency,
            len(dag.steps),
        )

        tasks = [asyncio.create_task(worker(mcp_backend)) for _ in range(worker_count)]

        await exec_queue.join()
        for task in tasks:
            task.cancel()
        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )
        return result
