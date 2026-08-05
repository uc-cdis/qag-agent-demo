from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from executor import ExecutionResult, Executor, NodeStatus
from mcp_executor import MCPExecutionBackend
from planner import PlanDag, Planner, PlannerResponse
from solver import Solver


class RunStatus:
    PLAN_DRAFTED = "PLAN_DRAFTED"
    EXECUTING = "EXECUTING"
    SOLVED = "SOLVED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Pending:
    run_id: str
    run_status: RunStatus
    user_query: str
    planner_response: PlannerResponse
    dag: PlanDag
    plan_summary: list[dict[str, Any]]
    token_usage: dict[str, int]
    feedback_history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    run_id: str
    run_status: RunStatus
    user_query: str
    planner_response: PlannerResponse
    dag: PlanDag
    plan_summary: list[dict[str, Any]]
    token_usage: dict[str, int]
    feedback_history: list[str] = field(default_factory=list)
    execution: ExecutionResult | None = None
    answer: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunView:
    run_id: str
    run_status: RunStatus
    user_query: str
    plan_summary: list[dict[str, Any]] | None = None
    execution_summary: dict[str, Any] | None = None
    answer: str | None = None
    errors: Any | None = None
    token_usage: dict[str, int] | None = None
    feedback_history: list[str] = field(default_factory=list)


@dataclass
class RewooResult:
    run_id: str
    run_status: RunStatus
    user_query: str
    answer: str
    plan: PlannerResponse
    dag: PlanDag
    execution: ExecutionResult
    token_usage: dict[str, int]
    feedback_history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewooConfig:
    max_concurrency: int = 8
    allow_partial_execution: bool = True
    require_all_steps: bool = False


class RewooRunStore:
    """run scoped object to create, store, update and retrieve current rewoo run status via records"""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, record: RunRecord) -> None:
        async with self._lock:
            if record.run_id in self._runs:
                raise ValueError(f"Duplicate run_id: {record.run_id}")
            self._runs[record.run_id] = record

    async def get(self, run_id: str) -> RunRecord:
        async with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as e:
                raise KeyError(f"Unknown ReWOO run_id: {run_id}") from e

    async def update_status(
        self, run_id: str, run_status: RunStatus, *, error: str | None = None
    ) -> None:
        async with self._lock:
            record = self._runs[run_id]
            record.run_status = run_status
            if error is not None:
                record.error = error

    async def replace_plan(
        self,
        run_id: str,
        *,
        user_query: str,
        planner_response: PlannerResponse,
        dag: PlanDag,
        plan_summary: list[dict[str, Any]],
        token_usage: dict[str, int],
        feedback: str,
    ) -> RunRecord:
        async with self._lock:
            record = self._runs[run_id]
            if record.run_status not in {
                RunStatus.PLAN_DRAFTED,
                RunStatus.FAILED,
            }:
                raise RuntimeError(
                    f"Cannot replan run {run_id!r} from run_status {record.run_status}"
                )

            record.user_query = user_query
            record.planner_response = planner_response
            record.dag = dag
            record.plan_summary = plan_summary
            record.token_usage = token_usage
            record.run_status = RunStatus.PLAN_DRAFTED
            record.execution = None
            record.answer = None
            record.error = None
            record.feedback_history.append(feedback)

            return record

    async def save_execution_result(
        self,
        run_id: str,
        *,
        execution: ExecutionResult,
        answer: str,
        token_usage: dict[str, int],
        run_status: RunStatus,
    ) -> RunRecord:
        async with self._lock:
            record = self._runs[run_id]
            record.execution = execution
            record.answer = answer
            record.token_usage = token_usage
            record.run_status = run_status
            return record


class RewooOrchestrator:
    """main ReWOO workflow orchestration class. exposes the plan creation, replanning and execution methods required to complete a ReWOO workflow"""

    def __init__(
        self,
        *,
        planner: Planner,
        executor: Executor,
        solver: Solver,
        mcp_url: str,
        config: RewooConfig | None = None,
        run_store: RewooRunStore | None = None,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.solver = solver
        self.mcp_url = mcp_url
        self.config = config or RewooConfig()
        self.run_store = run_store or RewooRunStore()
        self.orch_store = {}

    def _new_token_usage(self) -> dict[str, int]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def _register_pending(self, tool_call_id, run_id) -> None:
        self.orch_store[tool_call_id] = run_id
        return

    def _get_pending(self, tool_call_id) -> None:
        return self.orch_store.get(tool_call_id, None)

    def _summarize_plan(
        self, planner_response: PlannerResponse, dag: PlanDag
    ) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for step in planner_response.steps:
            compiled = dag.steps[step.evidence_id]
            summary.append(
                {
                    "evidence_id": step.evidence_id,
                    "plan": step.plan,
                    "tool_name": step.tool_name,
                    "execution_kind": str(compiled.execution_kind),
                    "executor_name": compiled.executor_name,
                    "tool_args": step.tool_args,
                    "predecessors": sorted(dag.predecessors[step.evidence_id]),
                    "successors": sorted(dag.successors[step.evidence_id]),
                }
            )
            # summary.append(
            #     {
            #         "plan": step.plan,
            #         "tool_name": step.tool_name,
            #     }
            # )
        return summary

    def _execution_summary(
        self,
        execution: ExecutionResult,
    ) -> dict[str, Any]:
        return {
            "statuses": {
                node: str(run_status) for node, run_status in execution.statuses.items()
            },
            "errors": {node: repr(error) for node, error in execution.errors.items()},
            "evidence_keys": sorted(execution.evidence.keys()),
        }

    def _select_final_evidence(
        self,
        execution: ExecutionResult,
    ) -> dict[str, Any]:
        """
        This return all evidence (TODO select final evidence)
        """
        return dict(execution.evidence)

    def _should_fail_execution(
        self,
        execution: ExecutionResult,
    ) -> bool:
        if self.config.require_all_steps:
            return any(
                run_status != NodeStatus.SUCCEEDED
                for run_status in execution.statuses.values()
            )

        if not self.config.allow_partial_execution:
            return bool(execution.errors)

        return False

    async def create_plan(
        self,
        user_query: str,
        *,
        message_history: list[Any] | None = None,
    ) -> PlannerResponse:
        """create a plan for ReWOO"""
        run_id = f"run_{uuid.uuid4().hex}"
        token_usage = self._new_token_usage()

        planner_response = self.planner.run(
            user_query=user_query,
            token_usage=token_usage,
        )
        dag = self.planner.compile_dag(planner_response)
        plan_summary = self._summarize_plan(
            planner_response=planner_response,
            dag=dag,
        )
        record = RunRecord(
            run_id=run_id,
            run_status=RunStatus.PLAN_DRAFTED,
            user_query=user_query,
            planner_response=planner_response,
            dag=dag,
            plan_summary=plan_summary,
            token_usage=token_usage,
            metadata={
                "message_history": message_history or [],
            },
        )
        await self.run_store.create(record)
        return Pending(
            run_id=record.run_id,
            run_status=record.run_status,
            user_query=record.user_query,
            planner_response=record.planner_response,
            dag=record.dag,
            plan_summary=record.plan_summary,
            token_usage=record.token_usage,
            feedback_history=record.feedback_history,
            metadata=record.metadata,
        )

    async def replan(
        self,
        run_id: str,
        feedback: str,
    ) -> Pending:
        record = await self.run_store.get(run_id)
        revised_query = (
            f"Original user query: \n{record.user_query}\n\n"
            f"User feedback on previous plan:\n{feedback}\n\n"
            "Revise the ReWOO plan accordingly."
        )
        token_usage = self._new_token_usage()
        planner_response = self.planner.run(
            user_query=revised_query,
            token_usage=token_usage,
        )
        dag = self.planner.compile_dag(planner_response)
        plan_summary = self._summarize_plan(
            planner_response=planner_response,
            dag=dag,
        )
        updated = await self.run_store.replace_plan(
            run_id,
            user_query=record.user_query,
            planner_response=planner_response,
            dag=dag,
            plan_summary=plan_summary,
            token_usage=token_usage,
            feedback=feedback,
        )
        return Pending(
            run_id=updated.run_id,
            run_status=updated.run_status,
            user_query=updated.user_query,
            planner_response=updated.planner_response,
            dag=updated.dag,
            plan_summary=updated.plan_summary,
            token_usage=updated.token_usage,
            feedback_history=updated.feedback_history,
            metadata=updated.metadata,
        )

    async def approve_and_execute(self, run_id: str) -> RewooResult:
        """execute the rest of the ReWOO workflow after Human In The Loop approval"""
        record = await self.run_store.get(run_id)
        if record.run_status != RunStatus.PLAN_DRAFTED:
            raise RuntimeError(
                f"Run {run_id!r} is not awaiting approval. "
                f"Current run_status: {record.run_status}"
            )

        await self.run_store.update_status(
            run_id,
            RunStatus.EXECUTING,
        )

        try:
            async with MCPExecutionBackend(self.mcp_url) as mcp_backend:
                await mcp_backend.load_tools()
                execution_result = await self.executor.execute_rewoo_run(
                    dag=record.dag,
                    mcp_backend=mcp_backend,
                    max_concurrency=self.config.max_concurrency,
                    token_usage=record.token_usage,
                )

            if self._should_fail_execution(execution_result):
                await self.run_store.update_status(
                    run_id,
                    RunStatus.FAILED,
                    error="Execution did not complete successfully.",
                )
                raise RuntimeError(
                    f"ReWOO execution failed for run {run_id!r}: "
                    f"{execution_result.errors}"
                )

            answer = self.solver.solve(
                task=record.user_query,
                plan_dag=record.dag,
                execution_result=execution_result,
                token_usage=record.token_usage,
            )
            final_status = RunStatus.SOLVED
            updated = await self.run_store.save_execution_result(
                run_id,
                execution=execution_result,
                answer=answer,
                token_usage=record.token_usage,
                run_status=final_status,
            )
            return RewooResult(
                run_id=updated.run_id,
                run_status=updated.run_status,
                user_query=record.user_query,
                answer=answer,
                plan=updated.planner_response,
                dag=updated.dag,
                execution=execution_result,
                token_usage=updated.token_usage,
                # final_evidence=self._select_final_evidence(execution_result),
                metadata=updated.metadata,
            )
        except Exception as e:
            await self.run_store.update_status(
                run_id,
                RunStatus.FAILED,
                error=repr(e),
            )
            raise

    async def execute_plan(self, user_query: str, approved_plan: PlannerResponse):
        dag = self.planner.compile_dag(approved_plan)
        async with MCPExecutionBackend(self.mcp_url) as mcp_backend:
            await mcp_backend.load_tools()
            execution_result = await self.executor.execute_rewoo_run(
                dag=dag,
                mcp_backend=mcp_backend,
                max_concurrency=self.config.max_concurrency,
            )
        answer = self.solver.solve(
            task=user_query,
            plan_dag=dag,
            execution_result=execution_result,
        )
        return RewooResult(
            answer=answer,
            plan=approved_plan,
            dag=dag,
            execution_result=execution_result,
        )

    async def cancel(self, run_id: str) -> RunView:
        record = await self.run_store.get(run_id)
        if record.run_status == RunStatus.EXECUTING:
            raise RuntimeError(f"Cannot cancel run {run_id!r}; it is already executing")
        await self.run_store.update_status(
            run_id,
            RunStatus.CANCELLED,
        )
        return await self.get_status(run_id)

    async def get_status(
        self,
        run_id: str,
    ) -> RunView:
        record = await self.run_store.get(run_id)

        execution_summary = (
            self._execution_summary(record.execution)
            if record.execution is not None
            else None
        )

        return RunView(
            run_id=record.run_id,
            run_status=record.run_status,
            user_query=record.user_query,
            plan_summary=record.plan_summary,
            execution_summary=execution_summary,
            answer=record.answer,
            errors=record.error,
            token_usage=record.token_usage,
            feedback_history=record.feedback_history,
        )

    async def run_end_to_end(self, user_query: str, *, message_history) -> RewooResult:
        """method to run entire ReWOO workflow end to end without any HITL approval gate - for endpoint use"""
        run_id = f"run_{uuid.uuid4().hex}"
        token_usage = self._new_token_usage()

        planner_response = self.planner.run(
            user_query=user_query,
            token_usage=token_usage,
        )
        dag = self.planner.compile_dag(planner_response)
        plan_summary = self._summarize_plan(
            planner_response=planner_response,
            dag=dag,
        )
        record = RunRecord(
            run_id=run_id,
            run_status=RunStatus.PLAN_DRAFTED,
            user_query=user_query,
            planner_response=planner_response,
            dag=dag,
            plan_summary=plan_summary,
            token_usage=token_usage,
            metadata={
                "message_history": message_history or [],
            },
        )
        await self.run_store.create(record)
        if record.run_status != RunStatus.PLAN_DRAFTED:
            raise RuntimeError(
                f"Run {run_id!r} is not awaiting approval. "
                f"Current run_status: {record.run_status}"
            )

        await self.run_store.update_status(
            run_id,
            RunStatus.EXECUTING,
        )

        try:
            async with MCPExecutionBackend(self.mcp_url) as mcp_backend:
                await mcp_backend.load_tools()
                execution_result = await self.executor.execute_rewoo_run(
                    dag=record.dag,
                    mcp_backend=mcp_backend,
                    max_concurrency=self.config.max_concurrency,
                    token_usage=record.token_usage,
                )

            if self._should_fail_execution(execution_result):
                await self.run_store.update_status(
                    run_id,
                    RunStatus.FAILED,
                    error="Execution did not complete successfully.",
                )
                raise RuntimeError(
                    f"ReWOO execution failed for run {run_id!r}: "
                    f"{execution_result.errors}"
                )

            answer = self.solver.solve(
                task=record.user_query,
                plan_dag=record.dag,
                execution_result=execution_result,
                token_usage=record.token_usage,
            )
            final_status = RunStatus.SOLVED
            updated = await self.run_store.save_execution_result(
                run_id,
                execution=execution_result,
                answer=answer,
                token_usage=record.token_usage,
                run_status=final_status,
            )
            return RewooResult(
                run_id=updated.run_id,
                run_status=updated.run_status,
                user_query=record.user_query,
                answer=answer,
                plan=updated.planner_response,
                dag=updated.dag,
                execution=execution_result,
                token_usage=updated.token_usage,
                metadata=updated.metadata,
            )
        except Exception as e:
            await self.run_store.update_status(
                run_id,
                RunStatus.FAILED,
                error=repr(e),
            )
            raise


def make_default_rewoo_orchestrator(
    *,
    mcp_url: str,
    model: str,
    client: Any,
) -> RewooOrchestrator:
    """default constrcutor for ReWOO orchestrator"""
    planner = Planner(
        client=client,
        model=model,
    )
    executor = Executor(
        rclient=client,
        rmodel=model,
    )
    solver = Solver(
        client=client,
        model=model,
    )
    return RewooOrchestrator(
        planner=planner, executor=executor, solver=solver, mcp_url=mcp_url
    )
