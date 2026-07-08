from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from _defines import PlannerResponse
from executor import ExecutionResult, Executor, NodeStatus
from planner import PlanDag, Planner
from solver import Solver


def format_plan(plan: PlannerResponse) -> str:
    return plan.model_dump_json(indent=2)


def json_compatible(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if isinstance(value, set):
        return sorted(value)

    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]

    return value


def format_execution_summary(
    execution: ExecutionResult,
) -> str:
    summary = {
        "statuses": {
            evidence_id: status.value
            for evidence_id, status in execution.statuses.items()
        },
        "evidence": {
            evidence_id: json_compatible(evidence)
            for evidence_id, evidence in execution.evidence.items()
        },
        "errors": {
            evidence_id: {
                "type": type(error).__name__,
                "message": str(error),
            }
            for evidence_id, error in execution.errors.items()
        },
    }

    return json.dumps(
        summary,
        indent=2,
        default=str,
    )


class OrchestrationError(Exception):
    """Base exception for orchestration failures."""


class PlanningError(OrchestrationError):
    pass


class CompilationError(OrchestrationError):
    pass


class ExecutionError(OrchestrationError):
    pass


class SolverError(OrchestrationError):
    pass


@dataclass
class OrchestratorResult:
    answer: str
    plan: PlannerResponse
    dag: PlanDag
    execution: ExecutionResult

    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorConfig:
    max_concurrency: int = 8
    allow_partial_execution: bool = True
    require_all_steps: bool = False


class Orchestrator:
    def __init__(
        self,
        *,
        planner: Planner,
        executor: Executor,
        solver: Solver,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.solver = solver
        self.config = config or OrchestratorConfig()

    async def run(
        self,
        user_query: str,
    ) -> OrchestratorResult:
        plan = await self._create_plan(user_query)

        print("\n========== GENERATED PLAN ==========")
        print(format_plan(plan))

        dag = self._compile_plan(plan)
        execution = await self._execute_plan(dag)

        print("\n========== EXECUTION SUMMARY ==========")
        print(format_execution_summary(execution))

        warnings = self._inspect_execution(execution)

        answer = await self._solve(
            user_query=user_query,
            dag=dag,
            execution=execution,
        )
        print("\n========== FINAL RESULT ==========")
        print(answer)
        print("==================================\n")

        return OrchestratorResult(
            answer=answer,
            plan=plan,
            dag=dag,
            execution=execution,
            warnings=warnings,
            metadata={
                "total_steps": len(dag.steps),
                "succeeded_steps": sum(
                    status == NodeStatus.SUCCEEDED
                    for status in execution.statuses.values()
                ),
                "failed_steps": sum(
                    status == NodeStatus.FAILED
                    for status in execution.statuses.values()
                ),
                "blocked_steps": sum(
                    status == NodeStatus.BLOCKED
                    for status in execution.statuses.values()
                ),
            },
        )

    async def _create_plan(
        self,
        user_query: str,
    ) -> PlannerResponse:
        try:
            plan = await asyncio.to_thread(
                self.planner.run,
                user_query,
            )
        except Exception as exc:
            raise PlanningError(f"Planner failed for query: {user_query!r}") from exc

        if plan is None:
            raise PlanningError("Planner returned no valid PlannerResponse")

        return plan

    def _compile_plan(
        self,
        plan: PlannerResponse,
    ) -> PlanDag:
        try:
            return self.planner.compile_dag(plan)
        except Exception as exc:
            raise CompilationError(
                "Planner output could not be compiled into a valid DAG"
            ) from exc

    async def _execute_plan(
        self,
        dag: PlanDag,
    ) -> ExecutionResult:
        try:
            return await self.executor.execute_plan(
                dag,
                max_concurrency=self.config.max_concurrency,
            )
        except Exception as exc:
            raise ExecutionError(
                "Plan execution failed at the scheduler level"
            ) from exc

    def _inspect_execution(
        self,
        execution: ExecutionResult,
    ) -> list[str]:
        failed = [
            evidence_id
            for evidence_id, status in execution.statuses.items()
            if status == NodeStatus.FAILED
        ]

        blocked = [
            evidence_id
            for evidence_id, status in execution.statuses.items()
            if status == NodeStatus.BLOCKED
        ]

        if self.config.require_all_steps and (failed or blocked):
            raise ExecutionError(
                "Execution did not complete successfully. "
                f"Failed={failed}, blocked={blocked}"
            )

        if not self.config.allow_partial_execution and failed:
            raise ExecutionError(f"Execution contains failed steps: {failed}")

        warnings: list[str] = []

        if failed:
            warnings.append(f"Failed evidence steps: {', '.join(failed)}")

        if blocked:
            warnings.append(f"Blocked evidence steps: {', '.join(blocked)}")

        return warnings

    async def _solve(
        self,
        *,
        user_query: str,
        dag: PlanDag,
        execution: ExecutionResult,
    ) -> str:
        try:
            return await asyncio.to_thread(
                self.solver.solve,
                user_query,
                dag,
                execution,
            )
        except Exception as exc:
            raise SolverError("Solver failed to generate the final answer") from exc
