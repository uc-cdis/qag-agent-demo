from __future__ import annotations

from typing import Any
from planner import PlanDag
from prompts import SOLVER_PROMPT
from executor import ExecutionResult, NodeStatus


class Solver:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        prompt: str = SOLVER_PROMPT,
    ) -> None:
        self.prompt = prompt
        self.client = client
        self.model = model
        pass

    def create_plan_evidence_history(
        self,
        plan_dag: PlanDag,
        execution_result: ExecutionResult,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []

        for index, (evidence_id, step) in enumerate(plan_dag.steps.items(), start=1):
            status = execution_result.statuses[evidence_id]
            entry: dict[str, Any] = {
                "step_id": f"#S{index}",
                "plan": step.plan,
                "evidence_id": evidence_id,
                "status": status.value,
            }
            if status == NodeStatus.SUCCEEDED:
                evidence = execution_result.evidence[evidence_id]

                if hasattr(evidence, "model_dump"):
                    evidence = evidence.model_dump(mode="json")

                entry["evidence"] = evidence

            elif status == NodeStatus.FAILED:
                error = execution_result.errors.get(evidence_id)

                entry["error"] = (
                    str(error) if error is not None else "Unknown execution error"
                )

            elif status == NodeStatus.BLOCKED:
                entry["error"] = "Failed dependency upstream"

            history.append(entry)

        return history

    def solve(
        self,
        task: str,
        plan_dag: PlanDag,
        execution_result: ExecutionResult,
        token_usage: dict[str, int],
    ) -> str:
        history = self.create_plan_evidence_history(
            plan_dag,
            execution_result,
        )

        response = self._solve(
            task=task,
            plan_evidence_history=history,
        )
        token_usage["input_tokens"] += response.usage.prompt_tokens
        token_usage["output_tokens"] += response.usage.completion_tokens
        token_usage["total_tokens"] += response.usage.total_tokens

        content = response.choices[0].message.content

        if content is None:
            raise ValueError("Solver returned empty content")

        print(content)
        return content

    def _solve(
        self,
        task: str,
        plan_evidence_history: list[dict],
    ):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt.format(
                        plan_evidence=plan_evidence_history,
                        task=task,
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=2048,
            seed=42,
        )
