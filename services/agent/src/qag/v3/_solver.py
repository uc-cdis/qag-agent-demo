from __future__ import annotations

import sys
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

from ._executor import ExecutionResult, NodeStatus
from ._planner import PlanDag

SOLVER_PROMPT = """
You are an AI assistant who solves problems given a concrete step by step plan and evidences for each step.
Your task is to briefly state the problem, the steps takens to arrive at a solution and then using the evidence
make a short final conclusion for your task. The main task is a Q&A about accurate variant statistics for the GDC.

RULES:
1. Return numerical answers specified to 2 decimal places, for example '42.05%'.
"""


def _format_markdown(item) -> str:
    return f"{item["evidence_id"]}: {item["evidence"]}"


class Solver:
    def __init__(self) -> None:
        self.agent = Agent(
            instructions=SOLVER_PROMPT,
        )

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

    async def solve(
        self,
        *,  # enforce kwargs
        task: str,
        plan_dag: PlanDag,
        execution_result: ExecutionResult,
        token_usage: dict[str, int],
        model: Model,
    ) -> str:
        history = self.create_plan_evidence_history(
            plan_dag,
            execution_result,
        )
        result = await self.agent.run(
            user_prompt=f"##Original Task##\n{task}\n\n##Plan & Evidence##\n{history}",
            model=model,
        )

        token_usage["input_tokens"] += result.usage.input_tokens
        token_usage["output_tokens"] += result.usage.output_tokens
        token_usage["total_tokens"] += result.usage.total_tokens

        # added generated evidences formatting for better ui rendering
        prefix = (
            "**Generated Evidences**"
            + "\n\n"
            + "\n".join([f"- {_format_markdown(item)}" for item in history])
        )
        # print(prefix + "\n\n" + result.output, file=sys.stderr) uncomment to view answer formatting in cli
        # final_output = prefix + "\n\n" + result.output
        final_output = result.output  # TODO: trying without evidence given tool calls
        return final_output
