from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from _defines import *
from prompts import *
from pydantic import ValidationError


@dataclass
class PlanDag:
    steps: dict[str, PlannerStep]
    predecessors: dict[str, set[str]]
    successors: dict[str, set[str]]
    in_degree: dict[str, int]


def extract_dependencies(dependency_list):
    if not dependency_list:
        return []
    return [d.ref for d in dependency_list]


class Planner:
    def __init__(
        self,
        prompt,
        tools,
        model,
        client,
        retry_reason: dict[int, str],
        max_retries: int,
    ):
        super().__init__()
        self.prompt = prompt
        self.tools = tools
        self.model = model
        self.client = client
        self.retry_reason = retry_reason

    def run(self, user_query: str) -> PlannerResponse:
        self.user_query = user_query
        prompt = self.prompt.format(tool_description=self.tools, task=user_query)
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=2048,
            extra_body={
                "structured_outputs": {
                    "json": PlannerResponse.model_json_schema(),
                }
            },
        )
        content = result.choices[0].message.content
        if content is None:
            raise ValueError("Planner returned empty")
        try:
            return PlannerResponse.model_validate_json(content)
        except ValidationError as e:
            raise ValueError(f"Planner returned invalid structured output: {e}") from e

    def compile_dag(self, plan: PlannerResponse) -> PlanDag:
        steps = {step.evidence_id: step for step in plan.steps}
        if len(steps) != len(plan.steps):
            raise ValueError("Plan contains duplicate evidence IDs")
        predecessors = {idx: set() for idx in steps}
        successors = defaultdict(set)

        for step in plan.steps:
            evidence_id = step.evidence_id
            dependencies = extract_dependencies(
                step.dependencies if step.dependencies else []
            )
            for dependency in dependencies:
                if dependency == evidence_id:
                    raise ValueError(f"{evidence_id} cannot depend on itself")
                if dependency not in steps:
                    raise ValueError(
                        f"{evidence_id} references unknown " f"dependency {dependency}"
                    )
                successors[dependency].add(evidence_id)
                predecessors[evidence_id].add(dependency)

        norm_successors = {
            evidence_id: set(successors[evidence_id]) for evidence_id in steps
        }

        in_degree = {
            evidence_id: len(dependencies)
            for evidence_id, dependencies in predecessors.items()
        }
        self._validate_acyclic(norm_successors, in_degree)

        return PlanDag(
            steps=steps,
            predecessors=predecessors,
            successors=norm_successors,
            in_degree=in_degree,
        )

    @staticmethod
    def _validate_acyclic(successors, in_degree) -> None:
        remaining = dict(in_degree)

        queue = [node for node, degree in remaining.items() if degree == 0]

        visited = 0
        while queue:
            node = queue.pop()
            visited += 1

            for child in successors[node]:
                remaining[child] -= 1

                if remaining[child] == 0:
                    queue.append(child)

        if visited != len(remaining):
            raise ValueError("Plan DAG contains a cycle")
