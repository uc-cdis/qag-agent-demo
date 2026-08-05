from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from _tool_desc import TOOLS_DESCRIPTION
from prompts import PLANNER_PROMPT
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ref: str = Field(
        description="Reference to an earlier evidence result, such as '#E1'",
        pattern=r"^#E[1-9]\d*$",
    )

    @property
    def index(self) -> int:
        return int(self.ref[2:])


ToolArgumentValue = str | int | float | bool | list[str] | EvidenceReference | None


class PlannerStep(BaseModel):
    plan: str = Field(
        min_length=1, description="What this step does and why it is needed."
    )
    evidence_id: str = Field(
        description="Unique evidence identifier produced by this step",
        pattern=r"^#E[1-9]\d*$",
        examples=["#E1"],
    )
    tool_name: str = Field(
        min_length=1,
        description="Exact name of the tool to invoke.",
    )
    tool_args: dict[str, ToolArgumentValue] = Field(
        default_factory=dict,
        description=(
            "Arguments passed to the tool. Values may be literals, lists, "
            "null, or references to earlier evidence."
        ),
    )
    dependencies: Optional[list[EvidenceReference]] = Field(
        min_length=1, description="Earlier evidence results needed by this step"
    )


class PlannerResponse(BaseModel):
    objective: str = Field(
        min_length=1, description="The objective the plan is intended to solve"
    )
    steps: list[PlannerStep] = Field(
        min_length=1, description="Ordered tool execution plan"
    )


class ExecutionKind(str, Enum):
    MCP = "mcp"
    RESOLVER = "resolver"
    LOCAL = "local"


EXECUTION_POLICIES = {
    "construct_cohort_filter": (ExecutionKind.RESOLVER, "cohort_resolver")
}


@dataclass
class PlanDag:
    steps: dict[str, CompiledStep]
    predecessors: dict[str, set[str]]
    successors: dict[str, set[str]]
    in_degree: dict[str, int]


@dataclass
class CompiledStep:
    plan: str
    evidence_id: str
    tool_name: str
    tool_args: dict[str, Any]
    execution_kind: ExecutionKind
    executor_name: str


# def extract_dependencies(dependency_list):
#     if not dependency_list:
#         return []
#     return [d.ref for d in dependency_list]


EVIDENCE_REF_PATTERN = re.compile(r"^#E[1-9]\d*$")


def extract_dependencies(value: Any) -> set[str]:
    deps: set[str] = set()

    if isinstance(value, str):
        if EVIDENCE_REF_PATTERN.fullmatch(value):
            deps.add(value)
        return deps

    if isinstance(value, Mapping):
        for item in value.values():
            deps.update(extract_dependencies(item))
        return deps

    if isinstance(value, list | tuple | set):
        for item in value:
            deps.update(extract_dependencies(item))
        return deps

    # Optional: handle EvidenceReference objects
    if hasattr(value, "ref"):
        ref = getattr(value, "ref")
        if isinstance(ref, str) and EVIDENCE_REF_PATTERN.fullmatch(ref):
            deps.add(ref)

    return deps


def compile_step(step: PlannerStep) -> CompiledStep:
    kind, executor_name = EXECUTION_POLICIES.get(
        step.tool_name,
        (
            ExecutionKind.MCP,
            step.tool_name,
        ),
    )

    return CompiledStep(
        plan=step.plan,
        evidence_id=step.evidence_id,
        tool_name=step.tool_name,
        tool_args=step.tool_args,
        execution_kind=kind,
        executor_name=executor_name,
    )


class Planner:
    """
    ReWOO planning module.
    Input: Natual language user query
    Output: A directed acyclic graph structured object with plan steps
    """

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        retry_reason: dict[int, str] = {},
        prompt: str = PLANNER_PROMPT,
        tools: str = TOOLS_DESCRIPTION,
    ):
        super().__init__()
        self.prompt = prompt
        self.tools = tools
        self.model = model
        self.client = client
        self.retry_reason = retry_reason

    def run(
        self,
        user_query: str,
        token_usage: dict[str, int],
    ) -> PlannerResponse:
        """produces a plan to address the user query via a structured outputs call to the LLM"""
        self.user_query = user_query
        prompt = self.prompt.format(tool_description=self.tools, task=user_query)
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2048,
            seed=42,
            extra_body={
                "structured_outputs": {
                    "json": PlannerResponse.model_json_schema(),
                }
            },
        )
        # naive token accounting
        token_usage["input_tokens"] += result.usage.prompt_tokens
        token_usage["output_tokens"] += result.usage.completion_tokens
        token_usage["total_tokens"] += result.usage.total_tokens

        # response validation
        content = result.choices[0].message.content
        if content is None:
            raise ValueError("Planner returned empty")
        try:
            return PlannerResponse.model_validate_json(content)
        except ValidationError as e:
            raise ValueError(f"Planner returned invalid structured output: {e}") from e

    def compile_dag(self, plan: PlannerResponse) -> PlanDag:
        """compiles a simple dag like object over the plan steps with evidences serving as dependency indicators"""
        steps = {step.evidence_id: step for step in plan.steps}
        compiled_steps = {step.evidence_id: compile_step(step) for step in plan.steps}
        if len(steps) != len(plan.steps):
            raise ValueError("Plan contains duplicate evidence IDs")
        predecessors = {idx: set() for idx in steps}
        successors = defaultdict(set)

        for step in plan.steps:
            evidence_id = step.evidence_id
            dependencies: set[str] = set()
            if step.dependencies:
                dependencies = extract_dependencies(step.dependencies)

            dependencies.update(extract_dependencies(step.tool_args))

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
            steps=compiled_steps,
            predecessors=predecessors,
            successors=norm_successors,
            in_degree=in_degree,
        )

    @staticmethod
    def _validate_acyclic(successors, in_degree) -> None:
        """simple cycle checking in a dag"""
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
