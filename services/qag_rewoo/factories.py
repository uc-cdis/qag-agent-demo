from __future__ import annotations

from collections import defaultdict
from typing import Any

from _defines import FIELD_REGISTRY, VALUE_REGISTRY
from _tools import (
    compute_case_intersection,
    compute_case_union,
    get_case_set_size,
    get_cases_by_cohort_filter,
    get_cases_with_available_genomic_data,
    get_copy_number_variant_occurrences,
    get_gene_expression_statistics,
    get_microsatellite_instability_occurrences,
    get_most_variably_expressed_genes,
    get_simple_somatic_mutation_occurrences,
    make_construct_cohort_filter_tool,
)
from executor import Executor, ToolExecutor
from orchestrator import Orchestrator, OrchestratorConfig
from planner import Planner
from prompts import PLANNER_PROMPT, RESOLVER_PROMPT, SOLVER_PROMPT
from solver import Solver


def make_planner(
    *,
    client: Any,
    model: str,
    tools_description: str,
    retry_reason: dict[str, str] = defaultdict(str),
    max_retries: int = 1,
) -> Planner:
    return Planner(
        prompt=PLANNER_PROMPT,
        tools=tools_description,
        model=model,
        client=client,
        retry_reason=retry_reason,
        max_retries=max_retries,
    )


def make_tool_executor(
    *,
    resolver_client: Any,
    resolver_model: str,
) -> ToolExecutor:
    registry = ToolExecutor()
    construct_cohort_filter = make_construct_cohort_filter_tool(
        client=resolver_client,
        model=resolver_model,
        resolver_prompt=RESOLVER_PROMPT,
        field_registry=FIELD_REGISTRY,
        value_registry=VALUE_REGISTRY,
    )

    registry.register(construct_cohort_filter, name="construct_cohort_filter")
    registry.register(get_cases_by_cohort_filter)
    registry.register(compute_case_intersection)
    registry.register(compute_case_union)
    registry.register(get_case_set_size)
    registry.register(get_cases_with_available_genomic_data)
    registry.register(get_simple_somatic_mutation_occurrences)
    registry.register(get_copy_number_variant_occurrences)
    registry.register(get_microsatellite_instability_occurrences)
    registry.register(get_gene_expression_statistics)
    registry.register(get_most_variably_expressed_genes)

    return registry


def make_executor(
    *,
    tool_executor: ToolExecutor,
) -> Executor:
    return Executor(
        exec_tool=tool_executor,
    )


def make_solver(
    *,
    client: Any,
    model: str,
) -> Solver:
    return Solver(
        prompt=SOLVER_PROMPT,
        client=client,
        model=model,
    )


def make_orchestrator(
    *,
    planner_client: Any,
    planner_model: str,
    resolver_client: Any,
    resolver_model: Any,
    solver_client: Any,
    solver_model: str,
    tools_description: str,
    max_execution_concurrency: int = 8,
) -> Orchestrator:
    planner = make_planner(
        client=planner_client,
        model=planner_model,
        tools_description=tools_description,
    )

    tool_executor = make_tool_executor(
        resolver_client=resolver_client,
        resolver_model=resolver_model,
    )

    executor = make_executor(
        tool_executor=tool_executor,
    )

    solver = make_solver(
        client=solver_client,
        model=solver_model,
    )

    return Orchestrator(
        planner=planner,
        executor=executor,
        solver=solver,
        config=OrchestratorConfig(
            max_concurrency=max_execution_concurrency,
            allow_partial_execution=True,
            require_all_steps=False,
        ),
    )


def make_default_orchestrator(
    *,
    client: Any,
    model: str,
    tools_description: str,
    max_execution_concurrency: int = 8,
) -> Orchestrator:
    return make_orchestrator(
        planner_client=client,
        planner_model=model,
        resolver_client=client,
        resolver_model=model,
        solver_client=client,
        solver_model=model,
        tools_description=tools_description,
        max_execution_concurrency=max_execution_concurrency,
    )
