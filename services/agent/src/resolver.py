from __future__ import annotations

import asyncio
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from pydantic_ai import Agent, AgentRunResult, RunContext

# isort: off
from _defines import FIELD_REGISTRY, VALUE_REGISTRY
from prompts import RESOLVER_PROMPT

# isort: on


def normalize_string_list(value: object) -> list[str] | None:
    """helper to ensure correct formatting for cohort args"""
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    if isinstance(value, list):
        normalized: list[str] = []

        for item in value:
            if not isinstance(item, str):
                raise TypeError("Expected a string or a list of strings")

            item = item.strip()

            if item:
                normalized.append(item)

        return normalized

    raise TypeError("Expected a string, a list of strings, or None")


StringList = Annotated[
    list[str] | None,
    BeforeValidator(normalize_string_list),
]


class ConstructCohortFilterArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: StringList | None = None
    primary_site: StringList | None = None
    disease_type: StringList | None = None
    project_id: StringList | None = None
    sex: StringList | None = None
    smoking_status: StringList | None = None

    min_age: int | None = Field(default=None, ge=0)
    max_age: int | None = Field(default=None, ge=0)


class ResolverResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: list[str]


class Resolver:
    """
    simple class to resolve cohort filter generation using simple LLM calls
    """

    def __init__(self) -> None:
        self.agent = Agent(
            instructions=RESOLVER_PROMPT,
            output_type=ResolverResponse,
        )

    async def resolve(
        self,
        args: ConstructCohortFilterArgs,
        ctx: RunContext,
        token_usage: dict[str, int],
    ) -> Any:
        """takes the input fields and match with exact GDC specific fields and values using LLM calls"""
        supplied_fields = {
            field: value
            for field, value in args.model_dump().items()
            if value is not None
        }
        resolved_fields: dict[str, list[str] | int | None] = {}

        # loop through fields and match it
        coros = []
        for field, value in supplied_fields.items():
            if field in {"min_age", "max_age"} and value:
                resolved_fields[field] = int(value * 365)
                continue

            if field not in FIELD_REGISTRY:
                raise ValueError(f"Unknown cohort field {field!r}")

            if field not in VALUE_REGISTRY:
                raise ValueError(f"No allowed-value registry for {field!r}")

            if not value or len(value) == 0:
                resolved_fields[field] = None
                continue

            coro = self.agent.run(
                user_prompt="Pick the closest value possible.\nSource Value: {value}\nPossible Values: {VALUE_REGISTRY[field]}",
                model=ctx.model,
            )
            coros.append(coro)
        results: list[AgentRunResult[ResolverResponse]] = await asyncio.gather(*coros)

        for field, result in zip(supplied_fields, results):
            resolved_fields[field] = result.output.value

            # naive token accounting
            token_usage["input_tokens"] += result.usage.input_tokens
            token_usage["output_tokens"] += result.usage.output_tokens
            token_usage["total_tokens"] += result.usage.total_tokens

        return resolved_fields
