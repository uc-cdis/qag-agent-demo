from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Annotated, Any

from _defines import FIELD_REGISTRY, VALUE_REGISTRY
from prompts import RESOLVER_PROMPT
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


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

    def __init__(
        self,
        client: Any,
        model: str,
        prompt: str = RESOLVER_PROMPT,
        field_registry: Mapping[str, str] = FIELD_REGISTRY,
        value_registry: Mapping[str, Any] = VALUE_REGISTRY,
    ) -> None:
        self.client = client
        self.model = model
        self.prompt = prompt
        self.field_registry = field_registry
        self.value_registry = value_registry

    async def resolve(
        self,
        args: ConstructCohortFilterArgs,
        token_usage: dict[str, int],
    ) -> Any:
        """takes the input fields and match with exact GDC specific fields and values using LLM calls"""
        supplied_fields = {
            field: value
            for field, value in args.model_dump().items()
            if value is not None
        }
        resolved_fields: dict[str, Any] = {}

        # loop through fields and match it
        for field, value in supplied_fields.items():
            if field in {"min_age", "max_age"} and value:
                resolved_fields[field] = int(value * 365)
                continue

            if field not in self.field_registry:
                raise ValueError(f"Unknown cohort field {field!r}")

            if field not in self.value_registry:
                raise ValueError(f"No allowed-value registry for {field!r}")

            if not value or len(value) == 0:
                resolved_fields[field] = None
                continue

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt.format(
                            source=value,
                            values=self.value_registry[field],
                        ),
                    }
                ],
                temperature=1.0,
                max_tokens=2048,
                extra_body={
                    "structured_outputs": {
                        "json": ResolverResponse.model_json_schema(),
                    }
                },
            )
            # naive token accounting
            token_usage["input_tokens"] += response.usage.prompt_tokens
            token_usage["output_tokens"] += response.usage.completion_tokens
            token_usage["total_tokens"] += response.usage.total_tokens
            content = response.choices[0].message.content
            if content is None:
                raise ValueError(f"Resolver returned no value for {field!r}")

            resolver_result = ResolverResponse.model_validate_json(content)
            resolved_fields[field] = resolver_result.value
        return resolved_fields
