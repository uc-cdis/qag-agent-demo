import inspect
import json
import sys
from typing import Any

import requests
from _defines import GDC_API, TOOL_CACHE_ID_TEMPLATES, CaseSetId


def get_tool_id_template() -> str:
    curr_frame = inspect.currentframe()
    assert curr_frame is not None
    frame = curr_frame.f_back
    assert frame is not None
    return TOOL_CACHE_ID_TEMPLATES[frame.f_code.co_name]


def suggest_tool_from_id(case_set_id: CaseSetId) -> str:
    match = match_template_and_params(case_set_id, TOOL_CACHE_ID_TEMPLATES)
    if match is not None:
        tool_name, tool_params = match
        tool_args = ", ".join([f"{k}={v}" for k, v in tool_params.items()])
        return (
            f"Based on the provided case_set_id={case_set_id}, "
            f"it looks like you need to use the following tool with these args: "
            f"{tool_name}({tool_args})"
        )
    return (
        f"The provided case_set_id={case_set_id} did not match any known patterns returned by our tools. "
        f"Are you sure the tools in this MCP server were used to query for cases?"
    )


def match_template_and_params(
    identifier: str, templates: dict[str, str]
) -> tuple[str, dict[str, str]] | None:
    for func_name, template in templates.items():
        prefix = template.split("-(")[0]
        if identifier.startswith(prefix):
            params = extract_top_level_params(identifier)
            names = extract_top_level_params(template)
            if len(params) != len(names):
                raise ValueError(
                    f"Number of parameters ({len(names)}) in matched template ({template}) "
                    f"does not match the number of parameters ({len(params)}) in the identifier ({identifier})"
                )
            return (func_name, {n.strip("{}"): p for n, p in zip(names, params)})
    return None


def extract_top_level_params(identifier: str) -> list[str]:
    params = []
    stack = []
    for i, c in enumerate(identifier):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if len(stack) == 0:
                raise IndexError("No matching closing parens at: " + str(i))
            opening_idx = stack.pop()
            closing_idx = i
            if len(stack) == 0:  # top level paren
                params.append(identifier[opening_idx + 1 : closing_idx])

    if len(stack) > 0:
        raise IndexError("No matching opening parens at: " + str(stack.pop()))

    return params


def truncate(x, max_len=100):
    x = str(x)
    if len(x) > max_len:
        x = x[:max_len]
    return x


def gdc_query_all(
    endpoint: str,
    filters: dict,
    cohort_filter: dict | str | None = None,
    fields: list[str] | None = None,
    page_size: int = 1000,
) -> list[dict[str, Any]]:
    url = f"{GDC_API}/{endpoint}"
    all_hits = []
    offset = 0

    print(
        f"querying endpoint: {endpoint} with filters: {truncate(filters)} and cohort_filter: {json.loads(cohort_filter) if isinstance(cohort_filter, str) else cohort_filter}",
        file=sys.stderr,
    )

    all_filters = filters
    if cohort_filter:  # not None or empty string
        if isinstance(cohort_filter, str):
            cohort_filter = json.loads(cohort_filter)
        if filters:
            all_filters = {
                "op": "and",
                "content": [
                    cohort_filter,
                    filters,
                ],
            }
        else:
            all_filters = cohort_filter

    while True:
        payload = {
            "filters": all_filters,
            "from": offset,
            "size": page_size,
        }

        if fields:
            # even if using post, fields needs to be a comma-separated string
            payload["fields"] = ",".join(fields)

        response = requests.post(url, json=payload)
        response.raise_for_status()
        resp_json = response.json()

        if resp_json["warnings"]:
            print(resp_json["warnings"], file=sys.stderr)

        data = resp_json["data"]

        hits = data["hits"]
        all_hits.extend(hits)

        pagination = data["pagination"]
        total = pagination["total"]

        offset += page_size
        if offset >= total:
            break

    return all_hits
