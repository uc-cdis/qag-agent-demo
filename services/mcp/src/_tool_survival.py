import sys

import requests
from cachetools import TTLCache
from mcp.server.fastmcp.exceptions import ToolError

# isort: off
from _defines import CaseSetId, GDC_API
from _utils import suggest_tool_from_id

# isort: on


def make_survival_tool(cache: TTLCache):
    def get_survival_data(
        case_set_id: CaseSetId,
    ) -> dict:
        """
        A tool to query the GDC API for survival data for a cohort of cases.
        """
        print(
            f"get_survival_data(case_set_id={repr(case_set_id)})",
            file=sys.stderr,
        )

        if case_set_id not in cache:
            raise ToolError(
                f"Case set {case_set_id} was not found in the server side cache, perhaps it expired? "
                f"Try requerying for those cases to cache it again. {suggest_tool_from_id(case_set_id)}"
            )
        # refresh
        cache[case_set_id] = cache[case_set_id]

        # compute and cache raw expression values
        case_ids = cache.get(case_set_id)
        response = requests.post(
            url=f"{GDC_API}/analysis/survival",
            json={
                "filters": {
                    "op": "=",
                    "content": {"field": "cases.case_id", "value": case_ids},
                },
            },
        )
        response.raise_for_status()

        return response.json()

    return get_survival_data
