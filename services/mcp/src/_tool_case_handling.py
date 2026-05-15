import sys

from cachetools import TTLCache
from mcp.server.fastmcp.exceptions import ToolError

# isort: off
from _defines import CaseCount, CaseSetId, CohortFilterID
from _utils import gdc_query_all, get_tool_id_template, suggest_tool_from_id

# isort: on


def make_case_handling_tools(cache: TTLCache):
    def get_cases_by_cohort_filter(cohort_filter: CohortFilterID) -> CaseSetId:
        """
        A tool to query the GDC API for cases matching a given cohort filter.
        The resulting case set is cached server side and can be referenced using the unique identifier returned by this tool.
        """
        print(
            f"get_cases_by_filter(cohort_filter={repr(cohort_filter)})",
            file=sys.stderr,
        )

        cache_id = get_tool_id_template().format(cohort_filter=cohort_filter)

        # if we've already done this retrieval, refresh it in the cache and shortcut return
        if cache_id in cache:
            cache[cache_id] = cache[cache_id]
            return cache_id

        # check cohort filters is in cache
        _cohort_filter = None
        if cohort_filter is not None:
            if cohort_filter not in cache:
                raise ToolError(
                    f"Cohort filter {cohort_filter} was not found in the server side cache, perhaps it expired? "
                    f"Try recreating that cohort filter to cache it again. {suggest_tool_from_id(cohort_filter)}"
                )
            # refresh
            _cohort_filter = cache[cohort_filter]
            cache[cohort_filter] = _cohort_filter

        hits = gdc_query_all(
            endpoint="cases",
            filters={},
            cohort_filter=_cohort_filter,
            fields=["project.project_id", "submitter_id"],
        )
        case_ids = set([hit["id"] for hit in hits])
        cache[cache_id] = list(case_ids)

        return cache_id

    def compute_case_intersection(
        case_set_id_A: CaseSetId,
        case_set_id_B: CaseSetId,
    ) -> CaseSetId:
        """
        A tool to compute the intersection between two case sets.
        The resulting intersected case set is cached server side and can be referenced using the unique identifier returned by this tool.
        """
        print(
            f"compute_case_intersection(case_set_id_A={repr(case_set_id_A)}, case_set_id_B={repr(case_set_id_B)})",
            file=sys.stderr,
        )

        cache_id = get_tool_id_template().format(
            case_set_id_A=case_set_id_A, case_set_id_B=case_set_id_B
        )

        # if we've already done this retrieval, refresh it in the cache and shortcut return
        if cache_id in cache:
            cache[cache_id] = cache[cache_id]
            return cache_id

        for case_set_id in [case_set_id_A, case_set_id_B]:
            if case_set_id not in cache:
                raise ToolError(
                    f"Case set {case_set_id} was not found in the server side cache, perhaps it expired? "
                    f"Try requerying for those cases to cache it again. {suggest_tool_from_id(case_set_id)}"
                )
            # refresh the subsets since we're using them
            cache[case_set_id] = cache[case_set_id]

        cases_A = set(cache[case_set_id_A])
        cases_B = set(cache[case_set_id_B])

        A_and_B = cases_A & cases_B
        cache[cache_id] = list(A_and_B)

        return cache_id

    def compute_case_union(
        case_set_id_A: CaseSetId,
        case_set_id_B: CaseSetId,
    ) -> CaseSetId:
        """
        A tool to compute the union between two case sets.
        The resulting unioned case set is cached server side and can be referenced using the unique identifier returned by this tool.
        """
        print(
            f"compute_case_union(case_set_id_A={repr(case_set_id_A)}, case_set_id_B={repr(case_set_id_B)})",
            file=sys.stderr,
        )

        cache_id = get_tool_id_template().format(
            case_set_id_A=case_set_id_A, case_set_id_B=case_set_id_B
        )

        # if we've already done this retrieval, refresh it in the cache and shortcut return
        if cache_id in cache:
            cache[cache_id] = cache[cache_id]
            return cache_id

        for case_set_id in [case_set_id_A, case_set_id_B]:
            if case_set_id not in cache:
                raise ToolError(
                    f"Case set {case_set_id} was not found in the server side cache, perhaps it expired? "
                    f"Try requerying for those cases to cache it again. {suggest_tool_from_id(case_set_id)}"
                )
            # refresh the subsets since we're using them
            cache[case_set_id] = cache[case_set_id]

        cases_A = set(cache[case_set_id_A])
        cases_B = set(cache[case_set_id_B])

        A_and_B = cases_A | cases_B
        cache[cache_id] = list(A_and_B)

        return cache_id

    def get_case_set_size(
        case_set_id: CaseSetId,
    ) -> CaseCount:
        """
        A tool to retireve the size of a precomputed case set.
        The case set must first be cached server side by one of the other tools provided by this MCP server.
        """
        print(
            f"get_case_set_size(case_set_id={repr(case_set_id)})",
            file=sys.stderr,
        )

        if case_set_id not in cache:
            raise ToolError(
                f"Case set {case_set_id} was not found in the server side cache, perhaps it expired? "
                f"Try requerying for those cases to cache it again. {suggest_tool_from_id(case_set_id)}"
            )

        # refresh the subsets since we're using them
        cache[case_set_id] = cache[case_set_id]

        return len(cache[case_set_id])

    return (
        get_cases_by_cohort_filter,
        compute_case_intersection,
        compute_case_union,
        get_case_set_size,
    )
