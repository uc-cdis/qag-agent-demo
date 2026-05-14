import sys

from cachetools import TTLCache
from mcp.server.fastmcp.exceptions import ToolError

# isort: off
from _defines import CaseSetId, CohortFilterID, MSIStatus
from _utils import gdc_query_all, get_tool_id_template, suggest_tool_from_id

# isort: on


def make_msi_tool(cache: TTLCache):
    def get_microsatellite_instability_occurrences(
        msi_status: MSIStatus,
        cohort_filter: CohortFilterID | None = None,
    ) -> CaseSetId:
        """
        A tool to query the GDC API for cases with microsatellite instability (MSI).
        This tool searches for either MSI or MSS status, optionally within a specific cohort.
        The resulting case set is cached server side and can be referenced using the unique identifier returned by this tool.
        """
        print(
            f"get_microsatellite_instability_occurrences(msi_status={repr(msi_status)}, cohort_filter={repr(cohort_filter)})",
            file=sys.stderr,
        )

        cache_id = get_tool_id_template().format(
            msi_status=msi_status, cohort_filter=cohort_filter
        )

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

        # MSI is stored at the file level, so we need to query the files and aggregate back up to the cases
        hits = gdc_query_all(
            endpoint="files",
            filters={
                "op": "and",
                "content": [
                    {
                        "op": "in",
                        "content": {
                            "field": "msi_status",
                            "value": [msi_status],
                        },
                    },
                    {"op": "in", "content": {"field": "data_format", "value": ["BAM"]}},
                ],
            },
            cohort_filter=_cohort_filter,
            fields=[
                "msi_status",
                "msi_score",
                "cases.submitter_id",
                "cases.case_id",
            ],
        )

        case_ids = set()
        for hit in hits:
            for case in hit.get("cases", []):
                case_ids.add(case["case_id"])
        cache[cache_id] = list(case_ids)

        return cache_id

    return get_microsatellite_instability_occurrences
