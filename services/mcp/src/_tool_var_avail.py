import sys

from cachetools import TTLCache
from mcp.server.fastmcp.exceptions import ToolError

# isort: off
from _defines import CaseSetId, VariantType, CohortFilterID
from _utils import gdc_query_all, get_tool_id_template, suggest_tool_from_id

# isort: on


def make_variant_case_availability_tool(
    *,
    cache: TTLCache,
    compute_case_union,
    get_microsatellite_instability_occurrences,
):
    def get_cases_with_available_variant_data(
        variant_type: VariantType,
        cohort_filter: CohortFilterID,
    ) -> CaseSetId:
        """
        A tool to retrieve cases with available data for a given genomic variant data type, for example 'ssm'.
        This tool should generally be used with a cohort filter as data availability without a specific cohort can be slow to compute.
        Some user queries may necessarily require computing data availability without a specific cohort, though this should be avoided if possible.
        The resulting case set is cached server side and can be referenced using the unique identifier returned by this tool.
        """
        print(
            f"get_cases_with_available_variant_data(variant_type={repr(variant_type)}, cohort_filter={repr(cohort_filter)})",
            file=sys.stderr,
        )

        cache_id = get_tool_id_template().format(
            variant_type=variant_type, cohort_filter=cohort_filter
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

        if variant_type in ["ssm", "cnv"]:
            hits = gdc_query_all(
                endpoint="cases",
                filters={
                    "op": "in",
                    "content": {
                        "field": "files.data_type",
                        "value": [
                            (
                                "Aggregated Somatic Mutation"
                                if variant_type == "ssm"
                                else "Allele-specific Copy Number Segment"
                            )
                        ],
                    },
                },
                cohort_filter=_cohort_filter,
                fields=["project.project_id", "submitter_id"],
            )
            case_ids = set([hit["id"] for hit in hits])
            cache[cache_id] = list(case_ids)
        elif variant_type in ["msi"]:
            _msi_ids = get_microsatellite_instability_occurrences("msi", cohort_filter)
            _mss_ids = get_microsatellite_instability_occurrences("mss", cohort_filter)
            _ms_any_ids = compute_case_union(_msi_ids, _mss_ids)
            cache[cache_id] = cache[_ms_any_ids]
        else:
            raise ToolError(
                f"Unknown how to find available cases for given variant type: {variant_type}"
            )

        return cache_id

    return get_cases_with_available_variant_data
