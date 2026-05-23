import sys

from cachetools import TTLCache
from mcp.server.fastmcp.exceptions import ToolError

# isort: off
from _defines import CaseSetId, CohortFilterID, GenomicDataType
from _utils import gdc_query_all, get_tool_id_template, suggest_tool_from_id

# isort: on


def make_case_availability_tool(
    *,
    cache: TTLCache,
    compute_case_union,
    get_microsatellite_instability_occurrences,
):
    def get_cases_with_available_genomic_data(
        genomic_data_type: GenomicDataType,
        cohort_filter: CohortFilterID,
    ) -> CaseSetId:
        """
        A tool to retrieve cases with available data for a given genomic data type, for example 'ssm'.
        This tool should always be used with a cohort filter, as data availability without a specific cohort can be slow to compute.
        Some user queries may necessarily require computing data availability over all cases.
        While such queries should be avoid if possible, if you absolutely must compute availability over all cases, create an empty cohort filter first.
        The resulting case set is cached server side and can be referenced using the unique identifier returned by this tool.
        """
        print(
            f"get_cases_with_available_genomic_data(genomic_data_type={repr(genomic_data_type)}, cohort_filter={repr(cohort_filter)})",
            file=sys.stderr,
        )

        cache_id = get_tool_id_template().format(
            genomic_data_type=genomic_data_type, cohort_filter=cohort_filter
        )

        # if we've already done this retrieval, refresh it in the cache and shortcut return
        if cache_id in cache:
            cache[cache_id] = cache[cache_id]
            return cache_id

        # check cohort filters is in cache
        if cohort_filter not in cache:
            raise ToolError(
                f"Cohort filter {cohort_filter} was not found in the server side cache, perhaps it expired? "
                f"Try recreating that cohort filter to cache it again. {suggest_tool_from_id(cohort_filter)}"
            )
        # refresh
        cohort_filter = cache[cohort_filter]
        cache[cohort_filter] = cohort_filter

        if genomic_data_type == "ssm":
            hits = gdc_query_all(
                endpoint="case_ssms",
                filters={
                    "op": "in",
                    "content": {"field": "available_variation_data", "value": ["ssm"]},
                },
                cohort_filter=cohort_filter,
                fields=["project.project_id", "submitter_id"],
            )
            case_ids = set([hit["id"] for hit in hits])
            cache[cache_id] = list(case_ids)
        elif genomic_data_type == "cnv" or genomic_data_type == "gene_expression":
            data_type_value = {
                "cnv": "Allele-specific Copy Number Segment",
                "gene_expression": "Gene Expression Quantification",
            }[genomic_data_type]
            hits = gdc_query_all(
                endpoint="cases",
                filters={
                    "op": "and",
                    "content": [
                        {
                            "op": "in",
                            "content": {
                                "field": "files.data_type",
                                "value": [data_type_value],
                            },
                        },
                        {
                            "op": "in",
                            "content": {
                                "field": "files.access",
                                "value": ["open"],
                            },
                        },
                    ],
                },
                cohort_filter=cohort_filter,
                fields=["project.project_id", "submitter_id"],
            )
            case_ids = set([hit["id"] for hit in hits])
            cache[cache_id] = list(case_ids)
        elif genomic_data_type == "msi":
            _msi_ids = get_microsatellite_instability_occurrences(cohort_filter, "msi")
            _mss_ids = get_microsatellite_instability_occurrences(cohort_filter, "mss")
            _ms_any_ids = compute_case_union(_msi_ids, _mss_ids)
            cache[cache_id] = cache[_ms_any_ids]
        else:
            raise ToolError(
                f"Unknown how to find available cases for given genomic data type: {genomic_data_type}"
            )

        return cache_id

    return get_cases_with_available_genomic_data
