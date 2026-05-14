import sys

from cachetools import TTLCache
from mcp.server.fastmcp.exceptions import ToolError

# isort: off
from _defines import CNVChange, CaseSetId, CohortFilterID, Gene
from _utils import gdc_query_all, get_tool_id_template, suggest_tool_from_id

# isort: on


def make_cnv_tool(cache: TTLCache):
    def get_copy_number_variant_occurrences(
        gene: Gene = None,
        cnv_change: CNVChange = None,
        cohort_filter: CohortFilterID = None,
    ) -> CaseSetId:
        """
        A tool to query the GDC API for cases with copy number variations (CNVs).
        This tool searches for CNVs within a given gene, optionally for a specific change type, optionally within a specific cohort.
        If only the gene is specified, this tool retrieves all cases with any change type for the given gene.
        If only the change type is specified, this tool retrieves all cases with the given change type in any gene.
        It is an error to use this tool without at least one of gene or change type.
        The resulting case set is cached server side and can be referenced using the unique identifier returned by this tool.
        """
        print(
            f"get_copy_number_variant_occurrences(gene={repr(gene)}, cnv_change={repr(cnv_change)}, cohort_filter={repr(cohort_filter)})",
            file=sys.stderr,
        )

        if gene is None and cnv_change is None:
            raise ToolError(
                f"Cannot retrieve copy number variation cases without either a specific gene, a change type, or both"
            )

        cache_id = get_tool_id_template().format(
            gene=gene, cnv_change=cnv_change, cohort_filter=cohort_filter
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

        # apparently `heterozygous deletion` change type is `loss` so remap it
        # keep the enum for the LLM to help it disambiguate the options
        # see: https://github.com/uc-cdis/gdc-qag/blob/07af83ecae960bd6f62347c473268ff73e725325/check_6k_mutations_withTable2.py#L541-L559
        if cnv_change == "heterozygous deletion":
            cnv_change = "loss"  # type: ignore

        ops = []
        if gene is not None:
            ops.append(
                {
                    "op": "in",
                    "content": {
                        "field": "cnv.consequence.gene.symbol",
                        "value": [gene],
                    },
                },
            )
        if cnv_change is not None:
            ops.append(
                {
                    "op": "in",
                    "content": {
                        "field": "cnv.cnv_change_5_category",
                        "value": [cnv_change],
                    },
                }
            )

        hits = gdc_query_all(
            endpoint="cnv_occurrences",
            filters={
                "op": "and",
                "content": ops,
            },
            cohort_filter=_cohort_filter,
            fields=[
                "cnv.cnv_change",
                "cnv.cnv_change_5_category",
                "cnv.consequence.gene.symbol",
                "case.submitter_id",
                "case.case_id",
            ],
        )
        case_ids = set([hit["case"]["case_id"] for hit in hits])
        cache[cache_id] = list(case_ids)

        return cache_id

    return get_copy_number_variant_occurrences
