import sys

from cachetools import TTLCache
from mcp.server.fastmcp.exceptions import ToolError

# isort: off
from _defines import AAChange, CaseSetId, CohortFilterID, Gene
from _utils import gdc_query_all, get_tool_id_template, suggest_tool_from_id

# isort: on


def make_ssm_tool(cache: TTLCache):
    def get_simple_somatic_mutation_occurrences(
        gene: Gene,
        aa_change: AAChange | None = None,
        cohort_filter: CohortFilterID | None = None,
    ) -> CaseSetId:
        """
        A tool to query the GDC API for cases with simple somatic mutations (SSMs).
        This tool searches for SSMs within a given gene, optionally resulting in an amino acid (AA) change, optionally within a specific cohort.
        It is an error to use this tool without a specific gene.
        The resulting case set is cached server side and can be referenced using the unique identifier returned by this tool.
        """
        print(
            f"get_simple_somatic_mutation_occurrences(gene={repr(gene)}, aa_change={repr(aa_change)}, cohort_filter={repr(cohort_filter)})",
            file=sys.stderr,
        )

        if gene is None:
            raise ToolError(
                f"Cannot retrieve simple somatic mutation cases without a specific gene"
            )

        cache_id = get_tool_id_template().format(
            gene=gene, aa_change=aa_change, cohort_filter=cohort_filter
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

        # first get SSMs for the given query parameters, either by gene & AA change or just gene
        if aa_change is not None:
            _filters = {
                "op": "in",
                "content": {
                    "field": "gene_aa_change",
                    "value": [f"{gene} {aa_change}"],  # e.g. 'BRAF V600E'
                },
            }
        else:
            _filters = {
                "op": "in",
                "content": {
                    "field": "consequence.transcript.gene.symbol",
                    "value": [gene],
                },
            }
        hits = gdc_query_all(
            endpoint="ssms",
            filters=_filters,
            cohort_filter=_cohort_filter,
            fields=["gene_aa_change", "ssm_id"],
        )

        # then get cases matching those SSMs
        ssm_ids = [hit["ssm_id"] for hit in hits]
        ssm_id_filter = {
            "op": "in",
            "content": {
                "field": "ssm.ssm_id",
                "value": ssm_ids,
            },
        }
        hits = gdc_query_all(
            endpoint="ssm_occurrences",
            filters=ssm_id_filter,
            cohort_filter=_cohort_filter,
            fields=["ssm.ssm_id", "case.submitter_id", "case.case_id"],
        )
        case_ids = set([hit["case"]["case_id"] for hit in hits])
        cache[cache_id] = list(case_ids)

        return cache_id

    return get_simple_somatic_mutation_occurrences
