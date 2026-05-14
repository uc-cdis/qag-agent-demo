import argparse
import json
import sys
from textwrap import dedent

import requests
from _defines import (
    AAChange,
    CaseCount,
    CaseSetId,
    CNVChange,
    CohortFilterID,
    DataType,
    Gene,
    MSIStatus,
    Project,
)
from _utils import gdc_query_all, get_tool_id_template, suggest_tool_from_id
from cachetools import TTLCache
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

cache = TTLCache(maxsize=1000, ttl=10800)


def get_simple_somatic_mutation_occurrences(
    gene: Gene,
    aa_change: AAChange = None,
    cohort_filter: CohortFilterID = None,
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


def get_microsatellite_instability_occurrences(
    msi_status: MSIStatus,
    cohort_filter: CohortFilterID = None,
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


def get_cases_with_available_data(
    data_type: DataType, cohort_filter: CohortFilterID
) -> CaseSetId:
    """
    A tool to retrieve cases with available data for a given genomic variant data type, for example 'ssm'.
    This tool should generally be used with a cohort filter as data availability without a specific cohort can be slow to compute.
    Some user queries may necessarily require computing data availability without a specific cohort, though this should be avoided if possible.
    The resulting case set is cached server side and can be referenced using the unique identifier returned by this tool.
    """
    print(
        f"get_cases_with_available_data(data_type={repr(data_type)}, cohort_filter={repr(cohort_filter)})",
        file=sys.stderr,
    )

    cache_id = get_tool_id_template().format(
        data_type=data_type, cohort_filter=cohort_filter
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

    if data_type in ["ssm", "cnv"]:
        hits = gdc_query_all(
            endpoint="cases",
            filters={
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": [
                        (
                            "Aggregated Somatic Mutation"
                            if data_type == "ssm"
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
    elif data_type in ["msi"]:
        _msi_ids = get_microsatellite_instability_occurrences("msi", cohort_filter)
        _mss_ids = get_microsatellite_instability_occurrences("mss", cohort_filter)
        _ms_any_ids = compute_case_union(_msi_ids, _mss_ids)
        cache[cache_id] = cache[_ms_any_ids]
    else:
        raise ToolError(
            f"Unknown how to find available cases for given data type: {data_type}"
        )

    return cache_id


def get_cohort_filter_by_project(project: Project) -> CohortFilterID:
    """
    A tool to construct a GDC cohort filter by project, for example 'TCGA-BRCA'.
    The resulting JSON filter is cached server side and can be referenced using the unique identifier returned by this tool.
    """
    print(
        f"get_cohort_filter_by_project(project={repr(project)})",
        file=sys.stderr,
    )

    cache_id = get_tool_id_template().format(project=project)

    # if we've already created this filter, refresh it in the cache and shortcut return
    if cache_id in cache:
        cache[cache_id] = cache[cache_id]
        return cache_id

    filter_str = json.dumps(
        {
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": [project],
            },
        },
        indent=4,
    )

    cache[cache_id] = filter_str

    return cache_id


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


def make_generate_filter_v2(backend):
    def generate_filter_v2(query: str) -> str:
        """
        Converts a free text description of a cancer cohort into a GDC structured cohort filter.

        Args:
            query (str): The free text cohort description

        Returns:
            str: JSON structured GDC cohort filter
        """

        resp = requests.post(
            "http://localhost:8100/invoke",
            json={"query": query, "backend": backend},
        )
        cohort_filter = json.dumps(resp.json()["filter"], indent=4)

        return cohort_filter

    return generate_filter_v2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--transport",
        required=True,
        choices=["sse", "stdio", "streamable-http"],
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
    )
    args = parser.parse_args()

    mcp = FastMCP(
        name="GDC API MCP Server",
        instructions=dedent(
            """
            Use the tools of the GDC MCP server to answer questions about genomic variant statistics from data in the NCI's Genomic Data Commons.

            ## Terminology & Disambiguation

            Because these tools are meant to answer highly specific biological questions, we need to be clear on terminology.
            You must clarify any ambiguous terms before proceeding in order to provide accurate answers.
            **Strictly follow these rules:**
            - Do NOT assume that 'mutation' means simple somatic mutations.
            - Do NOT assume that 'somatic mutation' means simple somatic mutations.
            - You should never assume what the user means by 'mutation' or 'somatic mutation'.
            - If the user's meaning of 'mutation' or 'somatic mutation' is unclear, you must stop and ask the user to specify if they mean simple somatic mutation (SSM), copy number variation (CNV), or both.
            - Even if the user asks for 'mutations in gene X', this is still ambiguous as the user could mean CNV in that gene and you must stop and clarify what the user means.

            ## Architecture & Mental Model

            These tools query and operate over case sets.
            The raw case sets are never returned to the client, instead case sets are cached on the server and referenced by a unique identifier returned by the tools.

            ## Tool Categories & Workflow

            The tools are broadly grouped into three categories: querying case sets, set operations over case sets, and returning information about case sets.
            The general workflow is as follows:
            1. use the query tools to select case sets
            2. use the set manipulation tools to combine case sets
            3. use the information tools to get details about the curated case sets

            ## General Guidance

            - Currently, we support querying case sets by simple somatic mutation, copy number variant, microsatellite instability, and by GDC project.
            - Case counts can be returned for any retrieved or computed case sets.
            - When a user asks for a statistics of a specific type of data, you must report the metric relative to cases with available data for the given data type. For example, if a user asks for 'prevalence of SSM of gene X', you should report 'cases with SSM in gene X / cases with any SSM'.
            """
        ),
        host="0.0.0.0",
        port=args.port,
    )

    mcp.add_tool(get_simple_somatic_mutation_occurrences)
    mcp.add_tool(get_copy_number_variant_occurrences)
    mcp.add_tool(get_microsatellite_instability_occurrences)
    mcp.add_tool(get_cohort_filter_by_project)
    mcp.add_tool(get_cases_by_cohort_filter)
    mcp.add_tool(compute_case_intersection)
    mcp.add_tool(compute_case_union)
    mcp.add_tool(get_case_set_size)
    mcp.add_tool(get_cases_with_available_data)

    mcp.run(transport=args.transport)
