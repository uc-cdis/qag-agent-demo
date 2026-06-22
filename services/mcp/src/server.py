import argparse
from textwrap import dedent

from cachetools import TTLCache
from mcp.server.fastmcp import FastMCP

# fmt: off
# isort: off
from _tool_const_filt import make_construct_filter_tool
from _tool_case_handling import make_case_handling_tools
from _tool_avail import make_case_availability_tool
from _tool_var_cnv import make_cnv_tool
from _tool_var_msi import make_msi_tool
from _tool_var_ssm import make_ssm_tool
from _tool_gene_expr import make_gene_expression_tools
# isort: on
# fmt: on

cache = TTLCache(maxsize=1000, ttl=10800)

construct_cohort_filter = make_construct_filter_tool(cache)
(
    get_cases_by_cohort_filter,
    compute_case_intersection,
    compute_case_union,
    get_case_set_size,
) = make_case_handling_tools(cache)
get_simple_somatic_mutation_occurrences = make_ssm_tool(cache)
get_copy_number_variant_occurrences = make_cnv_tool(cache)
get_microsatellite_instability_occurrences = make_msi_tool(cache)
get_cases_with_available_genomic_data = make_case_availability_tool(
    cache=cache,
    compute_case_union=compute_case_union,
    get_microsatellite_instability_occurrences=get_microsatellite_instability_occurrences,
)
get_gene_expression_statistics, get_most_variably_expressed_genes = (
    make_gene_expression_tools(cache)
)


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
            - For questions involving single molecular alterations, you must report the metric relative to cases with available data for that type of molecular alteration. For example, if a user asks about the frequency of molecular alteration A in gene X, compute: '(cases with alteration A in gene X) / (cases with data available for that type of molecular alteration)'
            - For questions involving multiple molecular alterations, report metrics relative to the number of cases with available data for the alterations. For example, if a user asks about the frequency of molecular alteration A or alteration B in gene X, compute: '(cases with alteration A or B in gene X) / (cases with data available for alteration A or B or both)'.

            """
        ),
        host="0.0.0.0",
        port=args.port,
    )

    mcp.add_tool(construct_cohort_filter)
    mcp.add_tool(compute_case_intersection)
    mcp.add_tool(compute_case_union)
    mcp.add_tool(get_case_set_size)
    mcp.add_tool(get_cases_by_cohort_filter)
    mcp.add_tool(get_simple_somatic_mutation_occurrences)
    mcp.add_tool(get_copy_number_variant_occurrences)
    mcp.add_tool(get_microsatellite_instability_occurrences)
    mcp.add_tool(get_cases_with_available_genomic_data)
    mcp.add_tool(get_gene_expression_statistics)
    mcp.add_tool(get_most_variably_expressed_genes)

    mcp.run(transport=args.transport)
