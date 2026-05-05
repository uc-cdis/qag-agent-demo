import argparse
import json
import sys
from io import StringIO
from textwrap import dedent
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from cachetools import TTLCache
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from ._defines import (
    TOOL_CACHE_ID_TEMPLATES,
    AAChange,
    CaseCount,
    CaseSetId,
    CNVChange,
    CohortDescription,
    CohortFilterID,
    DataFrameDescribeOutput,
    DataType,
    DiseaseType,
    ExpressionSetId,
    Gene,
    MSIStatus,
    PrimarySite,
    Project,
    Sex,
    SmokingStatus,
    TSVUnit,
)
from ._utils import (
    AverageGeneExpressionResult,
    GeneExpressionAverage,
    GeneExpressionVariable,
    VariableGeneExpressionResult,
    gdc_query_all,
    get_ensembl_gene_ids,
    get_tool_id_template,
    suggest_tool_from_id,
)

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


def get_gene_expression_occurances(
    genes: List[Gene],
    cohort_filter: CohortFilterID,
) -> CaseSetId:
    """
    A tool to query the GDC API for cases with gene expression data.
    This tool searches for availability of gene expression data for a gene or set of genes, optionally within a specific cohort.
    The resulting case set is cached server side and can be referenced using the unique identifier returned by this tool.
    """
    print(
        f"get_gene_expression_occurances(genes={repr(genes)}, cohort_filter={repr(cohort_filter)})",
        file=sys.stderr,
    )

    cache_id = get_tool_id_template().format(genes=genes, cohort_filter=cohort_filter)

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

    # first convert the gene symbol to ENSEMBL id and then retrieve the available case set and cache
    gene_ids = get_ensembl_gene_ids(genes)
    cases_cache_id = get_cases_by_cohort_filter(cohort_filter)
    cases = cache.get(cases_cache_id)
    hits = requests.post(
        url="https://api.gdc.cancer.gov/gene_expression/availability",
        json={
            "case_ids": cases,
            "gene_ids": gene_ids,
        },
    )

    case_ids = set()
    for case in hits.json().get("cases", []).get("details", []):
        case_ids.add(case["case_id"])
    cache[cache_id] = list(case_ids)

    return cache_id


def get_gene_expression_values(
    cases: CaseSetId,
    genes: List[Gene],
    tsv_unit: TSVUnit,
) -> ExpressionSetId:
    """
    A tool to query the GDC API for gene expression values given a gene or set of genes.
    The resulting expression set is cached server side and can be referenced using identifier returned by this tool
    """
    if genes is None:
        raise ToolError(
            f"Cannot retrieve cases without either a specific gene or genes"
        )
    print(
        f"get_gene_expression_values(cases={repr(cases)}, genes={repr(genes)}, tsv_unit={repr(tsv_unit)})",
        file=sys.stderr,
    )

    cache_id = get_tool_id_template().format(
        cases=cases, genes=genes, tsv_unit=tsv_unit
    )

    # if we've already done this retrieval, refresh it in the cache and shortcut return
    if cache_id in cache:
        cache[cache_id] = cache[cache_id]
        return cache_id

    case_ids = cache.get(cases)

    # Compute Raw Expression values
    url = "https://api.gdc.cancer.gov/gene_expression/values"
    headers = {
        "accept": "text/tab-separated-values",
        "Content-Type": "application/json",
    }
    gene_ids = get_ensembl_gene_ids(genes)
    payload = {
        "case_ids": case_ids,
        "gene_ids": gene_ids,
        "tsv_units": tsv_unit,
        "format": "tsv",
    }
    response = requests.post(url, headers=headers, json=payload)
    cache[cache_id] = response.text

    return cache_id


def get_gene_expression_statistics(
    expressions: ExpressionSetId,
) -> Dict:
    """
    Computes basic gene expression statistics on a set of cases and set of genes.
    """
    print(
        f"get_gene_expression_statistics(expressions={repr(expressions)})",
        file=sys.stderr,
    )

    # extract the expression values and
    expression_values = cache[expressions]
    df = pd.read_csv(StringIO(expression_values), sep="\t")
    df = df.set_index("gene_id")
    summary = {}

    for gene, row in df.iterrows():
        values = pd.to_numeric(row, errors="coerce")

        non_missing = values.dropna()

        if len(non_missing) == 0:
            summary[str(gene)] = {
                "count": 0,
                "missing": int(values.isna().sum()),
                "mean": None,
                "std": None,
                "min": None,
                "p25": None,
                "median": None,
                "p75": None,
                "max": None,
            }
            continue

        summary[str(gene)] = {
            "count": int(non_missing.count()),
            "missing": int(values.isna().sum()),
            "mean": non_missing.mean(),
            "std": non_missing.std(),
            "min": non_missing.min(),
            "p25": non_missing.quantile(0.25),
            "median": non_missing.median(),
            "p75": non_missing.quantile(0.75),
            "max": non_missing.max(),
        }

    return summary


def get_average_gene_expression(
    cases: CaseSetId,
    genes: List[Gene],
    tsv_unit: TSVUnit,
) -> AverageGeneExpressionResult:
    """
    A tool to query GDC API for gene expression values given a gene or set of genes and return basic statistics.
    The result describes the gene expression statistics of the current case and gene set combinations.
    """
    if genes is None:
        raise ToolError(
            f"Cannot retrieve cases without either a specific gene, or genes"
        )

    case_ids = cache.get(cases)
    # Compute Raw Expression values
    url = "https://api.gdc.cancer.gov/gene_expression/values"
    headers = {
        "accept": "text/tab-separated-values",
        "Content-Type": "application/json",
    }
    gene_ids = get_ensembl_gene_ids(genes)
    payload = {
        "case_ids": case_ids,
        "gene_ids": gene_ids,
        "tsv_units": tsv_unit,
        "format": "tsv",
    }
    response = requests.post(url, headers=headers, json=payload)
    df = pd.read_csv(StringIO(response.text), sep="\t")
    # print(df.head())
    avg_gene_expr = df.set_index("gene_id").mean(axis=1)
    result = []
    for id, val in avg_gene_expr.items():
        result.append(
            GeneExpressionAverage(
                gene_id=str(id),
                avg_expression=val,
            )
        )

    return AverageGeneExpressionResult(genes=result)


def get_variable_gene_expression(
    cases: CaseSetId,
    genes: List[Gene],
    selection_size: int = 1,
) -> VariableGeneExpressionResult:
    """
    A tool to query GDC API for variable gene expression result given a gene or set of genes such as for 'most variably expressed gene in a certain cancer'
    The result is the variable gene expression.
    """
    if genes is None:
        raise ToolError(
            f"Cannot retrieve cases without either a specific gene, or genes"
        )

    case_ids = cache.get(cases)
    # Compute Raw Expression values
    url = "https://api.gdc.cancer.gov/gene_expression/gene_selection"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    gene_ids = get_ensembl_gene_ids(genes)
    payload = {
        "case_ids": case_ids,
        "gene_ids": gene_ids,
        "selection_size": selection_size,
    }
    response = requests.post(url, headers=headers, json=payload)
    print(response)
    result = []
    for item in response.json().get("gene_selection", []):
        result.append(
            GeneExpressionVariable(
                gene_id=item.get("gene_id", ""),
                symbol=item.get("symbol", ""),
                log2_uqfpkm_median=item.get("log2_uqfpkm_median", ""),
                log2_uqfpkm_stddev=item.get("log2_uqfpkm_stddev", ""),
            )
        )

    return VariableGeneExpressionResult(selections=result)


def get_cases_with_available_data(
    data_type: DataType,
    cohort_filter: CohortFilterID,
    genes: Optional[List[Gene]],
) -> CaseSetId:
    """
    A tool to retrieve cases with available data for a given genomic variant data type, for example 'ssm' or 'gene_expression'.
    This tool should generally be used with a cohort filter as data availability without a specific cohort can be slow to compute.
    Some user queries may necessarily require computing data availability without a specific cohort, though this should be avoided if possible.
    The resulting case set is cached server side and can be referenced using the unique identifier returned by this tool.
    """
    print(
        f"get_cases_with_available_data(data_type={repr(data_type)}, cohort_filter={repr(cohort_filter)}, genes={repr(genes)})",
        file=sys.stderr,
    )

    cache_id = get_tool_id_template().format(
        data_type=data_type, cohort_filter=cohort_filter, genes=genes
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
    elif data_type in ["gene_expression"]:
        if genes is not None:
            _gene_expr_ids = get_gene_expression_occurances(genes, cohort_filter)
            cache[cache_id] = cache[_gene_expr_ids]
        else:
            print(
                f"Error, requires a gene or set of genes to compute gene expression occurances"
            )
    else:
        raise ToolError(
            f"Unknown how to find available cases for given data type: {data_type}"
        )

    return cache_id

    # def get_cohort_filter_by_project(project: List[Project]) -> CohortFilterID:
    """
    A tool to construct a GDC cohort filter by project or multiple projects, for example 'TCGA-BRCA'.
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
                "value": project,
            },
        },
        indent=4,
    )

    cache[cache_id] = filter_str

    return cache_id


def get_cohort_filter_by_project(
    project: List[Project] = None,
    primary_site: List[PrimarySite] = None,
    disease_type: List[DiseaseType] = None,
    sex: List[Sex] = None,
    min_age: int = 0,
    max_age: int = 90,
    smoking_status: List[SmokingStatus] = None,
) -> CohortFilterID:
    """
    A tool to construct a GDC cohort filter by by project and other optional additional fields.
    This tool only creates the filter str, does not execute the API request. Pass the resulting CohortFilterID either into 'get_cases_with_cohort_filter'
    or 'get_cases_with_available_data'.
    The resulting JSON filter is cached server side and can be referenced using the unique identifier returned by this tool.
    Args:
        project: GDC project IDs. e.g ["TCGA-BRCA", "TCGA-LUAD"]
        primary_site: Primary site of a disease e.g. ["Lung", "Brain"]
        disease_type: Type of malignant disease e.g. ["adenomas and adenocarcinomas", "leukemia, nos"]
        sex: A person's sex at birth e.g. any / many of ["male", "female", "unknown"],
        min_age : Lower end of the age range. Age at the time of diagnosis, expressed in years. e.g 34
        max_age : Higher end of the age range. Age at the time of diagnosis, expressed in years. e.g 76
        smoking_status: Self reported current smoking status or smoking history of a patient. e.g. ["lifelong non-smoker", "current smoker"]
    """

    print(
        f"get_cohort_filter_by_fields(project={repr(project)}, primary_site={repr(primary_site)}, disease_type={repr(disease_type)}, sex={repr(sex)}, min_age={repr(min_age)}, max_age={repr(max_age)}, smoking_status={repr(smoking_status)})",
        file=sys.stderr,
    )

    cache_id = get_tool_id_template().format(
        project=project,
        primary_site=primary_site,
        disease_type=disease_type,
        sex=sex,
        min_age=min_age,
        max_age=max_age,
        smoking_status=smoking_status,
    )

    # if we've already created this filter, refresh it in the cache and shortcut return
    if cache_id in cache:
        cache[cache_id] = cache[cache_id]
        return cache_id

    field_dict = {
        "project": ("in", "cases.project.project_id"),
        "primary_site": ("in", "cases.primary_site"),
        "disease_type": ("in", "cases.disease_type"),
        "sex": ("in", "cases.demographic.sex_at_birth"),
        "min_age": (">=", "cases.diagnoses.age_at_diagnosis"),
        "max_age": ("<=", "cases.diagnoses.age_at_diagnosis"),
        "smoking_status": ("in", "cases.exposures.tobacco_smoking_status"),
    }

    value_dict = {
        "project": project,
        "primary_site": primary_site,
        "disease_type": disease_type,
        "sex": sex,
        "min_age": int(min_age * 365),
        "max_age": int(max_age * 365),
        "smoking_status": smoking_status,
    }

    filters = []
    for key, value in value_dict.items():
        if value is not None:
            filters.append(
                {
                    "op": field_dict[key][0],
                    "content": {
                        "field": field_dict[key][1],
                        "value": value,
                    },
                }
            )
    if len(filters) == 1:
        filter_str = json.dumps(filters[0], indent=4)
    else:
        filter_str = json.dumps({"op": "and", "content": filters}, indent=4)
    cache[cache_id] = filter_str

    print(filter_str)
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


def make_cohort_copilot_tool(generate_filter):
    # The method to generate_filter should be modular so that improvements to cohort copilot
    # can be reflected simply by using a revised implementation of generate_filter
    def get_cohort_filter_by_cohort_description(
        cohort_description: CohortDescription,
    ) -> CohortFilterID:
        """
        A tool to construct a GDC cohort filter of a described cohort, for example 'cases from the TCGA-BRCA project' or 'cases of non-smoking patients'.
        The resulting JSON filter is cached server side and can be referenced using the unique identifier returned by this tool.
        While this tool can take natural language inputs, it is best used with very simple, single attribute descriptions.
        Do NOT provide gene filters to this tool.
        """
        print(
            f"get_cohort_filter_by_cohort_description(cohort_description={repr(cohort_description)})",
            file=sys.stderr,
        )

        cache_id = get_tool_id_template().format(cohort_description=cohort_description)

        # if we've already created this filter, refresh it in the cache and shortcut return
        if cache_id in cache:
            cache[cache_id] = cache[cache_id]
            return cache_id

        filter_str = generate_filter(cohort_description)

        cache[cache_id] = filter_str

        return cache_id

    return get_cohort_filter_by_cohort_description


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
    parser.add_argument(
        "--use-cohort-copilot",
        default=False,
        action="store_true",
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
            - For questions regarding gene expression always clarify which genes are being asked / discussed, do not assume all genes in GDC
            - Additionally, for gene expression questions that are asking for average expression always clarify the 'tsv_units' preference for the user

            ## Architecture & Mental Model

            These tools query and operate over case sets.
            The raw case sets are never returned to the client, instead case sets are cached on the server and referenced by a unique identifier returned by the tools.


            ## Tool Categories & Workflow

            The tools are broadly grouped into three categories: querying case sets, set operations over case sets, and returning information about case sets.
            The general workflow is as follows:
            1. use the query tools to select case sets
            2. use the set manipulation tools to combine case sets
            3. use the information tools to get details about the curated case sets

            Before calling a tool, state:
            - Tool name: exactly as written
            - Parameters: list each as "name (type): value"

            ## General Guidance

            - Currently, we support querying case sets by simple somatic mutation, copy number variant, microsatellite instability, gene expression and by {cohort_tool_type}.
            - Case counts can be returned for any retrieved or computed case sets.
            - When a user asks for a statistics of a specific type of data, you must report the metric relative to cases with available data for the given data type. For example, if a user asks for 'prevalence of SSM of gene X', you should report 'cases with SSM in gene X / cases with any SSM'.
            """.format(
                cohort_tool_type=(
                    "cohort description" if args.use_cohort_copilot else "GDC Project"
                )
            )
        ),
        port=args.port,
    )

    mcp.add_tool(get_simple_somatic_mutation_occurrences)
    mcp.add_tool(get_copy_number_variant_occurrences)
    mcp.add_tool(get_microsatellite_instability_occurrences)
    # mcp.add_tool(get_average_gene_expression)
    mcp.add_tool(get_gene_expression_values)
    mcp.add_tool(get_gene_expression_statistics)
    mcp.add_tool(get_variable_gene_expression)
    if args.use_cohort_copilot:
        # NOTE: with cohort copilot v1, importing will load a model onto GPU so defer until needed
        # from .cohort_copilot import generate_filter

        # NOTE: with cohort copilot v2, we can serve with different backends
        # however, when integrating with this QAG MCP server, the agent backend using this QAG MCP
        # can be different from the backend driving copilot v2
        # NOTE: for the sake of the demo, we'd like the backends to be unified so there's
        # some logic on the agent to switch which tool is available, depending on the agent's backend
        get_cohort_filter_by_cohort_description = make_cohort_copilot_tool(
            # cohort copilot v2 with claude backend
            make_generate_filter_v2("anthropic"),
        )
        mcp.add_tool(
            get_cohort_filter_by_cohort_description,
            name="get_cohort_filter_by_cohort_description",
        )

        # NOTE we have a hacky way to separate out the nearly identical versions of copilot v2
        # while preserving the natural language naming scheme
        get_cohort_filter_by_cohort_description = make_cohort_copilot_tool(
            # cohort copilot v2 with local backend
            make_generate_filter_v2("openai"),
        )
        mcp.add_tool(
            get_cohort_filter_by_cohort_description,
            name="generate_cohort_filter_using_cohort_description",
        )
    else:
        mcp.add_tool(get_cohort_filter_by_project)

    mcp.add_tool(get_cases_by_cohort_filter)
    mcp.add_tool(compute_case_intersection)
    mcp.add_tool(compute_case_union)
    mcp.add_tool(get_case_set_size)
    mcp.add_tool(get_cases_with_available_data)

    mcp.run(transport=args.transport)
