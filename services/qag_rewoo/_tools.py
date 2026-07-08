import asyncio
import sys
from collections.abc import Mapping
from io import StringIO
from typing import Annotated, Any, Literal

import pandas as pd
from _defines import *
from _utils import *
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError
from schema import GDCCohortSchema


def normalize_string_list(value: object) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    if isinstance(value, list):
        normalized: list[str] = []

        for item in value:
            if not isinstance(item, str):
                raise TypeError("Expected a string or a list of strings")

            item = item.strip()

            if item:
                normalized.append(item)

        return normalized

    raise TypeError("Expected a string, a list of strings, or None")


StringList = Annotated[
    list[str] | None,
    BeforeValidator(normalize_string_list),
]


class ConstructCohortFilterArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: StringList | None = None
    primary_site: StringList | None = None
    disease_type: StringList | None = None
    project_id: StringList | None = None
    sex: StringList | None = None
    smoking_status: StringList | None = None

    min_age: int | None = Field(default=None, ge=0)
    max_age: int | None = Field(default=None, ge=0)


class ResolverResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: list[str]


def make_construct_cohort_filter_tool(
    *,
    client: Any,
    model: str,
    resolver_prompt: str,
    field_registry: Mapping[str, str],
    value_registry: Mapping[str, Any],
):
    async def _construct(
        args: ConstructCohortFilterArgs,
    ) -> GDCCohortSchema:
        supplied_fields = {
            field: value
            for field, value in args.model_dump().items()
            if value is not None
        }

        resolved_fields: dict[str, Any] = {}

        for field, value in supplied_fields.items():
            if field in {"min_age", "max_age"}:
                resolved_fields[field] = int(value * 365)
                continue

            if field not in field_registry:
                raise ValueError(f"Unknown cohort field {field!r}")

            if field not in value_registry:
                raise ValueError(f"No allowed-value registry for {field!r}")

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": resolver_prompt.format(
                            source=value,
                            values=value_registry[field],
                        ),
                    }
                ],
                temperature=1.0,
                max_tokens=2048,
                extra_body={
                    "structured_outputs": {
                        "json": ResolverResponse.model_json_schema(),
                    }
                },
            )

            content = response.choices[0].message.content

            if content is None:
                raise ValueError(f"Resolver returned no value for {field!r}")

            resolver_result = ResolverResponse.model_validate_json(content)
            resolved_fields[field_registry[field]] = resolver_result.value

        sub_filters: list[dict[str, Any]] = []

        for field, value in resolved_fields.items():
            if field in {"min_age", "max_age"}:
                sub_filters.append(
                    {
                        "op": ">=" if field == "min_age" else "<=",
                        "content": {
                            "field": field_registry[field],
                            "value": value,
                        },
                    }
                )
            else:
                sub_filters.append(
                    {
                        "op": "in",
                        "content": {
                            "field": field,
                            "value": value,
                        },
                    }
                )

        return GDCCohortSchema(op="and", content=sub_filters)

    async def construct_cohort_filter(
        project: StringList | None = None,
        primary_site: StringList | None = None,
        disease_type: StringList | None = None,
        project_id: StringList | None = None,
        sex: StringList | None = None,
        smoking_status: StringList | None = None,
        min_age: int | None = None,
        max_age: int | None = None,
    ) -> str:
        args = ConstructCohortFilterArgs(
            project=project,
            primary_site=primary_site,
            disease_type=disease_type,
            project_id=project_id,
            sex=sex,
            min_age=min_age,
            max_age=max_age,
            smoking_status=smoking_status,
        )

        return await _construct(args)

    return construct_cohort_filter


# CASE TOOLS
def get_cases_by_cohort_filter(cohort_filter: GDCCohortSchema) -> CaseSet:
    """
    A tool to query the GDC API for cases matching a given cohort filter.
    """
    print(
        f"get_cases_by_filter(cohort_filter={repr(cohort_filter)})",
        file=sys.stderr,
    )
    hits = gdc_query_all(
        endpoint="cases",
        filters={},
        cohort_filter=cohort_filter,
        fields=["project.project_id", "submitter_id"],
    )
    case_ids = set([hit["id"] for hit in hits])
    return case_ids


def compute_case_intersection(
    case_set_A: CaseSet,
    case_set_B: CaseSet,
) -> CaseSet:
    """
    A tool to compute the intersection between two case sets.
    """
    cases_A = set(case_set_A)
    cases_B = set(case_set_B)

    A_and_B = cases_A & cases_B

    return list(A_and_B)


def compute_case_union(
    case_set_A: CaseSet,
    case_set_B: CaseSet,
) -> CaseSet:
    """
    A tool to compute the union between two case sets.
    """

    cases_A = set(case_set_A)
    cases_B = set(case_set_B)

    A_and_B = cases_A | cases_B

    return list(A_and_B)


def get_case_set_size(
    case_set: CaseSet,
) -> CaseCount:
    """
    A tool to retireve the size of a precomputed case set.
    """

    return len(case_set)


# VARIANT TOOLS
def get_simple_somatic_mutation_occurrences(
    cohort_filter: GDCCohortSchema,
    gene: Gene,
    aa_change: AAChange | None = None,
) -> CaseSet:
    """
    A tool to query the GDC API for cases with simple somatic mutations (SSMs).
    This tool searches within a specific cohort for SSMs within a given gene, optionally resulting in an amino acid (AA) change.
    The resulting case set is cached server side and can be referenced using the unique identifier returned by this tool.
    """
    print(
        f"get_simple_somatic_mutation_occurrences(gene={repr(gene)}, aa_change={repr(aa_change)}, cohort_filter={repr(cohort_filter)})",
        file=sys.stderr,
    )

    if gene is None:
        raise ValidationError(
            f"Cannot retrieve simple somatic mutation cases without a specific gene"
        )

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
    cohort_filter = cohort_filter.model_dump(mode="json")
    hits = gdc_query_all(
        endpoint="ssms",
        filters=_filters,
        cohort_filter=cohort_filter,
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
        cohort_filter=cohort_filter,
        fields=["ssm.ssm_id", "case.submitter_id", "case.case_id"],
    )
    case_ids = set([hit["case"]["case_id"] for hit in hits])

    return list(case_ids)


def get_copy_number_variant_occurrences(
    cohort_filter: GDCCohortSchema,
    gene: Gene | None = None,
    cnv_change: CNVChange | None = None,
) -> CaseSet:
    """
    A tool to query the GDC API for cases with copy number variations (CNVs).
    This tool searches within a specific cohort for CNVs, optionally within a given gene, optionally for a specific change type.
    If only the gene is specified, this tool retrieves all cases in the cohort with any change type for the given gene.
    If only the change type is specified, this tool retrieves all cases in the cohort with the given change type in any gene.
    It is an error to use this tool without at least one of gene or change type.
    """
    print(
        f"get_copy_number_variant_occurrences(gene={repr(gene)}, cnv_change={repr(cnv_change)}, cohort_filter={repr(cohort_filter)})",
        file=sys.stderr,
    )

    if gene is None and cnv_change is None:
        raise ValidationError(
            f"Cannot retrieve copy number variation cases without either a specific gene, a change type, or both"
        )

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

    cohort_filter = cohort_filter.model_dump(mode="json")
    hits = gdc_query_all(
        endpoint="cnv_occurrences",
        filters={
            "op": "and",
            "content": ops,
        },
        cohort_filter=cohort_filter,
        fields=[
            "cnv.cnv_change",
            "cnv.cnv_change_5_category",
            "cnv.consequence.gene.symbol",
            "case.submitter_id",
            "case.case_id",
        ],
    )
    case_ids = set([hit["case"]["case_id"] for hit in hits])

    return list(case_ids)


def get_microsatellite_instability_occurrences(
    cohort_filter: GDCCohortSchema,
    msi_status: MSIStatus,
) -> CaseSet:
    """
    A tool to query the GDC API for cases with microsatellite instability (MSI).
    This tool searches within a specific cohort for either MSI or MSS status.
    """
    print(
        f"get_microsatellite_instability_occurrences(msi_status={repr(msi_status)}, cohort_filter={repr(cohort_filter)})",
        file=sys.stderr,
    )
    cohort_filter = cohort_filter.model_dump(mode="json")
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
        cohort_filter=cohort_filter,
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

    return list(case_ids)


def get_cases_with_available_genomic_data(
    genomic_data_type: GenomicDataType,
    cohort_filter: GDCCohortSchema,
) -> CaseSet:
    cohort_filter = cohort_filter.model_dump(mode="json")
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
        return case_ids
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
        return case_ids
    elif genomic_data_type == "msi":
        _msi_ids = get_microsatellite_instability_occurrences(cohort_filter, "msi")
        _mss_ids = get_microsatellite_instability_occurrences(cohort_filter, "mss")
        _ms_any_ids = compute_case_union(_msi_ids, _mss_ids)
        return _ms_any_ids
    else:
        raise TypeError(
            f"Unknown how to find available cases for given genomic data type: {genomic_data_type}"
        )


# GENE EXPRESSION TOOLS
def _get_ensembl_gene_ids(genes: list[Gene]) -> dict[str, Gene]:
    # convert standard gene names to ensembl IDs as a mapping of ensembl ID -> input name
    ensembl_gene_ids = dict()
    for gene in genes:
        filters = {
            "op": "and",
            "content": [{"op": "=", "content": {"field": "symbol", "value": gene}}],
        }
        try:
            response = requests.get(
                url=f"{GDC_API}/genes",
                params={
                    "filters": json.dumps(filters),
                    "fields": ",".join(["gene_id"]),
                    "size": 100,
                },
            )
            data = response.json()

            ensembl_id = data["data"]["hits"][0]["gene_id"]
            ensembl_gene_ids[ensembl_id] = gene
        except Exception as e:
            print("unable to execute GDC API request {}".format(str(e)))
    return ensembl_gene_ids


def get_gene_expression_statistics(
    case_set: CaseSet,
    genes: list[Gene],
    normalization: Normalization,
) -> dict[str, dict[str, float]]:
    """
    A tool to query the GDC API for gene expression statistics for a list of genes within a set of cases.
    Gene expression values are normalized either using FPKM-UQ or Median-centered log2(FPKM-UQ+1).
    """

    if len(genes) == 0:
        raise ValidationError("Must provide at least one gene")

    # compute and cache raw expression values
    gene_mapping = _get_ensembl_gene_ids(genes)
    response = requests.post(
        url=f"{GDC_API}/gene_expression/values",
        headers={
            "accept": "text/tab-separated-values",
            "Content-Type": "application/json",
        },
        json={
            "case_ids": case_set,
            "gene_ids": list(gene_mapping),
            "tsv_units": normalization,
            "format": "tsv",
        },
    )
    expr_str = response.text
    df = pd.read_csv(StringIO(expr_str), sep="\t")

    # map back to user-given names/order as best we can
    df["gene_id"] = [gene_mapping[g] for g in df["gene_id"]]
    df = df.set_index("gene_id")
    df = df.loc[list(gene_mapping.values())]

    summary = df.T.describe().to_dict()
    return summary  # type: ignore


def get_most_variably_expressed_genes(
    case_set: CaseSet,
    top_k: int = 1,
) -> list[Gene]:
    """
    A tool to query GDC API for the top-k most variably expressed genes within a set of cases.'
    """
    # do query
    response = requests.post(
        url=f"{GDC_API}/gene_expression/gene_selection",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "case_ids": case_set,
            "selection_size": top_k,
            "gene_type": "protein_coding",
        },
    )
    if not response.ok:
        raise ValidationError(f"GDC API returned status {response.status_code}")
    data = response.json()
    genes = []
    if "gene_selection" not in data:
        raise ValidationError("GDC API did not return genes")
    for item in data["gene_selection"]:
        if "symbol" not in item:
            print(f"Gene did not have symbol: {item}")
            continue
        genes.append(item["symbol"])

    return genes
