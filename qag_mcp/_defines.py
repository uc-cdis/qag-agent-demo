from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field

ENUMS_PATH = "/opt/gpudata/anirudh/git-repos/qag-agent-demo/qag_mcp/enumerations.yaml"

with open(ENUMS_PATH) as f:
    _CATALOG = yaml.safe_load(f)

GDC_API = "https://api.gdc.cancer.gov"

DataType = Annotated[
    Literal["ssm", "cnv", "msi", "gene_expression"],
    Field(description="Genomic variant data modality."),
]
Project = Annotated[
    str,
    Field(
        description="GDC project name, for example 'TCGA-BRCA'.",
    ),
]
CohortDescription = Annotated[
    str,
    Field(
        description="Natural language description of a cohort of cases, for example 'cases from male patients with lung cancer'.",
    ),
]
CohortFilterID = Annotated[
    str | None,
    Field(
        description=(
            "(Optional) ID reprsenting a JSON filter for describing a GDC cohort, for example 'Filter-Cohort-(TCGA-BRCA)'. "
            "The JSON filters are cached server side and are referenced by this identifier."
        )
    ),
]
Gene = Annotated[
    str | None,
    Field(
        description="(Optional) Gene to query, for example 'BRAF'.",
    ),
]
TSVUnit = Annotated[
    Literal["uqfpkm", "median_centered_log2_uqfpkm"],
    Field(description="(Optional) TSV units for gene expression data."),
]
AAChange = Annotated[
    str | None,
    Field(
        description="(Optional) Amino acid change to query (in HGVS protein notation), for example 'V600E'.",
    ),
]
CNVChange = Annotated[
    Literal["gain", "amplification", "heterozygous deletion", "homozygous deletion"]
    | None,
    Field(
        description="(Optional) Copy number variation change to query. Valid options are 'gain', 'amplification', 'heterozygous deletion', or 'homozygous deletion'.",
    ),
]
MSIStatus = Annotated[
    Literal["msi", "mss"],
    Field(
        description="Microsatellite instability status. Valid options are 'msi' for microsatellite instable or 'mss' for microsatellite stable.",
    ),
]
PrimarySite = Annotated[
    Literal[*_CATALOG["cases.primary_site"]["values"]],
    Field(
        description="The text term used to describe the primary site of disease, as categorized by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O). This categorization groups cases into general categories. Use 'search_primary_sites' to find valid values"
    ),
]
DiseaseType = Annotated[
    Literal[*_CATALOG["cases.disease_type"]["values"]],
    Field(
        description="The text term used to describe the type of malignant disease, as categorized by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O). Use 'search_disease_type' to find valid values"
    ),
]
Sex = Annotated[
    Literal["male", "female", "unknown"],
    Field(description="A textual description of a person's sex at birth."),
]
# Age = Annotated[
#     int,
#     Field(
#         description="Age at the time of diagnosis expressed in number of days since birth. Minimum is 0, Maximum is 32872"
#     )
# ]
SmokingStatus = Annotated[
    Literal[
        "current reformed smoker for > 15 yrs",
        "current reformed smoker for < or = 15 yrs",
        "current reformed smoker, duration not specified",
        "current smoker",
        "lifelong non-smoker",
        "not reported",
        "smoking history not documented",
        "unknown",
    ],
    Field(
        description="Category describing current smoking status and smoking history as self-reported by a patient."
    ),
]

CaseSetId = Annotated[
    str,
    Field(
        description=(
            "ID representing a set of cases matching a specified query. "
            "The retrieved cases are cached server side and are referenced by this identifier. "
            "The ID additionally encodes the method by which the case set was computed, for example 'Cases-SSM-(BRAF)-(V600E)'."
        ),
    ),
]
ExpressionSetId = Annotated[
    str,
    Field(
        description=(
            "ID representing a set of cases and genes matching a specified query about gene expression "
            "The retrieved genes are cached server side and are referenced by this identifier. "
        ),
    ),
]
CaseCount = Annotated[
    int,
    Field(
        description="Number of cases for a retrieved or computed case set.",
    ),
]


class DataFrameDescribeOutput(BaseModel):
    row_count: int
    column_count: int
    columns: list[str]
    describe: dict[str, dict[str, Any]]
    markdown: str = Field(description="Human-readable describe table")


TOOL_CACHE_ID_TEMPLATES = {
    "compute_case_intersection": "Cases-Intersect-({case_set_id_A})-({case_set_id_B})",
    "compute_case_union": "Cases-Union-({case_set_id_A})-({case_set_id_B})",
    "get_simple_somatic_mutation_occurrences": "Cases-SSM-({gene})-({aa_change})-({cohort_filter})",
    "get_copy_number_variant_occurrences": "Cases-CNV-({gene})-({cnv_change})-({cohort_filter})",
    "get_microsatellite_instability_occurrences": "Cases-MSI-({msi_status})-({cohort_filter})",
    "get_cases_with_available_data": "Cases-Data-({data_type})-({cohort_filter})-({genes})",
    "get_cases_by_cohort_filter": "Cases-Filter-({cohort_filter})",
    "get_cohort_filter_by_project": "Filter-Project-({project})-({primary_site})-({disease_type})-({sex})-({min_age})-({max_age})-({smoking_status})",
    "get_cohort_filter_by_cohort_description": "Filter-Cohort-({cohort_description})",
    "get_gene_expression_occurances": "Cases-EXPR-({genes})-({cohort_filter})",
    "get_gene_expression_values": "Cases-EXPR-({cases})-({genes})-({tsv_unit})",
}
