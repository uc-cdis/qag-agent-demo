from typing import Annotated, Literal

from pydantic import Field

GDC_API = "https://api.gdc.cancer.gov"

DataType = Annotated[
    Literal["ssm", "cnv", "msi"],
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
CaseCount = Annotated[
    int,
    Field(
        description="Number of cases for a retrieved or computed case set.",
    ),
]

TOOL_CACHE_ID_TEMPLATES = {
    "compute_case_intersection": "Cases-Intersect-({case_set_id_A})-({case_set_id_B})",
    "compute_case_union": "Cases-Union-({case_set_id_A})-({case_set_id_B})",
    "get_simple_somatic_mutation_occurrences": "Cases-SSM-({gene})-({aa_change})-({cohort_filter})",
    "get_copy_number_variant_occurrences": "Cases-CNV-({gene})-({cnv_change})-({cohort_filter})",
    "get_microsatellite_instability_occurrences": "Cases-MSI-({msi_status})-({cohort_filter})",
    "get_cases_with_available_data": "Cases-Data-({data_type})-({cohort_filter})",
    "get_cases_by_cohort_filter": "Cases-Filter-({cohort_filter})",
    "get_cohort_filter_by_project": "Filter-Project-({project})",
    "get_cohort_filter_by_cohort_description": "Filter-Cohort-({cohort_description})",
}
