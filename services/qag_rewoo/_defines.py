from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class EndpointResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    warnings: list[str]
    metadata: dict[str, Any]


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ref: str = Field(
        description="Reference to an earlier evidence result, such as '#E1'",
        pattern=r"^#E[1-9]\d*$",
    )

    @property
    def index(self) -> int:
        return int(self.ref[2:])


ToolArgumentValue = str | int | float | bool | list[str] | EvidenceReference | None


class PlannerStep(BaseModel):
    plan: str = Field(
        min_length=1, description="What this step does and why it is needed."
    )
    evidence_id: str = Field(
        description="Unique evidence identifier produced by this step",
        pattern=r"^#E[1-9]\d*$",
        examples=["#E1"],
    )
    tool_name: str = Field(
        min_length=1,
        description="Exact name of the tool to invoke.",
    )
    tool_arguments: dict[str, ToolArgumentValue] = Field(
        default_factory=dict,
        description=(
            "Arguments passed to the tool. Values may be literals, lists, "
            "null, or references to earlier evidence."
        ),
    )
    dependencies: Optional[list[EvidenceReference]] = Field(
        min_length=1, description="Earlier evidence results needed by this step"
    )


class PlannerResponse(BaseModel):
    objective: str = Field(
        min_length=1, description="The objective the plan is intended to solve"
    )
    steps: list[PlannerStep] = Field(
        min_length=1, description="Ordered tool execution plan"
    )


# =============================================================================#
CaseSet = Annotated[
    list[str],
    Field(
        description=("A set of cases matching a specific cohort or query"),
    ),
]
CaseCount = Annotated[
    int,
    Field(
        description="Number of cases for a retrieved or computed case set.",
    ),
]

Gene = Annotated[
    str,
    Field(
        description="Gene to query, for example 'BRAF'.",
    ),
]
AAChange = Annotated[
    str,
    Field(
        description="Amino acid change to query (in HGVS protein notation), for example 'V600E'.",
    ),
]
CNVChange = Annotated[
    Literal["gain", "amplification", "heterozygous deletion", "homozygous deletion"],
    Field(
        description="Copy number variation change to query. Valid options are 'gain', 'amplification', 'heterozygous deletion', or 'homozygous deletion'.",
    ),
]
MSIStatus = Annotated[
    Literal["msi", "mss"],
    Field(
        description="Microsatellite instability status. Valid options are 'msi' for microsatellite instable or 'mss' for microsatellite stable.",
    ),
]
Normalization = Annotated[
    Literal["uqfpkm", "median_centered_log2_uqfpkm"],
    Field(description="Gene expression normalization method."),
]
GenomicDataType = Annotated[
    Literal["ssm", "cnv", "msi", "gene_expression"],
    Field(description="Genomic data modality."),
]
# =============================================================================#
PROJECT_VALUES = """
"APOLLO-LUAD",
"APOLLO-OV",
"BEATAML1.0-COHORT",
"BEATAML1.0-CRENOLANIB",
"CDDP_EAGLE-1",
"CGCI-BLGSP",
"CGCI-HTMCP-CC",
"CGCI-HTMCP-DLBCL",
"CGCI-HTMCP-LC",
"CMI-ASC",
"CMI-MBC",
"CMI-MPC",
"CPTAC-2",
"CPTAC-3",
"CTSP-DLBCL1",
"EXCEPTIONAL_RESPONDERS-ER",
"FM-AD",
"HCMI-CMDC",
"MATCH-B",
"MATCH-C1",
"MATCH-H",
"MATCH-I",
"MATCH-N",
"MATCH-P",
"MATCH-Q",
"MATCH-R",
"MATCH-S1",
"MATCH-S2",
"MATCH-U",
"MATCH-W",
"MATCH-Y",
"MATCH-Z1A",
"MATCH-Z1B",
"MATCH-Z1D",
"MATCH-Z1I",
"MMRF-COMMPASS",
"MP2PRT-ALL",
"MP2PRT-WT",
"NCICCR-DLBCL",
"OHSU-CNL",
"ORGANOID-PANCREATIC",
"REBC-THYR",
"TARGET-ALL-P1",
"TARGET-ALL-P2",
"TARGET-ALL-P3",
"TARGET-AML",
"TARGET-CCSK",
"TARGET-NBL",
"TARGET-OS",
"TARGET-RT",
"TARGET-WT",
"TCGA-ACC",
"TCGA-BLCA",
"TCGA-BRCA",
"TCGA-CESC",
"TCGA-CHOL",
"TCGA-COAD",
"TCGA-DLBC",
"TCGA-ESCA",
"TCGA-GBM",
"TCGA-HNSC",
"TCGA-KICH",
"TCGA-KIRC",
"TCGA-KIRP",
"TCGA-LAML",
"TCGA-LGG",
"TCGA-LIHC",
"TCGA-LUAD",
"TCGA-LUSC",
"TCGA-MESO",
"TCGA-OV",
"TCGA-PAAD",
"TCGA-PCPG",
"TCGA-PRAD",
"TCGA-READ",
"TCGA-SARC",
"TCGA-SKCM",
"TCGA-STAD",
"TCGA-TGCT",
"TCGA-THCA",
"TCGA-THYM",
"TCGA-UCEC",
"TCGA-UCS",
"TCGA-UVM",
"TRIO-CRU",
"VAREPOP-APOLLO",
"WCDT-MCRPC",
"""
PRIMARY_SITE_VALUES = """
"accessory sinuses",
"adrenal gland",
"anus and anal canal",
"base of tongue",
"bladder",
"bones, joints and articular cartilage of limbs",
"bones, joints and articular cartilage of other and unspecified sites",
"brain",
"breast",
"bronchus and lung",
"cervix uteri",
"colon",
"connective, subcutaneous and other soft tissues",
"corpus uteri",
"esophagus",
"eye and adnexa",
"floor of mouth",
"gallbladder",
"gum",
"heart, mediastinum, and pleura",
"hematopoietic and reticuloendothelial systems",
"hypopharynx",
"kidney",
"larynx",
"lip",
"liver and intrahepatic bile ducts",
"lymph nodes",
"meninges",
"nasal cavity and middle ear",
"nasopharynx",
"not reported",
"oropharynx",
"other and ill-defined digestive organs",
"other and ill-defined sites",
"other and ill-defined sites in lip, oral cavity and pharynx",
"other and ill-defined sites within respiratory system and intrathoracic organs",
"other and unspecified female genital organs",
"other and unspecified major salivary glands",
"other and unspecified male genital organs",
"other and unspecified parts of biliary tract",
"other and unspecified parts of mouth",
"other and unspecified parts of tongue",
"other and unspecified urinary organs",
"other endocrine glands and related structures",
"ovary",
"palate",
"pancreas",
"parotid gland",
"penis",
"peripheral nerves and autonomic nervous system",
"prostate gland",
"rectosigmoid junction",
"rectum",
"renal pelvis",
"retroperitoneum and peritoneum",
"skin",
"small intestine",
"spinal cord, cranial nerves, and other parts of central nervous system",
"stomach",
"testis",
"thymus",
"thyroid gland",
"tonsil",
"trachea",
"unknown",
"ureter",
"uterus, nos",
"vagina",
"vulva",
"""
DISEASE_TYPE_VALUES = """
"acinar cell neoplasms",
"acute lymphoblastic leukemia",
"adenomas and adenocarcinomas",
"adnexal and skin appendage neoplasms",
"basal cell neoplasms",
"blood vessel tumors",
"chronic myeloproliferative disorders",
"complex epithelial neoplasms",
"complex mixed and stromal neoplasms",
"cystic, mucinous and serous neoplasms",
"ductal and lobular neoplasms",
"epithelial neoplasms, nos",
"fibroepithelial neoplasms",
"fibromatous neoplasms",
"germ cell neoplasms",
"gliomas",
"leukemias, nos",
"lipomatous neoplasms",
"lymphoid leukemias",
"mature b-cell lymphomas",
"mature t- and nk-cell lymphomas",
"meningiomas",
"mesothelial neoplasms",
"miscellaneous bone tumors",
"miscellaneous tumors",
"mucoepidermoid neoplasms",
"myelodysplastic syndromes",
"myeloid leukemias",
"myomatous neoplasms",
"neoplasms, nos",
"nerve sheath tumors",
"neuroepitheliomatous neoplasms",
"nevi and melanomas",
"not applicable",
"not reported",
"osseous and chondromatous neoplasms",
"paragangliomas and glomus tumors",
"plasma cell tumors",
"soft tissue tumors and sarcomas, nos",
"specialized gonadal neoplasms",
"squamous cell neoplasms",
"synovial-like neoplasms",
"thymic epithelial neoplasms",
"transitional cell papillomas and carcinomas",
"unknown",
"""
SEX_VALUES = """
"male", 
"female", 
"unknown"
"""
SMOKING_STATUS_VALUES = """
"current reformed smoker for > 15 yrs",
"current reformed smoker for < or = 15 yrs",
"current reformed smoker, duration not specified",
"current smoker",
"lifelong non-smoker",
"not reported",
"smoking history not documented",
"unknown",
"""
# =============================================================================#
FIELD_REGISTRY = {
    "project": "cases.project.project_id",
    "primary_site": "cases.primary_site",
    "disease_type": "cases.disease_type",
    "sex": "cases.demographic.sex_at_birth",
    "min_age": "cases.diagnoses.age_at_diagnosis",
    "max_age": "cases.diagnoses.age_at_diagnosis",
    "smoking_status": "cases.exposures.tobacco_smoking_status",
}
VALUE_REGISTRY = {
    "project": PROJECT_VALUES,
    "primary_site": PRIMARY_SITE_VALUES,
    "disease_type": DISEASE_TYPE_VALUES,
    "sex": SEX_VALUES,
    "smoking_status": SMOKING_STATUS_VALUES,
}
# =============================================================================#
