from typing import Annotated, Literal

from pydantic import BeforeValidator, Field

GDC_API = "https://api.gdc.cancer.gov"

TOOL_CACHE_ID_TEMPLATES = {
    "compute_case_intersection": "Cases-Intersect-({case_set_id_A})-({case_set_id_B})",
    "compute_case_union": "Cases-Union-({case_set_id_A})-({case_set_id_B})",
    "get_simple_somatic_mutation_occurrences": "Cases-SSM-({gene})-({aa_change})-({cohort_filter})",
    "get_copy_number_variant_occurrences": "Cases-CNV-({gene})-({cnv_change})-({cohort_filter})",
    "get_microsatellite_instability_occurrences": "Cases-MSI-({msi_status})-({cohort_filter})",
    "get_cases_with_available_genomic_data": "Cases-Data-({genomic_data_type})-({cohort_filter})",
    "get_cases_by_cohort_filter": "Cases-Filter-({cohort_filter})",
    "construct_cohort_filter": "Filter-({project})-({primary_site})-({disease_type})-({sex})-({min_age})-({max_age})-({smoking_status})",
}

#####################################

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

#####################################

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

#####################################

CohortFilterID = Annotated[
    str,
    Field(
        description=(
            "ID reprsenting a JSON filter for describing a GDC cohort. "
            "The JSON filters are cached server side and are referenced by this identifier."
        )
    ),
]
Project = Annotated[
    Literal[
        "APOLLO-LUAD",
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
    ],
    Field(
        description="The GDC project name.",
    ),
]
PrimarySite = Annotated[
    Literal[
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
    ],
    BeforeValidator(str.lower),
    Field(description="The primary site of disease."),
]
DiseaseType = Annotated[
    Literal[
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
    ],
    BeforeValidator(str.lower),
    Field(description="The type of cancer."),
]
Sex = Annotated[
    Literal["male", "female", "unknown"],
    BeforeValidator(str.lower),
    Field(description="The patient's sex."),
]
Age = Annotated[
    int,
    Field(description="The patient's age at the time of diagnosis (in years)."),
]
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
    BeforeValidator(str.lower),
    Field(description="The patient's self-reported smoking status and history."),
]
