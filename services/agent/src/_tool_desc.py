TOOLS_DESCRIPTION = """
This is list of the available tool and tool sets and their descriptions:

Basic Tools:
1. `add`: -> float
Compute addition (a + b)
Args:
    a: float
    b: float
2. `subtract`: -> float
Compute subtraction (a - b)
Args:
    a: float
    b: float
3. `multiply`: -> float
Compute multiplication (a * b)
Args:
    a: float
    b: float
4. `divide`: -> float
Compute division (a / b)
Args:
    a: float
    b: float

Case Handling Tools:
1. `get_cases_by_cohort_filter`: -> CaseSetId
A tool to query the GDC API for cases matching a given cohort filter.
Args:
    cohort_filter: GDC cohort filter
2. `compute_case_intersection`: -> CaseSetId
A tool to compute the intersection between two case sets.
Args:
    case_set_id_A: CaseSetId
    case_set_id_B: CaseSetId
3. `compute_case_union`: -> CaseSetId
A tool to compute the union between two case sets.
Args:
    case_set_id_A: CaseSetId
    case_set_id_B: CaseSetId
4. `get_case_set_size`: -> CaseCount
A tool to retrieve the size of a precomputed case set
Args:
    case_set_id: CaseSetId

Cohort Creation Tools:
1. `construct_cohort_filter`: -> CohortFilter
A tool to construct a GDC cohort filter for case metadata.
The resulting filter is used to retrieve actual case counts using GDC API
Args:
    project: GDC project IDs. e.g ["TCGA-BRCA", "TCGA-LUAD"]
    primary_site: Primary site of a disease e.g. ["Lung", "Brain"]
    disease_type: Type of malignant disease e.g. ["adenomas and adenocarcinomas", "leukemia, nos"]
    sex: A person's sex at birth e.g. any / many of ["male", "female", "unknown"],
    min_age : Lower end of the age range, inclusive. Age at the time of diagnosis, expressed in years. e.g 34
    max_age : Higher end of the age range, inclusive. Age at the time of diagnosis, expressed in years. e.g 76
    smoking_status: Self reported current smoking status or smoking history of a patient. e.g. ["lifelong non-smoker", "current smoker"]

Case Availability Tools:
1. `get_cases_with_available_genomic_data`: -> CaseSetId
A tool to retrieve cases with available data for a given genomic data type, for example 'ssm'.
This tool should always be used with a cohort filter, as data availability without a specific cohort can be slow to compute.
Some user queries may necessarily require computing data availability over all cases.
While such queries should be avoid if possible, if you absolutely must compute availability over all cases, create an empty cohort filter first.
Args:
    genomic_data_type: Valid options are "ssm", "cnv", "msi", "gene_expression"
    cohort_filter: GDC cohort Filter

Variant Tools:
1. `get_simple_somatic_mutation_occurrences`: -> CaseSetId
A tool to query the GDC API for cases with simple somatic mutations (SSMs).
This tool searches within a specific cohort for SSMs within a given gene, optionally resulting in an amino acid (AA) change.
Args:
    cohort_filter: GDC cohort filter
    gene: Gene to query, for example, "BRAF"
    aa_change: Amino acid change to query (in HGVS protein notation), for example 'V600E'.

2. `get_copy_number_variant_occurrences`: -> CaseSetId
A tool to query the GDC API for cases with copy number variations (CNVs).
This tool searches within a specific cohort for CNVs, optionally within a given gene, optionally for a specific change type.
If only the gene is specified, this tool retrieves all cases in the cohort with any change type for the given gene.
If only the change type is specified, this tool retrieves all cases in the cohort with the given change type in any gene.
It is an error to use this tool without at least one of gene or change type.
Args:
    cohort_filter: GDC Cohort Filter
    gene: Gene to query, for example, "BRAF"
    cnv_change: Copy number variation change to query. Valid options are 'gain', 'amplification', 'heterozygous deletion', or 'homozygous deletion'."

3. `get_microsatellite_instability_occurrences`: -> CaseSetId
A tool to query the GDC API for cases with microsatellite instability (MSI).
This tool searches within a specific cohort for either MSI or MSS status.
Args:
    cohort_filter: GDC Cohort Filter
    msi_status: Microsatellite instability status. Valid options are 'msi' for microsatellite instable or 'mss' for microsatellite stable.

Gene Expression Tools
1. `get_gene_expression_statistics`:
A tool to query the GDC API for gene expression statistics for a list of genes within a set of cases.
Gene expression values are normalized either using FPKM-UQ or Median-centered log2(FPKM-UQ+1)
Args:
    case_set_id: A unique ID representing a set of cases usually pertaining to a cohort filter
    genes: A list of gene names to retrieve statistics for. Example ["BRAF", "BRCA"]
    normalization: Gene expression normalization method. Valid options are 'uqfpkm' and 'median_centered_log2_uqfpkm'

2. `get_most_variably_expressed_genes`:
A tool to query GDC API for the top-k most variably expressed genes within a set of cases.
Args:
    case_set_id: A unique ID representing a set of cases usually pertaining to a cohort filter
    top_k : An integer value representing the top k most varaibly expressed genes. Example '5'.
"""
