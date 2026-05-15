import json
import sys
from io import StringIO

import pandas as pd
import requests
from cachetools import TTLCache
from mcp.server.fastmcp.exceptions import ToolError

# isort: off
from _defines import CaseSetId, GDC_API, Gene, Normalization

# isort: on


def make_gene_expression_tools(cache: TTLCache):
    def get_gene_expression_statistics(
        case_set_id: CaseSetId,
        genes: list[Gene],
        normalization: Normalization,
    ) -> dict[str, dict[str, float]]:
        """
        A tool to query the GDC API for gene expression statistics for a list of genes within a set of cases.
        Gene expression values are normalized either using FPKM-UQ or Median-centered log2(FPKM-UQ+1).
        """
        if len(genes) == 0:
            raise ToolError("Must provide at least one gene")
        print(
            f"get_gene_expression_values(case_set_id={repr(case_set_id)}, genes={repr(genes)})",
            file=sys.stderr,
        )

        # this cache isn't exposed to the client so just define statically here
        # (no need to use the utilities for cache IDs that are part of the interface)
        cache_id = f"Expression-{case_set_id}-{genes}-{normalization}"

        if cache_id in cache:
            # if we've already done this retrieval, refresh it in the cache
            cache[cache_id] = cache[cache_id]
        else:
            # compute and cache raw expression values
            case_ids = cache.get(case_set_id)
            gene_mapping = _get_ensembl_gene_ids(genes)
            response = requests.post(
                url=f"{GDC_API}/gene_expression/values",
                headers={
                    "accept": "text/tab-separated-values",
                    "Content-Type": "application/json",
                },
                json={
                    "case_ids": case_ids,
                    "gene_ids": list(gene_mapping),
                    "tsv_units": normalization,
                    "format": "tsv",
                },
            )
            # store anything that required expensive calls
            # - mapping/filtering to ensembl IDs
            # - retrieving expression values
            cache[cache_id] = response.text, gene_mapping
        expr_str, gene_mapping = cache[cache_id]
        df = pd.read_csv(StringIO(expr_str), sep="\t")

        # map back to user-given names/order as best we can
        df["gene_id"] = [gene_mapping[g] for g in df["gene_id"]]
        df = df.set_index("gene_id")
        df = df.loc[list(gene_mapping.values())]

        summary = df.T.describe().to_dict()
        return summary  # type: ignore

    return get_gene_expression_statistics


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
