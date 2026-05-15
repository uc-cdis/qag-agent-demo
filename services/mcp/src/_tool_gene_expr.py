import json
import sys
from io import StringIO

import pandas as pd
import requests
from cachetools import TTLCache
from mcp.server.fastmcp.exceptions import ToolError

# isort: off
from _defines import CaseSetId, GDC_API, Gene, Normalization
from _utils import suggest_tool_from_id

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
        print(
            f"get_gene_expression_statistics(case_set_id={repr(case_set_id)}, genes={repr(genes)}, normalization={repr(normalization)})",
            file=sys.stderr,
        )

        if len(genes) == 0:
            raise ToolError("Must provide at least one gene")

        if case_set_id not in cache:
            raise ToolError(
                f"Case set {case_set_id} was not found in the server side cache, perhaps it expired? "
                f"Try requerying for those cases to cache it again. {suggest_tool_from_id(case_set_id)}"
            )
        # refresh
        cache[case_set_id] = cache[case_set_id]

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

    def get_most_variably_expressed_genes(
        case_set_id: CaseSetId,
        top_k: int = 1,
    ) -> list[Gene]:
        """
        A tool to query GDC API for the top-k most variably expressed genes within a set of cases.'
        """
        print(
            f"get_most_variably_expressed_genes(case_set_id={repr(case_set_id)}, top_k={repr(top_k)})",
            file=sys.stderr,
        )

        if case_set_id not in cache:
            raise ToolError(
                f"Case set {case_set_id} was not found in the server side cache, perhaps it expired? "
                f"Try requerying for those cases to cache it again. {suggest_tool_from_id(case_set_id)}"
            )
        # refresh
        cache[case_set_id] = cache[case_set_id]

        # this cache isn't exposed to the client so just define statically here
        # (no need to use the utilities for cache IDs that are part of the interface)
        cache_id = f"Variability-{case_set_id}-{top_k}"

        if cache_id in cache:
            # if we've already done this retrieval, refresh it in the cache
            cache[cache_id] = cache[cache_id]
        else:
            # do query
            response = requests.post(
                url=f"{GDC_API}/gene_expression/gene_selection",
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={
                    "case_ids": cache.get(case_set_id),
                    "selection_size": top_k,
                    "gene_type": "protein_coding",
                },
            )
            if not response.ok:
                raise ToolError(f"GDC API returned status {response.status_code}")
            data = response.json()
            genes = []
            if "gene_selection" not in data:
                raise ToolError("GDC API did not return genes")
            for item in data["gene_selection"]:
                if "symbol" not in item:
                    print(f"Gene did not have symbol: {item}")
                    continue
                genes.append(item["symbol"])
            cache[cache_id] = genes

        return cache[cache_id]

    return get_gene_expression_statistics, get_most_variably_expressed_genes


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
