import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx2
import lancedb
import pandas as pd
import torch
from cachetools import TTLCache
from fastmcp.server.dependencies import get_http_request
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from transformers import AutoModel, AutoTokenizer

TOP_K = 25
MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# TODO refactor as mcp resource
cache = TTLCache(maxsize=1000, ttl=10800)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
m = AutoModel.from_pretrained(MODEL_NAME).eval()
db = lancedb.connect("/tmp/gdc_lancedb")
if "fields" in db.table_names():
    print("Using existing `fields` table", flush=True)
    fields_table = db.open_table("fields")
else:
    rel_dir = Path(os.path.dirname(__file__))
    # precomputed embeddings with sapbert for both cases and files _mapping derived fields
    field_embs: dict[str, torch.Tensor] = torch.load(
        rel_dir / "static/gdc_mapping_embs.pt"
    )
    print(f"Loaded {len(field_embs)} fields", flush=True)
    df = pd.DataFrame.from_dict(
        {
            "field": list(field_embs.keys()),
            "vector": [x.tolist() for x in field_embs.values()],
        }
    )
    fields_table = db.create_table("fields", df)
    fields_table.create_fts_index("field", replace=True)

# TODO refactor filter def as mcp prompt
INSTRUCTIONS = """Use the tools of this MCP server to do search and retrieval for GDC data.

Your workflow should generally be:
1. Identify the right fields to create filters using
2. Select values to filter to under each field
3. Retrieve a cohort's the case set
4. Select the files you want in the cohort and get the corresponding manifest file 

For tools which require a GDC filter:

A GDC cohort filter is a single top-level and operation whose content is a list of independent clauses,
each constraining one field. Each clause is typically an in operation naming a dotted field path
(e.g. cases.project.project_id, cases.primary_site) and a list of allowed values for that field;
an entity must match every clause to be included. Values within a clause are OR'd together,
so multiple allowed values for the same field belong in one clause rather than separate ones.

Here is an example GDC filter:
{
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-LUAD", "TCGA-LUSC"]}},
        {"op": "in", "content": {"field": "cases.primary_site", "value": ["bronchus and lung"]}}
    ]
}

If you are ever unsure about the allowable fields or values, use the corresponding tools to look them up rather than guessing.
"""


# Initialize the server
mcp = FastMCP(
    name="GDC Search and Retrieval",
    instructions=INSTRUCTIONS,
)

# constants
GDC_API_BASE = "https://api.gdc.cancer.gov/"
USER_AGENT = "gdc_mcp/1.0"
CASES = "cases"
FILES = "files"


async def make_gdc_request(url: str, params: dict = None) -> dict[str, Any] | None:
    """Make a request to the GDC API with proper error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    async with httpx2.AsyncClient() as client:
        try:
            response = await client.get(
                url, headers=headers, params=params, timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return None


def format_mapping(mapping: dict) -> list[str]:
    """Format an alert feature into a readable string."""
    props = mapping["_mapping"]
    return [str(k) for k in props.keys()]


def format_facets(facet: dict, term: str) -> list[str]:
    """Format an alert feature into a readable string."""
    props = facet["data"]["aggregations"][term]["buckets"]
    return [str(k["key"]) for k in props]


def format_cases(cases: dict) -> list[str]:
    """format reponse of the GDC cases endpoint"""
    props = cases["data"]["hits"]
    return [str(k["id"]) for k in props]


async def get_all_gdc_cases(
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves ALL cases matching a specific GDC filter by automatically
    handling pagination.

    Args:
        filters (dict, optional): A GDC-compliant filter dictionary.
                                  If None, retrieves all cases in the GDC.

    Returns:
        list: A list containing all retrieved case dictionaries.
    """
    url = f"{GDC_API_BASE}{CASES}"

    # Define the fields you want to return per case.
    # Expand or contract this list based on your specific needs.
    fields = ["case_id"]

    params: Dict[str, Any] = {
        "fields": ",".join(fields),
        "format": "JSON",
        "size": 1000,  # Max size allowed by GDC per page to minimize API roundtrips
        "from": 0,
    }

    # Attach filters if they were provided
    if filters:
        params["filters"] = json.dumps(filters)

    all_cases: List[Dict[str, Any]] = []

    records_left = True

    while records_left:
        # Utilizing your existing request function
        response_data = await make_gdc_request(url, params)

        # Parse out the data block
        data = response_data.get("data", {})
        hits = data.get("hits", [])
        pagination = data.get("pagination", {})

        # Append current page hits to our master list
        all_cases.extend(hits)

        # Calculate pagination state
        current_from = pagination.get("from", 0)
        page_size = pagination.get("size", 0)
        total_records = pagination.get("total", 0)

        # If the next expected index exceeds or equals total records, we are done
        if current_from + page_size >= total_records or not hits:
            records_left = False

        # Move the offset tracker forward for the next loop iteration
        params["from"] += page_size

    return all_cases


async def get_all_gdc_file_ids(
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves ALL cases matching a specific GDC filter by automatically
    handling pagination.

    Args:
        filters (dict, optional): A GDC-compliant filter dictionary.
                                  If None, retrieves all cases in the GDC.

    Returns:
        list: A list containing all retrieved case dictionaries.
    """
    url = f"{GDC_API_BASE}{FILES}"

    fields = [
        "file_id",
        "file_name",
    ]

    params = {"fields": ",".join(fields), "format": "JSON", "size": 1000, "from": 0}

    if filters:
        params["filters"] = json.dumps(filters)

    all_files: List[Dict[str, Any]] = []

    records_left = True

    while records_left:
        # Utilizing your existing request function
        response_data = await make_gdc_request(url, params)

        # Parse out the data block
        data = response_data.get("data", {})
        hits = data.get("hits", [])
        pagination = data.get("pagination", {})

        # Append current page hits to our master list
        all_files.extend(hits)

        # Calculate pagination state
        current_from = pagination.get("from", 0)
        page_size = pagination.get("size", 0)
        total_records = pagination.get("total", 0)

        # If the next expected index exceeds or equals total records, we are done
        if current_from + page_size >= total_records or not hits:
            records_left = False

        # Move the offset tracker forward for the next loop iteration
        params["from"] += page_size

    return all_files


async def make_gdc_manifest_request(url: str, data) -> str:
    """
    Make a request to the gdc for a file manifest
    """
    headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}

    async with httpx2.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()
            return response.text
        except Exception as e:
            raise ToolError(repr(e))


async def get_gdc_manifest(filters: dict | None, case_ids: list[str]) -> str:
    """
    Retrieves ALL cases matching a specific GDC filter by automatically
    handling pagination.

    Args:
        filters : A GDC-compliant filter dictionary formatted as a string.

    Returns:
        list: A list containing all retrieved case dictionaries.
    """
    url = f"{GDC_API_BASE}files"
    and_terms = [
        {
            "op": "in",
            "content": {"field": "cases.case_id", "value": case_ids},
        }
    ]
    if filters:
        and_terms.append(filters)
    data = {
        "filters": {"op": "and", "content": and_terms},
        "return_type": "manifest",
        "size": 0,
    }
    response_data = await make_gdc_manifest_request(url, data)
    return response_data


@mcp.tool()
async def search_field_names(search_term: str) -> list[str]:
    """
    Search the GDC data dictionary for candidate fields which correspond to a given search term.
    The resulting fields can be used to construct cohort filters for the GDC API.
    """
    x = tok(
        search_term,
        padding="max_length",
        max_length=25,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        out: torch.Tensor = m(**x)
    emb = out[0][0, 0, :].tolist()  # (768,)
    q = fields_table.search(query_type="hybrid").vector(emb).text(search_term)
    res = q.limit(TOP_K).to_pandas()
    return res["field"].tolist()


@mcp.tool()
async def get_field_values(field: str) -> list[str]:
    """
    Get allowed values for the given field in the GDC.
    The resulting values can be used to specify values to filter for a given field in a cohort filter for the GDC API.
    """

    # get the root for the api
    api_root = field.split(".")[0]
    # get the field to search over
    field = field.removeprefix(api_root + ".")

    url = f"{GDC_API_BASE}/{api_root}?facets={field}&size=0"
    data = await make_gdc_request(url)

    if not data or "data" not in data:
        raise ToolError("Unable to fetch facets or no facet was found.")

    if not data["data"]:
        raise ToolError("No facets for this query.")

    values = format_facets(data, field)

    return values


# TODO refactor cohort handle as mcp resource?
@mcp.tool()
async def raw_case_set(cohort_handle: str) -> list[str]:
    """
    This tool returns the raw UUID list defining a case set after it has been
    retrieved by other tools. Beware that this tool should be rarely be used.
    """
    payload: list[str] = cache[cohort_handle]
    cache[cohort_handle] = payload  # refresh in cache
    return payload


@mcp.tool()
async def get_cases(filters: dict) -> dict:
    """
    Retrieve all GDC cases specified by a cohort filter.
    To prevent needless context window consumption, the raw UUID list defining
    the case set is not returned by this tool. Instead, this tool returns a
    cohort handle to reference the case set which can be used by other tools.

    Args:
        filters dict: A GDC cohort filter.

    Returns:
        dict: Metadata regarding the retrieved case set, including a cohort handle to reference the case set by.
    """
    try:
        cases = await get_all_gdc_cases(filters)
        case_ids = [x["id"] for x in cases]
        handle = secrets.token_urlsafe(8)
        cache[handle] = case_ids
        return {
            "handle": handle,
            "count": len(cases),
        }
    except Exception as e:
        raise ToolError(str(e))


@mcp.tool()
async def get_files(cohort_handle: str, filters: dict) -> str:
    """
    Retrieve a cohort's files from the GDC, with files specified by a filter JSON.
    Note that filtering on the cohort does not filter the files in the manifest.
    You must filter the files you want included in the manifest separately from the cohort.
    This tool does not execute the data download, it only retrieves a file manifest.
    The manifest file can be used by the user to download the data directly from the GDC; we recommend using the GDC Data Transfer Tool.
    To prevent needless context window consumption, the manifest file is not returned by this tool.
    Instead, this tool returns a URL from which the user can download the manifest file from.

    Args:
        cohort_handle str: A cohort handle returned by other tools to reference a case set by.
        filters dict: A GDC file filter.

    Returns:
        str: A URL to the downloadable manifest file for the retrieved files.
    """
    case_ids = cache[cohort_handle]
    cache[cohort_handle] = case_ids
    data = await get_gdc_manifest(filters, case_ids)
    manifest_handle = secrets.token_urlsafe(8)
    fpath = f"/tmp/gdc-manifests/{manifest_handle}.tsv"
    with open(fpath, "w+") as f:
        f.write(data)
    mcp_request = get_http_request()
    return f"{str(mcp_request.url)[:-8]}-manifests/download/{manifest_handle}.tsv"


os.makedirs("/tmp/gdc-manifests", exist_ok=True)
app = Starlette(
    routes=[
        Mount("/download", app=StaticFiles(directory="/tmp/gdc-manifests")),
    ],
)


if __name__ == "__main__":
    mcp.run(transport="stdio")
