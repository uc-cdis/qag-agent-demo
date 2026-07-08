import json
import sys
from typing import Any

import requests

GDC_API = "https://api.gdc.cancer.gov"


def truncate(x, max_len=100):
    x = str(x)
    if len(x) > max_len:
        x = x[:max_len]
    return x


def gdc_query_all(
    endpoint: str,
    filters: dict,
    cohort_filter: dict | str | None = None,
    fields: list[str] | None = None,
    page_size: int = 1000,
) -> list[dict[str, Any]]:
    GDC_API
    url = f"{GDC_API}/{endpoint}"
    all_hits = []
    offset = 0

    print(
        f"querying endpoint: {endpoint} with filters: {truncate(filters)} and cohort_filter: {json.loads(cohort_filter) if isinstance(cohort_filter, str) else cohort_filter}",
        file=sys.stderr,
    )

    all_filters = filters
    if cohort_filter:  # not None or empty string
        if isinstance(cohort_filter, str):
            cohort_filter = json.loads(cohort_filter)
        if filters:
            all_filters = {
                "op": "and",
                "content": [
                    cohort_filter,
                    filters,
                ],
            }
        else:
            all_filters = cohort_filter

    while True:
        payload = {
            "filters": all_filters,
            "from": offset,
            "size": page_size,
        }

        print(json.dumps(payload))
        if fields:
            # even if using post, fields needs to be a comma-separated string
            payload["fields"] = ",".join(fields)

        response = requests.post(url, json=payload)
        response.raise_for_status()
        resp_json = response.json()

        if resp_json["warnings"]:
            print(resp_json["warnings"], file=sys.stderr)

        data = resp_json["data"]

        hits = data["hits"]
        all_hits.extend(hits)

        pagination = data["pagination"]
        total = pagination["total"]

        # print(f"Retrieved {len(all_hits)} / {total}", file=sys.stderr)

        offset += page_size
        if offset >= total:
            break

    return all_hits
