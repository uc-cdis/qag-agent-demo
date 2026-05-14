import json
import sys

from cachetools import TTLCache

# isort: off
from _defines import Project, CohortFilterID
from _utils import get_tool_id_template

# isort: on


def make_construct_filter_tool(cache: TTLCache):
    def construct_cohort_filter(project: Project) -> CohortFilterID:
        """
        A tool to construct a GDC cohort filter by project, for example 'TCGA-BRCA'.
        The resulting JSON filter is cached server side and can be referenced using the unique identifier returned by this tool.
        """
        print(
            f"get_cohort_filter_by_project(project={repr(project)})",
            file=sys.stderr,
        )

        cache_id = get_tool_id_template().format(project=project)

        # if we've already created this filter, refresh it in the cache and shortcut return
        if cache_id in cache:
            cache[cache_id] = cache[cache_id]
            return cache_id

        filter_str = json.dumps(
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [project],
                },
            },
            indent=4,
        )

        cache[cache_id] = filter_str

        return cache_id

    return construct_cohort_filter
