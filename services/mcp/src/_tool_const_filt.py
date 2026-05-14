import json
import sys

from cachetools import TTLCache

# isort: off
from _defines import (
    CohortFilterID,
    Project,
    PrimarySite,
    DiseaseType,
    Sex,
    Age,
    SmokingStatus,
)
from _utils import get_tool_id_template

# isort: on


def make_construct_filter_tool(cache: TTLCache):
    def construct_cohort_filter(
        project: list[Project] | None = None,
        primary_site: list[PrimarySite] | None = None,
        disease_type: list[DiseaseType] | None = None,
        sex: list[Sex] | None = None,
        min_age: Age | None = None,
        max_age: Age | None = None,
        smoking_status: list[SmokingStatus] | None = None,
    ) -> CohortFilterID:
        """
        A tool to construct a GDC cohort filter for case metadata.
        The resulting JSON filter is cached server side and can be referenced using the unique identifier returned by this tool.

        Args:
            project: GDC project IDs. e.g ["TCGA-BRCA", "TCGA-LUAD"]
            primary_site: Primary site of a disease e.g. ["Lung", "Brain"]
            disease_type: Type of malignant disease e.g. ["adenomas and adenocarcinomas", "leukemia, nos"]
            sex: A person's sex at birth e.g. any / many of ["male", "female", "unknown"],
            min_age : Lower end of the age range, inclusive. Age at the time of diagnosis, expressed in years. e.g 34
            max_age : Higher end of the age range, inclusive. Age at the time of diagnosis, expressed in years. e.g 76
            smoking_status: Self reported current smoking status or smoking history of a patient. e.g. ["lifelong non-smoker", "current smoker"]
        """
        print(
            f"construct_cohort_filter(project={repr(project)}, primary_site={repr(primary_site)}, disease_type={repr(disease_type)}, sex={repr(sex)}, min_age={repr(min_age)}, max_age={repr(max_age)}, smoking_status={repr(smoking_status)})",
            file=sys.stderr,
        )

        cache_id = get_tool_id_template().format(
            project=project,
            primary_site=primary_site,
            disease_type=disease_type,
            sex=sex,
            min_age=min_age,
            max_age=max_age,
            smoking_status=smoking_status,
        )

        # if we've already created this filter, refresh it in the cache and shortcut return
        if cache_id in cache:
            cache[cache_id] = cache[cache_id]
            return cache_id

        field_dict = {
            # fmt: off
            "project":        ("in", "cases.project.project_id"),
            "primary_site":   ("in", "cases.primary_site"),
            "disease_type":   ("in", "cases.disease_type"),
            "sex":            ("in", "cases.demographic.sex_at_birth"),
            "min_age":        (">=", "cases.diagnoses.age_at_diagnosis"),
            "max_age":        ("<=", "cases.diagnoses.age_at_diagnosis"),
            "smoking_status": ("in", "cases.exposures.tobacco_smoking_status"),
            # fmt: on
        }

        value_dict = {
            "project": project,
            "primary_site": primary_site,
            "disease_type": disease_type,
            "sex": sex,
            "min_age": int(min_age * 365) if min_age is not None else None,
            "max_age": int(max_age * 365) if max_age is not None else None,
            "smoking_status": smoking_status,
        }

        filters = []
        for key, value in value_dict.items():
            if value is not None:
                filters.append(
                    {
                        "op": field_dict[key][0],
                        "content": {
                            "field": field_dict[key][1],
                            "value": value,
                        },
                    }
                )

        filter_str = json.dumps({"op": "and", "content": filters}, indent=4)
        cache[cache_id] = filter_str
        return cache_id

    return construct_cohort_filter
