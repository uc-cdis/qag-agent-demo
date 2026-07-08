from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator, Field, StringConstraints, Tag


def content_discriminator(v):
    if isinstance(v, BaseModel):
        field = getattr(v, "field", "")
    elif isinstance(v, dict):
        field = v.get("field", "")
    else:
        return "str_content"

    match field:
        case "cases.diagnoses.age_at_diagnosis":
            return "age_dx"
        case "cases.diagnoses.year_of_diagnosis":
            return "year_dx"
        case "cases.exposures.cigarettes_per_day":
            return "cig_day"
        case "cases.exposures.pack_years_smoked":
            return "pack_yr"
        case "cases.exposures.tobacco_smoking_onset_year":
            return "cig_start"
        case _:
            return "str_content"


class Op(Enum):
    IN = "in"
    EQ = "="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="


class StrContent(BaseModel):
    field: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    value: list[Annotated[str, StringConstraints(min_length=1, max_length=128)]]


class AgeDxContent(BaseModel):
    field: Literal["cases.diagnoses.age_at_diagnosis"]
    value: Annotated[int, Field(ge=0, le=32872)]


class YearDxContent(BaseModel):
    field: Literal["cases.diagnoses.year_of_diagnosis"]
    value: Annotated[int, Field(ge=1900, le=2050)]


class CigDayContent(BaseModel):
    field: Literal["cases.exposures.cigarettes_per_day"]
    value: Annotated[int, Field(ge=0, le=999999)]


class PackYrContent(BaseModel):
    field: Literal["cases.exposures.pack_years_smoked"]
    value: Annotated[int, Field(ge=0, le=999999)]


class CigStrtContent(BaseModel):
    field: Literal["cases.exposures.tobacco_smoking_onset_year"]
    value: Annotated[int, Field(ge=1900, le=2050)]


class Filter(BaseModel):
    op: Op
    content: Annotated[
        Union[
            Annotated[AgeDxContent, Tag("age_dx")],
            Annotated[YearDxContent, Tag("year_dx")],
            Annotated[CigDayContent, Tag("cig_day")],
            Annotated[PackYrContent, Tag("pack_yr")],
            Annotated[CigStrtContent, Tag("cig_start")],
            Annotated[StrContent, Tag("str_content")],
        ],
        Discriminator(content_discriminator),
    ]


class GDCCohortSchema(BaseModel):
    op: Literal["and"]
    content: list[Filter]
