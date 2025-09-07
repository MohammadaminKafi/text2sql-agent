from typing import List, Literal, TypedDict, Optional

class MeasureTD(TypedDict):
    field: str
    agg: Literal["sum", "mean", "count", "min", "max", "median"]

class SortByTD(TypedDict):
    field: str
    order: Literal["asc", "desc"]

class PlanTD(TypedDict, total=False):
    plot_id: str
    plot_type: Literal["plot", "bar", "scatter", "hist", "pie", "box"]
    description: str
    aggregate: bool
    groupby: List[str]
    measures: List[MeasureTD]
    sort_by: Optional[SortByTD]          
    limit: Optional[int]

class PlansOutputTD(TypedDict):
    plans: List[PlanTD]