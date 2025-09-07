from typing import Any, List, Optional
from pydantic import BaseModel, Field

class ReportRequest(BaseModel):
    prompt: str = Field(..., description="User's NL question")
    schema_id: Optional[str] = None
    include_plots: bool = True  # keep plots supported

class DataFramePayload(BaseModel):
    columns: List[str]
    rows: List[List[Any]]
    rowCount: int

class PlotImage(BaseModel):
    format: str = "png"
    b64: str
    width: Optional[int] = None
    height: Optional[int] = None
    caption: Optional[str] = None

class ReportResponse(BaseModel):
    summary: str
    sql: str
    data: DataFramePayload
    plots: list[PlotImage] = []
    warnings: list[str] = []
    