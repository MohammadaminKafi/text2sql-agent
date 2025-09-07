from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import ORJSONResponse, StreamingResponse
from backend.app.config import settings
from backend.app.models import ReportRequest, ReportResponse, DataFramePayload, PlotImage
from backend.app.services.text2sql import run_report
from backend.app.services.dataframe import to_json_payload, stream_csv
from backend.app.services.plotting import serialize_plots

router = APIRouter(prefix="/api/v1", tags=["report"])

@router.post("/report", response_class=ORJSONResponse, response_model=ReportResponse)
def report(req: ReportRequest):
    try:
        df, sql, summary, viz = run_report(req.prompt, req.schema_id)
        cols, rows = to_json_payload(df, settings.JSON_ROW_LIMIT)
        plots = serialize_plots(viz) if req.include_plots else []
        return ReportResponse(
            summary=summary,
            sql=sql,
            data=DataFramePayload(columns=cols, rows=rows, rowCount=len(rows)),
            plots=[PlotImage(**p) for p in plots],
            warnings=[],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to satisfy prompt: {e}")

@router.get("/report.csv")
def report_csv(
    prompt: str = Query(..., description="User's NL question"),
    schema_id: str | None = Query(None),
    limit: int | None = Query(None, ge=1, description="Optional row cap for CSV"),
):
    """
    CSV download endpoint.
    - If `limit` is omitted, uses settings.CSV_DEFAULT_LIMIT (None = no cap).
    - Always streams; suitable for large results.
    """
    try:
        df, _, _, _ = run_report(prompt, schema_id)
        row_limit = settings.CSV_DEFAULT_LIMIT if limit is None else limit
        gen = stream_csv(df, row_limit)
        headers = {"Content-Disposition": 'attachment; filename="report.csv"'}
        return StreamingResponse(gen, media_type="text/csv", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV generation failed: {e}")
