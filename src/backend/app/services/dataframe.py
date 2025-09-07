from __future__ import annotations
from typing import Any, Iterable, Iterator
import csv, io
import numpy as np
import pandas as pd

def to_json_payload(df: "pd.DataFrame", limit: int) -> tuple[list[str], list[list[Any]]]:
    out = df.head(limit)
    cols = [str(c) for c in out.columns]
    rows = [ [_jsonify_cell(v) for v in row] for row in out.itertuples(index=False, name=None) ]
    return cols, rows

def _jsonify_cell(v: Any) -> Any:
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if pd.isna(f) else f
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        # handle numpy scalars etc.
        return v.item()  # type: ignore[attr-defined]
    except Exception:
        return v

def stream_csv(df: "pd.DataFrame", limit: int | None = None) -> Iterator[str]:
    """Yield CSV chunks (header + rows)."""
    data = df if limit is None else df.head(limit)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([str(c) for c in data.columns])
    yield buf.getvalue(); buf.seek(0); buf.truncate(0)
    for row in data.itertuples(index=False, name=None):
        writer.writerow(row)
        yield buf.getvalue(); buf.seek(0); buf.truncate(0)
