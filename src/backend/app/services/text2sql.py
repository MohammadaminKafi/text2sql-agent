from __future__ import annotations
from typing import Any, Tuple
import pandas as pd
from backend.app.deps import get_flow

def run_report(prompt: str, schema_id: str | None = None) -> Tuple[pd.DataFrame, str, str, list[Any]]:
    """
    Calls your existing flow: returns (df, sql, summary, viz)
    """
    flow = get_flow()

    # If you support schema switching, you can set it on the flow or engine here.
    df, sql, summary, viz = flow(prompt)  # your existing interface
    if not isinstance(df, pd.DataFrame):
        # normalize if flow returns something else
        df = pd.DataFrame(df)

    return df, sql, summary, viz
