import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------- Utilities ----------

def infer_schema(df: pd.DataFrame) -> Dict[str, str]:
    """
    Map column -> {'type': 'numeric'|'categorical'|'datetime'|'boolean'|'unknown'}
    """
    schema = {}
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            schema[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            schema[col] = "datetime"
        elif pd.api.types.is_bool_dtype(dtype):
            schema[col] = "boolean"
        else:
            schema[col] = "categorical"
    return schema

def head_as_dict(df: pd.DataFrame, n: int = 5) -> Dict[str, List[Any]]:
    return df.head(n).to_dict(orient="list")

# ---------- Helpers: Aggregator and Drawer ----------

def apply_filters(df: pd.DataFrame, filters: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    if not filters:
        return df
    out = df.copy()
    for f in filters:
        field = f.get("field")
        op = f.get("op")
        value = f.get("value")
        if field not in out.columns:
            continue
        if op == "==":
            out = out[out[field] == value]
        elif op == "!=":
            out = out[out[field] != value]
        elif op == ">":
            out = out[out[field] > value]
        elif op == "<":
            out = out[out[field] < value]
        elif op == ">=":
            out = out[out[field] >= value]
        elif op == "<=":
            out = out[out[field] <= value]
        elif op == "in" and isinstance(value, list):
            out = out[out[field].isin(value)]
        elif op == "not in" and isinstance(value, list):
            out = out[~out[field].isin(value)]
        # silently ignore unknown ops
    return out

def aggregate_dataframe(df: pd.DataFrame, plan: Dict[str, Any]) -> pd.DataFrame:
    """
    Aggregates df according to the planner fields. Supports:
      - groupby (list[str])
      - series_by (optional)
      - measures (list of {"field","agg"})
      - sort_by {field, order}
      - limit (top-k)
    If plan["aggregate"] is False, returns df (after optional filters).
    """
    df2 = apply_filters(df, plan.get("filters"))

    if not plan.get("aggregate"):
        # Even when not aggregating, optional sort/limit can apply
        sort_by = plan.get("sort_by")
        if sort_by and sort_by.get("field") in df2.columns:
            ascending = sort_by.get("order", "asc") == "asc"
            df2 = df2.sort_values(by=sort_by["field"], ascending=ascending)
        if plan.get("limit"):
            df2 = df2.head(int(plan["limit"]))
        return df2

    groupby = list(plan.get("groupby") or [])
    series_by = plan.get("series_by")
    measures = plan.get("measures") or []

    keys = groupby + ([series_by] if series_by else [])
    keys = [k for k in keys if k and k in df2.columns]

    if not measures:
        # default to count
        grouped = df2.groupby(keys, dropna=False).size().reset_index(name="count")
    else:
        agg_map: Dict[str, Any] = {}
        for m in measures:
            field = m.get("field")
            agg = (m.get("agg") or "sum").lower()
            if field in df2.columns:
                agg_map.setdefault(field, []).append(agg)
        if not agg_map:
            grouped = df2.groupby(keys, dropna=False).size().reset_index(name="count")
        else:
            grouped = (
                df2.groupby(keys, dropna=False)
                   .agg(agg_map)
                   .reset_index()
            )
            # flatten MultiIndex columns
            grouped.columns = [
                "_".join([c for c in col if c]) if isinstance(col, tuple) else col
                for col in grouped.columns
            ]

    sort_by = plan.get("sort_by")
    if sort_by and sort_by.get("field") in grouped.columns:
        ascending = sort_by.get("order", "asc") == "asc"
        grouped = grouped.sort_values(by=sort_by["field"], ascending=ascending)
    if plan.get("limit"):
        grouped = grouped.head(int(plan["limit"]))

    return grouped

def draw_plot(df: pd.DataFrame, spec: Dict[str, Any]):
    """
    Draw a matplotlib plot based on the normalized spec.
    Supports: plot (line), bar, scatter, hist, pie, box
    Returns (fig, ax)
    """
    plot_type = spec.get("plot_type")
    mappings = spec.get("mappings", {})
    title = spec.get("title", "")
    annotations = spec.get("annotations") or []

    fig, ax = plt.subplots()

    if plot_type == "plot":  # line
        x = mappings.get("x")
        y = mappings.get("y")
        series_by = mappings.get("series_by")
        order_by = mappings.get("order_by") or x

        plot_df = df.copy()
        if order_by in plot_df.columns:
            plot_df = plot_df.sort_values(order_by)

        if series_by and series_by in plot_df.columns:
            for key, g in plot_df.groupby(series_by, dropna=False):
                ax.plot(g[x], g[y], label=str(key))
            ax.legend()
        else:
            ax.plot(plot_df[x], plot_df[y])

        ax.set_xlabel(x or "")
        ax.set_ylabel(y or "")

    elif plot_type == "bar":
        x = mappings.get("x")
        y = mappings.get("y")
        series_by = mappings.get("series_by")
        stacked = bool(mappings.get("stacked", False))

        plot_df = df.copy()
        if series_by and series_by in plot_df.columns:
            piv = plot_df.pivot_table(index=x, columns=series_by, values=y, aggfunc="sum", fill_value=0)
            piv.sort_index(inplace=True)
            if stacked:
                piv.plot(kind="bar", stacked=True, ax=ax, legend=True)
            else:
                piv.plot(kind="bar", stacked=False, ax=ax, legend=True)
        else:
            ax.bar(plot_df[x], plot_df[y])

        ax.set_xlabel(x or "")
        ax.set_ylabel(y or "")

    elif plot_type == "scatter":
        x = mappings.get("x")
        y = mappings.get("y")
        hue = mappings.get("hue")
        size = mappings.get("size")

        plot_df = df.copy()
        if hue and hue in plot_df.columns:
            # draw simple multiple series
            for key, g in plot_df.groupby(hue, dropna=False):
                if size and size in g.columns:
                    ax.scatter(g[x], g[y], s=g[size], label=str(key))
                else:
                    ax.scatter(g[x], g[y], label=str(key))
            ax.legend()
        else:
            if size and size in plot_df.columns:
                ax.scatter(plot_df[x], plot_df[y], s=plot_df[size])
            else:
                ax.scatter(plot_df[x], plot_df[y])

        ax.set_xlabel(x or "")
        ax.set_ylabel(y or "")

    elif plot_type == "hist":
        x = mappings.get("x")
        bins = mappings.get("bins", 30)
        by = mappings.get("by")

        plot_df = df.copy()
        if by and by in plot_df.columns:
            for key, g in plot_df.groupby(by, dropna=False):
                ax.hist(g[x], bins=bins, alpha=0.5, label=str(key))
            ax.legend()
        else:
            ax.hist(plot_df[x], bins=bins)

        ax.set_xlabel(x or "")
        ax.set_ylabel("count")

    elif plot_type == "pie":
        labels_col = mappings.get("labels")
        sizes_col = mappings.get("sizes")

        plot_df = df.copy()
        labels = plot_df[labels_col].astype(str).tolist()
        sizes = plot_df[sizes_col].tolist()
        ax.pie(sizes, labels=labels, autopct="%1.1f%%")

    elif plot_type == "box":
        x = mappings.get("x")   # optional categorical
        y = mappings.get("y")   # required numeric
        by = mappings.get("by") # optional

        plot_df = df.copy()
        if x and x in plot_df.columns:
            # pandas style boxplot w/ by
            if by and by in plot_df.columns:
                plot_df.boxplot(column=y, by=[x, by], ax=ax)
            else:
                plot_df.boxplot(column=y, by=x, ax=ax)
            ax.set_xlabel(x)
        else:
            plot_df.boxplot(column=y, ax=ax)
            ax.set_xlabel("")
        ax.set_ylabel(y or "")

    else:
        plt.close(fig)
        raise ValueError(f"Unsupported plot_type: {plot_type}")

    if title:
        ax.set_title(title)
        # For pandas boxplot with "by", pandas auto-sets a suptitle:
        if plot_type == "box":
            fig.suptitle("")

    for note in annotations:
        logger.info("Note for plot: %s", note)

    fig.tight_layout()
    return fig, ax

