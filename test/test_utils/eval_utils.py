from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd
import csv


def compare_dataframes_as_dataframe_safe(gt_df: pd.DataFrame, out_df: pd.DataFrame) -> pd.DataFrame:
    result_dict = {
        'Metric': [
            'gt_rows', 'out_rows', 'gt_not_in_out', 'out_not_in_gt', 'common_rows',
            'gt_cols', 'out_cols', 'gt_not_in_out_cols', 'out_not_in_gt_cols', 'common_cols',
            'exact_match', 'gt_in_out', 'out_in_gt', 'ordered_same', 'cols_type_match'
        ],
        'Value': []
    }

    try:
        gt_rows = len(gt_df)
    except Exception:
        gt_rows = None
    result_dict['Value'].append(gt_rows)

    try:
        out_rows = len(out_df)
    except Exception:
        out_rows = None
    result_dict['Value'].append(out_rows)

    try:
        gt_not_in_out = len(pd.merge(gt_df, out_df, how='left', indicator=True).query('_merge == "left_only"'))
    except Exception:
        gt_not_in_out = None
    result_dict['Value'].append(gt_not_in_out)

    try:
        out_not_in_gt = len(pd.merge(out_df, gt_df, how='left', indicator=True).query('_merge == "left_only"'))
    except Exception:
        out_not_in_gt = None
    result_dict['Value'].append(out_not_in_gt)

    try:
        common_rows = len(pd.merge(gt_df, out_df, how='inner'))
    except Exception:
        common_rows = None
    result_dict['Value'].append(common_rows)

    try:
        gt_cols = len(gt_df.columns)
    except Exception:
        gt_cols = None
    result_dict['Value'].append(gt_cols)

    try:
        out_cols = len(out_df.columns)
    except Exception:
        out_cols = None
    result_dict['Value'].append(out_cols)

    try:
        gt_not_in_out_cols = len(set(gt_df.columns) - set(out_df.columns))
    except Exception:
        gt_not_in_out_cols = None
    result_dict['Value'].append(gt_not_in_out_cols)

    try:
        out_not_in_gt_cols = len(set(out_df.columns) - set(gt_df.columns))
    except Exception:
        out_not_in_gt_cols = None
    result_dict['Value'].append(out_not_in_gt_cols)

    try:
        common_cols = len(set(gt_df.columns) & set(out_df.columns))
    except Exception:
        common_cols = None
    result_dict['Value'].append(common_cols)

    try:
        exact_match = gt_df.equals(out_df)
    except Exception:
        exact_match = None
    result_dict['Value'].append(exact_match)

    try:
        gt_in_out = gt_df.shape[0] <= out_df.shape[0] and gt_df.columns.isin(out_df.columns).all() and gt_df.equals(out_df.iloc[:gt_df.shape[0], :])
    except Exception:
        gt_in_out = None
    result_dict['Value'].append(gt_in_out)

    try:
        out_in_gt = out_df.shape[0] <= gt_df.shape[0] and out_df.columns.isin(gt_df.columns).all() and out_df.equals(gt_df.iloc[:out_df.shape[0], :])
    except Exception:
        out_in_gt = None
    result_dict['Value'].append(out_in_gt)

    try:
        ordered_same = gt_df.equals(out_df)
    except Exception:
        ordered_same = None
    result_dict['Value'].append(ordered_same)

    try:
        cols_type_match = (gt_df.dtypes == out_df.dtypes).all()
    except Exception:
        cols_type_match = None
    result_dict['Value'].append(cols_type_match)

    return pd.DataFrame(result_dict)


def generate_final_report(model_dir: Path) -> None:
    final_path = model_dir / "final_report.csv"
    header = [
        "category",
        "cases",
        "jaccard",
        "row_match_rate",
        "match_rate",
        "exact_match_rate",
    ]
    all_frames = []
    with open(final_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for cat_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            summary_path = cat_dir / "summary.csv"
            if not summary_path.exists():
                continue
            df = pd.read_csv(summary_path)
            all_frames.append(df)
            metrics = [
                df["jaccard"].mean(),
                df["row_match"].mean(),
                df["match"].mean(),
                df["exact_match"].mean(),
            ]
            writer.writerow(
                [cat_dir.name, len(df)] + [f"{m:.3f}" if m == m else "0.000" for m in metrics]
            )

        if all_frames:
            df_all = pd.concat(all_frames, ignore_index=True)
            metrics = [
                df_all["jaccard"].mean(),
                df_all["row_match"].mean(),
                df_all["match"].mean(),
                df_all["exact_match"].mean(),
            ]
            writer.writerow(
                ["OVERALL", len(df_all)] + [f"{m:.3f}" if m == m else "0.000" for m in metrics]
            )


def generate_language_summary(model_dir: Path, languages: List[str]) -> None:
    """Aggregate OVERALL metrics for each language into a CSV."""
    rows = []
    for lang in languages:
        final = model_dir / lang / "final_report.csv"
        if not final.exists():
            continue
        df = pd.read_csv(final)
        if df.empty:
            continue
        overall = df[df["category"] == "OVERALL"].iloc[0]
        rows.append([
            lang,
            overall["cases"],
            overall["jaccard"],
            overall["row_match_rate"],
            overall["match_rate"],
            overall["exact_match_rate"],
        ])
    if rows:
        out_path = model_dir / "language_summary.csv"
        header = [
            "language",
            "cases",
            "jaccard",
            "row_match_rate",
            "match_rate",
            "exact_match_rate",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
