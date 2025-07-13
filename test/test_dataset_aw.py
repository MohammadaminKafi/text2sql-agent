# coding: utf-8
"""Test AdventureWorks2022 prompts against Vanna models.

This script executes all SQL files in the dataset to create ground truth
results and then evaluates the output from Vanna's ask functions. Reports,
SQL, and intermediate data are stored in the ``logs`` directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import time
from typing import Callable, Dict, Iterable, List, Tuple
import argparse
from datetime import datetime
import csv
import pandas as pd

from openai import OpenAI
from vanna.src.vanna.base.base import VannaBase
from vanna.src.vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.src.vanna.openai.openai_chat import OpenAI_Chat

try:
    from no_commit_utils.credentials_utils import read_avalai_api_key, read_metis_api_key
except Exception:
    def read_avalai_api_key() -> str:
        return os.environ.get("OPENAI_API_KEY", "")
    def read_metis_api_key() -> str:
        return os.environ.get("OPENAI_API_KEY", "")

API_KEY = read_metis_api_key()
API_BASE_URL = "https://api.metisai.ir/openai/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    """Minimal Vanna wrapper used for dataset testing."""

    def __init__(
        self,
        openai_config: Dict[str, str],
        llm_config: Dict[str, str] | None = None,
        vdb_config: Dict[str, str] | None = None,
    ) -> None:
        client = OpenAI(
            api_key=openai_config["api_key"],
            base_url=openai_config.get("base_url", API_BASE_URL),
        )
        ChromaDB_VectorStore.__init__(self, config=vdb_config or {})
        OpenAI_Chat.__init__(self, config=llm_config or {}, client=client)


DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "dataset_AdventureWorks2022"
LOG_DIR = Path(__file__).resolve().parent / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Default connection string
CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=AdventureWorks2022;"
    "Trusted_Connection=yes;"
)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset tests")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR,
                        help="Directory containing prompt/query pairs")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR,
                        help="Directory to write logs")
    parser.add_argument("--conn-str", default=CONN_STR,
                        help="ODBC connection string for ground truth queries")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model name to evaluate")
    parser.add_argument("--method", choices=["ask", "ask_agent"], default="ask",
                        help="Vanna method to invoke")
    parser.add_argument("--level", type=int, default=1,
                        help="Number of prompt variants to evaluate")
    parser.add_argument(
        "--language",
        choices=["en", "fa"],
        default="en",
        help="Language of the prompt files (en or fa)",
    )
    return parser.parse_args()

def load_prompt(path: Path) -> Dict[str, str]:
    """Return prompts from JSON file."""

    import json

    return json.loads(path.read_text())

def collect_tests(
    dataset_dir: Path, language: str = "en"
) -> Dict[str, List[Tuple[Path, Path, Dict[str, str]]]]:
    """Collect pairs of query and prompt files grouped by category."""

    tests: Dict[str, List[Tuple[Path, Path, Dict[str, str]]]] = {}
    for category in sorted(p.name for p in dataset_dir.iterdir() if p.is_dir()):
        cat_dir = dataset_dir / category
        queries = sorted(cat_dir.glob("query*.sql"))
        pattern = "prompt*.json" if language == "en" else "persian_prompt*.json"
        prompts = sorted(cat_dir.glob(pattern))
        cases: List[Tuple[Path, Path, Dict[str, str]]] = []
        for q, pth in zip(queries, prompts):
            cases.append((q, pth, load_prompt(pth)))
        tests[category] = cases
    return tests

def compare_dataframes_as_dataframe_safe(gt_df: pd.DataFrame, out_df: pd.DataFrame):
    result_dict = {
        'Metric': [
            'gt_rows', 'out_rows', 'gt_not_in_out', 'out_not_in_gt', 'common_rows',
            'gt_cols', 'out_cols', 'gt_not_in_out_cols', 'out_not_in_gt_cols', 'common_cols',
            'exact_match', 'gt_in_out', 'out_in_gt', 'ordered_same', 'cols_type_match'
        ],
        'Value': []
    }
    
    # Safe calculation for each field with try-except blocks
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
        ordered_same = gt_df.equals(out_df)  # checks both values and order
    except Exception:
        ordered_same = None
    result_dict['Value'].append(ordered_same)

    try:
        cols_type_match = (gt_df.dtypes == out_df.dtypes).all()
    except Exception:
        cols_type_match = None
    result_dict['Value'].append(cols_type_match)

    # Return the result as a DataFrame
    return pd.DataFrame(result_dict)

def generate_final_report(model_dir: Path) -> None:
    """Create a final aggregate report with accuracy metrics."""

    final_path = model_dir / "final_report.csv"
    header = [
        "category",
        "cases",
        "accuracy",
        "precision",
        "recall",
        "f1",
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
                df["accuracy"].mean(),
                df["precision"].mean(),
                df["recall"].mean(),
                df["f1"].mean(),
                df["match"].mean(),
                df["exact_match"].mean(),
            ]
            writer.writerow(
                [cat_dir.name, len(df)] + [f"{m:.3f}" if m == m else "0.000" for m in metrics]
            )

        if all_frames:
            df_all = pd.concat(all_frames, ignore_index=True)
            metrics = [
                df_all["accuracy"].mean(),
                df_all["precision"].mean(),
                df_all["recall"].mean(),
                df_all["f1"].mean(),
                df_all["match"].mean(),
                df_all["exact_match"].mean(),
            ]
            writer.writerow(
                ["OVERALL", len(df_all)] + [f"{m:.3f}" if m == m else "0.000" for m in metrics]
            )

def run_test_case(
    vn: VannaBase,
    prompt: str,
    ground_sql: str,
    gt_df: pd.DataFrame,
    method: str,
) -> pd.DataFrame | None:
    """Execute a single test case and return the resulting dataframe."""

    try:
        if prompt is None:
            return None
        if method == "ask_agent":
            result = vn.ask_agent(question=prompt)
            if isinstance(result, tuple):
                df = result[1] if len(result) > 1 else None
            elif isinstance(result, pd.DataFrame):
                df = result
            elif isinstance(result, str):
                df = vn.run_sql(result)
            else:
                df = None
        else:
            _, df, _ = vn.ask(question=prompt, print_results=False, visualize=False)
    except Exception:
        return None
    if df is None:
        return None
    return df


def main() -> None:
    args = parse_args()

    print("\nArguments:")
    for key, value in vars(args).items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    dataset_dir = args.dataset_dir
    log_dir = args.log_dir
    log_dir.mkdir(exist_ok=True)

    openai_cfg = {"api_key": API_KEY}
    all_tests = collect_tests(dataset_dir, language=args.language)
    total = sum(len(v) for v in all_tests.values())
    print(f"\nNumber of tests: {total}")

    vn = MyVanna(openai_config=openai_cfg, llm_config={"model": args.model})
    vn.connect_to_mssql(odbc_conn_str=args.conn_str)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = log_dir / f"{args.model}-{args.method}-{current_time}"
    model_dir.mkdir(exist_ok=True)

    if vn.run_sql_is_set:
        print("\nVanna is connected to the database")

    if vn.test_llm_connection():
        print("\nVanna is connected to the LLM provider")

    test_count = 0


    for category, cases in all_tests.items():
        cat_dir = model_dir / category
        cat_dir.mkdir(exist_ok=True)
        summary_path = cat_dir / "summary.csv"
        
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "case",
                "prompt_type",
                "sql_path",
                "output_path",
                "match",
                "gt_rows", "out_rows", "gt_not_in_out", "out_not_in_gt", "common_rows",
                "gt_cols", "out_cols", "gt_not_in_out_cols", "out_not_in_gt_cols", "common_cols",
                "exact_match", "gt_in_out", "out_in_gt", "ordered_same", "cols_type_match",
                "accuracy", "precision", "recall", "f1"
            ])

            test_count = 0
            total = len(all_tests)

            for idx, (sql_path, prompt_path, prompts) in enumerate(cases, start=0):
                gt_path = cat_dir / f"case{idx:02d}_gt.csv"
                if gt_path.exists():
                    gt_df = pd.read_csv(gt_path)
                else:
                    gt_df = vn.run_sql(sql_path.read_text())
                    gt_df.to_csv(gt_path, index=False)

                prompt_order = [
                    ("well", prompts.get("well-explained")),
                    ("poor", prompts.get("poorly-explained")),
                    ("under", prompts.get("underspecified")),
                ]
                prompt_order = [p for p in prompt_order if p[1] is not None]
                prompt_order = prompt_order[:args.level]

                for p_type, p_text in prompt_order:
                    test_count += 1
                    print(f"{args.model} | {category}-{idx} {p_type}")
                    
                    df_out = run_test_case(
                        vn, p_text, sql_path.read_text(), gt_df, args.method
                    )
                    out_path = cat_dir / f"case{idx:02d}_{p_type}.csv"

                    if df_out is not None:
                        df_out.to_csv(out_path, index=False)

                    # Compare the dataframes and aggregate the result
                    comparison_result = compare_dataframes_as_dataframe_safe(gt_df, df_out)

                    # Flatten the comparison result to append it to the CSV row
                    comparison_values = comparison_result['Value'].tolist()

                    metrics = dict(zip(comparison_result['Metric'], comparison_values))
                    gt_rows = metrics.get('gt_rows') or 0
                    out_rows = metrics.get('out_rows') or 0
                    common_rows = metrics.get('common_rows') or 0
                    precision = common_rows / out_rows if out_rows else 0
                    recall = common_rows / gt_rows if gt_rows else 0
                    accuracy = common_rows / max(gt_rows, out_rows) if max(gt_rows, out_rows) else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

                    match = bool(df_out is not None and gt_rows == out_rows and metrics.get('gt_cols') == metrics.get('out_cols'))

                    comparison_values += [accuracy, precision, recall, f1]

                    # Write the results in the summary CSV
                    writer.writerow([idx, p_type, sql_path.name, out_path.name, match] + comparison_values)


    generate_final_report(model_dir)

if __name__ == "__main__":
    main()
