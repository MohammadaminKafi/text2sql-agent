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

from test_utils.dataset_utils import collect_tests, verify_dataset
from test_utils.eval_utils import (
    compare_dataframes_as_dataframe_safe,
    generate_final_report,
    generate_language_summary,
)

from openai import OpenAI
from vanna.src.vanna.base.base import VannaBase
from vanna.src.vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.src.vanna.openai.openai_chat import OpenAI_Chat

try:
    from no_commit_utils.credentials_utils import read_credentials
except Exception:
    def read_avalai_api_key() -> str:
        return os.environ.get("OPENAI_API_KEY", "")
    def read_metis_api_key() -> str:
        return os.environ.get("OPENAI_API_KEY", "")

use_avalai = True

metis_base_url = "https://api.metisai.ir/openai/v1"
avalai_base_url = "https://api.avalai.ir/v1"

API_KEY = read_credentials("avalai.key") if use_avalai else read_credentials("metis.key")
API_BASE_URL = avalai_base_url if use_avalai else read_credentials("metis.key")
DEFAULT_MODEL = "gpt-4o-mini" # "gemma-3-27b-it"
DEFAULT_MODE = "ask_agent"
AGENT_TOOLKIT = ["run_sql", "query_rag"]


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
    parser.add_argument("--method", choices=["ask", "ask_agent"], default=DEFAULT_MODE,
                        help="Vanna method to invoke")
    parser.add_argument("--level", type=int, default=1,
                        help="Number of prompt variants to evaluate")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help=(
            "Languages of the prompt files. Additional prompts must follow the "
            "<lang>_promptXX.json naming convention."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Specific test categories to run. Defaults to all except 'misc'",
    )
    return parser.parse_args()


def run_test_case(
    vn: VannaBase,
    prompt: str,
    ground_sql: str,
    gt_df: pd.DataFrame,
    method: str,
) -> Tuple[pd.DataFrame | None, str, float, str]:
    """Execute a single test case and return dataframe, generated SQL, duration, and error."""

    start = time.time()
    generated_sql = ""
    error = ""
    df: pd.DataFrame | None = None
    try:
        if prompt is None:
            raise ValueError("Prompt is None")
        if method == "ask_agent":
            result = vn.ask_agent(question=prompt, print_results=False)
            if isinstance(result, tuple):
                if len(result) > 0 and isinstance(result[0], str):
                    generated_sql = result[0]
                df = result[1] if len(result) > 1 else None
            elif isinstance(result, pd.DataFrame):
                df = result
            elif isinstance(result, str):
                generated_sql = result
                df = vn.run_sql(result)
        else:
            generated_sql, df, _ = vn.ask(question=prompt, print_results=False, visualize=False)
    except Exception as e:
        error = str(e)
        df = None
    duration = time.time() - start
    return df, generated_sql, duration, error


def main() -> None:
    args = parse_args()

    print("\nArguments:")
    for key, value in vars(args).items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    dataset_dir = args.dataset_dir
    log_dir = args.log_dir
    log_dir.mkdir(exist_ok=True)

    all_categories = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    categories = args.categories or [c for c in all_categories if c != "misc"]
    unknown = set(categories) - set(all_categories)
    if unknown:
        raise ValueError(f"Unknown categories: {', '.join(sorted(unknown))}")

    openai_cfg = {"api_key": API_KEY}
    verify_dataset(dataset_dir, args.languages)

    vn = MyVanna(openai_config=openai_cfg, llm_config={"model": args.model})
    vn.connect_to_mssql(odbc_conn_str=args.conn_str)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = log_dir / f"{args.model}-{args.method}-{current_time}"
    model_dir.mkdir(exist_ok=True)
    
    if args.method == "ask_agent":
        vn.create_agent(
            model=DEFAULT_MODEL,
            api_base=API_BASE_URL,
            api_key=API_KEY,
            agent_toolkit=AGENT_TOOLKIT
        )

    if vn.run_sql_is_set:
        print("\nVanna is connected to the database")

    if vn.test_llm_connection():
        print("\nVanna is connected to the LLM provider")

    for language in args.languages:
        print(f"\nRunning language: {language}")
        all_tests = collect_tests(dataset_dir, language=language, categories=categories)
        total = sum(len(v) for v in all_tests.values())
        print(f"Number of tests: {total}")
        lang_dir = model_dir / language
        lang_dir.mkdir(exist_ok=True)

        for category, cases in all_tests.items():
            cat_dir = lang_dir / category
            cat_dir.mkdir(exist_ok=True)
            summary_path = cat_dir / "summary.csv"

            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "case",
                    "prompt_type",
                    "sql_path",
                    "output_path",
                    "duration_sec",
                    "generated_sql",
                    "prompt_file",
                    "error",
                    "match",
                    "row_match",
                    "gt_rows", "out_rows", "gt_not_in_out", "out_not_in_gt", "common_rows",
                    "gt_cols", "out_cols", "gt_not_in_out_cols", "out_not_in_gt_cols", "common_cols",
                    "exact_match", "gt_in_out", "out_in_gt", "ordered_same", "cols_type_match",
                    "jaccard"
                ])

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
                        print(f"{args.model} | {language} | {category}-{idx} {p_type}")

                        df_out, gen_sql, duration, error = run_test_case(
                            vn, p_text, sql_path.read_text(), gt_df, args.method
                        )
                        out_path = cat_dir / f"case{idx:02d}_{p_type}.csv"
                        gen_sql_path = cat_dir / f"case{idx:02d}_{p_type}_gen.sql"
                        prompt_txt_path = cat_dir / f"case{idx:02d}_{p_type}_prompt.txt"

                        if df_out is not None:
                            df_out.to_csv(out_path, index=False)

                        gen_sql_path.write_text(gen_sql or "", encoding="utf-8")
                        prompt_txt_path.write_text(p_text or "", encoding="utf-8")

                        comparison_result = compare_dataframes_as_dataframe_safe(gt_df, df_out)

                        comparison_values = comparison_result['Value'].tolist()

                        metrics = dict(zip(comparison_result['Metric'], comparison_values))
                        gt_rows = metrics.get('gt_rows') or 0
                        out_rows = metrics.get('out_rows') or 0
                        common_rows = metrics.get('common_rows') or 0

                        row_match = gt_rows == out_rows
                        union_rows = gt_rows + out_rows - common_rows
                        jaccard = common_rows / union_rows if union_rows else 0
                        match = bool(df_out is not None and row_match and metrics.get("gt_cols") == metrics.get("out_cols"))
                        comparison_values.append(jaccard)

                        writer.writerow([
                            idx,
                            p_type,
                            sql_path.name,
                            out_path.name,
                            f"{duration:.2f}",
                            gen_sql_path.name,
                            prompt_txt_path.name,
                            error,
                            match,
                            row_match,
                        ] + comparison_values)

                        status = "SUCCESS" if not error else f"ERROR: {error}"
                        print(
                            f"-> {status}, match={match}, row_match={row_match}, jaccard={jaccard:.3f}"
                        )

        generate_final_report(lang_dir)

    generate_language_summary(model_dir, args.languages)

if __name__ == "__main__":
    main()
