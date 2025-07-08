# coding: utf-8
"""Test AdventureWorks2022 prompts against Vanna models.

This script executes all SQL files in the dataset to create ground truth
results and then evaluates the output from Vanna's ask functions. Reports,
SQL, and intermediate data are stored in the ``logs`` directory.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd
from openai import OpenAI

from vanna.src.vanna.base.base import VannaBase
from vanna.src.vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.src.vanna.openai.openai_chat import OpenAI_Chat

try:
    from no_commit_utils.credentials_utils import read_avalai_api_key
except Exception:  # pragma: no cover - optional dependency
    def read_avalai_api_key() -> str:
        return os.environ.get("OPENAI_API_KEY", "")


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
            base_url=openai_config.get("base_url", "https://api.avalapis.ir/v1"),
        )
        ChromaDB_VectorStore.__init__(self, config=vdb_config or {})
        OpenAI_Chat.__init__(self, config=llm_config or {}, client=client)


DATASET_DIR = Path(__file__).resolve().parent / "dataset_AdventureWorks2022"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# [model_name, vanna_method, test_level]
TEST_CONFIGS: List[List] = [
    ["gpt-4o", "ask", 1],
]

CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=AdventureWorks2022;"
    "Trusted_Connection=yes;"
)


def load_prompt(path: Path) -> Dict[str, str]:
    """Return prompts from file."""

    text = path.read_text().strip().splitlines()
    sections: Dict[str, str] = {}
    current = None
    buffer: List[str] = []
    for line in text:
        line = line.strip()
        if line.endswith("Prompt"):
            if current:
                sections[current] = "\n".join(buffer).strip().strip('"')
            current = line.split()[0].lower()
            buffer = []
        else:
            buffer.append(line)
    if current:
        sections[current] = "\n".join(buffer).strip().strip('"')
    return sections


def collect_tests(dataset_dir: Path) -> Dict[str, List[Tuple[Path, Path, Dict[str, str]]]]:
    """Collect pairs of query and prompt files grouped by category."""

    tests: Dict[str, List[Tuple[Path, Path, Dict[str, str]]]] = {}
    for category in sorted(p.name for p in dataset_dir.iterdir() if p.is_dir()):
        cat_dir = dataset_dir / category
        queries = sorted(cat_dir.glob("query*.sql"))
        prompts = sorted(cat_dir.glob("prompt*.txt"))
        cases: List[Tuple[Path, Path, Dict[str, str]]] = []
        for q, pth in zip(queries, prompts):
            cases.append((q, pth, load_prompt(pth)))
        tests[category] = cases
    return tests


def compare_frames(gt: pd.DataFrame, out: pd.DataFrame) -> str:
    """Rough dataframe comparison returning a status string."""

    try:
        if out.equals(gt):
            return "exact_match"
    except Exception:
        pass

    gt_cols = list(gt.columns)
    out_cols = list(out.columns)
    parts: List[str] = []

    if out_cols != gt_cols:
        if set(out_cols) == set(gt_cols):
            parts.append("wrong_order")
        else:
            if len(out_cols) > len(gt_cols):
                parts.append("more_cols")
            if len(out_cols) < len(gt_cols):
                parts.append("less_cols")
            if not set(out_cols).issubset(set(gt_cols)) and not set(gt_cols).issubset(
                set(out_cols)
            ):
                parts.append("cols_partial")

    if len(out) != len(gt):
        if len(out) > len(gt):
            parts.append("more_rows")
        else:
            parts.append("less_rows")

    merged = pd.merge(out, gt, how="inner")
    if merged.empty:
        parts.append("rows_no_match")
    elif len(merged) < min(len(gt), len(out)):
        parts.append("rows_partial")

    return ",".join(parts) if parts else "mismatch"


def run_test_case(
    vn: VannaBase,
    prompt: str,
    ground_sql: str,
    gt_df: pd.DataFrame,
) -> Tuple[pd.DataFrame | None, str]:
    """Execute a single test case and return the resulting dataframe and status."""

    try:
        if prompt is None:
            return None, "no_prompt"
        sql, df, _ = vn.ask(question=prompt, print_results=False, visualize=False)
    except Exception:
        return None, "failed_run"
    if df is None:
        return None, "failed_run"
    status = compare_frames(gt_df, df)
    return df, status


def main() -> None:
    openai_cfg = {"api_key": read_avalai_api_key()}
    all_tests = collect_tests(DATASET_DIR)
    total = sum(len(v) for v in all_tests.values())

    for model_name, method_name, level in TEST_CONFIGS:
        vn = MyVanna(openai_config=openai_cfg, llm_config={"model": model_name})
        vn.connect_to_mssql(odbc_conn_str=CONN_STR)
        model_dir = LOG_DIR / model_name
        model_dir.mkdir(exist_ok=True)

        test_count = 0
        start = time.time()

        for category, cases in all_tests.items():
            cat_dir = model_dir / category
            cat_dir.mkdir(exist_ok=True)
            summary_path = cat_dir / "summary.csv"
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "case",
                    "prompt_type",
                    "status",
                    "sql_path",
                    "output_path",
                ])

                for idx, (sql_path, prompt_path, prompts) in enumerate(cases, start=1):
                    gt_df = vn.run_sql(sql_path.read_text())
                    gt_path = cat_dir / f"case{idx:02d}_gt.csv"
                    gt_df.to_csv(gt_path, index=False)

                    prompt_order = [
                        ("well", prompts.get("Well-Explained")),
                        ("poor", prompts.get("Poorly-Explained")),
                        ("under", prompts.get("Underspecified")),
                    ]
                    prompt_order = [p for p in prompt_order if p[1] is not None]
                    prompt_order = prompt_order[:level]

                    for p_type, p_text in prompt_order:
                        test_count += 1
                        prog = (test_count / (total * level)) * 100
                        elapsed = time.time() - start
                        eta = (elapsed / test_count) * ((total * level) - test_count)
                        print(
                            f"{model_name} | {category} case {idx} {p_type}: {prog:.1f}% ETA {eta:.1f}s"
                        )
                        df_out, status = run_test_case(
                            vn, p_text, sql_path.read_text(), gt_df
                        )
                        out_path = cat_dir / f"case{idx:02d}_{p_type}.csv"
                        if df_out is not None:
                            df_out.to_csv(out_path, index=False)
                        writer.writerow([idx, p_type, status, sql_path.name, out_path.name])


if __name__ == "__main__":
    main()
