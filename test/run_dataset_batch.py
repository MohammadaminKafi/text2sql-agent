#!/usr/bin/env python3
"""Run test_dataset_aw.py with multiple argument sets.

This helper reads a text file containing argument strings and executes
``test_dataset_aw.py`` for each line. Lines beginning with ``#`` or blank
lines are ignored.

Example::

    python run_dataset_batch.py args.txt

"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset tests sequentially")
    parser.add_argument(
        "args_file",
        type=Path,
        help="Text file with argument strings, one per line",
    )
    parsed = parser.parse_args()

    if not parsed.args_file.is_file():
        parser.error(f"{parsed.args_file} does not exist")

    with parsed.args_file.open(encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    script = Path(__file__).parent / "test_dataset_aw.py"

    for idx, line in enumerate(lines, start=1):
        print(f"\n=== Running configuration {idx}: {line}")
        cmd = [sys.executable, str(script)] + line.split()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Configuration {idx} exited with code {result.returncode}")


if __name__ == "__main__":
    main()
