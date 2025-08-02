from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json

EXPECTED_KEYS = {"well-explained", "poorly-explained", "underspecified"}


def load_prompt(path: Path) -> Dict[str, str]:
    """Return prompts from JSON file ensuring required keys exist."""
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = EXPECTED_KEYS - data.keys()
    if missing:
        raise ValueError(f"{path} missing keys: {', '.join(sorted(missing))}")
    return data


def prompt_pattern(language: str) -> str:
    if language == "en":
        return "prompt*.json"
    if language == "fa":
        return "persian_prompt*.json"
    return f"{language}_prompt*.json"


def collect_tests(
    dataset_dir: Path,
    language: str = "en",
    categories: Iterable[str] | None = None,
) -> Dict[str, List[Tuple[Path, Path, Dict[str, str]]]]:
    """Collect query/prompt pairs grouped by category."""

    tests: Dict[str, List[Tuple[Path, Path, Dict[str, str]]]] = {}
    pattern = prompt_pattern(language)

    for category in sorted(p.name for p in dataset_dir.iterdir() if p.is_dir()):
        if categories is not None and category not in categories:
            continue
        cat_dir = dataset_dir / category
        queries = sorted(cat_dir.glob("query*.sql"))
        prompts = sorted(cat_dir.glob(pattern))
        cases: List[Tuple[Path, Path, Dict[str, str]]] = []
        for q, pth in zip(queries, prompts):
            cases.append((q, pth, load_prompt(pth)))
        tests[category] = cases
    return tests


def verify_dataset(dataset_dir: Path, languages: List[str]) -> None:
    """Ensure all queries have prompt files for each language and valid keys."""
    for category in sorted(p.name for p in dataset_dir.iterdir() if p.is_dir()):
        if category == "misc":
            continue
        cat_dir = dataset_dir / category
        queries = sorted(cat_dir.glob("query*.sql"))
        for lang in languages:
            pattern = prompt_pattern(lang)
            prompts = sorted(cat_dir.glob(pattern))
            if len(queries) != len(prompts):
                raise ValueError(f"Mismatch in {cat_dir} for language {lang}")
            for p in prompts:
                load_prompt(p)
