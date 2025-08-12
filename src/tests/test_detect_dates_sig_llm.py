# tests/test_detect_dates_sig_llm.py
import json
import pytest
import dspy

from components.dspy_signatures import DetectDatesSig


@pytest.mark.llm
def test_detect_dates_sig_returns_valid_items_json(lm):
    """End-to-end LLM check: schema + basic behavior across Jalali & Gregorian."""
    assert lm is not None, "LM fixture should be initialized when --llm is used."

    # A prompt that contains a Jalali month-year (Persian) and a Gregorian day
    prompt = "Report sales for خرداد ۱۴۰۲ and also for 2023-06-05."

    planner = dspy.ChainOfThought(DetectDatesSig)
    out = planner(original_prompt=prompt, database_calendar="Gregorian")

    # Must be parseable JSON and a list
    try:
        items = json.loads(out.items_json or "[]")
    except Exception as exc:
        raise AssertionError(f"items_json is not valid JSON: {out.items_json!r}") from exc

    assert isinstance(items, list), f"Expected a list, got: {type(items)}"
    assert len(items) > 0, "Expected at least one detected date-like item."

    # Each item must follow the specified schema (presence + types)
    required_keys = {"text", "src_calendar", "granularity", "year", "month", "day"}
    for idx, it in enumerate(items):
        missing = required_keys - set(it.keys())
        assert not missing, f"Item {idx} missing keys: {missing}"

        # Soft type checks (allow None for month/day in month/year granularities)
        assert it["src_calendar"] in ("gregorian", "jalali", "Gregorian", "Jalali")
        assert it["granularity"] in ("day", "month", "year")

    # Pragmatic behavior checks: we expect *some* Jalali and *some* Gregorian
    has_jalali_month = any(
        (it["src_calendar"].lower() == "jalali" and it["granularity"] == "month")
        for it in items
    )
    has_gregorian_day = any(
        (it["src_calendar"].lower() == "gregorian" and it["granularity"] == "day")
        for it in items
    )

    assert has_jalali_month, "Expected a Jalali month to be detected (e.g., خرداد ۱۴۰۲)."
    assert has_gregorian_day, "Expected a Gregorian day to be detected (e.g., 2023-06-05)."
