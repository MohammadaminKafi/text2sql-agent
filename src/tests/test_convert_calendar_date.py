# test_convert_calendar_date.py
# pytest-style tests + a human-readable log runner.
# pip install pytest jdatetime

import re
from pprint import pformat

from components.utils.date_utils import convert_calendar_date


ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_iso_date(s: str) -> bool:
    return bool(ISO_DATE_RE.match(s or ""))


def _log(title: str, src: str, tgt: str, date_input, year_hint, result):
    print("=" * 72)
    print(title)
    print(f"Source: {src}  ->  Target: {tgt}")
    print(f"Input : {pformat(date_input)}  year_hint={year_hint}")
    print("Output:", pformat(result, width=88))


# --------------------------- EXACT ---------------------------

def test_exact_solar_to_greg_full_date():
    res = convert_calendar_date("solar", "gregorian", "۱۵ اردیبهشت ۱۳۹۲")
    _log("EXACT SH→G (full date)", "solar", "gregorian", "۱۵ اردیبهشت ۱۳۹۲", None, res)
    assert res["kind"] == "exact"
    assert res["target_calendar"] == "gregorian"
    assert res["date"] == "2013-05-05"  # canonical: 1392/02/15 -> 2013-05-05


def test_exact_greg_to_solar_full_date():
    res = convert_calendar_date("gregorian", "solar", "21st Mar 2013")
    _log("EXACT G→SH (full date with ordinal)", "gregorian", "solar", "21st Mar 2013", None, res)
    assert res["kind"] == "exact"
    assert res["target_calendar"] == "solar"
    assert res["date"] == "1392-01-01"  # Nowruz 2013


def test_exact_same_calendar_greg():
    res = convert_calendar_date("gregorian", "gregorian", {"year": 2012, "month": "آوریل", "day": 12})
    _log("EXACT G→G (FA month name)", "gregorian", "gregorian",
         {"year": 2012, "month": "آوریل", "day": 12}, None, res)
    assert res["kind"] == "exact"
    assert res["date"] == "2012-04-12"


def test_exact_same_calendar_solar():
    res = convert_calendar_date("solar", "solar", {"year": 1402, "month": 2, "day": 1})
    _log("EXACT SH→SH", "solar", "solar", {"year": 1402, "month": 2, "day": 1}, None, res)
    assert res["kind"] == "exact"
    assert res["date"] == "1402-02-01"


# --------------------------- RANGE ---------------------------

def test_range_greg_to_solar_year_month():
    res = convert_calendar_date("gregorian", "solar", "2013-03")
    _log("RANGE G→SH (Y/M)", "gregorian", "solar", "2013-03", None, res)
    assert res["kind"] == "range"
    # March 2013 spans SH 1391-12-11 .. 1392-01-11
    assert res["start"] == "1391-12-11"
    assert res["end"] == "1392-01-11"


def test_range_solar_to_greg_year_month():
    res = convert_calendar_date("solar", "gregorian", {"year": 1401, "month": "آبان"})
    _log("RANGE SH→G (Y/M)", "solar", "gregorian", {"year": 1401, "month": "آبان"}, None, res)
    assert res["kind"] == "range"
    assert res["start"] == "2022-10-23"
    assert res["end"] == "2022-11-21"


def test_range_solar_to_greg_year_only():
    res = convert_calendar_date("solar", "gregorian", {"year": 1392})
    _log("RANGE SH→G (Y only)", "solar", "gregorian", {"year": 1392}, None, res)
    assert res["kind"] == "range"
    assert res["start"] == "2013-03-21"
    assert res["end"] == "2014-03-20"


def test_range_same_calendar_greg_y_m():
    res = convert_calendar_date("gregorian", "gregorian", {"year": 2013, "month": 7})
    _log("RANGE G→G (Y/M)", "gregorian", "gregorian", {"year": 2013, "month": 7}, None, res)
    assert res["kind"] == "range"
    assert res["start"] == "2013-07-01"
    assert res["end"] == "2013-07-31"


def test_range_same_calendar_solar_y_m():
    res = convert_calendar_date("solar", "solar", {"year": 1402, "month": 2})
    _log("RANGE SH→SH (Y/M)", "solar", "solar", {"year": 1402, "month": 2}, None, res)
    assert res["kind"] == "range"
    # Ordibehesht has 31 days
    assert res["start"] == "1402-02-01"
    assert res["end"] == "1402-02-31"


def test_range_same_calendar_greg_y_only():
    res = convert_calendar_date("gregorian", "gregorian", {"year": 2013})
    _log("RANGE G→G (Y only)", "gregorian", "gregorian", {"year": 2013}, None, res)
    assert res["kind"] == "range"
    assert res["start"] == "2013-01-01"
    assert res["end"] == "2013-12-31"


def test_range_same_calendar_solar_y_only():
    res = convert_calendar_date("solar", "solar", {"year": 1392})
    _log("RANGE SH→SH (Y only)", "solar", "solar", {"year": 1392}, None, res)
    assert res["kind"] == "range"
    assert res["start"] == "1392-01-01"
    assert _is_iso_date(res["end"]) and res["end"].startswith("1392-12-")


# --------------------------- APPROX ---------------------------

def test_approx_solar_to_greg_month_day_no_year():
    res = convert_calendar_date("solar", "gregorian", {"month": "مرداد", "day": 13})
    _log("APPROX SH→G (M/D w/o year)", "solar", "gregorian",
         {"month": "مرداد", "day": 13}, None, res)
    assert res["kind"] == "approx"
    approx = res["approx"]
    for key in ("mode_mmdd", "window_mmdd", "support", "distinct_outcomes",
                "sample_years", "anchor_year_example", "mode_support"):
        assert key in approx
    # Mode is very often 08-04; don't hard-lock it, just sanity-check the shape.
    assert isinstance(approx["mode_mmdd"], str) and len(approx["mode_mmdd"]) == 5
    assert "start" in approx["window_mmdd"] and "end" in approx["window_mmdd"]
    assert approx["support"] >= 1
    assert approx["mode_support"] <= approx["support"]
    assert _is_iso_date(approx["anchor_year_example"]["date"])


def test_approx_greg_to_solar_month_day_no_year():
    res = convert_calendar_date("gregorian", "solar", "Aug 15")
    _log("APPROX G→SH (M/D w/o year)", "gregorian", "solar", "Aug 15", None, res)
    assert res["kind"] == "approx"
    approx = res["approx"]
    # Often SH mode might be '05-24' (Mordad 24). Keep assertions generic.
    assert isinstance(approx["mode_mmdd"], str) and len(approx["mode_mmdd"]) == 5
    assert "start" in approx["window_mmdd"] and "end" in approx["window_mmdd"]
    assert approx["support"] >= 1
    assert _is_iso_date(approx["anchor_year_example"]["date"])


def test_approx_resolved_with_year_hint_to_exact():
    res = convert_calendar_date("solar", "gregorian", {"month": "مرداد", "day": 13}, year_hint=1392)
    _log("APPROX (resolved by hint) SH→G", "solar", "gregorian",
         {"month": "مرداد", "day": 13}, 1392, res)
    assert res["kind"] == "exact"
    assert res["date"] == "2013-08-04"


# --------------------------- NEEDS_YEAR ---------------------------

def test_needs_year_cross_calendar_month_only():
    res = convert_calendar_date("solar", "gregorian", {"month": "آبان"})
    _log("NEEDS_YEAR SH→G (month-only)", "solar", "gregorian", {"month": "آبان"}, None, res)
    assert res["kind"] == "needs_year"
    assert "year" in res.get("suggested_hint_key", "year")


def test_needs_year_cross_calendar_day_only():
    res = convert_calendar_date("gregorian", "solar", {"day": 15})
    _log("NEEDS_YEAR G→SH (day-only)", "gregorian", "solar", {"day": 15}, None, res)
    assert res["kind"] == "needs_year"


def test_needs_year_same_calendar_month_day():
    res = convert_calendar_date("solar", "solar", {"month": "آبان", "day": 13})
    _log("NEEDS_YEAR SH→SH (M/D same calendar)", "solar", "solar",
         {"month": "آبان", "day": 13}, None, res)
    assert res["kind"] == "needs_year"


# --------------------------- NORMALIZATION ---------------------------

def test_persian_digits_parsing_full_solar():
    res = convert_calendar_date("solar", "gregorian", "۱۳۹۲/۰۲/۱۵")
    _log("NORMALIZE Persian digits SH→G", "solar", "gregorian", "۱۳۹۲/۰۲/۱۵", None, res)
    assert res["kind"] == "exact"
    assert res["date"] == "2013-05-05"


def test_numeric_ambiguous_defaults_to_md():
    # "03/07" as solar should be parsed as month/day (since no year-like number),
    # then converted w/o year -> needs_year
    res = convert_calendar_date("solar", "gregorian", "03/07")
    _log("NUMERIC ambiguity (defaults M/D)", "solar", "gregorian", "03/07", None, res)
    assert res["kind"] in {"approx", "needs_year"}  # cross-calendar M/D no year -> approx now
    # With the updated behavior, it should be approx:
    assert res["kind"] == "approx"


# --------------------------- UNSUPPORTED / ERROR ---------------------------

def test_unsupported_unparseable():
    res = convert_calendar_date("gregorian", "solar", "not-a-date at all")
    _log("UNSUPPORTED", "gregorian", "solar", "not-a-date at all", None, res)
    assert res["kind"] == "unsupported"


def test_invalid_full_date_raises():
    # Invalid Gregorian date should raise when full Y/M/D is provided (e.g., Feb 29 on non-leap 2019)
    raised = False
    try:
        convert_calendar_date("gregorian", "solar", {"year": 2019, "month": 2, "day": 29})
    except ValueError:
        raised = True
    _log("ERROR path (invalid full date raises)", "gregorian", "solar",
         {"year": 2019, "month": 2, "day": 29}, None, "ValueError raised" if raised else "no error")
    assert raised


# --------------------------- Smoke (your snippet) ---------------------------

def test_smoke_examples_from_prompt():
    r1 = convert_calendar_date("solar", "gregorian", {"month": "مرداد", "day": 13})
    r2 = convert_calendar_date("gregorian", "solar", "Aug 15")
    _log("SMOKE 1", "solar", "gregorian", {"month": "مرداد", "day": 13}, None, r1)
    _log("SMOKE 2", "gregorian", "solar", "Aug 15", None, r2)
    assert r1["kind"] == "approx" and r2["kind"] == "approx"


# --------------------------- Human-readable log runner ---------------------------

def run_demo_logs():
    cases = [
        ("EXACT SH→G",  ("solar", "gregorian", "۱۵ اردیبهشت ۱۳۹۲", None)),
        ("EXACT G→SH",  ("gregorian", "solar", "21st Mar 2013", None)),
        ("RANGE G→SH",  ("gregorian", "solar", "2013-03", None)),
        ("RANGE SH→G",  ("solar", "gregorian", {"year": 1401, "month": "آبان"}, None)),
        ("RANGE SH Y",  ("solar", "gregorian", {"year": 1392}, None)),
        ("RANGE G Y",   ("gregorian", "gregorian", {"year": 2013}, None)),
        ("APPROX SH→G", ("solar", "gregorian", {"month": "مرداد", "day": 13}, None)),
        ("APPROX G→SH", ("gregorian", "solar", "Aug 15", None)),
        ("HINT→EXACT",  ("solar", "gregorian", {"month": "مرداد", "day": 13}, 1392)),
        ("NEEDS_YEAR",  ("solar", "gregorian", {"month": "آبان"}, None)),
        ("UNSUPPORTED", ("gregorian", "solar", "not-a-date at all", None)),
        ("FA digits",   ("solar", "gregorian", "۱۳۹۲/۰۲/۱۵", None)),
    ]
    for title, (src, tgt, di, hint) in cases:
        res = None
        try:
            res = convert_calendar_date(src, tgt, di, hint)
            _log(f"[LOG] {title}", src, tgt, di, hint, res)
        except Exception as e:
            _log(f"[LOG] {title} (EXC)", src, tgt, di, hint, f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    # Running this file directly prints readable logs for all major cases.
    run_demo_logs()
