import re
import pytest
import dspy
from components.dspy_signatures import ExtractDatesSig


# --------------------------
# Fixtures & helpers
# --------------------------

@pytest.fixture(scope="module")
def extractor(llm):
    """Bind DSPy to the live LLM once, then build a predictor for the signature."""
    dspy.configure(lm=llm)
    return dspy.Predict(ExtractDatesSig)


_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

def _to_ascii_digits(s: str) -> str:
    return "".join((c.translate(_DIGIT_MAP) if c in "۰۱۲۳۴۵۶۷۸۹" else c) for c in s)

def _is_num_1_31(val: str) -> bool:
    if not val:
        return False
    s = _to_ascii_digits(val)
    if not s.isdigit():
        return False
    n = int(s)
    return 1 <= n <= 31

def _is_num_1_12(val: str) -> bool:
    if not val:
        return False
    s = _to_ascii_digits(val)
    if not s.isdigit():
        return False
    n = int(s)
    return 1 <= n <= 12

def _is_year_yyyy(val: str) -> bool:
    if not val:
        return False
    # Accept ASCII or Persian digits
    return bool(re.fullmatch(r"(?:\d{4}|[۰-۹]{4})", val))

def _empty(val) -> bool:
    return (val is None) or (isinstance(val, str) and val.strip() == "")

def _invoke(extractor, prompt: str):
    out = extractor(user_prompt=prompt)
    assert hasattr(out, "dates"), "Signature must return `dates` field"
    assert isinstance(out.dates, dict), "dates must be a dict"
    return out.dates

def _assert_contract(dates: dict):
    """Contract checks per spec for structure + fields."""
    assert isinstance(dates, dict)
    for k, v in dates.items():
        # title key
        assert isinstance(k, str) and k.strip(), "title key must be non-empty"
        assert "\n" not in k
        # value dict & required fields
        assert isinstance(v, dict), "each value must be a dict"
        for fld in ("source_calendar", "day", "month", "year"):
            assert fld in v, f"missing field: {fld}"
            assert isinstance(v[fld], str), f"{fld} must be str"
        assert v["source_calendar"] in ("solar", "gregorian"), "invalid source_calendar"
        # day range if present (allow Persian digits)
        if v["day"]:
            assert _is_num_1_31(v["day"]), f"invalid day: {v['day']}"
        # month numeric or a name
        if v["month"]:
            if re.fullmatch(r"(?:\d{1,2}|[۰-۹]{1,2})", v["month"]):
                assert _is_num_1_12(v["month"]), f"invalid month: {v['month']}"
            else:
                # assume it's a (corrected) name
                assert isinstance(v["month"], str) and len(v["month"]) <= 30
        # year must be 4 digits (ASCII or Persian) if present
        if v["year"]:
            assert _is_year_yyyy(v["year"]), f"invalid year: {v['year']}"

def _find_by_substring(dates: dict, substr: str):
    for k, v in dates.items():
        if substr in k:
            return k, v
    return None, None


# --------------------------
# 0) Smoke / empty-case
# --------------------------
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt",
    [
        "No dates here, just a sentence.",
        "لطفا فقط یک توضیح کلی بده بدون تاریخ.",
        "Tell me something interesting.",
    ],
    ids=["en_none", "fa_none", "en_generic"],
)
def test_empty_case_returns_empty_dict(extractor, prompt):
    dates = _invoke(extractor, prompt)
    assert dates == {} or len(dates) == 0


@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt",
    [
        "Between March 3, 2019 and April 10, 2020.",
        "گزارش برای ۱۴۰۲/۰۷/۱۵",
    ],
    ids=["en_simple", "fa_simple"],
)
def test_contract_on_simple_cases(extractor, prompt):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)


# --------------------------
# 1) Persian defaults to SOLAR
# --------------------------
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt, substrings",
    [
        ("گزارش فروش در ۱۴۰۲/۰۷/۱۵ و ۱۴۰۲/۰۹/۰۱", ["۱۴۰۲/۰۷/۱۵", "۱۴۰۲/۰۹/۰۱"]),
        ("آمار 1401-12-29 و 1402-01-01", ["1401-12-29", "1402-01-01"]),
        ("پرداخت‌ها در ۲۰ اسفند ۱۴۰۱ و ۱ فروردین ۱۴۰۲", ["۲۰ اسفند ۱۴۰۱", "۱ فروردین ۱۴۰۲"]),
    ],
    ids=["fa_solar_slash", "fa_solar_dash", "fa_solar_names"],
)
def test_farsi_default_solar(extractor, prompt, substrings):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)
    for sub in substrings:
        k, v = _find_by_substring(dates, sub)
        assert k, f"missing mention: {sub}"
        assert v["source_calendar"] == "solar", f"expected solar for {sub}"
        assert not _empty(v["day"]) and not _empty(v["month"]) and not _empty(v["year"])


# --------------------------
# 2) Farsi explicit GREGORIAN override
# --------------------------
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt, substrings",
    [
        ("گزارش کاربران در 2023/11/05 میلادی و 2023/12/01", ["2023/11/05", "2023/12/01"]),
        ("فروش در ژانویه 2024 و دسامبر 2024 میلادی", ["ژانویه 2024", "دسامبر 2024"]),
        ("در May 2022 و June 2022 (میلادی) بررسی کن", ["May 2022", "June 2022"]),
    ],
    ids=["fa_greg_token", "fa_greg_fa_month_names", "fa_greg_en_month_names"],
)
def test_farsi_explicit_gregorian_override(extractor, prompt, substrings):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)
    for sub in substrings:
        k, v = _find_by_substring(dates, sub)
        assert k, f"missing mention: {sub}"
        assert v["source_calendar"] == "gregorian"
        assert not _empty(v["month"])
        # year preferred when present in the text
        if any(ch.isdigit() or ch in "۰۱۲۳۴۵۶۷۸۹" for ch in sub):
            assert not _empty(v["year"])


# --------------------------
# 3) English ranges: split into endpoints, preserve order
# --------------------------
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt, first_sub, second_sub",
    [
        ("Between March 3, 2019 and April 10, 2020.", "March 3, 2019", "April 10, 2020"),
        ("From 01-02-2020 to 03-04-2020.", "01-02-2020", "03-04-2020"),
        ("Jan 5, 2018 – Feb 6, 2018", "Jan 5, 2018", "Feb 6, 2018"),
    ],
    ids=["en_between_and", "en_from_to_dash_fmt", "en_en_dash"],
)
def test_english_range_split_and_order(extractor, prompt, first_sub, second_sub):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)
    titles = list(dates.keys())
    k1, v1 = _find_by_substring(dates, first_sub)
    k2, v2 = _find_by_substring(dates, second_sub)
    assert k1 and k2, f"missing endpoints: {first_sub}, {second_sub}"
    assert titles.index(k1) < titles.index(k2), "range endpoints must keep order"
    assert v1["source_calendar"] == "gregorian" and v2["source_calendar"] == "gregorian"


# --------------------------
# 4) Farsi ranges: از … تا … / الی
# --------------------------
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt, first_sub, second_sub",
    [
        ("از ۱۴۰۲/۰۱/۰۵ تا ۱۴۰۲/۰۱/۱۰ گزارش بده", "۱۴۰۲/۰۱/۰۵", "۱۴۰۲/۰۱/۱۰"),
        ("از ۱۴۰۱/۱۲/۲۵ الی ۱۴۰۲/۰۱/۰۵", "۱۴۰۱/۱۲/۲۵", "۱۴۰۲/۰۱/۰۵"),
    ],
    ids=["fa_az_ta", "fa_az_eli"],
)
def test_farsi_range_split_and_order(extractor, prompt, first_sub, second_sub):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)
    titles = list(dates.keys())
    k1, v1 = _find_by_substring(dates, first_sub)
    k2, v2 = _find_by_substring(dates, second_sub)
    assert k1 and k2
    assert v1["source_calendar"] == "solar" and v2["source_calendar"] == "solar"
    assert titles.index(k1) < titles.index(k2)


# --------------------------
# 5) Typo correction for month names
# --------------------------
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt, misspell, expect_any_of",
    [
        ("On Feberuary 12, 2020 we shipped v1.", "Feberuary", {"February", "Feb", "02", "2"}),
        ("Septmber 3, 2021 was the launch.", "Septmber", {"September", "Sept", "09", "9"}),
        ("We kicked off in Janaury 2019.", "Janaury", {"January", "Jan", "01", "1"}),
        ("Agust 10, 2022 was quiet.", "Agust", {"August", "Aug", "08", "8"}),
    ],
    ids=["feb_misspell", "sep_misspell", "jan_misspell", "aug_misspell"],
)
def test_month_typo_correction(extractor, prompt, misspell, expect_any_of):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)
    k, v = _find_by_substring(dates, misspell)
    assert k, f"missing misspelled title: {misspell}"
    assert v["source_calendar"] == "gregorian"
    assert not _empty(v["day"]) and not _empty(v["year"])
    # month corrected to name or normalized to MM
    assert v["month"] in expect_any_of, f"unexpected month normalization: {v['month']}"


# --------------------------
# 6) Mixed-language prompt => per-mention calendars
# --------------------------
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt, triples",
    [
        (
            "آمار بین March 2020، ۱۴۰۱/۰۵/۰۱ و August 2021 را نشان بده.",
            [("March 2020", "gregorian"), ("۱۴۰۱/۰۵/۰۱", "solar"), ("August 2021", "gregorian")],
        ),
        (
            "Between ژوئن 2022 و ۱۴۰۱/۰۹/۱۵ and July 2023.",
            [("ژوئن 2022", "gregorian"), ("۱۴۰۱/۰۹/۱۵", "solar"), ("July 2023", "gregorian")],
        ),
    ],
    ids=["fa_en_mix_1", "fa_en_mix_2"],
)
def test_mixed_language_per_mention_calendar(extractor, prompt, triples):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)
    titles = list(dates.keys())
    found_keys = []
    for sub, cal in triples:
        k, v = _find_by_substring(dates, sub)
        assert k, f"missing mention: {sub}"
        assert v["source_calendar"] == cal, f"calendar mismatch for {sub}"
        found_keys.append(k)
    # order check
    assert titles.index(found_keys[0]) < titles.index(found_keys[1]) < titles.index(found_keys[2])


# 7) Day-only (ordinal/positional) — day set, month/year empty
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt, substrings",
    [
        ("Report for the 7th day.", ["7th day"]),
        ("فقط روز هفتم را بررسی کن.", ["روز هفتم"]),
        ("روز ۱۵ ام را تحلیل کن.", ["روز ۱۵"]),
    ],
    ids=["en_7th_day", "fa_rooz_haftom", "fa_rooz_15am"],
)
def test_day_only(extractor, prompt, substrings):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)
    for sub in substrings:
        k, v = _find_by_substring(dates, sub)
        assert k, f"missing day-only mention: {sub}"
        assert not _empty(v["day"]) and _is_num_1_31(v["day"])
        assert _empty(v["month"])
        assert _empty(v["year"])


# 8) Month-only (ordinal name/number) — month set, day/year empty
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt, substrings",
    [
        ("Handle the 2nd month.", ["2nd month"]),
        ("ماه دوازدهم را گزارش کن.", ["ماه دوازدهم"]),
        ("Focus on February.", ["February"]),
    ],
    ids=["en_2nd_month", "fa_maah_12", "en_february"],
)
def test_month_only(extractor, prompt, substrings):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)
    for sub in substrings:
        k, v = _find_by_substring(dates, sub)
        assert k, f"missing month-only mention: {sub}"
        assert not _empty(v["month"])
        assert _empty(v["day"])
        assert _empty(v["year"])


# 9) Year-only — year set, day/month empty; calendar by language/label
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt, substrings, expected_cal",
    [
        ("Growth in 2021 was strong.", ["2021"], "gregorian"),
        ("در سال ۱۴۰۱ استخدام‌ها افزایش داشت.", ["۱۴۰۱"], "solar"),
        ("Milestones for Gregorian 2019.", ["2019"], "gregorian"),
    ],
    ids=["en_2021", "fa_1401", "en_greg_label"],
)
def test_year_only(extractor, prompt, substrings, expected_cal):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)
    for sub in substrings:
        k, v = _find_by_substring(dates, sub)
        assert k, f"missing year-only mention: {sub}"
        assert _empty(v["day"]) and _empty(v["month"])
        assert not _empty(v["year"]) and _is_year_yyyy(v["year"])
        assert v["source_calendar"] == expected_cal


# 10) Day–Month (no year) — day & month set, year empty
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt, substrings, expected_cal",
    [
        ("Ship on March 3rd.", ["March 3"], "gregorian"),
        ("گزارش برای ۳ خرداد تهیه کن.", ["۳ خرداد"], "solar"),
        ("روز 5 مهر را بررسی کن.", ["5 مهر", "۵ مهر"], "solar"),
    ],
    ids=["en_mar_3", "fa_3_khordad", "fa_5_mehr"],
)
def test_day_month_no_year(extractor, prompt, substrings, expected_cal):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)
    # Accept either representation when there are two substrings (e.g., Persian/ASCII digits)
    found_any = False
    for sub in substrings:
        k, v = _find_by_substring(dates, sub)
        if k:
            found_any = True
            assert v["source_calendar"] == expected_cal
            assert not _empty(v["day"]) and not _empty(v["month"])
            assert _empty(v["year"])
            break
    assert found_any, f"none of the expected mentions found: {substrings}"


# 11) Month–Year (no day) — month & year set, day empty
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt, substrings, expected_cal",
    [
        ("Deploy in April 2020.", ["April 2020"], "gregorian"),
        ("آمار ژانویه ۲۰۱۹ را بده.", ["ژانویه ۲۰۱۹"], "gregorian"),
        ("گزارش خرداد ۱۴۰۲ را آماده کن.", ["خرداد ۱۴۰۲"], "solar"),
    ],
    ids=["en_apr_2020", "fa_jan_2019", "fa_khordad_1402"],
)
def test_month_year_no_day(extractor, prompt, substrings, expected_cal):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)
    for sub in substrings:
        k, v = _find_by_substring(dates, sub)
        assert k, f"missing month-year mention: {sub}"
        assert v["source_calendar"] == expected_cal
        assert _empty(v["day"])
        assert not _empty(v["month"]) and not _empty(v["year"])


# 12) Full date — day, month, year all set
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt, substrings, expected_cal",
    [
        ("We launched on 2019-03-03.", ["2019-03-03"], "gregorian"),
        ("Meeting: March 3, 2019.", ["March 3, 2019"], "gregorian"),
        ("تحویل در ۱۴۰۲/۰۷/۱۵ است.", ["۱۴۰۲/۰۷/۱۵"], "solar"),
    ],
    ids=["en_iso", "en_long", "fa_solar_full"],
)
def test_full_date(extractor, prompt, substrings, expected_cal):
    dates = _invoke(extractor, prompt)
    _assert_contract(dates)
    for sub in substrings:
        k, v = _find_by_substring(dates, sub)
        assert k, f"missing full date mention: {sub}"
        assert v["source_calendar"] == expected_cal
        assert not _empty(v["day"]) and not _empty(v["month"]) and not _empty(v["year"])