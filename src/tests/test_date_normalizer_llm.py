# tests/test_date_normalizer_llm.py
import re
import pytest

from components.dspy_modules import DateNormalizer


@pytest.mark.llm
def test_normalizer_jalali_month_to_gregorian_month(lm):
    """If DB calendar is Gregorian, a Jalali month-year should become YYYY-MM (target)."""
    dn = DateNormalizer(database_calendar="Gregorian")
    prompt = "Show sales for خرداد ۱۴۰۲ and compare with July 2023."

    normalized, meta = dn(prompt)

    # We expect *some* replacement / normalization
    assert isinstance(normalized, str)
    assert isinstance(meta, list)
    assert any(m.get("src_calendar") for m in meta), "Expected at least one metadata item."

    # Expect a YYYY-MM in normalized text (the Jalali month converted using day=1 rule)
    assert re.search(r"\b\d{4}-\d{2}\b", normalized), (
        f"Expected a YYYY-MM in normalized string, got: {normalized!r}"
    )

    # Find the jalali month conversion entry
    jmonth = [m for m in meta if (m.get("src_calendar", "").lower() == "jalali" and m.get("granularity") == "month")]
    assert jmonth, f"Expected a Jalali-month conversion in metadata. Got: {meta!r}"
    assert re.fullmatch(r"\d{4}-\d{2}", jmonth[0]["normalized"]), (
        f"Expected YYYY-MM for month granularity, got: {jmonth[0]['normalized']!r}"
    )
    # Status should be ok unless a parse failure occurred
    assert jmonth[0].get("status") in ("ok", "left-unchanged (parse-failure)")


@pytest.mark.llm
def test_normalizer_gregorian_day_to_jalali_day_when_target_is_solar(lm):
    """If DB calendar is Solar (Jalali), a Gregorian day should become YYYY-MM-DD (Jalali)."""
    dn = DateNormalizer(database_calendar="Solar")
    prompt = "Filter orders placed on 2023-06-05 and in شهریور ۱۴۰۲."

    normalized, meta = dn(prompt)

    # Normalized should contain a Jalali-looking day for the Gregorian date
    assert re.search(r"\b1[34]\d{2}-\d{2}-\d{2}\b", normalized), (
        f"Expected a Jalali YYYY-MM-DD in normalized string, got: {normalized!r}"
    )

    # Metadata should include a Gregorian->Jalali day conversion
    gday = [
        m for m in meta
        if (m.get("src_calendar", "").lower() == "gregorian" and m.get("granularity") == "day")
    ]
    assert gday, f"Expected a Gregorian day conversion in metadata. Got: {meta!r}"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", gday[0]["normalized"]), (
        f"Expected YYYY-MM-DD for day granularity, got: {gday[0]['normalized']!r}"
    )
    assert gday[0].get("tgt_calendar") == "solar"
