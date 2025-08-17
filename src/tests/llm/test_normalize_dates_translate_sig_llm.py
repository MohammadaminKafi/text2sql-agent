import re
import pytest
import dspy
from components.dspy_signatures import NormalizeDatesTranslateSig

# --------------------------
# Fixtures & helpers
# --------------------------

@pytest.fixture(scope="module")
def normalizer(llm):
    """Bind DSPy to the live LLM once, then build a predictor for the signature."""
    dspy.configure(lm=llm)
    return dspy.Predict(NormalizeDatesTranslateSig)


def _invoke(normalizer, prompt: str, converted_dates):
    """Run signature and return (english_prompt, language)."""
    out = normalizer(user_prompt=prompt, converted_dates=converted_dates)
    assert hasattr(out, "english_prompt")
    assert hasattr(out, "language")
    assert isinstance(out.english_prompt, str)
    assert isinstance(out.language, str)
    return out.english_prompt, out.language


def _apply_replacements_locally(text: str, converted_dates):
    """Reference replacement logic to compute exact expected passthrough strings for English prompts."""
    result = text
    for (_src_cal, _tgt_cal, original, converted) in converted_dates or []:
        result = result.replace(original, converted)
    return result


def _lang_norm(s: str) -> str:
    s = s.strip().lower()
    return {"farsi": "persian"}.get(s, s)  # treat "Farsi" == "Persian"


def _assert_contains_all(haystack: str, needles):
    for n in needles:
        assert n in haystack, f"expected to find '{n}' in: {haystack}"


def _assert_not_contains_any(haystack: str, needles):
    for n in needles:
        assert n not in haystack, f"did not expect '{n}' in: {haystack}"


# --------------------------
# 0) English passthrough (no translation). Exact string expected.
# --------------------------
@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt, converted_dates",
    [
        (
            "Sales report from 24 March 2024 to 13 March 2024.",
            [("solar","gregorian","24 March 2024","24-03-2024"),
             ("solar","gregorian","13 March 2024","13-03-2024")],
        ),
        (
            "Ship on May 10, compare with May 12.",
            [("solar","gregorian","May 10","10-05"),
             ("solar","gregorian","May 12","12-05")],
        ),
        (
            "No replacements here, already English.",
            [],  # empty list should be a pure passthrough
        ),
    ],
    ids=["en_two_repl", "en_two_partial_tokens", "en_empty_list"],
)
def test_english_passthrough_exact(normalizer, prompt, converted_dates):
    english_prompt, language = _invoke(normalizer, prompt, converted_dates)
    expected = _apply_replacements_locally(prompt, converted_dates)
    assert english_prompt == expected, "English input must be returned unchanged except exact replacements"
    assert _lang_norm(language) == "english"


# --------------------------
# 1) Replacement + translate (Farsi → English). Check dates replaced & language detected.
# --------------------------
@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt, converted_dates, must_appear, must_not_appear",
    [
        (
            "گزارش فروش از ۵ فروردین ۱۴۰۳ تا ۲۳ اسفند ۱۴۰۲",
            [("solar","gregorian","۵ فروردین ۱۴۰۳","24-03-2024"),
             ("solar","gregorian","۲۳ اسفند ۱۴۰۲","13-03-2024")],
            ["24-03-2024","13-03-2024"],
            ["۵ فروردین ۱۴۰۳","۲۳ اسفند ۱۴۰۲"],
        ),
        (
            "بین ۱۴۰۲/۰۹/۰۱ و ۱۴۰۲/۰۹/۱۵ مقایسه کن",
            [("solar","gregorian","۱۴۰۲/۰۹/۰۱","22-11-2023"),
             ("solar","gregorian","۱۴۰۲/۰۹/۱۵","06-12-2023")],
            ["22-11-2023","06-12-2023"],
            ["۱۴۰۲/۰۹/۰۱","۱۴۰۲/۰۹/۱۵"],
        ),
        (
            "در ۲۰ اسفند ۱۴۰۱ بررسی کن",
            [("solar","gregorian","۲۰ اسفند ۱۴۰۱","11-03-2023")],
            ["11-03-2023"],
            ["۲۰ اسفند ۱۴۰۱"],
        ),
    ],
    ids=["fa_range_names", "fa_range_numeric", "fa_single_name"],
)
def test_farsi_replacement_and_translate(normalizer, prompt, converted_dates, must_appear, must_not_appear):
    english_prompt, language = _invoke(normalizer, prompt, converted_dates)
    assert _lang_norm(language) in {"persian"}  # allow "Farsi" -> normalized to "persian"
    _assert_contains_all(english_prompt, must_appear)
    _assert_not_contains_any(english_prompt, must_not_appear)
    # Sanity: output should contain ASCII letters (likely an English translation)
    assert re.search(r"[A-Za-z]", english_prompt), "expected English letters after translation"


# --------------------------
# 2) Translate-only (empty list). Ensure translation happens & language detected.
# --------------------------
@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt",
    [
        "گزارش ماه دوم را بده.",
        "آمار کاربران در ژانویه ۲۰۲۴ کجاست؟",
        "در مورد فروش توضیح بده.",
    ],
    ids=["fa_month_ordinal", "fa_greg_month", "fa_generic"],
)
def test_translate_only_empty_converted_dates(normalizer, prompt):
    english_prompt, language = _invoke(normalizer, prompt, [])
    assert _lang_norm(language) in {"persian"}
    assert english_prompt != prompt  # should be translated
    assert re.search(r"[A-Za-z]", english_prompt), "expected English letters after translation"


# --------------------------
# 3) Multiple replacements & counts (Farsi → English). Every occurrence replaced.
# --------------------------
@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt, converted_dates, count_pairs",
    [
        (
            "از ۱۴۰۲/۰۱/۰۱ تا ۱۴۰۲/۰۱/۰۱ و دوباره ۱۴۰۲/۰۱/۰۱ را بررسی کن",
            [("solar","gregorian","۱۴۰۲/۰۱/۰۱","21-03-2023")],
            [("21-03-2023", 3)],  # appears three times after replacement
        ),
        (
            "۵ فروردین ۱۴۰۳ و ۵ فروردین ۱۴۰۳ را مقایسه کن",
            [("solar","gregorian","۵ فروردین ۱۴۰۳","24-03-2024")],
            [("24-03-2024", 2)],
        ),
    ],
    ids=["fa_three_occ", "fa_two_occ"],
)
def test_multiple_replacements_counts(normalizer, prompt, converted_dates, count_pairs):
    english_prompt, language = _invoke(normalizer, prompt, converted_dates)
    assert _lang_norm(language) in {"persian"}
    for token, expected_count in count_pairs:
        assert english_prompt.count(token) >= expected_count, (
            f"expected '{token}' at least {expected_count} times in output"
        )
    # originals must be gone
    for (_s, _t, orig, _conv) in converted_dates:
        assert orig not in english_prompt


# --------------------------
# 4) Exact-match replacement: similar but different tokens remain untouched.
#     (Use English passthrough for a strict equality check.)
# --------------------------
@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt, converted_dates, should_remain",
    [
        (
            "Compare May 10 and May 10th.",
            [("solar","gregorian","May 10","10-05")],
            ["May 10th"],  # should remain untouched
        ),
        (
            "Launch on March 03, not March 3.",
            [("solar","gregorian","March 03","03-03")],
            ["March 3"],  # should remain untouched
        ),
        (
            "Two forms: 24 March 2024 and 24th March 2024.",
            [("solar","gregorian","24 March 2024","24-03-2024")],
            ["24th March 2024"],  # untouched
        ),
    ],
    ids=["may10_vs_10th", "mar03_vs_3", "24mar_vs_24thmar"],
)
def test_exact_match_only_on_english(normalizer, prompt, converted_dates, should_remain):
    english_prompt, language = _invoke(normalizer, prompt, converted_dates)
    expected = _apply_replacements_locally(prompt, converted_dates)
    assert _lang_norm(language) == "english"
    assert english_prompt == expected, "English passthrough must keep non-exact tokens unchanged"
    _assert_contains_all(english_prompt, should_remain)


# --------------------------
# 5) Mixed-language with English passthrough (already English)
#     ensures no translation but replacement still happens.
# --------------------------
@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt, converted_dates, must_appear",
    [
        (
            "Sales report for ۵ فروردین ۱۴۰۳.",
            [("solar","gregorian","۵ فروردین ۱۴۰۳","24-03-2024")],
            ["Sales report for 24-03-2024."],  # full-string equality expected
        ),
        (
            "Check ۲۳ اسفند ۱۴۰۲ and 24 March 2024.",
            [("solar","gregorian","۲۳ اسفند ۱۴۰۲","13-03-2024")],
            ["Check 13-03-2024 and 24 March 2024."],
        ),
    ],
    ids=["en_sentence_with_fa_date", "en_sentence_with_mixed_dates"],
)
def test_english_sentence_with_farsi_tokens_passthrough(normalizer, prompt, converted_dates, must_appear):
    english_prompt, language = _invoke(normalizer, prompt, converted_dates)
    expected = _apply_replacements_locally(prompt, converted_dates)
    assert _lang_norm(language) == "english"
    assert english_prompt == expected
    _assert_contains_all(english_prompt, must_appear)


# --------------------------
# 6) No-op replacements (orig not present). Should translate/passthrough without crashes.
# --------------------------
@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt, converted_dates, expect_lang",
    [
        ("گزارش کلی بده", [("solar","gregorian","۵ فروردین ۱۴۰۳","24-03-2024")], "persian"),
        ("All good here", [("solar","gregorian","Nonexistent","01-01")], "english"),
    ],
    ids=["fa_noop", "en_noop"],
)
def test_noop_replacements(normalizer, prompt, converted_dates, expect_lang):
    english_prompt, language = _invoke(normalizer, prompt, converted_dates)
    assert _lang_norm(language) == expect_lang
    # English input: exact passthrough (unchanged, since no matches)
    if expect_lang == "english":
        assert english_prompt == prompt
    else:
        # Farsi input: should translate; not equal to original; include ASCII letters
        assert english_prompt != prompt
        assert re.search(r"[A-Za-z]", english_prompt)
