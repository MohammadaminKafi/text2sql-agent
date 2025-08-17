import re
import pytest
import dspy
from components.dspy_signatures import DetectAmbiguitySig


# --------------------------
# Fixture
# --------------------------

@pytest.fixture(scope="module")
def detect_amb(llm):
    """Bind DSPy to the live LLM once, then build a predictor for the signature."""
    dspy.configure(lm=llm)
    return dspy.Predict(DetectAmbiguitySig)


# --------------------------
# Helpers
# --------------------------

QUESTION_CUES = (
    "how many", "how much", "which", "what", "when", "where",
    "please specify", "do you mean", "should we", "clarify",
    "define", "starting from", "ending at", "from which year",
)

def _is_question_like(s: str) -> bool:
    low = s.strip().lower()
    return "?" in s or any(cue in low for cue in QUESTION_CUES)

def _assert_contract(out):
    assert hasattr(out, "has_ambiguity"), "missing has_ambiguity"
    assert hasattr(out, "ambiguities"), "missing ambiguities"
    assert isinstance(out.has_ambiguity, (bool, int)), "has_ambiguity must be bool"
    assert isinstance(out.ambiguities, dict), "ambiguities must be a dict"
    for k, v in out.ambiguities.items():
        assert isinstance(k, str) and k.strip(), "ambiguity label must be non-empty string"
        assert isinstance(v, str) and v.strip(), "question must be non-empty string"
        assert _is_question_like(v), f"question should look like a question: {v}"


# --------------------------
# 0) Contract & shape (sanity)
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "sql_ready_prompt",
    [
        "Return total revenue per year for 2020–2022; shipped orders only; order by year ascending.",
        "Show order count by customer for January 2024 in US region; top 5 customers.",
        "Sum revenue per category for 2023; exclude cancelled; order by revenue descending.",
    ],
    ids=["rev_by_year", "orders_by_customer_top5", "sum_rev_by_category"],
)
def test_contract_and_shape(detect_amb, sql_ready_prompt):
    out = detect_amb(sql_ready_prompt=sql_ready_prompt)
    _assert_contract(out)


# --------------------------
# 1) Ambiguous → should flag TRUE
# --------------------------
# Material ambiguities: vague recency, missing counts/thresholds, undefined metrics or time windows.

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "sql_ready_prompt, expected_cues",
    [
        (
            "Return revenue by year for recent years; shipped orders only.",
            ("where to start", "which years", "recent years"),
        ),
        (
            "Show top customers by total revenue; last year; order by revenue descending.",
            ("how many", "top customers"),
        ),
        (
            "Return largest orders; group by customer.",
            ("how many", "threshold", "largest"),
        ),
        (
            "Show active users; group by week.",
            ("time window", "which period", "which timeframe"),
        ),
        (
            "Return high-value customers by region.",
            ("define high-value", "threshold"),
        ),
    ],
    ids=[
        "recent_years_ambiguous",
        "top_customers_no_n",
        "largest_orders_no_n",
        "active_users_no_window",
        "high_value_no_threshold",
    ],
)
def test_ambiguous_true(detect_amb, sql_ready_prompt, expected_cues):
    out = detect_amb(sql_ready_prompt=sql_ready_prompt)
    _assert_contract(out)
    assert out.has_ambiguity is True, f"expected ambiguity for: {sql_ready_prompt}"
    # Require at least one question referencing any of the expected cues (lenient)
    joined_qs = " ".join(out.ambiguities.values()).lower()
    assert any(cue in joined_qs for cue in (c.lower() for c in expected_cues)), \
        f"expected one of {expected_cues} in clarification questions, got: {out.ambiguities}"


# --------------------------
# 2) Multiple ambiguities in one prompt → should surface >1 item
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "sql_ready_prompt, min_count, cue_sets",
    [
        (
            "Top customers in recent years; revenue per customer; shipped only.",
            2,
            (("how many", "top"), ("which years", "recent years", "start year")),
        ),
        (
            "Largest orders for recent months; group by customer segment.",
            2,
            (("how many", "largest"), ("which months", "recent months", "time window")),
        ),
    ],
    ids=["top_customers_and_recent_years", "largest_orders_recent_months"],
)
def test_multiple_ambiguities(detect_amb, sql_ready_prompt, min_count, cue_sets):
    out = detect_amb(sql_ready_prompt=sql_ready_prompt)
    _assert_contract(out)
    assert out.has_ambiguity is True
    assert len(out.ambiguities) >= min_count, f"expected at least {min_count} ambiguities"
    all_qs = " ".join(out.ambiguities.values()).lower()
    for cues in cue_sets:
        assert any(c in all_qs for c in cues), f"missing cues {cues} in questions: {out.ambiguities}"


# --------------------------
# 3) Unambiguous → should be FALSE with empty ambiguities
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "sql_ready_prompt",
    [
        "Return total revenue per year for 2020–2022; shipped orders only; order by year ascending; limit to top 5 customers per year.",
        "Show top 10 customers by total revenue for 2023; exclude cancelled; order by revenue descending.",
        "Daily active users for 2024-03-01 to 2024-03-31; group by day; US region only.",
        "Order count by customer segment for January 2024; online channel only; order by count descending.",
    ],
    ids=[
        "fully_specified_years_top_per_year",
        "top10_customers_in_2023",
        "dau_explicit_range",
        "orders_seg_jan2024_online",
    ],
)
def test_unambiguous_false(detect_amb, sql_ready_prompt):
    out = detect_amb(sql_ready_prompt=sql_ready_prompt)
    _assert_contract(out)
    assert out.has_ambiguity is False, f"should be unambiguous: {sql_ready_prompt}"
    assert out.ambiguities == {} or len(out.ambiguities) == 0


# --------------------------
# 4) Avoid nit-picking
# --------------------------
# These are reasonably precise; the model should *not* invent tiny clarifications.

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "sql_ready_prompt",
    [
        "Monthly revenue for 2021; group by month; order by month ascending.",
        "Sum revenue per product category in 2023; order by revenue descending.",
        "Order count by customer for March 2024; shipped orders only.",
        "Average order value per region for 2022; exclude cancelled.",
    ],
    ids=["monthly_rev_2021", "sum_rev_category_2023", "orders_by_customer_mar2024", "aov_by_region_2022"],
)
def test_avoid_nit_picking(detect_amb, sql_ready_prompt):
    out = detect_amb(sql_ready_prompt=sql_ready_prompt)
    _assert_contract(out)
    assert out.has_ambiguity is False
    assert out.ambiguities == {} or len(out.ambiguities) == 0


# --------------------------
# 5) Non-ambiguous variants of previously ambiguous patterns
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "sql_ready_prompt",
    [
        "Show top 5 customers by total revenue; 2021–2023; shipped orders only.",
        "Return largest 20 orders; 2023 only; group by customer.",
        "Active users for the last 30 days ending 2024-03-31; group by day.",
        "High-value customers defined as revenue > 1000; by region; 2022.",
    ],
    ids=["top5_with_range", "largest20_2023", "dau_last30_anchored", "high_value_defined_threshold"],
)
def test_resolved_ambiguities_become_false(detect_amb, sql_ready_prompt):
    out = detect_amb(sql_ready_prompt=sql_ready_prompt)
    _assert_contract(out)
    assert out.has_ambiguity is False
    assert out.ambiguities == {} or len(out.ambiguities) == 0
