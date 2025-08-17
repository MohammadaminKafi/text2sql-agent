# tests/llm/test_quick_gate_sig_llm.py

import pytest
import dspy
from components.dspy_signatures import QuickGateSig


@pytest.fixture(scope="module")
def quick_gate(llm):
    """Bind DSPy to the live LLM once, then build a predictor for the signature."""
    dspy.configure(lm=llm)
    return dspy.Predict(QuickGateSig)


def _invoke(quick_gate, prompt: str):
    """Run the signature and normalize outputs."""
    out = quick_gate(user_prompt=prompt)
    is_text2sql = bool(out.is_text2sql)
    confidence = float(out.confidence)
    cause = str(out.cause or "").strip()
    return is_text2sql, confidence, cause


def _assert_contract(is_text2sql, confidence, cause):
    """Contract checks that should hold for any output."""
    assert isinstance(is_text2sql, bool)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0, f"confidence out of range: {confidence}"
    assert isinstance(cause, str) and len(cause) > 0, "cause should be a non-empty one-liner"
    assert "\n" not in cause, "cause should be a single line"
    assert len(cause) <= 200, "keep the cause short"


@pytest.mark.llm
def test_contract_and_schema(quick_gate):
    """Smoke test: outputs should follow the schema and be well-formed."""
    is_t2s, conf, cause = _invoke(quick_gate, "Show average order total per month in 2024.")
    _assert_contract(is_t2s, conf, cause)


# --------------------------
# POSITIVE: Clearly Text-to-SQL
# --------------------------
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt",
    [
        # original style
        "List the top 10 customers by total revenue this year.",
        "Count orders placed in March 2023.",
        "Join employees with departments and show headcount per department.",
        # extra English
        "Show average order total per month in 2024.",
        "Return the daily active users for the last 30 days.",
        "Which products have not been sold in the past quarter?",
        "Total revenue by region and quarter for 2023; include regions with zero sales.",
        # Persian (FA)
        "فهرست ۱۰ مشتری برتر از نظر مبلغ فروش در سال جاری را نشان بده.",
        "میانگین مبلغ سفارش به تفکیک ماه در ۲۰۲۳ را گزارش کن.",
    ],
    ids=[
        "top10_customers_revenue",
        "count_orders_march_2023",
        "join_emp_dept_headcount",
        "avg_order_2024_by_month",
        "dau_last_30_days",
        "unsold_products_past_quarter",
        "rev_by_region_quarter_2023_including_zeros",
        "fa_top10_customers",
        "fa_avg_order_by_month_2023",
    ],
)
def test_positive_cases_prefer_true(quick_gate, prompt):
    """
    For clearly text-to-SQL prompts, the signature should classify TRUE.
    If it doesn't, it should at least show low confidence (<= 0.35),
    honoring the 'prefer TRUE' guidance.
    """
    is_t2s, conf, cause = _invoke(quick_gate, prompt)
    _assert_contract(is_t2s, conf, cause)

    if not is_t2s:
        assert conf <= 0.35, f"False negative but too confident: conf={conf}, cause={cause}"


# --------------------------
# NEGATIVE: Clearly Non-DB
# --------------------------
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt",
    [
        # original style
        "Write a haiku about wind.",
        "Implement quicksort in Python.",
        "Translate 'hello' to French.",
        "What is the capital of Peru?",
        "Give me career advice for becoming a designer.",
        # more negatives
        "How do I install SQL Server on Ubuntu?",
        "What's new in Python 3.12?",
        "Summarize the latest news about Tesla.",
        "Generate a logo for my coffee shop.",
        "Refactor this code for readability.",
        # SQL-keyword but not a DB lookup
        "Explain the difference between SELECT and UPDATE.",
        "How to design a star schema for a data warehouse?",
        # Persian (FA)
        "یک شعر کوتاه درباره باد بنویس.",
        "تفاوت بین SELECT و UPDATE را توضیح بده.",
        "چطور در ویندوز SQL Server را نصب کنم؟",
    ],
    ids=[
        "haiku",
        "quicksort",
        "translate_hello",
        "capital_of_peru",
        "career_advice",
        "install_sqlserver_ubuntu",
        "python_312_news",
        "tesla_news",
        "generate_logo",
        "refactor_code",
        "explain_select_update",
        "design_star_schema",
        "fa_poem_wind",
        "fa_explain_select_update",
        "fa_install_sqlserver_windows",
    ],
)
def test_negative_cases_prefer_false(quick_gate, prompt):
    """
    For clearly non-DB prompts, the signature should classify FALSE.
    If it predicts TRUE, it must be with low confidence (<= 0.35).
    """
    is_t2s, conf, cause = _invoke(quick_gate, prompt)
    _assert_contract(is_t2s, conf, cause)

    if is_t2s:
        assert conf <= 0.35, f"False positive but too confident: conf={conf}, cause={cause}"


# --------------------------
# AMBIGUOUS: Prefer TRUE but modest confidence (tripled to 9; includes FA)
# --------------------------
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt",
    [
        # original style
        "Tell me about sales.",
        "users report?",
        "How are things going?",
        # more ambiguous
        "Revenue?",
        "Make me a report.",
        "What should I look at this week?",
        # Persian (FA)
        "گزارش فروش؟",
        "کاربران؟",
        "چطور پیش می‌رود؟",
    ],
    ids=[
        "tell_me_about_sales",
        "users_report_fragment",
        "how_are_things",
        "revenue_single_token",
        "make_me_a_report",
        "what_to_look_this_week",
        "fa_sales_report_q",
        "fa_users_q",
        "fa_hows_it_going",
    ],
)
def test_ambiguous_cases_modest_confidence(quick_gate, prompt):
    """
    Ambiguous prompts: spec says 'prefer TRUE' but with modest confidence.
    We don't force the label, but we require modest confidence.
    """
    is_t2s, conf, cause = _invoke(quick_gate, prompt)
    _assert_contract(is_t2s, conf, cause)
    assert 0.0 <= conf <= 0.5, f"Expected modest confidence for ambiguous: conf={conf}, cause={cause}"


# --------------------------
# DB-NEGATIVE: Keyword-bait negatives (talking about SQL, not asking DB)
# --------------------------
@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt",
    [
        "Explain SQL joins with examples.",
        "What does SELECT DISTINCT do?",
        "How to write a GROUP BY clause?",
        "Differences between WHERE and HAVING.",
        "Show me SQL syntax for INNER JOIN.",
        "Translate this SQL to English: SELECT * FROM Orders.",
        # Persian (FA)
        "JOIN در SQL را با مثال توضیح بده.",
        "GROUP BY چه کاری انجام می‌دهد؟",
        "نحوۀ نوشتن WHERE در SQL چگونه است؟",
    ],
    ids=[
        "explain_joins",
        "what_is_select_distinct",
        "how_to_group_by",
        "where_vs_having",
        "syntax_inner_join",
        "translate_sql_to_english",
        "fa_explain_join",
        "fa_what_group_by_does",
        "fa_how_to_write_where",
    ],
)
def test_keyword_bait_negatives(quick_gate, prompt):
    """
    These include SQL keywords but are meta/educational about SQL,
    not database data retrieval tasks. Expect FALSE or very low confidence if TRUE.
    """
    is_t2s, conf, cause = _invoke(quick_gate, prompt)
    _assert_contract(is_t2s, conf, cause)

    if is_t2s:
        assert conf <= 0.35, f"False positive on SQL-meta with high confidence: conf={conf}, cause={cause}"
