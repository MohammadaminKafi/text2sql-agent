import re
import pytest
import dspy
from components.dspy_signatures import SqlReadyPromptSig


# --------------------------
# Fixtures & helpers
# --------------------------

@pytest.fixture(scope="module")
def sql_ready(llm):
    """Bind DSPy to the live LLM once, then build a predictor for the signature."""
    dspy.configure(lm=llm)
    return dspy.Predict(SqlReadyPromptSig)


SQL_BANNED = (
    r"\bselect\b",
    r"\bwhere\b",
    r"\bfrom\b",
    r"\bgroup\s+by\b",
    r"\border\s+by\b",
    r"\bhaving\b",
    r"\blimit\b",
)
CODE_BANNED_PATTS = (
    r"```",      # code fences
    r"`",        # inline code
)

def _assert_no_sql_or_code(s: str):
    low = s.lower()
    for patt in SQL_BANNED:
        assert not re.search(patt, low), f"Should not contain SQL keyword: /{patt}/ → {s}"
    for patt in CODE_BANNED_PATTS:
        assert not re.search(patt, s), f"Should not contain code fences/inline code: {s}"

def _sentences(s: str):
    # crude but effective for our constraint (<= 2 sentences)
    parts = [p.strip() for p in re.split(r"[.!?]+", s) if p.strip()]
    return parts

def _assert_concise(s: str, max_chars=280, max_sentences=2):
    assert isinstance(s, str) and s.strip(), "Output must be non-empty string"
    assert len(s) <= max_chars, f"Too long for a 'crisp instruction' ({len(s)} chars)"
    assert len(_sentences(s)) <= max_sentences, f"Should be <= {max_sentences} sentences: {s}"

def _must_contain_any(s: str, options):
    low = s.lower()
    assert any(opt.lower() in low for opt in options), f"Expected one of {options} in: {s}"

def _must_contain_all(s: str, tokens):
    low = s.lower()
    for t in tokens:
        assert t.lower() in low, f"Missing required token '{t}' in: {s}"

def _must_not_contain_any(s: str, tokens):
    low = s.lower()
    for t in tokens:
        assert t.lower() not in low, f"Unexpected token '{t}' present: {s}"


# --------------------------
# 0) Contract & bans (structure sanity)
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt",
    [
        "Top 5 customers by total revenue in 2023; shipped orders only.",
        "Monthly revenue trend in 2021 for US region.",
        "Compare average order value by customer segment last quarter.",
        "Daily shipped order count for March 2024.",
    ],
    ids=["top5_rev_2023", "monthly_rev_region", "aov_by_segment_qtr", "daily_shipped_count"],
)
def test_contract_and_bans(sql_ready, prompt):
    out = sql_ready(user_prompt=prompt)
    s = out.sql_ready_prompt
    _assert_concise(s)
    _assert_no_sql_or_code(s)


# --------------------------
# 1) Happy paths – clear measures/filters/group/order (parameterized)
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt, measures_any, filters_all, grouping_any, ordering_any",
    [
        (
            "Top 5 customers by total revenue in 2023; shipped orders only.",
            ["total revenue"],                                  # measures
            ["shipped", "2023"],                                # filters (status & year)
            ["by customer", "per customer"],                    # grouping
            ["top 5", "first 5", "limit to 5", "return 5"],     # ordering/limit phrasing (non-SQL)
        ),
        (
            "Monthly revenue trend in 2021 for US region.",
            ["revenue"],                                        # measures
            ["2021", "US"],                                     # filters
            ["by month", "per month"],                          # grouping
            ["ascending", "chronological", "by month ascending"],  # ordering hint
        ),
        (
            "Daily shipped order count for March 2024.",
            ["shipped order count", "order count", "shipped orders"], # measures
            ["march 2024", "shipped"],                             # filters
            ["by day", "per day"],                                 # grouping
            ["ascending", "chronological", "by day ascending"],    # ordering
        ),
        (
            "Sum revenue by product category last quarter; sort highest to lowest.",
            ["sum revenue", "total revenue", "revenue"],           # measures
            ["last quarter"],                                      # filters
            ["by category", "per product category"],               # grouping
            ["descending", "highest to lowest", "sort by revenue descending"], # ordering
        ),
        (
            "Average order value by year for 2020–2022, exclude cancelled.",
            ["average order value", "avg order value"],            # measures
            ["2020", "2021", "2022", "exclude cancelled"],         # filters
            ["by year", "per year"],                               # grouping
            ["ascending", "chronological"],                        # ordering
        ),
    ],
    ids=[
        "top5_rev_shipped_2023",
        "monthly_rev_2021_us",
        "daily_shipped_count_mar2024",
        "sum_rev_by_category_desc",
        "aov_by_year_range",
    ],
)
def test_happy_paths(sql_ready, prompt, measures_any, filters_all, grouping_any, ordering_any):
    s = sql_ready(user_prompt=prompt).sql_ready_prompt
    _assert_concise(s)
    _assert_no_sql_or_code(s)
    _must_contain_any(s, measures_any)
    _must_contain_all(s, filters_all)
    _must_contain_any(s, grouping_any)
    _must_contain_any(s, ordering_any)


# --------------------------
# 2) Filter extraction & clarity (parameterized)
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt, must_filters",
    [
        ("Revenue for orders between 2021-01-01 and 2021-12-31, status shipped.",
         ["2021-01-01", "2021-12-31", "shipped"]),
        ("Order count for January 2024 in Canada (online channel only).",
         ["january 2024", "canada", "online"]),
        ("Total refunds for returned items in EMEA last month.",
         ["returned", "EMEA", "last month"]),
        ("Customer list with > 5 orders and revenue over 1000 in 2023.",
         ["> 5", "over 1000", "2023"]),  # GK: numeric constraints should be echoed
    ],
    ids=["date_range_shipped", "jan2024_canada_online", "refunds_returned_emea_last_month", "thresholds_2023"],
)
def test_filters_are_spelled_out(sql_ready, prompt, must_filters):
    s = sql_ready(user_prompt=prompt).sql_ready_prompt
    _assert_concise(s)
    _assert_no_sql_or_code(s)
    _must_contain_all(s, must_filters)


# --------------------------
# 3) Ordering / limits phrased naturally (parameterized)
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt, ordering_any",
    [
        ("Show the 10 most recent orders by order date.", ["most recent", "newest first", "sort by order date descending"]),
        ("Top 3 categories by total revenue.", ["top 3", "highest", "largest"]),
        ("Bottom 5 products by profit margin.", ["bottom 5", "lowest", "smallest"]),
        ("Order results by year ascending.", ["ascending", "earliest first", "oldest to newest"]),
    ],
    ids=["10_most_recent", "top3_categories", "bottom5_margin", "year_ascending"],
)
def test_ordering_limits_natural_language(sql_ready, prompt, ordering_any):
    s = sql_ready(user_prompt=prompt).sql_ready_prompt
    _assert_concise(s)
    _assert_no_sql_or_code(s)
    _must_contain_any(s, ordering_any)


# --------------------------
# 4) Prompts containing SQL/table names → output must be clean (parameterized)
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt, forbidden_literals",
    [
        ("SELECT * FROM Sales.SalesOrderHeader WHERE Status = 'Shipped'", ["select", "from", "where", "sales.salesorderheader"]),
        ("Compute monthly revenue from dbo.Orders joined to dbo.Customers", ["dbo.orders", "dbo.customers"]),
        ("Group by year and order by revenue desc", ["group by", "order by", "desc"]),
    ],
    ids=["raw_sql", "schema_qualified", "sql_phrases"],
)
def test_sqly_prompts_are_sanitized(sql_ready, prompt, forbidden_literals):
    s = sql_ready(user_prompt=prompt).sql_ready_prompt
    _assert_concise(s)
    _assert_no_sql_or_code(s)
    _must_not_contain_any(s, forbidden_literals)


# --------------------------
# 5) Vague prompts become crisp instructions (parameterized)
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "prompt, must_have_any_measure, must_have_any_grouping",
    [
        ("Sales trend last year.", ["revenue", "sales amount", "sales"], ["by month", "per month", "by quarter", "per quarter"]),
        ("Orders by customer.", ["order count", "orders"], ["by customer", "per customer"]),
        ("How are shipments doing recently?", ["shipped order count", "shipment count", "shipment volume"], ["by week", "per week", "by month"]),
        ("Which regions are performing best?", ["revenue", "sales"], ["by region", "per region"]),
    ],
    ids=["sales_trend_last_year", "orders_by_customer", "shipments_recently", "regions_best"],
)
def test_vague_prompts_become_crisp(sql_ready, prompt, must_have_any_measure, must_have_any_grouping):
    s = sql_ready(user_prompt=prompt).sql_ready_prompt
    _assert_concise(s)
    _assert_no_sql_or_code(s)
    _must_contain_any(s, must_have_any_measure)
    _must_contain_any(s, must_have_any_grouping)