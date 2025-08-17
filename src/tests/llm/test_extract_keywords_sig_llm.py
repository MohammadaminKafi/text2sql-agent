import re
import pytest
import dspy
from components.dspy_signatures import ExtractKeywordsSig


# --------------------------
# Fixture
# --------------------------

@pytest.fixture(scope="module")
def kw_extractor(llm):
    dspy.configure(lm=llm)
    return dspy.Predict(ExtractKeywordsSig)


# --------------------------
# Helpers – strong hygiene checks (very sensitive)
# --------------------------

AGG_VERBS = {
    "sum", "average", "avg", "count", "median", "min", "max", "total",
    "aggregate", "aggregated", "mean"
}

CAL_TIME_WORDS = {
    "date", "year", "month", "quarter", "week", "day", "hour", "minute",
    "q1", "q2", "q3", "q4", "fy", "fy2024", "today", "yesterday", "tomorrow",
    "recent", "last", "this", "next", "current"
}

MONTHS = {
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
    "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec"
}

DOW = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday","mon","tue","wed","thu","fri","sat","sun"}

STOP_NOT_FIELDS = {
    "top","bottom","most","least","by","in","for","of","to","with","without","only",
    "and","or","vs","per","across","between","from","until","through"
}

LITERAL_VALUE_HINTS = {
    "us","usa","emea","germany","online","offline","shipped","cancelled","returned","completed","pending"
}

TOKEN_OK_PATTERN = re.compile(r"^[A-Za-z0-9 _\-]+$")

def _is_date_like(tok: str) -> bool:
    t = tok.lower().strip()
    if re.search(r"\d{4}-\d{2}-\d{2}", t): return True
    if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", t): return True
    if t in MONTHS or t in DOW: return True
    if t in CAL_TIME_WORDS: return True
    if re.fullmatch(r"q[1-4]", t): return True
    if re.fullmatch(r"fy\d{4}", t): return True
    return False

def _is_number_like(tok: str) -> bool:
    return bool(re.fullmatch(r"[0-9.,]+", tok.strip()))

def _has_quotes(tok: str) -> bool:
    return ("'" in tok) or ('"' in tok)

def _looks_like_non_field(tok: str) -> bool:
    t = tok.lower().strip()
    if t in STOP_NOT_FIELDS: return True
    if t in LITERAL_VALUE_HINTS: return True
    if t in AGG_VERBS: return True
    if _is_number_like(t): return True
    if _is_date_like(t): return True
    if _has_quotes(tok): return True
    if not TOKEN_OK_PATTERN.match(tok): return True
    return False

def _assert_keywords_clean(keywords, max_k):
    # Contract
    assert isinstance(keywords, list), "keywords must be a list"
    assert len(keywords) <= max_k, f"must honor max_keywords={max_k}"

    # No empties, trimmed, unique (case-insensitive)
    lowered = set()
    for k in keywords:
        assert isinstance(k, str) and k.strip(), "keyword must be a non-empty string"
        assert k == k.strip(), "no leading/trailing spaces"
        kl = k.lower()
        assert kl not in lowered, f"duplicate keyword: {k}"
        lowered.add(kl)

        # High-sensitivity hygiene
        assert not _looks_like_non_field(k), f"not a field-like keyword: {k}"

        # Avoid aggregator prefixes like "total revenue" → prefer "revenue"
        for agg in AGG_VERBS:
            assert not kl.startswith(agg + " "), f"avoid aggregator phrasing in keywords: {k}"
            assert kl != agg, f"pure aggregator not allowed: {k}"


# --------------------------
# 0) Contract & hygiene on simple prompt
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "sql_prompt,max_keywords,allowed_any",
    [
        ("total sales by region in 2024 by month", 3, {"sales","region"}),
        ("count shipped orders by customer for January 2023 in US", 3, {"orders","customer"}),
        ("average order value per product category, exclude cancelled", 4, {"order","order value","product","category","product category"}),
    ],
    ids=["sales_region", "orders_customer", "aov_product_category"],
)
def test_contract_and_hygiene_basic(kw_extractor, sql_prompt, max_keywords, allowed_any):
    out = kw_extractor(sql_prompt=sql_prompt, max_keywords=max_keywords)
    kws = out.keywords
    _assert_keywords_clean(kws, max_keywords)
    # Must include at least one of the expected field-like concepts
    assert any(kk.lower() in {a.lower() for a in allowed_any} for kk in kws), f"expected one of {allowed_any}, got {kws}"
    # Ensure time-like concepts were not returned
    banned_time = {"month","year","date","week","january","2023","2024"}
    for kk in kws:
        assert kk.lower() not in banned_time, f"time-like token leaked: {kk}"


# --------------------------
# 1) Exclude values, numbers, dates, time-like, aggregations
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "sql_prompt,max_keywords,required_subset,banned_any",
    [
        (
            "orders where country = 'Germany' and channel = 'Online' grouped by city",
            4,
            {"orders","country","channel","city"},
            {"germany","online","grouped","'germany'","'online'"},
        ),
        (
            "median, sum, average across sales by region",
            3,
            {"sales","region"},
            {"median","sum","average","across"},
        ),
        (
            "show me the # of invoices for Q2 FY2024 in EMEA",
            3,
            {"invoices"},
            {"q2","fy2024","emea","#"},
        ),
        (
            "shipment date between 2021-01-01 and 2021-03-31; group by week",
            3,
            {"shipment","orders"},  # allow one of these if model extracts a base entity
            {"date","2021-01-01","2021-03-31","week","group"},
        ),
    ],
    ids=["values_and_grouping", "aggregation_verbs", "quarter_fy_region", "range_and_week"],
)
def test_exclusions_strict(kw_extractor, sql_prompt, max_keywords, required_subset, banned_any):
    out = kw_extractor(sql_prompt=sql_prompt, max_keywords=max_keywords)
    kws = out.keywords
    _assert_keywords_clean(kws, max_keywords)
    # At least one of required base entities shows up
    assert any(k.lower() in {r.lower() for r in required_subset} for k in kws), f"need one of {required_subset}, got {kws}"
    # None of the banned pieces should appear
    for bad in banned_any:
        assert all(bad.lower() != k.lower() for k in kws), f"banned token present: {bad} in {kws}"


# --------------------------
# 2) Field lists & identifiers with underscores – preserve field-like shape
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "sql_prompt,max_keywords,allowed_only",
    [
        ("product_id, product name, order_id", 5, {"product_id","product name","order_id"}),
        ("customer_id and account_id by region", 3, {"customer_id","account_id","region"}),
    ],
    ids=["ids_and_names", "ids_by_region"],
)
def test_identifiers_and_multiword_fields(kw_extractor, sql_prompt, max_keywords, allowed_only):
    out = kw_extractor(sql_prompt=sql_prompt, max_keywords=max_keywords)
    kws = out.keywords
    _assert_keywords_clean(kws, max_keywords)
    # Everything returned must be subset of allowed_only (strict)
    for k in kws:
        assert k.lower() in {a.lower() for a in allowed_only}, f"unexpected keyword leaked: {k} (allowed: {allowed_only})"


# --------------------------
# 3) Ranking & max_keywords honored – keep most useful fields
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "sql_prompt,max_keywords,important_priority,allowed_set",
    [
        (
            "sales revenue profit margin cost discounts returns region country city channel product category customer",
            3,
            ["sales","revenue","customer"],  # top priority candidates
            {"sales","revenue","profit","margin","cost","discounts","returns","region","country","city","channel","product","category","customer"},
        ),
        (
            "orders items order value product category customer segment store region",
            2,
            ["orders","order value","customer"],  # prefer orders/order value/customer
            {"orders","items","order","order value","product","category","customer","segment","store","region"},
        ),
    ],
    ids=["many_candidates_pick3", "many_candidates_pick2"],
)
def test_ranking_and_cap(kw_extractor, sql_prompt, max_keywords, important_priority, allowed_set):
    out = kw_extractor(sql_prompt=sql_prompt, max_keywords=max_keywords)
    kws = out.keywords
    _assert_keywords_clean(kws, max_keywords)
    # All returned keywords must be from allowed_set
    for k in kws:
        assert k.lower() in {a.lower() for a in allowed_set}, f"unexpected keyword: {k}"
    # Strong suggestion: at least one of the top priorities included
    assert any(k.lower() in {p.lower() for p in important_priority} for k in kws), \
        f"expected a priority field among {important_priority}, got {kws}"


# --------------------------
# 4) Edge: force exclusion of calendar/time words explicitly present
# --------------------------

@pytest.mark.llm
@pytest.mark.signature
@pytest.mark.parametrize(
    "sql_prompt,max_keywords",
    [
        ("total sales by region in 2024 by month and by week", 4),
        ("orders per day by month and quarter, in 2021 and 2022", 5),
        ("revenue by year and month; weekly trend for 2023", 3),
    ],
    ids=["month_week_present", "day_month_quarter_present", "year_month_weekly_present"],
)
def test_calendar_tokens_excluded(kw_extractor, sql_prompt, max_keywords):
    out = kw_extractor(sql_prompt=sql_prompt, max_keywords=max_keywords)
    kws = out.keywords
    _assert_keywords_clean(kws, max_keywords)
    cal_tokens = CAL_TIME_WORDS | MONTHS | DOW | {"weekly","monthly","yearly"}
    for k in kws:
        assert k.lower() not in cal_tokens, f"calendar/time token leaked: {k}"
