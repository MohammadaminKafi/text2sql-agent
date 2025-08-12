import os
import json
import logging
import re
import time
import urllib.parse
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import dspy
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect
from urllib.parse import urlparse
import sqlparse
from datetime import date as py_date, datetime as py_datetime
import calendar
import jdatetime

from dspy import (ChainOfThought, InputField, Module, OutputField, Predict,
                  Signature)


# ─────────────────────────────  LOGGING SETUP  ───────────────────────────── #
LOG_FORMAT = "%(asctime)s  [%(levelname)s]  %(name)s › %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt="%H:%M:%S")


class TruncateLongMsgs(logging.Filter):
    def __init__(self, max_len: int = 300):
        super().__init__()
        self.max_len = max_len

    def filter(self, record: logging.LogRecord) -> bool:
        if len(record.getMessage()) > self.max_len:
            record.msg = record.getMessage()[: self.max_len] + " …(truncated)"
        return True


logging.getLogger().addFilter(TruncateLongMsgs(100))

# Silence chatty third-party loggers
NOISY = (
    "LiteLLM",
    "litellm",
    "httpx",
    "urllib3",
    "httpcore",
    "openai",
    "openai._base_client",
)
for name in NOISY:
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.debug("🚀 Logging initialised (level=DEBUG)")


# ──────────────────────────  DATE / CALENDAR HELPERS  ───────────────────── #

CAL_GREGORIAN = "Gregorian"
CAL_SOLAR = "Solar"  # Solar Hijri (Shamsi / Jalali)

# Persian and Arabic-Indic digits → Latin
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_eastern_digits(s: str) -> str:
    """Convert Persian/Arabic-Indic numerals in a string to ASCII digits."""
    return s.translate(_PERSIAN_DIGITS).translate(_ARABIC_INDIC_DIGITS)


# Month name maps (minimal but practical set)
GREG_MONTHS_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Solar Hijri months (Persian script + common Latin transliterations)
JALALI_MONTHS = {
    "فروردین": 1, "ordibehesht": 2, "اردیبهشت": 2, "khordad": 3, "خرداد": 3,
    "tir": 4, "تیر": 4, "mordad": 5, "مرداد": 5, "shahrivar": 6, "شهریور": 6,
    "mehr": 7, "مهر": 7, "aban": 8, "آبان": 8, "azar": 9, "آذر": 9,
    "dey": 10, "دی": 10, "bahman": 11, "بهمن": 11, "esfand": 12, "اسفند": 12,
}

def jalali_days_in_month(year: int, month: int) -> int:
    """Number of days in a given Jalali month (uses jdatetime leap logic)."""
    if month < 1 or month > 12:
        raise ValueError("Invalid Jalali month")
    if month <= 6:
        return 31
    if 7 <= month <= 11:
        return 30
    # Esfand (12)
    # jdatetime leap: create a date and call isleap()
    if jdatetime is None:
        # conservative default if lib missing
        return 29
    leap = jdatetime.date(year, 1, 1).isleap()
    return 30 if leap else 29

def gregorian_days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _ensure_jdatetime():
    if jdatetime is None:
        raise RuntimeError("jdatetime is not installed. Please: pip install jdatetime")


def jalali_to_gregorian(y: int, m: int, d: int) -> py_date:
    """Exact conversion Jalali → Gregorian date."""
    _ensure_jdatetime()
    return jdatetime.date(y, m, d).togregorian()

def gregorian_to_jalali(y: int, m: int, d: int) -> jdatetime.date:
    """Exact conversion Gregorian → Jalali date."""
    _ensure_jdatetime()
    return jdatetime.date.fromgregorian(year=y, month=m, day=d)

def format_iso(y: int, m: int | None = None, d: int | None = None) -> str:
    """ISO-ish string with available granularity."""
    if m is None:
        return f"{y:04d}"
    if d is None:
        return f"{y:04d}-{m:02d}"
    return f"{y:04d}-{m:02d}-{d:02d}"


# ──────────────────────────  Utility helpers  ────────────────────────────── #


def list_schemas(engine: sa.Engine) -> List[str]:
    logger.debug("🔍 list_schemas: inspecting database for schema names")
    inspector = inspect(engine)
    schemas = inspector.get_schema_names()
    logger.debug("🔍 list_schemas: found %s", schemas)
    return schemas


def list_tables(engine: sa.Engine, schema: str) -> List[str]:
    logger.debug("🔍 list_tables: schema=%s", schema)
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema=schema)
    logger.debug("🔍 list_tables: %s → %s", schema, tables)
    return tables


def list_columns(engine, schema: str, table: str) -> List[Tuple[str, str]]:
    logger.debug("🔍 list_columns: %s.%s", schema, table)

    inspector = inspect(engine)
    cols = [
        (col["name"], str(col["type"]))
        for col in inspector.get_columns(table, schema=schema)
    ]

    logger.debug("🔍 list_columns: %s.%s → %s", schema, table, cols)
    return cols


def get_pk_fk_pairs(
    engine: sa.Engine,
    tables: List[Tuple[str, str]],
) -> List[Tuple[str, str, str]]:
    """
    Return (child_table, parent_table, constraint_name) triples for every
    FK that originates in tables.
    """
    logger.debug("🔍 get_pk_fk_pairs: %s tables provided", len(tables))

    inspector = inspect(engine)
    relations: list[tuple[str, str, str]] = []

    for schema, table in tables:
        fk_list = inspector.get_foreign_keys(table, schema=schema)
        logger.debug("   🗄️  %s.%s — %s FKs found", schema, table, len(fk_list))

        for fk in fk_list:
            src_fq = f"{schema}.{table}"
            tgt_schema = fk.get("referred_schema") or schema
            tgt_table = fk["referred_table"]
            tgt_fq = f"{tgt_schema}.{tgt_table}"
            cname = fk.get("name", "")

            # Skip self-references
            if src_fq.lower() == tgt_fq.lower():
                continue

            relations.append((src_fq, tgt_fq, cname))
            logger.debug("      🔗 %s → %s  (constraint=%s)", src_fq, tgt_fq, cname)

    # ── de-duplicate (same edge can appear twice) ─────────────────────
    dedup = list(
        {(a.lower(), b.lower(), c): (a, b, c) for a, b, c in relations}.values()
    )

    logger.debug("🔍 get_pk_fk_pairs: discovered %s relation(s)", len(dedup))
    return dedup


def extract_sql(llm_response: str) -> str:
    logger.debug("🧹 extract_sql: received %s chars", len(llm_response))

    patterns: List[tuple[str, str]] = [
        (r"\bCREATE\s+TABLE\b.*?\bAS\b.*?;", "CREATE TABLE AS"),
        (r"\bWITH\b .*?;", "WITH / CTE"),
        (r"\bSELECT\b .*?;", "SELECT"),
        (r"```sql\s*\n(.*?)```", "```sql fenced"),
        (r"```(.*?)```", "generic fenced"),
    ]

    for pat, label in patterns:
        matches = re.findall(pat, llm_response, re.DOTALL | re.IGNORECASE)
        if matches:
            sql = matches[-1].strip()
            logger.debug("🧹 extract_sql: matched %s block (%s chars)", label, len(sql))
            return sql

    logger.debug("🧹 extract_sql: no pattern matched; returning full text")
    return llm_response.strip()


def is_sql_valid(sql: str) -> bool:
    logger.debug("🔎 is_sql_valid: validating SQL (%s chars)", len(sql))
    for stmt in sqlparse.parse(sql):
        if stmt.get_type().upper() == "SELECT":
            logger.debug("🔎 is_sql_valid: found SELECT → valid")
            return True
    logger.debug("🔎 is_sql_valid: no SELECT found → invalid")
    return False


def execute_query(engine: sa.Engine, query: str) -> pd.DataFrame:
    logger.info("🪄 execute_query: %s", query.replace("\n", " "))
    with engine.connect() as conn:
        result = conn.execute(sa.text(query))  # Use sa.text() for raw SQL
        rows = result.fetchall()
        columns = result.keys()
    df = pd.DataFrame(rows, columns=columns)
    logger.info("🪄 execute_query: returned %s rows × %s cols", df.shape[0], df.shape[1])
    return df


_PRESET_ANSWERS = [
    "Yes",
    "No",
    "I don't know",
    "Doesn't matter",
    "Maybe",
    "Absolutely!",
    "Absolutely not",
]


def ask_user(question: str) -> str:
    """
    Print `question`, then show a numbered list of preset replies plus an
    "Other…" option. Return the chosen answer (or free-form input).

    Works in any CLI / notebook environment that supports `input()`.
    """
    # Display the question
    print("\n" + "─" * 60)
    print(f"Agent asked for more clarification ➜ {question}\n")

    # Show the preset menu
    for idx, ans in enumerate(_PRESET_ANSWERS, start=1):
        print(f"[{idx}] {ans}")
    print("[0] Other…")  # sentinel for custom input

    # Keep asking until we get a valid response
    while True:
        choice = input("\nChoose a number or press Enter for 'Other': ").strip()

        if choice == "" or choice == "0":
            # Free-form path
            custom = input("Your custom answer: ").strip()
            if custom:
                return custom
            print("⚠️  Empty input—please type something.")
            continue

        # Numeric choice → preset answer
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(_PRESET_ANSWERS):
                return _PRESET_ANSWERS[idx - 1]

        # Anything else is invalid ⇒ loop again
        print("Invalid selection. Try again.")


# ───────────────────────────  DSPy Tools  ────────────────────────────────── #

interact_user = dspy.Tool(
    func=ask_user,
    name="ask_user",
    desc="Ask the human-in-the-loop a question for more clarification and return the answer.",
)

# ───────────────────────────  DSPy Signatures  ───────────────────────────── #


class QuickGateSig(Signature):
    """
    Decide if the user's request is likely a Text-to-SQL task.

    Behaviors (LOOSE):
    - Return TRUE when the prompt seems to ask for data retrieval, filtering,
      grouping, joining, aggregation, KPIs, tabular reports, or trends that are
      typically answered from an internal database — in ANY language.
    - Do NOT require the word "SQL".
    - If ambiguous, prefer TRUE but use modest confidence (≈0.35–0.55).

    Return FALSE when the prompt is clearly not a database question
    (creative writing, general coding unrelated to data access, web/news lookup,
     personal advice, casual chat, or questions about LLMs/tools themselves).

    Output:
    - is_text2sql: boolean
    - confidence: 0.0–1.0 (calibrated, not necessarily linear)
    - cause: one short line explaining the decision
    """

    user_prompt: str = InputField(desc="Raw user prompt in any language")

    is_text2sql: bool = OutputField(desc="True if likely a Text-to-SQL request")
    confidence: float = OutputField(desc="0.0–1.0 confidence in the decision")
    cause: str = OutputField(desc="One short line explaining the decision")


class DetectDatesSig(Signature):
    """
    Find date-like mentions in the user's prompt and decide how to normalize them
    to the target database calendar.

    Behavior:
    - Identify explicit dates (e.g., 2024-05-03, 03/05/2024, 1402/03/10), month-year
      (e.g., June 2023, Khordad 1402 / خرداد ۱۴۰۲), and year-only (e.g., 2021, ۱۴۰۲).
    - For each item, infer source calendar (gregorian|jalali) from context (month names,
      language, or year range). When unclear, make your best call.
    - Choose target_calendar based on database_calendar input.
    - Return a compact JSON array. Each element must be:
        {
          "text": "<verbatim span from the prompt>",
          "src_calendar": "gregorian" | "jalali",
          "granularity": "day" | "month" | "year",
          "year": 2024,
          "month": 5 | null,
          "day": 3 | null
        }
      Only include items you are confident in and that should be normalized.

    Notes:
    - Keep numbers exactly as they appear in the prompt in "text".
    - Use Latin digits in year/month/day fields.
    """

    original_prompt: str = InputField(
        desc="User's raw prompt (any language, any digits)"
    )
    database_calendar: str = InputField(
        desc="'Gregorian' or 'Solar' (Solar Hijri/Jalali)"
    )
    items_json: str = OutputField(
        desc="JSON array of date objects as specified"
    )


class TranslatePromptSig(Signature):
    """
    If the user prompt is not in English, translate it;
    otherwise return it unchanged. Also detect and return the language.
    """

    user_prompt: str = InputField(
        desc="Original user prompt in any language"
    )

    english_prompt: str = OutputField(
        desc="Prompt translated to English, or the original text if already English"
    )

    language: str = OutputField(
        desc="One-word language of the user_prompt, e.g. English, Farsi, etc."
    )


class SqlReadyPromptSig(Signature):
    """
    Re-express a raw business question as a single, well-structured
    natural-language instruction that makes it obvious to an LLM-to-SQL
    stage what to build—without actually writing SQL.

    Requirements for `sql_ready_prompt`:
    - Clearly list the *measures or columns* to return.
      – Prefer readable column labels over SQL snippets
      – e.g. “year (order_date)”, “shipped order count”, “total revenue”

    - Spell out any *filters* in plain words (“status is shipped”,
      “order date between 2021-01-01 and 2021-12-31”).

    - State grouping or aggregation intent (“group by year”,
      “sum revenue per customer”).

    - Mention *ordering / limits* if relevant (“order by year ascending”).

    - No SQL keywords (`SELECT`, `WHERE`, `GROUP BY`, …), no code blocks,
      no table names. Think of it as a crystal-clear spec a DB engineer
      could translate into SQL in one pass.

    One concise sentence (or two short clauses) is enough.
    """

    user_prompt: str = InputField(
        desc="User's raw English request (may be vague or poorly structured)"
    )
    sql_ready_prompt: str = OutputField(
        desc="Crisp, DB-oriented instruction in plain language (no SQL syntax)"
    )


class DetectAmbiguitySig(Signature):
    """
    Examine the SQL-ready prompt and flag only material ambiguities
    (omit nit-picking).

    Examples of ambiguities to catch
    • “recent years” without saying which years are considered recent → ask: “Where to start recent years?”
    • “top customers” without a count → ask: “How many top customers do you want?”
    """

    sql_ready_prompt: str = InputField()
    has_ambiguity: bool = OutputField(
        desc="True if the prompt needs clarification; False otherwise"
    )
    ambiguities: Dict[str, str] = OutputField(
        desc="Map {ambiguity_label: question_for_user}; empty if none"
    )


class TranslateQuestionContextSig(Signature):
    """
    Translate a question (in English) into the user's language.
    Use the original user prompt to preserve the context for the user.
    """

    original_prompt: str = InputField(
        desc="Original user prompt in any language"
    )
    user_language: str = InputField(
        desc="Detected language of the user's original prompt, e.g. English, Farsi, etc."
    )
    question_en: str = InputField(
        desc="Clarification question in English"
    )

    question_translated: str = OutputField(
        desc="Question translated to user's language"
    )


class ClarifyPromptSig(Signature):
    """
    Combine the original SQL-ready prompt with user clarifications
    to produce a single, unambiguous instruction string.

    The output prompt should:
    - Maintain the same structure as the sql_ready_prompt
    - Be in English

    Note:
    - The clarification question-answer pairs may be in any language.
    - The language of these pairs is provided for proper processing.
    """

    sql_ready_prompt: str = InputField(desc="Original SQL-ready prompt with structure to preserve")

    clarification_qs: List[str] = InputField(
        desc="List of clarification questions (may not be in English)"
    )
    clarification_as: List[str] = InputField(
        desc="List of corresponding clarification answers (may not be in English)"
    )
    clarifications_language: str = InputField(
        desc="Language code (ISO) of the clarification question-answer pairs"
    )

    clarified_prompt: str = OutputField(
        desc="Rewritten prompt in English, preserving the structure of sql_ready_prompt and incorporating all clarifications"
    )


class ExtractKeywordsSig(Signature):
    """
    From an English prompt, pull only the field-like keywords that correspond to potential table or column names;
    ignore literal filter values, numbers, date-like, calender-like, time-like and aggregation verbs.
      e.g.  'total sales by region in 2024 by month'  →  ['sales', 'region']
    Then rank them by importance and usefulness and only retuned up to `max_keywords` of them
    """

    sql_prompt: str = InputField(desc="SQL-ready prompt")
    max_keywords: int = InputField(desc="Maximum number of keywords to return")
    keywords: List[str] = OutputField(
        desc="List of bare nouns / entities likely matching DB fields"
    )


class KeywordSchemaSig(Signature):
    """
    Given a single keyword, pick the most relevant DB schemas.
    Sort them by relevancy and usefulness and return up to `max_chosen_schemas`.
    """

    keyword: str = InputField(desc="Single keyword we want to satisfy")
    db_schemas: List[str] = InputField(
        desc="All schema names that exist in the database"
    )
    max_chosen_schemas: int = InputField(desc="Maximum number of schemas to return")
    related_schemas: List[str] = OutputField(
        desc="Schemas related to the keyword (no commentary)"
    )


class KeywordTableSig(Signature):
    """
    Pick the most relevant tables in a given schema for a single keyword.
    Sort them by relevancy and usefulness and return up to `max_chosen_tables`.
    """

    keyword: str = InputField(desc="Single keyword we want to satisfy")
    schema_table_names: str = InputField(desc="Tbale names of the schema inspected")
    max_chosen_tables: int = InputField(desc="Maximum number of tables to return")
    related_tables: List[str] = OutputField(
        desc="Tables related to the keyword (no commentary)"
    )


class TableColumnSig(Signature):
    """
    For a given SQL prompt and table, return the column names most likely
    to satisfy the prompt. Both column names and
    their data-types are provided so it can pick the best match (e.g. favour numeric
    columns for “amount”, date columns for “year”, etc.).
    """

    sql_prompt: str = InputField(desc="Natural-language question we want to satisfy")
    schema_name: str = InputField(desc="Schema that owns the table")
    table_columns_info: List[Tuple[str, str]] = InputField(
        desc="List of (column_name, data_type) tuples for this table"
    )
    related_columns: List[str] = OutputField(
        desc="Column names that best match the prompt (no commentary)"
    )


class GenSqlSig(Signature):
    """
    Write a runnable T-SQL query that answers the request.
    
    Instructions for the LLM:
    - Use only the provided schemas, tables, columns, and relations in the context.
    - Never join tables or fields that do not have an explicitly provided relation in the context.
    - Terminate the query with a semicolon.
    - Do NOT wrap the SQL in markdown fences or any extra formatting.
    - Use proper SQL syntax and alias tables when necessary for clarity.
    - Ensure the query returns only the requested information, no extra columns or rows.
    """

    sql_prompt: str = InputField(desc="SQL-ready instruction string")
    context: Dict[str, Any] = InputField(
        desc="Dict with schemas/tables/columns/relations"
    )
    generated_sql: str = OutputField(desc="Executable SQL text")


class EvaluateSig(Signature):
    """
    Decide whether the dataframe answers the prompt.

    Special case:
    - If the dataframe is empty but the SQL and columns appear correct,
      treat the result as VALID (empty result can be valid).

    If INVALID, provide a clear and straightforward cause
    explaining why, avoiding jargon or vague statements.
    """

    user_prompt: str = InputField(desc="Original user prompt")
    sql_prompt: str = InputField(desc="SQL-ready instruction string")
    dataframe_json: str = InputField(
        desc="Small JSON sample (≈5 rows) of the query result"
    )
    verdict: str = OutputField(desc="'VALID' or 'INVALID'")
    cause: str = OutputField(
        desc="Clear, straightforward explanation when verdict is INVALID; empty otherwise"
    )


class RefineSqlSig(Signature):
    """
    Improve a faulty SQL query given the evaluator’s reason.
    Keep the same tables when possible; fix only what is necessary.
    Use the provided context (schemas/tables/columns/relations)
    to help generate correct SQL.
    """

    sql_prompt: str = InputField(desc="Original SQL-ready prompt")
    last_sql: str = InputField(desc="Previous (failing) SQL query")
    cause: str = InputField(desc="Why the query was judged INVALID")
    context: Dict[str, Any] = InputField(
        desc="Dict with schemas/tables/columns/relations"
    )
    improved_sql: str = OutputField(desc="Corrected SQL query")


class ReportSig(Signature):
    """
    Turn a validated SQL query into a reader-friendly report query
    and produce a concise plain-language summary of what the result
    set will show.
      • Alias ID columns with joined name columns (CustomerID → CustomerName).
      • Rename columns to human terms (‘OrderYear’, ‘TotalSales’, …).
      • Add ORDER BY or ranking columns where helpful.
      • No markdown fences; return bare SQL.
    """

    user_prompt: str = InputField(desc="Original user prompt")
    sql_prompt: str = InputField(desc="Cleaned prompt used for SQL generation")
    generated_sql: str = InputField(desc="Previously validated SQL query")
    readable_sql: str = OutputField(
        desc="Human-readable SQL with friendly column names, joins, ordering"
    )
    report: str = OutputField(
        desc="1-2 sentence description of what the resulting dataframe contains, "
        "phrased for the end user"
    )


# ────────────────────────────  DSPy Modules  ─────────────────────────────── #


class QuickText2SQLGate(Module):
    def __init__(self, min_confidence_true: float = 0.35):
        super().__init__()
        self.pred = Predict(QuickGateSig)
        self.min_confidence_true = min_confidence_true  # keeps it loose

    def forward(self, user_prompt: str) -> tuple[bool, float, str]:
        logger.debug("🚪 Gate.in: %s chars", len(user_prompt or ""))

        # Call LLM (module_name helps token accounting wrappers)
        out = self.pred(user_prompt=user_prompt, module_name="QuickText2SQLGate")

        is_sql = bool(getattr(out, "is_text2sql", False))
        conf_raw = float(getattr(out, "confidence", 0.0) or 0.0)
        cause = (getattr(out, "cause", "") or "").strip()

        conf = conf_raw
        if is_sql and conf < self.min_confidence_true:
            logger.debug("⬆️  Confidence floor applied: %.2f → %.2f",
                         conf, self.min_confidence_true)
            conf = self.min_confidence_true

        logger.info("🚪 Gate.out: is_text2sql=%s  conf=%.2f (raw=%.2f)  cause=%s",
                    is_sql, conf, conf_raw, cause or "—")

        return is_sql, conf, cause


class DateNormalizer(Module):
    """
    Pre-translation stage.
    Uses the LLM to find date-like mentions and this module to do *exact*
    Gregorian↔Jalali conversion with jdatetime. Replaces found spans inline
    with ISO-like normalized values in the database calendar.

    Returns (normalized_prompt, metadata)
    metadata: list of dicts with 'original', 'normalized', 'granularity', 'src_calendar', 'tgt_calendar'
    """

    def __init__(self, database_calendar: str):
        super().__init__()
        if database_calendar not in (CAL_GREGORIAN, CAL_SOLAR):
            raise ValueError("database_calendar must be 'Gregorian' or 'Solar'")
        self.database_calendar = database_calendar
        self.detect = ChainOfThought(DetectDatesSig)

    def _convert_one(self, item: dict) -> tuple[str, dict]:
        """
        Perform exact conversion using jdatetime where possible.
        Returns (normalized_string, metadata)
        """
        text = item.get("text", "")
        src = (item.get("src_calendar") or "").lower()
        gran = (item.get("granularity") or "").lower()
        y = item.get("year")
        m = item.get("month")
        d = item.get("day")

        # Normalize digits (LLM should give Latin digits, but be safe)
        try:
            y = int(normalize_eastern_digits(str(y))) if y is not None else None
            m = int(normalize_eastern_digits(str(m))) if m is not None else None
            d = int(normalize_eastern_digits(str(d))) if d is not None else None
        except Exception:
            # if parsing fails, don't convert
            return text, {
                "original": text, "normalized": text, "granularity": gran,
                "src_calendar": src, "tgt_calendar": self.database_calendar.lower(),
                "status": "left-unchanged (parse-failure)"
            }

        tgt = CAL_SOLAR.lower() if self.database_calendar == CAL_SOLAR else CAL_GREGORIAN.lower()

        # If granularity is day → exact convert.
        if gran == "day" and y and m and d:
            try:
                if src == "jalali" and tgt == "gregorian":
                    g = jalali_to_gregorian(y, m, d)
                    norm = format_iso(g.year, g.month, g.day)
                elif src == "gregorian" and tgt == "jalali":
                    j = gregorian_to_jalali(y, m, d)
                    norm = format_iso(j.year, j.month, j.day)
                else:
                    # already in target calendar
                    norm = format_iso(y, m, d)
            except Exception as exc:
                logger.debug("Date conversion failed for %s → %s: %s", text, tgt, exc)
                norm = format_iso(y, m, d)  # fallback: keep as-is
        # Month granularity → convert first day, then keep YYYY-MM in target.
        elif gran == "month" and y and m:
            try:
                if src == "jalali" and tgt == "gregorian":
                    g = jalali_to_gregorian(y, m, 1)
                    # Normalize as YYYY-MM in target calendar
                    norm = f"{g.year:04d}-{g.month:02d}"
                elif src == "gregorian" and tgt == "jalali":
                    j = gregorian_to_jalali(y, m, 1)
                    norm = f"{j.year:04d}-{j.month:02d}"
                else:
                    norm = f"{y:04d}-{m:02d}"
            except Exception as exc:
                logger.debug("Month conversion failed for %s → %s: %s", text, tgt, exc)
                norm = f"{y:04d}-{m:02d}"
        # Year-only → convert first day of year, then keep YYYY in target.
        elif gran == "year" and y:
            try:
                if src == "jalali" and tgt == "gregorian":
                    g = jalali_to_gregorian(y, 1, 1)
                    norm = f"{g.year:04d}"
                elif src == "gregorian" and tgt == "jalali":
                    j = gregorian_to_jalali(y, 1, 1)
                    norm = f"{j.year:04d}"
                else:
                    norm = f"{y:04d}"
            except Exception as exc:
                logger.debug("Year conversion failed for %s → %s: %s", text, tgt, exc)
                norm = f"{y:04d}"
        else:
            # Unknown granularity or insufficient parts — leave unchanged
            norm = text

        meta = {
            "original": text, "normalized": norm, "granularity": gran,
            "src_calendar": src, "tgt_calendar": tgt, "status": "ok"
        }
        return norm, meta

    def forward(self, user_prompt: str) -> tuple[str, list[dict]]:
        """
        1) Ask LLM to detect/plan date conversions.
        2) Convert with jdatetime.
        3) Replace the detected spans inline (first occurrence per span).
        """
        if not user_prompt:
            return user_prompt, []

        # Keep a digits-normalized copy to help later stages, but we always replace on the original text
        planned = self.detect(
            original_prompt=user_prompt,
            database_calendar=self.database_calendar
        )

        try:
            items = json.loads(planned.items_json or "[]")
            if not isinstance(items, list):
                items = []
        except Exception:
            items = []

        normalized_prompt = user_prompt
        metadata: list[dict] = []

        # Replace one-by-one to avoid shifting indices; escape literal text
        for item in items:
            norm_str, meta = self._convert_one(item)
            metadata.append(meta)

            original_span = item.get("text", "")
            if not original_span:
                continue

            # Only replace the first occurrence to stay aligned with the user's intent
            try:
                pattern = re.escape(original_span)
                normalized_prompt, n = re.subn(pattern, norm_str, normalized_prompt, count=1)
                if n == 0:
                    # Try again with digit-normalized variant (covers Eastern digits)
                    alt = normalize_eastern_digits(original_span)
                    if alt != original_span:
                        pattern = re.escape(alt)
                        normalized_prompt = re.sub(pattern, norm_str, normalized_prompt, count=1)
            except re.error:
                # If regex fails, skip replacement
                pass

        # Also normalize number digits globally to reduce translation errors (optional, mild)
        normalized_prompt = normalize_eastern_digits(normalized_prompt)

        # Log a compact summary
        if metadata:
            logger.debug("🗓️ DateNormalizer applied %d conversion(s): %s",
                         len(metadata),
                         "; ".join([f"{m['original']}→{m['normalized']}" for m in metadata]))
        else:
            logger.debug("🗓️ DateNormalizer found no convertible dates")

        return normalized_prompt, metadata


class TranslatePrompt(Module):
    """Translate to English (or passthrough) and detect the language."""

    def __init__(self):
        super().__init__()
        self.pred = Predict(TranslatePromptSig)

    def forward(self, user_prompt: str) -> tuple[str, str]:
        out = self.pred(user_prompt=user_prompt)
        logger.debug(
            "🌐 TranslatePrompt: %s→%s chars", len(user_prompt), len(out.english_prompt)
        )
        logger.debug("📜 Translated prompt:\n%s\n", out.english_prompt)
        logger.debug("🗣 Detected language: %s", out.language)
        return out.english_prompt, out.language


class SqlPromptCleaner(Module):
    """Produce a crisp SQL-ready instruction string."""

    def __init__(self):
        super().__init__()
        self.pred = Predict(SqlReadyPromptSig)

    def forward(self, english_prompt: str) -> str:
        out = self.pred(user_prompt=english_prompt)
        logger.debug(
            "🧹 SqlPromptCleaner: %s→%s chars",
            len(english_prompt),
            len(out.sql_ready_prompt),
        )
        logger.debug("⚙️ SQL-ready prompt:\n%s\n", out.sql_ready_prompt)
        return out.sql_ready_prompt


class AmbiguityResolver(Module):
    """
    Stage – detect material ambiguities and, if present, interactively
    ask the user for clarifications, translating questions into
    the user's language.
    """

    def __init__(self):
        super().__init__()
        self.detect = Predict(DetectAmbiguitySig)
        self.translator = Predict(TranslateQuestionContextSig)

    def forward(
        self, sql_ready_prompt: str, original_prompt: str, user_language: str
    ) -> Dict[str, List[str]]:
        # Let the LLM spot ambiguities
        out = self.detect(sql_ready_prompt=sql_ready_prompt)

        if not out.has_ambiguity:
            logger.debug("🔎 AmbiguityResolver: no significant ambiguities detected")
            return {"questions": [], "answers": []}

        questions, answers = [], []
        logger.debug(
            "🤔 AmbiguityResolver detected %s ambiguity(ies)", len(out.ambiguities)
        )

        for label, question_en in out.ambiguities.items():
            # Translate question to user's language with context preservation
            trans_out = self.translator(
                original_prompt=original_prompt,
                user_language=user_language,
                question_en=question_en,
            )
            question_user_lang = trans_out.question_translated

            # Ask the user in their language
            user_ans = ask_user(question_user_lang)

            questions.append(question_user_lang)
            answers.append(user_ans)
            logger.debug("🙋 User answered [%s]: %s → %s", label, question_user_lang, user_ans)

        return {"questions": questions, "answers": answers}
    

class PromptClarifier(Module):
    """Generate the final clarified SQL prompt."""

    def __init__(self):
        super().__init__()
        self.pred = Predict(ClarifyPromptSig)

    def forward(
        self,
        base_prompt: str,
        questions: List[str],
        answers: List[str],
        user_language: str
    ) -> str:
        out = self.pred(
            sql_ready_prompt=base_prompt,
            clarification_qs=questions,
            clarification_as=answers,
            clarifications_language=user_language
        )
        logger.debug("✅ PromptClarifier produced %s chars", len(out.clarified_prompt))
        logger.debug("📝 Clarified prompt:\n%s\n", out.clarified_prompt)
        return out.clarified_prompt


class KeywordExtractor(Module):
    """Extract field-like keywords (ignore literal values)."""

    def __init__(self, max_keywords: int = 4):
        super().__init__()
        self.max_keywords = max_keywords
        self.think = ChainOfThought(ExtractKeywordsSig)

    def forward(self, sql_ready_prompt: str) -> List[str]:
        out = self.think(sql_prompt=sql_ready_prompt, max_keywords=self.max_keywords)
        logger.debug(
            "🔑 KeywordExtractor: %s keyword(s) → \n%s\n\n",
            len(out.keywords),
            ", ".join(out.keywords),
        )
        return out.keywords


class MatchSchemas(Module):
    """Map each keyword to its most relevant database schemas."""

    def __init__(self, engine: sa.Engine, max_schema_per_kw: int = 2):
        super().__init__()
        self.engine = engine
        self.max_schema_per_kw = max_schema_per_kw
        self.think = ChainOfThought(KeywordSchemaSig)

    def forward(self, keywords: List[str]) -> Dict[str, List[str]]:
        db_schemas = list_schemas(self.engine)
        logger.debug(
            "📥 MatchSchemas: %s keyword(s) %s | %s schema(s) in DB",
            len(keywords),
            keywords,
            len(db_schemas),
        )

        result: Dict[str, List[str]] = defaultdict(list)

        for kw in keywords:
            out = self.think(
                keyword=kw,
                db_schemas=db_schemas,
                max_chosen_schemas=self.max_schema_per_kw,
            )

            # parse the LLM output safely
            schemas = (
                json.loads(out.related_schemas)
                if isinstance(out.related_schemas, str)
                else out.related_schemas
            )

            # keep only valid schemas and at most `max_schema_per_kw`
            chosen = [s for s in schemas if s in db_schemas][: self.max_schema_per_kw]
            result[kw].extend(chosen)

            logger.debug("   ↳ %s → %s", kw, chosen or "∅")

        logger.debug(
            "📤 MatchSchemas result: %s keyword(s) mapped, %s total schema refs",
            len(result),
            sum(len(v) for v in result.values()),
        )

        return result


class MatchTables(Module):
    """Choose relevant tables for every (keyword, schema) pair that survived the previous stage."""

    def __init__(self, engine: sa.Engine, max_tbl_per_kw_schema: int = 4):
        super().__init__()
        self.max_tbl_per_kw_schema = max_tbl_per_kw_schema
        self.engine = engine
        self.think = ChainOfThought(KeywordTableSig)

    def forward(self, schema_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
        logger.debug(
            "📥 MatchTables: %s keyword(s) → %s schema refs",
            len(schema_map),
            sum(len(v) for v in schema_map.values()),
        )

        insp = inspect(self.engine)
        result: Dict[str, List[str]] = defaultdict(list)

        for kw, schemata in schema_map.items():
            for schema in schemata:
                # candidate table list
                all_tbls = insp.get_table_names(schema=schema)
                logger.debug(
                    "🔍 %s | %s: %s table candidates", kw, schema, len(all_tbls)
                )

                # ask the LLM to rank tables
                tbl_json = json.dumps(all_tbls)
                out = self.think(
                    keyword=kw,
                    schema_table_names=tbl_json,
                    max_chosen_tables=self.max_tbl_per_kw_schema,
                )

                tbls = (
                    out.related_tables
                    if not isinstance(out.related_tables, str)
                    else json.loads(out.related_tables)
                )

                # keep unique order-preserving
                fresh = [t for t in tbls if t not in result[schema]]
                if fresh:
                    result[schema].extend(fresh)
                    logger.debug("   ↳ %s → %s.%s", kw, schema, fresh)

        total_tbls = sum(len(v) for v in result.values())
        logger.debug(
            "📤 MatchTables result: %s schema(s), %s total table refs -> %s",
            len(result),
            total_tbls,
            result,
        )
        return result


class ColumnSelector(Module):
    """
    Choose relevant columns (with data-types) for every table that the previous
    stage selected.

    Behaviors:
      - pass_all=True      → skip LLM; return ALL columns for all tables.
      - pass_all_cols=True → ask LLM; for tables with ≥1 related column, return ALL columns.
      - otherwise          → ask LLM; return only the related columns.

    Input
    -----
    sql_prompt : str
        The original natural-language question or SQL request.
    table_map  : dict[str, list[str]]
        {schema → [unique table names]}

    Output
    ------
    {schema: {table: [(col, dtype), …]}}
    """

    def __init__(self, engine: sa.Engine, pass_all: bool = True, pass_all_cols: bool = True):
        super().__init__()
        self.engine = engine
        self.pass_all = pass_all
        self.pass_all_cols = pass_all_cols
        self.pred = Predict(TableColumnSig)

    def forward(
        self,
        sql_prompt: str,
        table_map: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:

        total_pairs = sum(len(tables) for tables in table_map.values())
        logger.debug(
            "📥 ColumnSelector: %s schema(s) → %s (schema,table) pairs",
            len(table_map),
            total_pairs,
        )

        out_map: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for schema, tables in table_map.items():
            # ensure tables are unique per schema
            for table in dict.fromkeys(tables):  # preserves order, no dups
                # 1) pull (name, dtype) pairs from the DB
                col_pairs = list_columns(self.engine, schema, table)

                # 2) pass_all overrides everything
                if self.pass_all:
                    for col_tuple in col_pairs:
                        if col_tuple not in out_map[schema][table]:
                            out_map[schema][table].append(col_tuple)
                    logger.debug(
                        "   ↳ pass_all=True: returning ALL columns for %s.%s (%d cols)",
                        schema, table, len(col_pairs),
                    )
                    continue

                # 3) Otherwise, consult the LLM
                o = self.pred(
                    sql_prompt=sql_prompt,
                    schema_name=schema,
                    table_columns_info=col_pairs,
                )
                col_names = (
                    json.loads(o.related_columns)
                    if isinstance(o.related_columns, str)
                    else o.related_columns
                ) or []

                # 4) pass_all_cols logic: if there is ANY related column, include ALL columns
                if self.pass_all_cols and len(col_names) > 0:
                    for col_tuple in col_pairs:
                        if col_tuple not in out_map[schema][table]:
                            out_map[schema][table].append(col_tuple)
                    logger.debug(
                        "   ↳ pass_all_cols=True and LLM found %d related in %s.%s → returning ALL (%d cols)",
                        len(col_names), schema, table, len(col_pairs),
                    )
                    continue

                # 5) Default: only include the columns the LLM selected
                dtype_map = {name: dtype for name, dtype in col_pairs}
                for name in col_names:
                    col_tuple = (name, dtype_map.get(name, "UNKNOWN"))
                    if col_tuple not in out_map[schema][table]:
                        out_map[schema][table].append(col_tuple)
                        logger.debug(
                            "   ↳ %-18s → %s.%s.%s",
                            sql_prompt[:15] + ("…" if len(sql_prompt) > 15 else ""),
                            schema, table, col_tuple,
                        )

        if out_map:
            logger.debug("📤 ColumnSelector result:")
            for schema, tables in out_map.items():
                logger.debug("   %s", schema)
                for tbl, cols in tables.items():
                    logger.debug("      %s: %s", tbl, cols)
        else:
            logger.debug("📤 ColumnSelector result: ∅ (no columns selected)")

        return out_map


class GenerateSQL(Module):
    """Create first-draft SQL using only column and relation context."""

    def __init__(self):
        super().__init__()
        self.think = ChainOfThought(GenSqlSig)

    def forward(
        self,
        sql_prompt: str,
        table_columns: Dict[str, Dict[str, List[Tuple[str, str]]]],
        relations: Dict[str, list],
    ) -> str:
        ctx = {
            "columns": table_columns,
            "relations": relations,
        }
        sql = self.think(sql_prompt=sql_prompt, context=ctx).generated_sql
        logger.info("📝 GenerateSQL produced %s chars", len(sql))
        return sql


class ValidateAndRepairSQL(Module):
    """Run, evaluate, and iteratively repair SQL until VALID."""

    def __init__(self, engine: sa.Engine, max_attempts: int = 5):
        super().__init__()
        self.engine = engine
        self.max_attempts = max_attempts
        self.evaluate = ChainOfThought(EvaluateSig)
        self.refine = ChainOfThought(RefineSqlSig)

    def forward(
        self,
        user_prompt: str,
        sql_prompt: str,
        sql: str,
        table_columns: Dict[str, Dict[str, List[Tuple[str, str]]]],
        relations: Dict[str, list],
    ) -> Tuple[pd.DataFrame, str]:
        ctx = {
            "columns": table_columns,
            "relations": relations,
        }

        for attempt in range(1, self.max_attempts + 1):
            logger.info("🔄 Validate attempt %s/%s", attempt, self.max_attempts)

            sql_clean = extract_sql(sql)
            if not is_sql_valid(sql_clean):
                verdict, cause = "INVALID", "Query is not a SELECT statement."
            else:
                try:
                    df = execute_query(self.engine, sql_clean)
                except Exception as exc:
                    verdict, cause = "INVALID", f"Execution error: {exc}"
                else:
                    sample = df.head(3).to_json()
                    ev = self.evaluate(
                        user_prompt=user_prompt,
                        sql_prompt=sql_prompt,
                        dataframe_json=sample,
                    )
                    verdict = ev.verdict.upper()
                    cause = ev.cause or ""

            if verdict == "VALID":
                logger.info("✅ SQL validated")
                return df, sql_clean

            logger.info("🔧 Refining (%s)", cause)
            sql = self.refine(
                sql_prompt=sql_prompt,
                last_sql=sql_clean,
                cause=cause,
                context=ctx,
            ).improved_sql

        raise RuntimeError("Could not craft a valid SQL query.")
    

class MakeReportQuery(Module):
    """Transform working SQL into human-readable report + summary."""

    def __init__(self):
        super().__init__()
        self.pred = Predict(ReportSig)

    def forward(self, user_prompt: str, sql_prompt: str, sql: str) -> Tuple[str, str]:
        out = self.pred(
            user_prompt=user_prompt, sql_prompt=sql_prompt, generated_sql=sql
        )
        human_sql = extract_sql(out.readable_sql)
        summary = out.report
        return human_sql, summary


# ───────────────────────────  Orchestrator  ──────────────────────────────


class Text2SQLFlow(Module):
    """Full pipeline: NL → SQL → validated report."""

    def __init__(self, engine: sa.Engine, lm: dspy.LM):
        super().__init__()
        dspy.configure(lm=lm)

        self.engine = engine

        self.database_calendar = "Gregorian"

        self.max_keywords = 3
        self.max_schema_per_keyword = 1
        self.max_table_per_schema = 4
        self.max_columns_per_table = 10
        self.max_sql_tables = 12

        self.gate = QuickText2SQLGate(min_confidence_true=0.2)
        self.normalize_date = DateNormalizer(database_calendar=self.database_calendar)
        self.translate = TranslatePrompt()
        self.clean_prompt = SqlPromptCleaner()
        self.detect_ambiguity = AmbiguityResolver()
        self.clarify_prompt = PromptClarifier()
        self.extract_keywords = KeywordExtractor(max_keywords=self.max_keywords)
        self.match_schemas = MatchSchemas(self.engine, max_schema_per_kw=self.max_schema_per_keyword)
        self.match_tables = MatchTables(self.engine, max_tbl_per_kw_schema=self.max_table_per_schema)
        self.match_columns = ColumnSelector(self.engine)
        self.generate_sql_draft = GenerateSQL()
        self.validate = ValidateAndRepairSQL(self.engine)
        self.report = MakeReportQuery()

    def forward(self, user_prompt: str) -> Tuple[pd.DataFrame, str, str]:

        # Sanity checks
        is_sql_request, gate_confidence, gate_cause = self.gate(user_prompt)
        if not is_sql_request:
            return pd.DataFrame(), "", f"Request does not seem to be a SQL request (confidence={gate_confidence}): {gate_cause}"
        
        # Prompt reforming
        user_prompt_norm, date_meta = self.normalize_date(user_prompt)
        input()
        english_prompt, original_prompt_language = self.translate(user_prompt)
        sql_ready = self.clean_prompt(english_prompt)
        ambiguity = self.detect_ambiguity(sql_ready, user_prompt, original_prompt_language)
        if len(ambiguity["questions"]) != 0:
            sql_ready = self.clarify_prompt(
                sql_ready,
                ambiguity["questions"],
                ambiguity["answers"],
                original_prompt_language
            )
        keywords = self.extract_keywords(sql_ready)

        # Database inspection
        schema_map = self.match_schemas(keywords)
        table_map = self.match_tables(schema_map=schema_map)
        column_map = self.match_columns(sql_prompt=sql_ready, table_map=table_map)

        survived_tables: List[Tuple[str, str]] = [
            (schema, table)
            for schema, tables in column_map.items()
            for table in tables.keys()
        ]

        relations = get_pk_fk_pairs(engine=self.engine, tables=survived_tables)

        # SQL generation
        sql_draft = self.generate_sql_draft(
            sql_prompt=sql_ready, table_columns=column_map, relations=relations
        )

        df, working_sql = self.validate(
            user_prompt=user_prompt, sql_prompt=sql_ready, sql=sql_draft, table_columns=column_map, relations=relations
        )

        # Report preparation
        readable_sql, summary = self.report(
            user_prompt=user_prompt, sql_prompt=sql_ready, sql=working_sql
        )

        return df, readable_sql, summary


# ───────────────────  Convenience LM creator ─────────────────── #
def create_dspy_lm(
    model: str = "openai/gpt-4o-mini",
    api_key: str | None = None,
    api_base: str = "https://api.avalapis.ir/v1",
):

    api_key = api_key or os.getenv("AvalAI_API_KEY")
    urlparse(api_base)

    logger.debug("🌟 Initialising dspy.LM: model=%s  api_base=%s", model, api_base)
    lm = dspy.LM(
        model=model, 
        api_key=api_key, 
        api_base=api_base,
        temperature=0.3,
        #max_tokens=8000,
        cache=False,
        #cache_in_memory=False,
        #num_retries=5,
    )

    return lm


# ────────────────────────────────  CLI  ───────────────────────────────────── #
if __name__ == "__main__":

    engine_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=AdventureWorks2022;"
        "Trusted_Connection=yes;"
    )
    logger.debug("🔧 SQLAlchemy engine URL: %s", engine_url)
    engine = create_engine(engine_url)

    # ollama_chat/              ->  http://199.168.172.141:11434/v1 (api_key='none')
    # gemma3:27b                ->  192.168.172.141:11434/v1
    # openai/gemma-3-27b-it     ->  https://api.avalapis.ir/v1, https://api.avalai.ir/v1
    lm = create_dspy_lm(
        model="openai/gpt-4o-mini",
        api_base="https://api.avalai.ir/v1",
    )

    flow = Text2SQLFlow(engine=engine, lm=lm)

    while True:
        try:
            prompt = input("\nAsk> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        try:
            df, readable_sql, summary = flow(prompt)
            print("\n— Final report SQL —")
            print(readable_sql)
            print("\n— Preview —")
            print(df.head())
            print("\n— Summary —")
            print(summary)
        except Exception as exc:
            logger.exception("💥 Failed to satisfy prompt: %s", exc)
