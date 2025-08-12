from dspy import Signature, InputField, OutputField
from typing import Any, Dict, List, Tuple


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
