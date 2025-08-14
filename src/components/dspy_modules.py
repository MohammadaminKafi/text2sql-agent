import logging
import re
import json
import sqlalchemy as sa
import pandas as pd
import matplotlib as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

from dspy import Module, Predict, ChainOfThought

from components.dspy_signatures import (
    QuickGateSig,
    ExtractDatesSig,
    NormalizeDatesTranslateSig,
    SqlReadyPromptSig,
    DetectAmbiguitySig,
    TranslateQuestionContextSig,
    ClarifyPromptSig,
    ExtractKeywordsSig,
    KeywordSchemaSig,
    KeywordTableSig,
    TableColumnSig,
    GenSqlSig,
    EvaluateSig,
    RefineSqlSig,
    DFSummarySig,
    VizPlanSig,
    VizSpecSig,
    ReportSig
)
from components.utils.date_utils import (
    convert_calendar
)
from components.utils.helpers import (
    list_schemas,
    list_columns,
    extract_sql,
    is_sql_valid,
    execute_query,
    ask_user,
)
from components.utils.vis_utils import (
    infer_schema,
    head_as_dict,
    aggregate_dataframe,
    draw_plot
)

logger = logging.getLogger(__name__)

# ────────────────────────────  DSPy Modules  ─────────────────────────────── #


class QuickText2SQLGate(Module):
    def __init__(self, min_confidence_true: float = 0.35):
        super().__init__()
        self.pred = Predict(QuickGateSig)
        self.min_confidence_true = min_confidence_true  # keeps it loose

        # --- Heuristic patterns (fast path) ---
        # SQL/code signals
        self.re_sql_sniff = re.compile(
            r"\b(select|from|where|group\s+by|order\s+by|join|left\s+join|inner\s+join|having|limit)\b",
            re.IGNORECASE | re.UNICODE,
        )
        self.re_code_fence = re.compile(r"```.*?\n", re.DOTALL)

        # Business / analytics verbs (en + fa transliterations)
        self.re_data_verbs = re.compile(
            r"\b(show|list|fetch|get|count|sum|avg|average|rank|top|trend|compare|report|kpi|breakdown|by|per)\b"
            r"|گزارش|فیلتر|گروه(?:‌|)بندی|مقایسه|میانگین|جمع|تعداد|روند|براساس|بر اساس|به تفکیک",
            re.IGNORECASE | re.UNICODE,
        )

        # Schema/table/column nouns
        self.re_db_objects = re.compile(
            r"\b(table|tables|column|columns|field|fields|schema|database|dataset)\b"
            r"|جدول|ستون|فیلد|شِما|پایگاه\s*داده",
            re.IGNORECASE | re.UNICODE,
        )

        # Time/date hints (YYYY, months incl. Persian)
        self.re_date_hints = re.compile(
            r"\b(19\d{2}|20\d{2}|21\d{2})\b"
            r"|january|february|march|april|may|june|july|august|september|october|november|december"
            r"|فروردین|اردیبهشت|خرداد|تیر|مرداد|شهریور|مهر|آبان|آذر|دی|بهمن|اسفند",
            re.IGNORECASE | re.UNICODE,
        )

        # Clear negatives (creative writing, chit-chat, coding not about data access, news, meta-LLM)
        self.re_negative = re.compile(
            r"\b(poem|poetry|story|song|lyrics|joke|advise|advice|feel|therapy|news|headline|who\s+is|define|explain|"
            r"translate|summarize|paraphrase|regex|bug|stacktrace|compile|deploy|docker|kubernetes|frontend|backend|"
            r"prompt|llm|chatgpt|model)\b"
            r"|شعر|داستان|ترجمه|اخبار|خبر|کیست|توضیح|احساس|مشاوره",
            re.IGNORECASE | re.UNICODE,
        )

    # ---- heuristics ---------------------------------------------------------
    def _heuristic_gate(self, text: str) -> Tuple[bool, float, str, bool]:
        """
        Returns (is_sql, confidence, cause, decisive)
        decisive=True → skip LLM; False → fallback to LLM
        """
        t = (text or "").strip()
        if not t:
            return (False, 0.20, "empty prompt", True)

        # crude token length
        tokens = re.findall(r"\w+", t, flags=re.UNICODE)
        if len(tokens) < 3 and not self.re_sql_sniff.search(t):
            return (False, 0.30, "too short to be a database question", True)

        # Hard negatives
        if self.re_negative.search(t):
            # Only decisive if there are no strong SQL signals
            if not (self.re_sql_sniff.search(t) or self.re_db_objects.search(t)):
                return (False, 0.65, "matches non-database intent", True)

        # Strong positives
        strong_pos = 0
        if self.re_sql_sniff.search(t) or self.re_code_fence.search(t):
            strong_pos += 2
        if self.re_data_verbs.search(t):
            strong_pos += 1
        if self.re_db_objects.search(t):
            strong_pos += 1
        if self.re_date_hints.search(t):
            strong_pos += 1

        # Decisive positive
        if strong_pos >= 3:
            # looks like data retrieval/reporting
            conf = min(0.85, 0.55 + 0.1 * strong_pos)
            return (True, conf, "matches database/reporting patterns", True)

        # Decisive negative
        if strong_pos == 0 and len(tokens) < 6:
            return (False, 0.50, "no database cues in short prompt", True)

        # Not decisive → let the LLM decide
        return (False, 0.0, "", False)

    # ---- main ---------------------------------------------------------------
    def forward(self, user_prompt: str) -> tuple[bool, float, str]:
        logger.debug("🚪 Gate.in: %s chars", len(user_prompt or ""))

        # Heuristic fast path
        is_sql_h, conf_h, cause_h, decisive = self._heuristic_gate(user_prompt)
        if decisive:
            is_sql = is_sql_h
            conf = conf_h
            cause = cause_h
            if is_sql and conf < self.min_confidence_true:
                logger.debug("⬆️  Confidence floor applied (heuristic): %.2f → %.2f",
                             conf, self.min_confidence_true)
                conf = self.min_confidence_true
            logger.info("🚪 Gate.out (heuristic): is_text2sql=%s  conf=%.2f  cause=%s",
                        is_sql, conf, cause or "—")
            return is_sql, conf, cause

        # Fallback to LLM gate
        out = self.pred(user_prompt=user_prompt)
        is_sql = bool(getattr(out, "is_text2sql", False))
        conf_raw = float(getattr(out, "confidence", 0.0) or 0.0)
        cause = (getattr(out, "cause", "") or "").strip()

        conf = conf_raw
        if is_sql and conf < self.min_confidence_true:
            logger.debug("⬆️  Confidence floor applied: %.2f → %.2f", conf, self.min_confidence_true)
            conf = self.min_confidence_true

        logger.info("🚪 Gate.out (LLM): is_text2sql=%s  conf=%.2f (raw=%.2f)  cause=%s",
                    is_sql, conf, conf_raw, cause or "—")
        return is_sql, conf, cause
    
#--------------- Prompt reforming Stage
class ConvertDates(Module):
    """
    Apply ExtractDatesSig to the prompt, convert each date to the target calendar,
    and return a list of tuples:

        (source_calendar, target_calendar, original_date_text, converted_date)

    Notes:
      - `original_date_text` is exactly as it appeared in the user prompt.
      - If convert_calendar returns kind="invalid", that date is skipped.
      - Order matches the order of mentions in the original prompt.
    """

    def __init__(self, target_calendar: str = "gregorian"):
        super().__init__()
        self.extract = ChainOfThought(ExtractDatesSig)
        self.target_calendar = self._norm_calendar(target_calendar)

    @staticmethod
    def _norm_calendar(name: Optional[str]) -> str:
        n = (name or "").strip().lower()
        if n in {"jalali", "shamsi", "solar", "hijri-shamsi", "persian", "iranian", "fa"}:
            return "solar"
        if n in {"gregorian", "miladi", "en"}:
            return "gregorian"
        return "gregorian"

    @staticmethod
    def _norm_day(day: Optional[str]) -> str:
        s = (day or "").strip()
        if s.isdigit():
            i = int(s)
            if 1 <= i <= 31:
                return f"{i:02d}"
        return ""

    @staticmethod
    def _norm_month(month: Optional[str]) -> str:
        """
        Accepts "01".."12" or any name (already typo-corrected by ExtractDatesSig).
        If numeric and valid, return zero-padded; otherwise pass the name through.
        """
        s = (month or "").strip()
        if not s:
            return ""
        if s.isdigit():
            i = int(s)
            return f"{i:02d}" if 1 <= i <= 12 else ""
        return s  # keep name as is

    @staticmethod
    def _norm_year(year: Optional[str]) -> str:
        s = (year or "").strip()
        return s if (len(s) == 4 and s.isdigit()) else ""

    def forward(
        self,
        user_prompt: str,
        target_calendar: Optional[str] = None
    ) -> List[Tuple[str, str, str, str]]:
        if target_calendar:
            self.target_calendar = self._norm_calendar(target_calendar)

        # 1) Extract raw dates from prompt
        out = self.extract(user_prompt=user_prompt)
        dates_dict: Dict[str, Dict[str, str]] = getattr(out, "dates", {}) or {}

        results: List[Tuple[str, str, str, str]] = []

        for original_text, parts in dates_dict.items():
            title = (original_text or "").strip()
            if not title:
                continue  # skip empty titles

            source_cal = self._norm_calendar(parts.get("source_calendar"))
            day = self._norm_day(parts.get("day"))
            month = self._norm_month(parts.get("month"))
            year = self._norm_year(parts.get("year"))

            try:
                conv = convert_calendar(
                    source_calendar=source_cal,
                    target_calendar=self.target_calendar,
                    day=day,
                    month=month,
                    year=year,
                )
            except Exception as exc:
                logger.exception("Calendar conversion error for %r: %s", title, exc)
                continue

            if not isinstance(conv, dict) or (conv.get("kind") or "").lower() == "invalid":
                logger.debug("Skipping invalid conversion for %r", title)
                continue

            # Prefer 'date' string if available; fallback to 'parsed'
            converted_date = ""
            if isinstance(conv.get("date"), str):
                converted_date = conv["date"]
            elif isinstance(conv.get("parsed"), str):
                converted_date = conv["parsed"]

            results.append((source_cal, self.target_calendar, title, converted_date))

        logger.info("🧮 Detected %d date%s", len(results), "" if len(results) == 1 else "s")
        if results:
            preview_items = [
                f"[{src}→{tgt}] {orig} → {conv}"
                for (src, tgt, orig, conv) in results[:5]
            ]
            preview = " | ".join(preview_items)
            if len(results) > 5:
                preview += f" | … +{len(results) - 5} more"
            logger.info("📅 Date conversions: %s", preview)

        return results
  

class NormalizeDatesTranslate(Module):
    """
    Replace original date substrings with converted dates ONLY, then translate.

    - If the prompt is already English, just return the replaced text.
    - Otherwise, translate the replaced text to English.
    - Language detection is returned by the LLM (via NormalizeDatesTranslateSig).
    """

    def __init__(self):
        super().__init__()
        self.pred = Predict(NormalizeDatesTranslateSig)

    @staticmethod
    def _replace_with_converted(
        text: str, pairs: List[Tuple[str, str, str, str]]
    ) -> str:
        """
        For each (src_cal, tgt_cal, original_text, converted_date),
        replace ALL exact occurrences of original_text with converted_date ONLY.
        Longer originals are processed first to avoid partial overlaps.
        """
        if not pairs:
            return text

        # Sort by length of original_text desc to avoid nested/partial replacements
        pairs_sorted = sorted(pairs, key=lambda t: len(t[2] or ""), reverse=True)

        out = text
        for _, _, original, converted in pairs_sorted:
            if not original or not converted:
                continue
            pattern = re.compile(re.escape(original), flags=re.UNICODE)
            out, n = pattern.subn(converted, out)
            if n > 0:
                logger.debug("Replaced %d occurrence(s) of %r → %r", n, original, converted)
        return out

    def forward(
        self,
        user_prompt: str,
        converted_dates: Optional[List[Tuple[str, str, str, str]]] = None,
    ) -> tuple[str, str]:
        pairs = converted_dates or []

        # Do deterministic replacement locally for robustness,
        # then let the LLM detect language + (if needed) translate.
        replaced = self._replace_with_converted(user_prompt, pairs)

        out = self.pred(user_prompt=replaced, converted_dates=pairs)

        logger.debug("🌐 NormalizeDatesTranslate: %s→%s chars", len(replaced), len(out.english_prompt))
        logger.debug("📜 English prompt:\n%s\n", out.english_prompt)
        logger.debug("🗣 Original language: %s", out.language)

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

#--------------- Schema Inspection Stage
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

        insp = sa.inspect(self.engine)
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

#--------------- SQL Generation Stage
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
    

class SummarizeDataFrameHead(Module):
    """Module that calls DFHeadSummarySig and enforces the two-line constraint."""

    def __init__(self):
        super().__init__()
        self.summarizer = ChainOfThought(DFSummarySig)

    def forward(self, df_head: Dict[str, List[Any]], user_prompt: str, max_line_len: int = 300) -> str:
        """
        Args:
            df_head: {column: [sample values]} for the head of the dataframe
            user_prompt: the initial NL request
            max_line_len: soft cap per line (will trim if exceeded)

        Returns:
            Summary string.
        """
        summary = self.summarizer(df_head=df_head, user_prompt=user_prompt).summary
        logger.info("🧾 SummarizeDataFrameHead produced %d chars", len(summary))
        return summary

#--------------- Visualization Stage
class PlanVisualizations(Module):
    """LLM planner that proposes plot plans."""

    def __init__(self):
        super().__init__()
        self.plan = ChainOfThought(VizPlanSig)

    def forward(
        self,
        df_head: Dict[str, List[Any]],
        dtypes: Dict[str, str],
        n_rows: int,
        description: str,
        max_plans: int = 3,
    ) -> List[Dict[str, Any]]:
        out = self.plan(
            df_head=df_head,
            dtypes=dtypes,
            n_rows=n_rows,
            description=description,
            max_plans=max_plans,
        ).plans_json
        plans_obj = json.loads(out)
        plans = plans_obj.get("plans", [])
        logger.info("🧭 PlanVisualizations produced %d plan(s)", len(plans))
        return plans[:max_plans]


class MapToMatplotlib(Module):
    """LLM mapper that converts a plan + (agg)df schema to concrete Matplotlib fields."""

    def __init__(self):
        super().__init__()
        self.map = ChainOfThought(VizSpecSig)

    def forward(
        self,
        plan: Dict[str, Any],
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        spec_text = self.map(
            plan=plan,
            df_head=head_as_dict(df),
            dtypes=infer_schema(df),
            n_rows=len(df),
        ).spec_json
        spec = json.loads(spec_text)
        logger.info("🧩 MapToMatplotlib produced spec for %s", spec.get("plot_type"))
        return spec


class VisualizeDataFrame(Module):
    """
    End-to-end DSPy visualization module.

    Flow:
        1) Plan plots (LLM)
        2) Aggregate (Python) as needed
        3) Map to Matplotlib fields (LLM)
        4) Draw with Matplotlib (Python)
    Returns a list of matplotlib Figure objects (and optional metadata).

    Safety:
        - Does not specify colors.
        - Validates presence of referenced columns.
    """

    def __init__(self):
        super().__init__()
        self.planner = PlanVisualizations()
        self.mapper = MapToMatplotlib()

    def forward(
        self,
        df: pd.DataFrame,
        description: str,
        max_plots: int = 3,
    ) -> Dict[str, Any]:
        # 1) Plan
        plans = self.planner(
            df_head=head_as_dict(df),
            dtypes=infer_schema(df),
            n_rows=len(df),
            description=description,
            max_plans=max_plots,
        )

        figures: List[Tuple[str, plt.Figure]] = []
        artifacts: List[Dict[str, Any]] = []

        for plan in plans:
            try:
                # 2) Aggregate (if required)
                working_df = aggregate_dataframe(df, plan)

                # Validate referenced columns early
                missing_cols = []
                for k in (plan.get("groupby") or []):
                    if k not in working_df.columns and k not in df.columns:
                        missing_cols.append(k)
                for m in (plan.get("measures") or []):
                    f = m.get("field")
                    if f and f not in working_df.columns and f not in df.columns:
                        missing_cols.append(f)
                if missing_cols:
                    logger.warning("Skipping plan due to missing columns: %s", missing_cols)
                    continue

                # 3) Map to concrete Matplotlib fields
                spec = self.mapper(plan=plan, df=working_df)

                # 4) Draw
                fig, ax = draw_plot(working_df, spec)
                figures.append((plan.get("plot_id", spec.get("title", "plot")), fig))
                artifacts.append({
                    "plan": plan,
                    "spec": spec,
                    "shape": tuple(working_df.shape),
                })

            except Exception as e:
                logger.exception("Plan failed: %s", e)
                continue

        logger.info("✅ VisualizeDataFrame produced %d figure(s)", len(figures))
        return {
            "figures": [f for _, f in figures],
            "labels": [name for name, _ in figures],
            "artifacts": artifacts,  # for debugging/inspection
        }

#--------------- Reporting Stage
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

