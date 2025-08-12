import logging
import re
import json
import sqlalchemy as sa
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

from dspy import Module, Predict, ChainOfThought

from components.dspy_signatures import (
    QuickGateSig,
    DetectDatesSig,
    TranslatePromptSig,
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
    ReportSig
)
from components.utils.date_utils import (
    normalize_eastern_digits,
    jalali_to_gregorian,
    gregorian_to_jalali,
    format_iso,
    CAL_GREGORIAN,
    CAL_SOLAR
)
from components.utils.helpers import (
    list_schemas,
    list_columns,
    extract_sql,
    is_sql_valid,
    execute_query,
    ask_user,
)

logger = logging.getLogger(__name__)

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
