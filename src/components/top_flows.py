import sqlalchemy as sa
import pandas as pd
from typing import Dict, List, Tuple

from dspy import Module
import dspy

from components.dspy_modules import (
    QuickText2SQLGate,
    ConvertDates,
    NormalizeDatesTranslate,
    SqlPromptCleaner,
    AmbiguityResolver,
    PromptClarifier,
    KeywordExtractor,
    MatchSchemas,
    MatchTables,
    ColumnSelector,
    GenerateSQL,
    ValidateAndRepairSQL,
    MakeReportQuery
)
from components.utils.helpers import (
    get_pk_fk_pairs
)

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
        self.convert_dates = ConvertDates(target_calendar=self.database_calendar)
        self.translate = NormalizeDatesTranslate()
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
        dates_list = self.convert_dates(user_prompt)
        english_prompt, original_prompt_language = self.translate(user_prompt, dates_list)
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