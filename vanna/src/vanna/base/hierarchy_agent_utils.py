import json
import logging
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import inspect, create_engine

import dspy
from dspy import InputField, OutputField, Signature, Predict, Module

# ---------- Utility helpers --------------------------------------------------

def list_schemas(engine: sa.Engine) -> List[str]:
    """Return list of schema names for the connected DB."""
    inspector = inspect(engine)
    return inspector.get_schema_names()


def list_tables(engine: sa.Engine, schema: str) -> List[str]:
    """Return list of tables for a given schema."""
    inspector = inspect(engine)
    return inspector.get_table_names(schema=schema)


def get_pk_fk_pairs(engine: sa.Engine, schema: str) -> List[Tuple[str, str, str]]:
    """Return (fk_table, pk_table, constraint_name) tuples in the schema."""
    inspector = inspect(engine)
    rels = []
    for table_name in inspector.get_table_names(schema=schema):
        fkeys = inspector.get_foreign_keys(table_name, schema=schema)
        for fk in fkeys:
            rels.append((table_name, fk["referred_table"], fk.get("name", "")))
    return rels


def execute_query(engine: sa.Engine, query: str) -> pd.DataFrame:
    """Safely execute a query and return a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


# ---------- DSPy Signatures ---------------------------------------------------

class ExtractKeywordSig(Signature):
    """Translate the prompt to English and extract main keywords."""
    prompt: str = InputField(
            desc="Original user prompt in any language"
    )
    english_prompt: str = OutputField(
        desc="Prompt translated to English, preserving meaning"
    )
    keywords: List[str] = OutputField(
        desc="List of keyword strings that capture entities, measures, dates, etc."
    )

class KeywordSchemaMatchSig(Signature):
    """Match each keyword to the most relevant database schemas."""
    keywords: List[str] = InputField(
        desc="List of extracted keyword strings"
    )
    schemas: List[str] = InputField(
        desc="List of all schema names available in the database"
    )
    schema_map: Dict[str, List[str]] = OutputField(
        desc="Dictionary mapping each keyword to a list of related schemas"
    )

class KeywordTableSig(Signature):
    """Pick the most relevant tables in a given schema for a single keyword."""
    keyword: str = InputField(
        desc="Single keyword we want to satisfy (e.g., 'orders', 'customer', 'date')"
    )
    schema: str = InputField(
        desc="Schema currently being inspected"
    )
    related_tables: List[str] = OutputField(
        desc="Up to three table names most likely to contain information for the keyword"
    )

class GenSqlSig(Signature):
    """Write a runnable SQL query that answers the question using provided context."""
    prompt: str = InputField()
    schema_context: str = InputField(
        desc="Selected schemas, tables, and relationships (JSON string)"
    )
    sql: str = OutputField(
        desc="Runnable SQL query (no code fences)"
    )

class EvaluateSig(Signature):
    """Check if result dataframe answers the prompt. Return VALID or INVALID + reason."""
    prompt: str = InputField()
    dataframe_json: str = InputField(
        desc="df.head(…​).to_json(), limited to ~5 rows"
    )
    verdict: str = OutputField(
        desc="'VALID' (fits prompt) or 'INVALID' plus short reason"
    )

class RefineSqlSig(Signature):
    """Given invalid verdict and reason, refine the SQL query."""
    prompt: str = InputField()
    last_sql: str = InputField()
    reason: str = InputField()
    improved_sql: str = OutputField(
        desc="Corrected SQL query"
    )

class ReportSig(Signature):
    """Rewrite a working SQL query into a user-friendly reporting query."""
    prompt: str = InputField()
    sql: str = InputField()
    readable_sql: str = OutputField(
        desc="Report-style SQL with friendly column names, joins for IDs, ordering, ranking"
    )


# ---------- DSPy Modules ------------------------------------------------------

class ExtractKeywords(Module):
    """translate + keyword extraction"""

    def __init__(self):
        super().__init__()
        self.pred = Predict(ExtractKeywordSig)

    def forward(self, prompt: str):
        return self.pred(prompt=prompt)  # returns .english_prompt, .keywords


class MatchSchemas(Module):
    """keyword → schemas mapping"""

    def __init__(self, engine: sa.Engine):
        super().__init__()
        self.engine = engine
        self.pred = Predict(KeywordSchemaMatchSig)

    def forward(self, keywords: List[str]) -> Dict[str, List[str]]:
        schemas = list_schemas(self.engine)
        mapping_json = self.pred(
            keywords=keywords,
            schemas=schemas  # DSPy auto-serialises list → JSON
        ).schema_map
        # Ensure data types are native Python:
        if isinstance(mapping_json, str):
            schema_map = json.loads(mapping_json)
        else:
            schema_map = mapping_json
        # Normalise – only keep valid schemas that truly exist:
        valid = set(schemas)
        return {kw: [s for s in schs if s in valid]
                for kw, schs in schema_map.items()}


class MatchTables(Module):
    """for each (keyword, schema) select candidate tables"""

    def __init__(self, engine: sa.Engine):
        super().__init__()
        self.engine = engine
        self.pred = Predict(KeywordTableSig)

    def forward(self, schema_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Returns schema_tables: {schema: sorted_unique_tables_list}
        """
        result: Dict[str, List[str]] = defaultdict(list)
        insp = inspect(self.engine)

        for kw, schemata in schema_map.items():
            for schema in schemata:
                # Build a mini prompt: we cram table list into the 'schema'
                # field so the model knows what's available.
                tables = insp.get_table_names(schema=schema)
                schema_prompt = json.dumps(
                    {"schema": schema, "tables": tables})
                out = self.pred(keyword=kw, schema=schema_prompt)
                # Ensure correct type:
                rel = out.related_tables
                rel = json.loads(rel) if isinstance(rel, str) else rel
                # Deduplicate while keeping order:
                for tbl in rel:
                    if tbl not in result[schema]:
                        result[schema].append(tbl)

        return result


class GenerateSQL(Module):
    """first SQL draft"""

    def __init__(self):
        super().__init__()
        self.pred = Predict(GenSqlSig)

    def forward(self,
                prompt: str,
                schema_tables: Dict[str, List[str]],
                relations: Dict[str, list]):
        ctx = json.dumps({"schemas": schema_tables,
                          "relations": relations})
        sql = self.pred(prompt=prompt, schema_context=ctx).sql
        return sql


class ValidateAndRepairSQL(Module):
    """execute, evaluate, optionally repair"""

    def __init__(self, engine: sa.Engine, max_attempts: int = 3):
        super().__init__()
        self.engine = engine
        self.max_attempts = max_attempts
        self.evaluate = Predict(EvaluateSig)
        self.refine = Predict(RefineSqlSig)

    def forward(self, prompt: str, sql: str) -> Tuple[pd.DataFrame, str]:
        for attempt in range(1, self.max_attempts + 1):
            try:
                df = execute_query(self.engine, sql)
            except Exception as exc:
                reason = f"Execution error: {exc}"
                verdict = 'INVALID'
            else:
                sample = df.head(5).to_json()
                verdict_text = self.evaluate(
                    prompt=prompt,
                    dataframe_json=sample
                ).verdict
                verdict = verdict_text.split()[0].upper()
                reason = verdict_text if verdict == 'INVALID' else ''

            if verdict == 'VALID':
                return df, sql

            logging.info("Attempt %s failed – reason: %s", attempt, reason)
            sql = self.refine(
                prompt=prompt,
                last_sql=sql,
                reason=reason
            ).improved_sql

        raise RuntimeError("Failed to produce a valid SQL query.")


class MakeReportQuery(Module):
    """prettify SQL for reporting"""

    def __init__(self):
        super().__init__()
        self.pred = Predict(ReportSig)

    def forward(self, prompt: str, sql: str) -> str:
        return self.pred(prompt=prompt, sql=sql).readable_sql


# ---------- Orchestrator ------------------------------------------------------

class Text2SQLFlow(Module):
    """
    High-level pipeline tying all modules together.
    Call as: df, report_sql = flow("Show me 2021 order counts per territory")
    """

    def __init__(self, engine: sa.Engine, lm: dspy.LM):
        super().__init__()
        dspy.configure(lm=lm)        # global config

        self.extract = ExtractKeywords()
        self.match_schemas = MatchSchemas(engine)
        self.match_tables = MatchTables(engine)
        self.gen_sql = GenerateSQL()
        self.validate = ValidateAndRepairSQL(engine)
        self.report = MakeReportQuery()

        self.engine = engine

    def forward(self, user_prompt: str):
        # 1. translate + keywords
        ek = self.extract(user_prompt)
        english_prompt, keywords = ek.english_prompt, ek.keywords

        # 2. keyword → schemas
        schema_map = self.match_schemas(keywords=keywords)

        # 3. (keyword, schema) → tables
        schema_tables = self.match_tables(schema_map=schema_map)

        # collect relations (optional, but helpful)
        relations = {s: get_pk_fk_pairs(self.engine, s)
                     for s in schema_tables}

        # 4. first SQL draft
        sql = self.gen_sql(
            prompt=english_prompt,
            schema_tables=schema_tables,
            relations=relations
        )

        # 5. execute, evaluate, repair loop
        df, working_sql = self.validate(
            prompt=english_prompt,
            sql=sql
        )

        # 6. final human-readable query
        report_sql = self.report(prompt=english_prompt, sql=working_sql)

        return df, report_sql

def create_dspy_lm(
    model: str = "openai/gpt-4o-mini",
    api_key: str | None = None,
    api_base: str = "https://api.avalapis.ir/v1",
):
    """Minimal LM factory (same as previous answer)."""
    import requests
    from urllib.parse import urlparse
    from no_commit_utils.credentials_utils import read_credentials

    api_key = api_key or read_credentials("avalai.key")
    if not api_key:
        raise ValueError("Missing API key.")

    urlparse(api_base)  # quick validation
    lm = dspy.LM(model=model, api_key=api_key, api_base=api_base)

    return lm


if __name__ == "__main__":
    engine = create_engine()
    lm = create_dspy_lm()

    flow = Text2SQLFlow(engine=engine, lm=lm)

    while True:
        prompt = input("Ask: ")
        
        df, sql = flow(prompt)

        print(sql)