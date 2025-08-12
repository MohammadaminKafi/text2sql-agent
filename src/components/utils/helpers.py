import logging
import re
import sqlparse
import sqlalchemy as sa
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def list_schemas(engine: sa.Engine) -> List[str]:
    logger.debug("🔍 list_schemas: inspecting database for schema names")
    inspector = sa.inspect(engine)
    schemas = inspector.get_schema_names()
    logger.debug("🔍 list_schemas: found %s", schemas)
    return schemas


def list_tables(engine: sa.Engine, schema: str) -> List[str]:
    logger.debug("🔍 list_tables: schema=%s", schema)
    inspector = sa.inspect(engine)
    tables = inspector.get_table_names(schema=schema)
    logger.debug("🔍 list_tables: %s → %s", schema, tables)
    return tables


def list_columns(engine, schema: str, table: str) -> List[Tuple[str, str]]:
    logger.debug("🔍 list_columns: %s.%s", schema, table)

    inspector = sa.inspect(engine)
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

    inspector = sa.inspect(engine)
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
