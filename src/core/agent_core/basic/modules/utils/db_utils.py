"""
Database utilities for agent core modules.

This module provides unified database operation functions using the new
database connector system.
"""
import logging
import re
import sqlparse
import sqlalchemy as sa
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

from core.smartlog import get_logger
from core.database import DatabaseConnector, get_default_connector, create_connector

logger = get_logger(__name__)

def list_schemas(engine_or_connector: Union[sa.Engine, DatabaseConnector]) -> List[str]:
    """
    List database schemas.
    
    Args:
        engine_or_connector: SQLAlchemy engine or DatabaseConnector instance
        
    Returns:
        List of schema names
    """
    if isinstance(engine_or_connector, DatabaseConnector):
        connector = engine_or_connector
        logger.flowdebug("🔍 list_schemas: using unified connector")
        try:
            schemas = connector.list_schemas()
            logger.flowdebug("🔍 list_schemas: found %s", schemas)
            return schemas
        except Exception as e:
            logger.warning(f"Could not get schema names via connector: {e}")
            return ["dbo"]  # Safe fallback
    
    # Legacy engine support
    engine = engine_or_connector
    logger.flowdebug("🔍 list_schemas: inspecting database for schema names (legacy)")
    inspector = sa.inspect(engine)
    try:
        schemas = inspector.get_schema_names()
        logger.flowdebug("🔍 list_schemas: found %s", schemas)
        return schemas
    except Exception as e:
        logger.warning(f"Could not get schema names: {e}")
        # Fallback: try to get current database name and use common schemas
        try:
            with engine.connect() as conn:
                result = conn.execute(sa.text("SELECT DB_NAME() as current_db"))
                current_db = result.scalar()
                if current_db and current_db.lower() == "adventureworks2022":
                    # Return common AdventureWorks schemas
                    logger.info("Using AdventureWorks2022 schema fallback")
                    return ["dbo", "HumanResources", "Person", "Production", "Purchasing", "Sales"]
                elif current_db and current_db.lower() == "master":
                    logger.warning("Connected to master database - limited schema access")
                    return ["dbo", "sys"]
                else:
                    logger.info(f"Using default schema for database: {current_db}")
                    return ["dbo"]  # Default schema
        except Exception as conn_e:
            logger.error(f"Could not determine current database: {conn_e}")
            return ["dbo"]  # Safe fallback


def list_tables(engine_or_connector: Union[sa.Engine, DatabaseConnector], schema: str) -> List[str]:
    """
    List tables in a schema.
    
    Args:
        engine_or_connector: SQLAlchemy engine or DatabaseConnector instance
        schema: Schema name
        
    Returns:
        List of table names
    """
    if isinstance(engine_or_connector, DatabaseConnector):
        connector = engine_or_connector
        logger.flowdebug("🔍 list_tables: schema=%s (using unified connector)", schema)
        try:
            tables = connector.list_tables(schema)
            logger.flowdebug("🔍 list_tables: %s → %s", schema, tables)
            return tables
        except Exception as e:
            logger.warning(f"Could not get table names for schema {schema} via connector: {e}")
            return []
    
    # Legacy engine support
    engine = engine_or_connector
    logger.flowdebug("🔍 list_tables: schema=%s (legacy)", schema)
    inspector = sa.inspect(engine)
    try:
        tables = inspector.get_table_names(schema=schema)
        logger.flowdebug("🔍 list_tables: %s → %s", schema, tables)
        return tables
    except Exception as e:
        logger.warning(f"Could not get table names for schema {schema}: {e}")
        # If we're in master database or having issues, return empty list
        try:
            with engine.connect() as conn:
                result = conn.execute(sa.text("SELECT DB_NAME() as current_db"))
                current_db = result.scalar()
                if current_db and current_db.lower() == "master":
                    logger.warning("Cannot list tables from master database")
                    return []
                else:
                    # Try a direct query approach
                    result = conn.execute(sa.text("""
                        SELECT TABLE_NAME 
                        FROM INFORMATION_SCHEMA.TABLES 
                        WHERE TABLE_SCHEMA = :schema_name AND TABLE_TYPE = 'BASE TABLE'
                    """), {"schema_name": schema})
                    tables = [row[0] for row in result]
                    logger.flowdebug("🔍 list_tables (fallback): %s → %s", schema, tables)
                    return tables
        except Exception as fallback_e:
            logger.error(f"Fallback table listing also failed: {fallback_e}")
            return []


def list_columns(engine_or_connector: Union[sa.Engine, DatabaseConnector], 
                schema: str, table: str) -> List[Tuple[str, str]]:
    """
    List columns in a table.
    
    Args:
        engine_or_connector: SQLAlchemy engine or DatabaseConnector instance
        schema: Schema name
        table: Table name
        
    Returns:
        List of (column_name, column_type) tuples
    """
    if isinstance(engine_or_connector, DatabaseConnector):
        connector = engine_or_connector
        logger.flowdebug("🔍 list_columns: %s.%s (using unified connector)", schema, table)
        try:
            columns = connector.list_columns(schema, table)
            # Convert to expected format
            cols = [(col['name'], col['type']) for col in columns]
            logger.flowdebug("🔍 list_columns: %s.%s → %s", schema, table, cols)
            return cols
        except Exception as e:
            logger.warning(f"Could not get columns for {schema}.{table} via connector: {e}")
            return []
    
    # Legacy engine support
    engine = engine_or_connector
    logger.flowdebug("🔍 list_columns: %s.%s (legacy)", schema, table)

    inspector = sa.inspect(engine)
    cols = [
        (col["name"], str(col["type"]))
        for col in inspector.get_columns(table, schema=schema)
    ]

    logger.flowdebug("🔍 list_columns: %s.%s → %s", schema, table, cols)
    return cols


def get_pk_fk_pairs(
    engine_or_connector: Union[sa.Engine, DatabaseConnector],
    tables: List[Tuple[str, str]],
) -> List[Tuple[str, str, str]]:
    """
    Return (child_table, parent_table, constraint_name) triples for every
    FK that originates in tables.
    
    Args:
        engine_or_connector: SQLAlchemy engine or DatabaseConnector instance
        tables: List of (schema, table) tuples
        
    Returns:
        List of (child_table_fqn, parent_table_fqn, constraint_name) tuples
    """
    logger.flowdebug("🔍 get_pk_fk_pairs: %s tables provided", len(tables))

    if isinstance(engine_or_connector, DatabaseConnector):
        connector = engine_or_connector
        logger.flowdebug("🔍 get_pk_fk_pairs: using unified connector")
        try:
            # Use the new connector's method if available
            relations = []
            for schema, table in tables:
                fk_info = connector.get_foreign_keys(schema, table)
                for fk in fk_info:
                    src_fq = f"{schema}.{table}"
                    tgt_fq = f"{fk['referenced_schema']}.{fk['referenced_table']}"
                    cname = fk.get('constraint_name', '')
                    
                    # Skip self-references
                    if src_fq.lower() == tgt_fq.lower():
                        continue
                    
                    relations.append((src_fq, tgt_fq, cname))
                    logger.flowdebug("      🔗 %s → %s  (constraint=%s)", src_fq, tgt_fq, cname)
            
            # De-duplicate
            dedup = list(
                {(a.lower(), b.lower(), c): (a, b, c) for a, b, c in relations}.values()
            )
            
            logger.flowdebug("🔍 get_pk_fk_pairs: discovered %s relation(s)", len(dedup))
            return dedup
            
        except Exception as e:
            logger.warning(f"Could not get FK relationships via connector: {e}")
            # Fallback to legacy approach with the connector's engine
            engine = connector.get_engine()
        
    else:
        engine = engine_or_connector

    # Legacy engine approach
    inspector = sa.inspect(engine)
    relations: list[tuple[str, str, str]] = []

    for schema, table in tables:
        fk_list = inspector.get_foreign_keys(table, schema=schema)
        logger.flowdebug("   🗄️  %s.%s — %s FKs found", schema, table, len(fk_list))

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
            logger.flowdebug("      🔗 %s → %s  (constraint=%s)", src_fq, tgt_fq, cname)

    # ── de-duplicate (same edge can appear twice) ─────────────────────
    dedup = list(
        {(a.lower(), b.lower(), c): (a, b, c) for a, b, c in relations}.values()
    )

    logger.flowdebug("🔍 get_pk_fk_pairs: discovered %s relation(s)", len(dedup))
    return dedup


def extract_sql(llm_response: str) -> str:
    """Extract SQL from LLM response using various patterns."""
    logger.flowdebug("🧹 extract_sql: received %s chars", len(llm_response))

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
            logger.flowdebug("🧹 extract_sql: matched %s block (%s chars)", label, len(sql))
            return sql

    logger.flowdebug("🧹 extract_sql: no pattern matched; returning full text")
    return llm_response.strip()


def is_sql_valid(sql: str) -> bool:
    """Check if SQL contains valid SELECT statement."""
    logger.flowdebug("🔎 is_sql_valid: validating SQL (%s chars)", len(sql))
    for stmt in sqlparse.parse(sql):
        if stmt.get_type().upper() == "SELECT":
            logger.flowdebug("🔎 is_sql_valid: found SELECT → valid")
            return True
    logger.flowdebug("🔎 is_sql_valid: no SELECT found → invalid")
    return False


def execute_query(engine_or_connector: Union[sa.Engine, DatabaseConnector], 
                 query: str) -> pd.DataFrame:
    """
    Execute a SQL query and return results as DataFrame.
    
    Args:
        engine_or_connector: SQLAlchemy engine or DatabaseConnector instance
        query: SQL query to execute
        
    Returns:
        DataFrame with query results
    """
    if isinstance(engine_or_connector, DatabaseConnector):
        connector = engine_or_connector
        logger.flowdebug("🪄 execute_query: %s (using unified connector)", query.replace("\n", " "))
        result = connector.execute_query(query)
        logger.flowdebug("🪄 execute_query: returned %s rows × %s cols", result.shape[0], result.shape[1])
        return result
    
    # Legacy engine support
    engine = engine_or_connector
    logger.flowdebug("🪄 execute_query: %s (legacy)", query.replace("\n", " "))
    with engine.connect() as conn:
        result = conn.execute(sa.text(query))  # Use sa.text() for raw SQL
        rows = result.fetchall()
        columns = result.keys()
    df = pd.DataFrame(rows, columns=columns)
    logger.flowdebug("🪄 execute_query: returned %s rows × %s cols", df.shape[0], df.shape[1])
    return df


# Convenience functions for the new unified system
def get_db_connector(profile_name: Optional[str] = None) -> DatabaseConnector:
    """
    Get a database connector.
    
    Args:
        profile_name: Optional profile name. If None, uses default.
        
    Returns:
        DatabaseConnector instance
    """
    if profile_name:
        return create_connector(profile_name)
    else:
        return get_default_connector()


def execute_sql_with_profile(query: str, profile_name: Optional[str] = None) -> pd.DataFrame:
    """
    Execute SQL using a specific database profile.
    
    Args:
        query: SQL query to execute
        profile_name: Optional profile name. If None, uses default.
        
    Returns:
        DataFrame with query results
    """
    connector = get_db_connector(profile_name)
    return execute_query(connector, query)



