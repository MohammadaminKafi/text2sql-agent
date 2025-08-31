import os
import urllib.parse
from typing import Optional
from sqlalchemy import create_engine, Engine as SAEngine
from components.smartlog import get_logger

logger = get_logger(__name__)

def connect_mssql_engine():
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=AdventureWorks2022;"
        "Trusted_Connection=yes;"
    )
    engine_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(conn_str)
    logger.sysdebug(f"Creating DB engine with URL: {engine_url}")
    return create_engine(engine_url)

def connect_snowflake_engine(
    warehouse: str,
    database: str,
    role: str = "PARTICIPANT",
) -> SAEngine:
    
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    account = os.getenv("SNOWFLAKE_ACCOUNT")

    if not all([user, password, account]):
        raise EnvironmentError(
            "Missing one or more Snowflake env vars: "
            "SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT"
        )

    # URL-encode credentials safely
    user_q = urllib.parse.quote_plus(user)
    pass_q = urllib.parse.quote_plus(password)
    warehouse_q = urllib.parse.quote_plus(warehouse)
    role_q = urllib.parse.quote_plus(role)

    # Example URL: snowflake://user:pass@account/database?warehouse=WH&role=ROLE
    engine_url = (
        f"snowflake://{user_q}:{pass_q}@{account}/"
        f"{database}?warehouse={warehouse_q}&role={role_q}"
    )

    logger.sysdebug(
        f"Creating Snowflake DB engine "
        f"(account={account}, database={database}, warehouse={warehouse}, role={role})"
    )
    return create_engine(engine_url)

def create_db_engine(
    dbms: str = "mssql",
    warehouse: Optional[str] = None,
    database: Optional[str] = None,
) -> SAEngine:
    if dbms == "mssql":
        return connect_mssql_engine()

    if dbms == "snowflake":
        if not warehouse:
            warehouse = "COMPUTE_WH_PARTICIPANT"
        if not database:
            raise ValueError("For Snowflake, please provide 'database' (e.g., database='YOUR_DB').")
        return connect_snowflake_engine(warehouse=warehouse, database=database)

    raise ValueError(f"Unsupported DBMS: {dbms}")