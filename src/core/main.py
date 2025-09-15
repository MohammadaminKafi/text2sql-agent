"""
Main database connection module - DEPRECATED

This module is deprecated. Use the new unified database system:
    from core.database import get_db_manager, create_connector

Legacy functions are preserved for backwards compatibility.
"""
import os
import urllib.parse
import warnings
from typing import Optional
from sqlalchemy import create_engine, Engine as SAEngine
from core.smartlog import get_logger

logger = get_logger(__name__)

# Import new unified system
from core.database import get_db_manager, create_connector, get_default_connector

def _issue_deprecation_warning(func_name: str):
    """Issue a deprecation warning for legacy functions."""
    warnings.warn(
        f"Function '{func_name}' is deprecated. Use the new unified database system: "
        "from core.database import create_connector",
        DeprecationWarning,
        stacklevel=3
    )

# --------------------------------------
# MSSQL (Windows auth; host dev machine)
# --------------------------------------
def connect_mssql_engine() -> SAEngine:
    """
    DEPRECATED: Use create_connector('mssql_adventureworks') instead.
    
    Windows Authentication to a local SQL Server on the host.
    Kept for backwards compatibility with 'mssql_win_auth'.
    """
    _issue_deprecation_warning('connect_mssql_engine')
    
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=AdventureWorks2022;"
        "Trusted_Connection=yes;"
    )
    engine_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(conn_str)
    logger.sysdebug(f"Creating DB engine (WinAuth) with URL: {engine_url}")
    return create_engine(engine_url)

# --------------------------------------
# MSSQL (SQL auth; Docker-compose setup)
# --------------------------------------
def connect_mssql_engine_docker() -> SAEngine:
    """
    DEPRECATED: Use create_connector('mssql_docker') instead.
    
    SQL Authentication for the Dockerized SQL Server service.
    Defaults match the docker-compose snippet:
      - host:     sqlserver
      - port:     1433
      - database: AdventureWorks2022
      - user:     sa
      - password: SA_PASSWORD (from .env)
    You can override via env: MSSQL_HOST, MSSQL_PORT, MSSQL_DB, MSSQL_USER, MSSQL_PASSWORD.
    Also supports MSSQL_ODBC_DRIVER (default: 'ODBC Driver 18 for SQL Server').
    """
    _issue_deprecation_warning('connect_mssql_engine_docker')
    
    host = os.getenv("MSSQL_HOST", "sqlserver")
    port = os.getenv("MSSQL_PORT", "1433")
    database = os.getenv("MSSQL_DB", "AdventureWorks2022")
    user = os.getenv("MSSQL_USER", "sa")

    # Prefer explicit MSSQL_PASSWORD; fall back to SA_PASSWORD for convenience
    password = os.getenv("MSSQL_PASSWORD") or os.getenv("SA_PASSWORD")
    if not password:
        raise EnvironmentError(
            "Missing MSSQL password. Set MSSQL_PASSWORD or SA_PASSWORD in your environment."
        )

    # Driver 18 is recommended on Linux containers. Allow override.
    driver = os.getenv("MSSQL_ODBC_DRIVER", "ODBC Driver 18 for SQL Server")

    # For local/dev: encrypt+trust (avoid cert hassles). For prod, manage certs properly.
    encrypt = os.getenv("MSSQL_ENCRYPT", "yes")  # 'yes'|'no'
    trust_cert = os.getenv("MSSQL_TRUST_SERVER_CERTIFICATE", "yes")  # 'yes'|'no'

    # Optional: connection timeout seconds
    timeout = os.getenv("MSSQL_LOGIN_TIMEOUT", "30")

    conn_kv = {
        "DRIVER": f"{{{driver}}}",
        "SERVER": f"{host},{port}",
        "DATABASE": database,
        "UID": user,
        "PWD": password,
        "Encrypt": encrypt,
        "TrustServerCertificate": trust_cert,
        "LoginTimeout": timeout,
    }

    # Build ODBC connection string safely, preserving spaces in driver name
    conn_str = ";".join(f"{k}={v}" for k, v in conn_kv.items()) + ";"

    engine_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(conn_str)

    # Do not log the password; show a redacted URL
    redacted = engine_url.replace(
        urllib.parse.quote_plus(password), "******"
    )
    logger.sysdebug(f"Creating DB engine (Docker MSSQL) with URL: {redacted}")

    return create_engine(engine_url)

# --------------------------------------
# Snowflake (unchanged)
# --------------------------------------
def connect_snowflake_engine(
    warehouse: str,
    database: str,
    role: str = "PARTICIPANT",
) -> SAEngine:
    """
    DEPRECATED: Use create_connector('snowflake_default') instead.
    """
    _issue_deprecation_warning('connect_snowflake_engine')
    
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

# --------------------------------------
# Factory
# --------------------------------------
def create_db_engine(
    dbms: str = "mssql_win_auth",
    warehouse: Optional[str] = None,
    database: Optional[str] = None,
) -> SAEngine:
    """
    DEPRECATED: Use create_connector() or get_default_connector() instead.
    
    Legacy compatibility function. Maps old DBMS strings to new profiles:
    - "mssql_win_auth" -> "mssql_adventureworks" (Windows Auth)
    - "mssql" -> "mssql_docker" (SQL Auth)  
    - "snowflake" -> "snowflake_default"
    """
    _issue_deprecation_warning('create_db_engine')
    
    # Map legacy DBMS values to new profile names
    profile_mapping = {
        "mssql_win_auth": "mssql_adventureworks",
        "mssql": "mssql_docker",
        "snowflake": "snowflake_default",
    }
    
    profile_name = profile_mapping.get(dbms)
    if not profile_name:
        raise ValueError(f"Unsupported DBMS: {dbms}. Use new unified system instead.")
    
    try:
        # Use new unified system
        connector = create_connector(profile_name)
        return connector.get_engine()
    except Exception as e:
        logger.warning(f"New unified system failed for profile '{profile_name}': {e}")
        logger.info("Falling back to legacy implementation...")
        
        # Fallback to legacy implementation
        if dbms == "mssql_win_auth":
            return connect_mssql_engine()

        if dbms == "mssql":
            # Dockerized SQL Server using SQL auth
            return connect_mssql_engine_docker()

        if dbms == "snowflake":
            if not warehouse:
                warehouse = "COMPUTE_WH_PARTICIPANT"
            if not database:
                raise ValueError("For Snowflake, please provide 'database' (e.g., database='YOUR_DB').")
            return connect_snowflake_engine(warehouse=warehouse, database=database)

        raise ValueError(f"Unsupported DBMS: {dbms}")


# New unified functions (recommended)
def get_engine(profile_name: Optional[str] = None) -> SAEngine:
    """
    Get a database engine using the new unified system.
    
    Args:
        profile_name: Name of the database profile. If None, uses default.
        
    Returns:
        SQLAlchemy engine instance
    """
    if profile_name:
        connector = create_connector(profile_name)
    else:
        connector = get_default_connector()
    
    return connector.get_engine()


def get_database_connector(profile_name: Optional[str] = None):
    """
    Get a database connector using the new unified system.
    
    Args:
        profile_name: Name of the database profile. If None, uses default.
        
    Returns:
        DatabaseConnector instance
    """
    if profile_name:
        return create_connector(profile_name)
    else:
        return get_default_connector()


def list_available_profiles():
    """List all available database profiles."""
    manager = get_db_manager()
    return manager.list_profiles()


def test_database_connection(profile_name: str) -> bool:
    """Test connection to a specific database profile."""
    manager = get_db_manager()
    return manager.test_connection(profile_name)