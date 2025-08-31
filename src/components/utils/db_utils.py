import urllib.parse
from sqlalchemy import create_engine
from components.smartlog import get_logger

logger = get_logger(__name__)

def create_db_engine():
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=AdventureWorks2022;"
        "Trusted_Connection=yes;"
    )
    engine_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(conn_str)
    logger.sysdebug(f"Creating DB engine with URL: {engine_url}")
    return create_engine(engine_url)
