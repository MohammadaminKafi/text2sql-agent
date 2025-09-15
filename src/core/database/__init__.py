"""
Database package providing unified database connectivity.

This package provides a unified interface for connecting to various database types
including MSSQL, Snowflake, and SQLite through a configuration-driven approach.

Usage:
    from core.database import get_db_manager, create_connector, execute_query
    
    # Get default connector
    manager = get_db_manager()
    default_conn = manager.get_default_connector()
    
    # Create specific connector
    conn = create_connector('mssql_adventureworks')
    
    # Execute query with default connector
    results = execute_query("SELECT * FROM table")
    
    # Execute query with specific profile
    results = execute_query("SELECT * FROM table", profile_name='snowflake_default')
"""

# Import main classes and functions for easy access
from core.database.base import DatabaseConnector, DatabaseException, ConnectionError, QueryError
from core.database.factory import (
    DatabaseFactory, 
    DatabaseManager, 
    get_db_manager, 
    reset_db_manager,
    create_connector,
    get_default_connector,
    execute_query
)

# Import specific connector classes (optional, for advanced usage)
from core.database.mssql import MSSQLConnector
from core.database.snowflake import SnowflakeConnector
from core.database.sqlite import SQLiteConnector

__all__ = [
    # Base classes
    'DatabaseConnector',
    'DatabaseException',
    'ConnectionError', 
    'QueryError',
    
    # Factory and management
    'DatabaseFactory',
    'DatabaseManager',
    'get_db_manager',
    'reset_db_manager',
    
    # Convenience functions
    'create_connector',
    'get_default_connector',
    'execute_query',
    
    # Specific connectors (for advanced usage)
    'MSSQLConnector',
    'SnowflakeConnector', 
    'SQLiteConnector',
]