"""
Abstract base classes and interfaces for database connectors.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union
import pandas as pd
from sqlalchemy import Engine, text

# Initialize smartlog if not already done
from core.smartlog import init_logging
try:
    init_logging(console=False)
except Exception:
    pass

from core.smartlog import get_logger

logger = get_logger(__name__)


class DatabaseConnector(ABC):
    """
    Abstract base class for all database connectors.
    
    Provides a unified interface for database operations across different database types.
    """
    
    def __init__(self, profile_name: str, config: Dict[str, Any]):
        """
        Initialize the database connector.
        
        Args:
            profile_name: Name of the configuration profile
            config: Resolved configuration dictionary
        """
        self.profile_name = profile_name
        self.config = config
        self.db_type = config.get('type')
        self.description = config.get('description', f'{self.db_type} database')
        self._engine: Optional[Engine] = None
        self._connected = False
    
    @abstractmethod
    def _create_engine(self) -> Engine:
        """Create and return a SQLAlchemy engine. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _build_connection_string(self) -> str:
        """Build database-specific connection string. Must be implemented by subclasses."""
        pass
    
    def connect(self) -> Engine:
        """
        Establish connection to the database.
        
        Returns:
            SQLAlchemy Engine instance
        """
        if self._engine is None:
            try:
                logger.sysdebug(f"Connecting to {self.description} (profile: {self.profile_name})")
                self._engine = self._create_engine()
                
                # Test the connection
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                self._connected = True
                logger.sysdebug(f"Successfully connected to {self.description}")
                
            except Exception as e:
                logger.error(f"Failed to connect to {self.description}: {e}")
                self._connected = False
                raise
        
        return self._engine
    
    def disconnect(self) -> None:
        """Close the database connection."""
        if self._engine is not None:
            try:
                self._engine.dispose()
                logger.sysdebug(f"Disconnected from {self.description}")
            except Exception as e:
                logger.warning(f"Error during disconnect from {self.description}: {e}")
            finally:
                self._engine = None
                self._connected = False
    
    def is_connected(self) -> bool:
        """Check if the connector is currently connected."""
        return self._connected and self._engine is not None
    
    def get_engine(self) -> Engine:
        """Get the SQLAlchemy engine, connecting if necessary."""
        if not self.is_connected():
            return self.connect()
        return self._engine
    
    @abstractmethod
    def list_schemas(self) -> List[str]:
        """List all schemas in the database."""
        pass
    
    @abstractmethod
    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """
        List all tables in the specified schema.
        
        Args:
            schema: Schema name. If None, uses default schema.
        """
        pass
    
    @abstractmethod
    def list_columns(self, table: str, schema: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        List all columns in the specified table.
        
        Args:
            table: Table name
            schema: Schema name. If None, uses default schema.
            
        Returns:
            List of (column_name, column_type) tuples
        """
        pass
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            pandas DataFrame with query results
        """
        try:
            engine = self.get_engine()
            logger.flowdebug(f"Executing query on {self.description}: {query[:100]}...")
            
            with engine.connect() as conn:
                if params:
                    result = pd.read_sql(query, conn, params=params)
                else:
                    result = pd.read_sql(query, conn)
            
            logger.flowdebug(f"Query returned {len(result)} rows × {len(result.columns)} columns")
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed on {self.description}: {e}")
            raise
    
    def get_table_info(self, table: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive information about a table.
        
        Args:
            table: Table name
            schema: Schema name
            
        Returns:
            Dictionary with table metadata
        """
        try:
            columns = self.list_columns(table, schema)
            
            # Get row count
            schema_prefix = f"{schema}." if schema else ""
            count_query = f"SELECT COUNT(*) as row_count FROM {schema_prefix}{table}"
            count_result = self.execute_query(count_query)
            row_count = count_result.iloc[0]['row_count']
            
            return {
                'schema': schema,
                'table': table,
                'columns': columns,
                'column_count': len(columns),
                'row_count': row_count,
                'connector': self.profile_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get table info for {schema}.{table}: {e}")
            raise
    
    def get_foreign_keys(self, table: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get foreign key relationships for a table.
        
        Args:
            table: Table name
            schema: Schema name
            
        Returns:
            List of foreign key dictionaries
        """
        try:
            engine = self.get_engine()
            from sqlalchemy import inspect
            
            inspector = inspect(engine)
            
            if hasattr(inspector, 'get_foreign_keys'):
                fks = inspector.get_foreign_keys(table, schema=schema)
                
                # Normalize the foreign key information
                normalized_fks = []
                for fk in fks:
                    normalized_fks.append({
                        'name': fk.get('name'),
                        'constrained_columns': fk.get('constrained_columns', []),
                        'referred_table': fk.get('referred_table'),
                        'referred_schema': fk.get('referred_schema'),
                        'referred_columns': fk.get('referred_columns', [])
                    })
                
                return normalized_fks
            else:
                logger.warning(f"Foreign key inspection not supported for {self.db_type}")
                return []
                
        except Exception as e:
            logger.warning(f"Failed to get foreign keys for {schema}.{table}: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Connection test failed for {self.description}: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def __repr__(self) -> str:
        """String representation of the connector."""
        status = "connected" if self.is_connected() else "disconnected"
        return f"{self.__class__.__name__}(profile='{self.profile_name}', status='{status}')"


class DatabaseException(Exception):
    """Custom exception for database-related errors."""
    
    def __init__(self, message: str, connector: Optional[DatabaseConnector] = None, 
                 original_error: Optional[Exception] = None):
        self.connector = connector
        self.original_error = original_error
        super().__init__(message)


class ConnectionError(DatabaseException):
    """Exception raised when database connection fails."""
    pass


class QueryError(DatabaseException):
    """Exception raised when query execution fails."""
    pass