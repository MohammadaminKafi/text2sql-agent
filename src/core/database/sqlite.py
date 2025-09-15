"""
SQLite database connector implementation.
"""
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sqlalchemy import create_engine, Engine, text, inspect
from core.database.base import DatabaseConnector, ConnectionError, QueryError
from core.smartlog import get_logger

logger = get_logger(__name__)


class SQLiteConnector(DatabaseConnector):
    """SQLite database connector."""
    
    def __init__(self, profile_name: str, config: Dict[str, Any]):
        super().__init__(profile_name, config)
        self.database_path = config.get('database_path', ':memory:')
        
        # Create directory if needed (except for :memory:)
        if self.database_path != ':memory:':
            db_path = Path(self.database_path)
            if not db_path.parent.exists():
                db_path.parent.mkdir(parents=True, exist_ok=True)
                logger.sysdebug(f"Created directory for SQLite database: {db_path.parent}")
    
    def _build_connection_string(self) -> str:
        """Build SQLite connection string."""
        if self.database_path == ':memory:':
            return "sqlite:///:memory:"
        else:
            # Use absolute path for file-based databases
            abs_path = os.path.abspath(self.database_path)
            return f"sqlite:///{abs_path}"
    
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine for SQLite."""
        try:
            connection_string = self._build_connection_string()
            
            logger.sysdebug(f"Creating SQLite engine: {self.database_path}")
            
            # SQLite-specific engine parameters
            engine_kwargs = {
                'pool_pre_ping': True,
                'connect_args': {
                    'check_same_thread': False,  # Allow multiple threads
                    'timeout': 30  # 30 second timeout
                }
            }
            
            return create_engine(connection_string, **engine_kwargs)
            
        except Exception as e:
            raise ConnectionError(f"Failed to create SQLite engine: {e}", self, e)
    
    def list_schemas(self) -> List[str]:
        """List all schemas in the SQLite database. SQLite only has 'main' schema."""
        # SQLite doesn't have multiple schemas in the same way as other databases
        # It has 'main' for the primary database and can have attached databases
        try:
            engine = self.get_engine()
            
            with engine.connect() as conn:
                result = conn.execute(text("PRAGMA database_list"))
                
                schemas = []
                for row in result:
                    # row format: (seq, name, file)
                    schema_name = row[1] if len(row) > 1 else 'main'
                    schemas.append(schema_name)
                
                logger.flowdebug(f"Found {len(schemas)} schemas in SQLite database")
                return schemas if schemas else ['main']
                
        except Exception as e:
            logger.warning(f"Could not list schemas: {e}")
            return ['main']  # Default SQLite schema
    
    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List all tables in the specified schema."""
        if schema is None:
            schema = 'main'
        
        try:
            engine = self.get_engine()
            
            with engine.connect() as conn:
                if schema == 'main':
                    # For main schema, use sqlite_master
                    result = conn.execute(text("""
                        SELECT name 
                        FROM sqlite_master 
                        WHERE type='table' AND name NOT LIKE 'sqlite_%'
                        ORDER BY name
                    """))
                else:
                    # For attached databases
                    result = conn.execute(text(f"""
                        SELECT name 
                        FROM {schema}.sqlite_master 
                        WHERE type='table' AND name NOT LIKE 'sqlite_%'
                        ORDER BY name
                    """))
                
                tables = [row[0] for row in result]
                logger.flowdebug(f"Found {len(tables)} tables in schema {schema}")
                return tables
                
        except Exception as e:
            logger.error(f"Failed to list tables in schema {schema}: {e}")
            return []
    
    def list_columns(self, table: str, schema: Optional[str] = None) -> List[Tuple[str, str]]:
        """List all columns in the specified table."""
        if schema is None:
            schema = 'main'
        
        try:
            engine = self.get_engine()
            
            with engine.connect() as conn:
                if schema == 'main':
                    result = conn.execute(text(f"PRAGMA table_info({table})"))
                else:
                    result = conn.execute(text(f"PRAGMA {schema}.table_info({table})"))
                
                columns = []
                for row in result:
                    # row format: (cid, name, type, notnull, dflt_value, pk)
                    col_name = row[1]
                    col_type = row[2] if row[2] else 'TEXT'  # Default to TEXT if no type
                    columns.append((col_name, col_type))
                
                logger.flowdebug(f"Found {len(columns)} columns in {schema}.{table}")
                return columns
                
        except Exception as e:
            logger.error(f"Failed to list columns for {schema}.{table}: {e}")
            return []
    
    def get_primary_keys(self, table: str, schema: Optional[str] = None) -> List[str]:
        """Get primary key columns for a table."""
        if schema is None:
            schema = 'main'
        
        try:
            engine = self.get_engine()
            
            with engine.connect() as conn:
                if schema == 'main':
                    result = conn.execute(text(f"PRAGMA table_info({table})"))
                else:
                    result = conn.execute(text(f"PRAGMA {schema}.table_info({table})"))
                
                pk_columns = []
                for row in result:
                    # row format: (cid, name, type, notnull, dflt_value, pk)
                    if row[5]:  # pk field is 1 for primary key columns
                        pk_columns.append(row[1])
                
                logger.flowdebug(f"Found {len(pk_columns)} primary key columns in {schema}.{table}")
                return pk_columns
                
        except Exception as e:
            logger.warning(f"Could not get primary keys for {schema}.{table}: {e}")
            return []
    
    def get_foreign_keys(self, table: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get foreign key relationships for a table."""
        if schema is None:
            schema = 'main'
        
        try:
            engine = self.get_engine()
            
            with engine.connect() as conn:
                if schema == 'main':
                    result = conn.execute(text(f"PRAGMA foreign_key_list({table})"))
                else:
                    result = conn.execute(text(f"PRAGMA {schema}.foreign_key_list({table})"))
                
                fks = []
                for row in result:
                    # row format: (id, seq, table, from, to, on_update, on_delete, match)
                    fks.append({
                        'name': f"fk_{table}_{row[0]}",  # SQLite doesn't name FKs
                        'constrained_columns': [row[3]],  # from column
                        'referred_table': row[2],         # table
                        'referred_schema': schema,
                        'referred_columns': [row[4]]      # to column
                    })
                
                logger.flowdebug(f"Found {len(fks)} foreign keys in {schema}.{table}")
                return fks
                
        except Exception as e:
            logger.warning(f"Could not get foreign keys for {schema}.{table}: {e}")
            return []
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get general information about the SQLite database."""
        try:
            with self.get_engine().connect() as conn:
                # Get SQLite version
                result = conn.execute(text("SELECT sqlite_version()"))
                version = result.scalar()
                
                # Get database file info
                if self.database_path == ':memory:':
                    file_size = 0
                    file_exists = True
                else:
                    db_path = Path(self.database_path)
                    file_exists = db_path.exists()
                    file_size = db_path.stat().st_size if file_exists else 0
                
                return {
                    'database_path': self.database_path,
                    'sqlite_version': version,
                    'file_exists': file_exists,
                    'file_size_bytes': file_size,
                    'connector_type': 'sqlite',
                    'profile': self.profile_name
                }
                
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {
                'database_path': self.database_path,
                'connector_type': 'sqlite',
                'profile': self.profile_name
            }
    
    def vacuum_database(self) -> bool:
        """Vacuum the SQLite database to reclaim space."""
        try:
            with self.get_engine().connect() as conn:
                conn.execute(text("VACUUM"))
            logger.info(f"Successfully vacuumed SQLite database: {self.database_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            return False
    
    def get_table_size_info(self, table: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Get size information for a specific table."""
        if schema is None:
            schema = 'main'
        
        try:
            with self.get_engine().connect() as conn:
                # Get row count
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                row_count = result.scalar()
                
                # SQLite doesn't have built-in table size functions like other databases
                # We can estimate based on page count for the whole database
                return {
                    'table': table,
                    'schema': schema,
                    'row_count': row_count,
                    'connector_type': 'sqlite'
                }
                
        except Exception as e:
            logger.error(f"Failed to get table size info for {table}: {e}")
            return {
                'table': table,
                'schema': schema,
                'connector_type': 'sqlite'
            }