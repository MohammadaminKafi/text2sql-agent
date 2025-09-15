"""
Snowflake database connector implementation.
"""
import urllib.parse
from typing import List, Tuple, Dict, Any, Optional
from sqlalchemy import create_engine, Engine, text, inspect
from core.database.base import DatabaseConnector, ConnectionError, QueryError
from core.smartlog import get_logger

logger = get_logger(__name__)


class SnowflakeConnector(DatabaseConnector):
    """Snowflake cloud database connector."""
    
    def __init__(self, profile_name: str, config: Dict[str, Any]):
        super().__init__(profile_name, config)
        self.account = config.get('account')
        self.warehouse = config.get('warehouse', 'COMPUTE_WH_PARTICIPANT')
        self.database = config.get('database')
        self.username = config.get('username')
        self.password = config.get('password')
        self.role = config.get('role', 'PARTICIPANT')
        
        # Validate required parameters
        if not self.account:
            raise ConnectionError("Snowflake account is required", self)
        if not self.username:
            raise ConnectionError("Username is required", self)
        if not self.password:
            raise ConnectionError("Password is required", self)
        if not self.database:
            raise ConnectionError("Database is required", self)
    
    def _build_connection_string(self) -> str:
        """Build Snowflake connection string."""
        # URL-encode credentials safely
        user_q = urllib.parse.quote_plus(self.username)
        pass_q = urllib.parse.quote_plus(self.password)
        warehouse_q = urllib.parse.quote_plus(self.warehouse)
        role_q = urllib.parse.quote_plus(self.role)
        
        # Example URL: snowflake://user:pass@account/database?warehouse=WH&role=ROLE
        connection_string = (
            f"snowflake://{user_q}:{pass_q}@{self.account}/"
            f"{self.database}?warehouse={warehouse_q}&role={role_q}"
        )
        
        return connection_string
    
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine for Snowflake."""
        try:
            connection_string = self._build_connection_string()
            
            logger.sysdebug(
                f"Creating Snowflake engine: account={self.account}, "
                f"database={self.database}, warehouse={self.warehouse}, role={self.role}"
            )
            
            return create_engine(connection_string, pool_pre_ping=True)
            
        except Exception as e:
            raise ConnectionError(f"Failed to create Snowflake engine: {e}", self, e)
    
    def list_schemas(self) -> List[str]:
        """List all schemas in the Snowflake database."""
        try:
            engine = self.get_engine()
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT SCHEMA_NAME 
                    FROM INFORMATION_SCHEMA.SCHEMATA 
                    WHERE CATALOG_NAME = CURRENT_DATABASE()
                    ORDER BY SCHEMA_NAME
                """))
                
                schemas = [row[0] for row in result]
                logger.flowdebug(f"Found {len(schemas)} schemas in {self.database}")
                return schemas
                
        except Exception as e:
            logger.error(f"Failed to list schemas: {e}")
            # Fallback to common Snowflake schemas
            return ["PUBLIC", "INFORMATION_SCHEMA"]
    
    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List all tables in the specified schema."""
        if schema is None:
            schema = "PUBLIC"
        
        try:
            engine = self.get_engine()
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT TABLE_NAME 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA = :schema_name 
                    AND TABLE_TYPE = 'BASE TABLE'
                    ORDER BY TABLE_NAME
                """), {"schema_name": schema.upper()})
                
                tables = [row[0] for row in result]
                logger.flowdebug(f"Found {len(tables)} tables in schema {schema}")
                return tables
                
        except Exception as e:
            logger.error(f"Failed to list tables in schema {schema}: {e}")
            return []
    
    def list_columns(self, table: str, schema: Optional[str] = None) -> List[Tuple[str, str]]:
        """List all columns in the specified table."""
        if schema is None:
            schema = "PUBLIC"
        
        try:
            engine = self.get_engine()
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = :schema_name 
                    AND TABLE_NAME = :table_name
                    ORDER BY ORDINAL_POSITION
                """), {"schema_name": schema.upper(), "table_name": table.upper()})
                
                columns = [(row[0], row[1]) for row in result]
                logger.flowdebug(f"Found {len(columns)} columns in {schema}.{table}")
                return columns
                
        except Exception as e:
            logger.error(f"Failed to list columns for {schema}.{table}: {e}")
            return []
    
    def get_primary_keys(self, table: str, schema: Optional[str] = None) -> List[str]:
        """Get primary key columns for a table."""
        if schema is None:
            schema = "PUBLIC"
        
        try:
            engine = self.get_engine()
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                    JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                    ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                    WHERE tc.TABLE_SCHEMA = :schema_name 
                    AND tc.TABLE_NAME = :table_name
                    AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                    ORDER BY kcu.ORDINAL_POSITION
                """), {"schema_name": schema.upper(), "table_name": table.upper()})
                
                pk_columns = [row[0] for row in result]
                logger.flowdebug(f"Found {len(pk_columns)} primary key columns in {schema}.{table}")
                return pk_columns
                
        except Exception as e:
            logger.warning(f"Could not get primary keys for {schema}.{table}: {e}")
            return []
    
    def get_foreign_keys(self, table: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get foreign key relationships for a table."""
        if schema is None:
            schema = "PUBLIC"
        
        try:
            engine = self.get_engine()
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        tc.CONSTRAINT_NAME,
                        kcu.COLUMN_NAME,
                        ccu.TABLE_SCHEMA AS FOREIGN_TABLE_SCHEMA,
                        ccu.TABLE_NAME AS FOREIGN_TABLE_NAME,
                        ccu.COLUMN_NAME AS FOREIGN_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                    JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                    ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                    JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
                    ON ccu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
                    WHERE tc.TABLE_SCHEMA = :schema_name 
                    AND tc.TABLE_NAME = :table_name
                    AND tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
                """), {"schema_name": schema.upper(), "table_name": table.upper()})
                
                fks = []
                for row in result:
                    fks.append({
                        'name': row[0],
                        'constrained_columns': [row[1]],
                        'referred_table': row[3],
                        'referred_schema': row[2],
                        'referred_columns': [row[4]]
                    })
                
                logger.flowdebug(f"Found {len(fks)} foreign keys in {schema}.{table}")
                return fks
                
        except Exception as e:
            logger.warning(f"Could not get foreign keys for {schema}.{table}: {e}")
            return []
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get general information about the Snowflake database."""
        try:
            with self.get_engine().connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        CURRENT_DATABASE() as database_name,
                        CURRENT_WAREHOUSE() as warehouse,
                        CURRENT_ROLE() as role,
                        CURRENT_USER() as user_name,
                        CURRENT_ACCOUNT() as account
                """))
                
                row = result.fetchone()
                
                return {
                    'database_name': row[0],
                    'warehouse': row[1],
                    'role': row[2],
                    'user_name': row[3],
                    'account': row[4],
                    'connector_type': 'snowflake',
                    'profile': self.profile_name
                }
                
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {
                'database_name': self.database,
                'warehouse': self.warehouse,
                'account': self.account,
                'connector_type': 'snowflake',
                'profile': self.profile_name
            }
    
    def get_warehouse_info(self) -> Dict[str, Any]:
        """Get information about the current warehouse."""
        try:
            with self.get_engine().connect() as conn:
                result = conn.execute(text("""
                    SHOW WAREHOUSES LIKE :warehouse_name
                """), {"warehouse_name": self.warehouse})
                
                # Note: Snowflake SHOW commands return different result structure
                warehouses = result.fetchall()
                
                if warehouses:
                    # Extract warehouse information (columns vary by Snowflake version)
                    return {
                        'warehouse_name': self.warehouse,
                        'status': 'active',
                        'connector_type': 'snowflake'
                    }
                else:
                    return {
                        'warehouse_name': self.warehouse,
                        'status': 'unknown',
                        'connector_type': 'snowflake'
                    }
                    
        except Exception as e:
            logger.warning(f"Could not get warehouse info: {e}")
            return {
                'warehouse_name': self.warehouse,
                'status': 'unknown',
                'connector_type': 'snowflake'
            }