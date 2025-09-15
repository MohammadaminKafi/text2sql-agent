"""
Database test fixtures and utilities.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from core.config.database_config import DatabaseConfig
from core.database.sqlite import SQLiteConnector


@pytest.fixture
def temp_database_config():
    """Create a temporary database configuration for testing."""
    config_data = {
        'default_profile': 'test_sqlite',
        'profiles': {
            'test_sqlite': {
                'type': 'sqlite',
                'database': ':memory:',
                'description': 'Test SQLite in-memory database'
            },
            'test_sqlite_file': {
                'type': 'sqlite',
                'database': '/tmp/test.db',
                'description': 'Test SQLite file database'
            },
            'test_mssql': {
                'type': 'mssql',
                'host': 'localhost',
                'port': 1433,
                'database': 'TestDB',
                'user': 'sa',
                'password': 'TestPassword123',
                'driver': 'ODBC Driver 18 for SQL Server',
                'encrypt': 'yes',
                'trust_server_certificate': 'yes',
                'description': 'Test MSSQL database'
            },
            'test_snowflake': {
                'type': 'snowflake',
                'account': 'test-account',
                'user': 'testuser',
                'password': 'testpass',
                'warehouse': 'TEST_WH',
                'database': 'TEST_DB',
                'role': 'TEST_ROLE',
                'description': 'Test Snowflake database'
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        with patch.object(DatabaseConfig, '_get_config_path', return_value=Path(config_path)):
            config = DatabaseConfig()
            yield config
    finally:
        os.unlink(config_path)


@pytest.fixture
def mock_sqlite_engine():
    """Mock SQLAlchemy engine for SQLite testing."""
    engine = Mock()
    engine.connect.return_value.__enter__ = Mock(return_value=Mock())
    engine.connect.return_value.__exit__ = Mock(return_value=None)
    return engine


@pytest.fixture
def mock_mssql_engine():
    """Mock SQLAlchemy engine for MSSQL testing."""
    engine = Mock()
    engine.connect.return_value.__enter__ = Mock(return_value=Mock())
    engine.connect.return_value.__exit__ = Mock(return_value=None)
    return engine


@pytest.fixture
def mock_snowflake_engine():
    """Mock SQLAlchemy engine for Snowflake testing."""
    engine = Mock()
    engine.connect.return_value.__enter__ = Mock(return_value=Mock())
    engine.connect.return_value.__exit__ = Mock(return_value=None)
    return engine


@pytest.fixture
def sqlite_memory_connector():
    """Create a real SQLite in-memory connector for testing."""
    config = {
        'type': 'sqlite',
        'database': ':memory:',
        'description': 'Test SQLite in-memory database'
    }
    connector = SQLiteConnector('test_sqlite', config)
    yield connector
    try:
        connector.disconnect()
    except Exception:
        pass


@pytest.fixture
def sample_table_data():
    """Sample table structure for testing."""
    return {
        'schemas': ['main', 'temp'],
        'tables': {
            'main': ['users', 'orders', 'products'],
            'temp': ['temp_table']
        },
        'columns': {
            'users': [
                {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True},
                {'name': 'username', 'type': 'VARCHAR(50)', 'nullable': False, 'primary_key': False},
                {'name': 'email', 'type': 'VARCHAR(100)', 'nullable': True, 'primary_key': False},
                {'name': 'created_at', 'type': 'TIMESTAMP', 'nullable': True, 'primary_key': False}
            ],
            'orders': [
                {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True},
                {'name': 'user_id', 'type': 'INTEGER', 'nullable': False, 'primary_key': False},
                {'name': 'total', 'type': 'DECIMAL(10,2)', 'nullable': False, 'primary_key': False},
                {'name': 'order_date', 'type': 'DATE', 'nullable': False, 'primary_key': False}
            ]
        },
        'foreign_keys': {
            'orders': [
                {
                    'constraint_name': 'fk_orders_user_id',
                    'column_name': 'user_id',
                    'referenced_schema': 'main',
                    'referenced_table': 'users',
                    'referenced_column': 'id'
                }
            ]
        }
    }


@pytest.fixture
def mock_pandas_dataframe():
    """Create a mock pandas DataFrame for testing."""
    with patch('pandas.DataFrame') as mock_df:
        mock_instance = Mock()
        mock_instance.shape = (3, 2)
        mock_instance.columns = ['col1', 'col2']
        mock_instance.to_dict.return_value = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
        mock_df.return_value = mock_instance
        yield mock_instance


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    from core.smartlog import init_logging
    try:
        init_logging(console=False)
    except Exception:
        pass  # Already initialized


class MockInspector:
    """Mock SQLAlchemy inspector for testing."""
    
    def __init__(self, sample_data: Dict[str, Any]):
        self.sample_data = sample_data
    
    def get_schema_names(self):
        return self.sample_data.get('schemas', ['main'])
    
    def get_table_names(self, schema=None):
        schema = schema or 'main'
        return self.sample_data.get('tables', {}).get(schema, [])
    
    def get_columns(self, table, schema=None):
        return self.sample_data.get('columns', {}).get(table, [])
    
    def get_foreign_keys(self, table, schema=None):
        return self.sample_data.get('foreign_keys', {}).get(table, [])


@pytest.fixture
def mock_inspector(sample_table_data):
    """Create a mock SQLAlchemy inspector with sample data."""
    return MockInspector(sample_table_data)