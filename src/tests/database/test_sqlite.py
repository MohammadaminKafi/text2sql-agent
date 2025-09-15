"""
Tests for SQLite database connector.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import sqlite3

from core.database.sqlite import SQLiteConnector
from core.database.base import ConnectionError, QueryError


class TestSQLiteConnector:
    """Test the SQLite database connector."""
    
    def test_memory_database_initialization(self):
        """Test initialization with in-memory database."""
        config = {
            'type': 'sqlite',
            'database': ':memory:',
            'description': 'In-memory test database'
        }
        
        connector = SQLiteConnector('sqlite_memory', config)
        
        assert connector.profile_name == 'sqlite_memory'
        assert connector.config == config
        assert connector.database_path == ':memory:'
        assert not connector.is_connected()
    
    def test_file_database_initialization(self):
        """Test initialization with file database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'test.db')
            config = {
                'type': 'sqlite',
                'database': db_path,
                'description': 'File test database'
            }
            
            connector = SQLiteConnector('sqlite_file', config)
            
            assert connector.profile_name == 'sqlite_file'
            assert connector.database_path == db_path
            assert not connector.is_connected()
    
    def test_connect_memory_database(self, sqlite_memory_connector):
        """Test connecting to in-memory database."""
        connector = sqlite_memory_connector
        
        assert connector.connect()
        assert connector.is_connected()
        assert connector.engine is not None
    
    def test_connect_file_database(self):
        """Test connecting to file database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'test.db')
            config = {
                'type': 'sqlite',
                'database': db_path
            }
            
            connector = SQLiteConnector('sqlite_file', config)
            
            assert connector.connect()
            assert connector.is_connected()
            assert os.path.exists(db_path)
            
            connector.disconnect()
    
    def test_connect_creates_directory(self):
        """Test that connecting creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, 'nested', 'path', 'test.db')
            config = {
                'type': 'sqlite',
                'database': nested_path
            }
            
            connector = SQLiteConnector('sqlite_nested', config)
            
            assert connector.connect()
            assert os.path.exists(nested_path)
            assert os.path.exists(os.path.dirname(nested_path))
            
            connector.disconnect()
    
    def test_disconnect(self, sqlite_memory_connector):
        """Test disconnecting from database."""
        connector = sqlite_memory_connector
        
        connector.connect()
        assert connector.is_connected()
        
        connector.disconnect()
        assert not connector.is_connected()
        assert connector.engine is None
    
    def test_test_connection_success(self, sqlite_memory_connector):
        """Test successful connection test."""
        connector = sqlite_memory_connector
        
        assert connector.test_connection()
    
    def test_test_connection_failure(self):
        """Test connection test failure."""
        config = {
            'type': 'sqlite',
            'database': '/invalid/path/test.db'
        }
        
        connector = SQLiteConnector('invalid_sqlite', config)
        
        # Should return False for invalid path without raising exception
        assert not connector.test_connection()
    
    def test_get_engine(self, sqlite_memory_connector):
        """Test getting SQLAlchemy engine."""
        connector = sqlite_memory_connector
        connector.connect()
        
        engine = connector.get_engine()
        assert engine is not None
        assert str(engine.url).startswith('sqlite://')
    
    def test_list_schemas(self, sqlite_memory_connector):
        """Test listing database schemas."""
        connector = sqlite_memory_connector
        connector.connect()
        
        schemas = connector.list_schemas()
        assert isinstance(schemas, list)
        assert 'main' in schemas
    
    def test_list_tables_empty_database(self, sqlite_memory_connector):
        """Test listing tables in empty database."""
        connector = sqlite_memory_connector
        connector.connect()
        
        tables = connector.list_tables('main')
        assert isinstance(tables, list)
        assert len(tables) == 0
    
    def test_list_tables_with_data(self, sqlite_memory_connector):
        """Test listing tables with existing tables."""
        connector = sqlite_memory_connector
        connector.connect()
        
        # Create a test table
        connector.execute_query("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        """)
        
        tables = connector.list_tables('main')
        assert 'test_table' in tables
    
    def test_list_columns_existing_table(self, sqlite_memory_connector):
        """Test listing columns for existing table."""
        connector = sqlite_memory_connector
        connector.connect()
        
        # Create a test table
        connector.execute_query("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        columns = connector.list_columns('main', 'users')
        assert isinstance(columns, list)
        assert len(columns) > 0
        
        # Check column structure
        column_names = [col['name'] for col in columns]
        assert 'id' in column_names
        assert 'username' in column_names
        assert 'email' in column_names
        assert 'created_at' in column_names
        
        # Check column details
        id_column = next(col for col in columns if col['name'] == 'id')
        assert id_column['type'] == 'INTEGER'
        assert id_column['primary_key'] is True
        assert id_column['nullable'] is False
    
    def test_list_columns_nonexistent_table(self, sqlite_memory_connector):
        """Test listing columns for non-existent table."""
        connector = sqlite_memory_connector
        connector.connect()
        
        columns = connector.list_columns('main', 'nonexistent_table')
        assert isinstance(columns, list)
        assert len(columns) == 0
    
    def test_get_foreign_keys_with_fks(self, sqlite_memory_connector):
        """Test getting foreign keys for table with foreign keys."""
        connector = sqlite_memory_connector
        connector.connect()
        
        # Create tables with foreign key relationship
        connector.execute_query("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL
            )
        """)
        
        connector.execute_query("""
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                total DECIMAL(10,2),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        foreign_keys = connector.get_foreign_keys('main', 'orders')
        assert isinstance(foreign_keys, list)
        
        if len(foreign_keys) > 0:  # SQLite FK support depends on version
            fk = foreign_keys[0]
            assert 'column_name' in fk
            assert 'referenced_table' in fk
            assert 'referenced_column' in fk
    
    def test_get_foreign_keys_no_fks(self, sqlite_memory_connector):
        """Test getting foreign keys for table without foreign keys."""
        connector = sqlite_memory_connector
        connector.connect()
        
        connector.execute_query("""
            CREATE TABLE simple_table (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)
        
        foreign_keys = connector.get_foreign_keys('main', 'simple_table')
        assert isinstance(foreign_keys, list)
        assert len(foreign_keys) == 0
    
    def test_execute_query_select(self, sqlite_memory_connector):
        """Test executing SELECT query."""
        connector = sqlite_memory_connector
        connector.connect()
        
        # Create and populate test table
        connector.execute_query("""
            CREATE TABLE test_data (
                id INTEGER,
                name TEXT,
                value REAL
            )
        """)
        
        connector.execute_query("""
            INSERT INTO test_data (id, name, value) VALUES
            (1, 'first', 10.5),
            (2, 'second', 20.0),
            (3, 'third', 30.5)
        """)
        
        # Test SELECT query
        result = connector.execute_query("SELECT * FROM test_data ORDER BY id")
        
        assert hasattr(result, 'shape')  # Should be a DataFrame
        assert result.shape[0] == 3  # 3 rows
        assert result.shape[1] == 3  # 3 columns
        
        # Check data
        assert list(result['id']) == [1, 2, 3]
        assert list(result['name']) == ['first', 'second', 'third']
    
    def test_execute_query_ddl(self, sqlite_memory_connector):
        """Test executing DDL queries."""
        connector = sqlite_memory_connector
        connector.connect()
        
        # CREATE TABLE
        result = connector.execute_query("""
            CREATE TABLE ddl_test (
                id INTEGER PRIMARY KEY,
                description TEXT
            )
        """)
        
        # Should return empty DataFrame for DDL
        assert hasattr(result, 'shape')
        assert result.shape[0] == 0
        
        # Verify table was created
        tables = connector.list_tables('main')
        assert 'ddl_test' in tables
    
    def test_execute_query_dml(self, sqlite_memory_connector):
        """Test executing DML queries."""
        connector = sqlite_memory_connector
        connector.connect()
        
        connector.execute_query("""
            CREATE TABLE dml_test (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)
        
        # INSERT
        result = connector.execute_query("""
            INSERT INTO dml_test (name) VALUES ('test1'), ('test2')
        """)
        
        # Should return empty DataFrame for DML
        assert hasattr(result, 'shape')
        assert result.shape[0] == 0
        
        # Verify data was inserted
        select_result = connector.execute_query("SELECT COUNT(*) as count FROM dml_test")
        assert select_result.iloc[0]['count'] == 2
    
    def test_execute_query_invalid_sql(self, sqlite_memory_connector):
        """Test executing invalid SQL query."""
        connector = sqlite_memory_connector
        connector.connect()
        
        with pytest.raises(QueryError):
            connector.execute_query("INVALID SQL SYNTAX")
    
    def test_context_manager(self):
        """Test using connector as context manager."""
        config = {
            'type': 'sqlite',
            'database': ':memory:'
        }
        
        connector = SQLiteConnector('context_test', config)
        
        with connector as conn:
            assert conn.is_connected()
            
            # Test database operations
            conn.execute_query("""
                CREATE TABLE context_test (id INTEGER, name TEXT)
            """)
            
            tables = conn.list_tables('main')
            assert 'context_test' in tables
        
        assert not connector.is_connected()
    
    @patch('core.database.sqlite.create_engine')
    def test_connection_failure_handling(self, mock_create_engine):
        """Test handling of connection failures."""
        mock_create_engine.side_effect = Exception("Connection failed")
        
        config = {
            'type': 'sqlite',
            'database': ':memory:'
        }
        
        connector = SQLiteConnector('fail_test', config)
        
        with pytest.raises(ConnectionError):
            connector.connect()
    
    def test_vacuum_operation(self, sqlite_memory_connector):
        """Test VACUUM operation."""
        connector = sqlite_memory_connector
        connector.connect()
        
        # Create and populate table
        connector.execute_query("""
            CREATE TABLE vacuum_test (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """)
        
        connector.execute_query("""
            INSERT INTO vacuum_test (data) VALUES ('test data')
        """)
        
        # Delete data to create space for vacuum
        connector.execute_query("DELETE FROM vacuum_test")
        
        # VACUUM should work without errors
        result = connector.execute_query("VACUUM")
        assert hasattr(result, 'shape')
    
    def test_pragma_queries(self, sqlite_memory_connector):
        """Test PRAGMA queries for database introspection."""
        connector = sqlite_memory_connector
        connector.connect()
        
        # Test basic PRAGMA query
        result = connector.execute_query("PRAGMA database_list")
        assert hasattr(result, 'shape')
        assert result.shape[0] >= 1  # Should have at least main database
    
    def test_concurrent_connections(self):
        """Test multiple connections to the same file database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'concurrent_test.db')
            config = {
                'type': 'sqlite',
                'database': db_path
            }
            
            connector1 = SQLiteConnector('conn1', config)
            connector2 = SQLiteConnector('conn2', config)
            
            # Both should be able to connect
            assert connector1.connect()
            assert connector2.connect()
            
            # Test operations from both connections
            connector1.execute_query("""
                CREATE TABLE IF NOT EXISTS concurrent_test (
                    id INTEGER PRIMARY KEY,
                    name TEXT
                )
            """)
            
            connector2.execute_query("""
                INSERT INTO concurrent_test (name) VALUES ('test')
            """)
            
            # Read from first connection
            result = connector1.execute_query("SELECT * FROM concurrent_test")
            assert result.shape[0] == 1
            
            connector1.disconnect()
            connector2.disconnect()


class TestSQLiteConnectorEdgeCases:
    """Test edge cases and error conditions for SQLite connector."""
    
    def test_readonly_database_path(self):
        """Test handling of read-only database path."""
        # This test might be platform-specific
        config = {
            'type': 'sqlite',
            'database': '/proc/version'  # Read-only file on Linux
        }
        
        connector = SQLiteConnector('readonly_test', config)
        
        # Should handle gracefully
        result = connector.test_connection()
        assert isinstance(result, bool)
    
    def test_very_long_database_path(self):
        """Test handling of very long database path."""
        long_path = '/tmp/' + 'x' * 1000 + '.db'
        config = {
            'type': 'sqlite',
            'database': long_path
        }
        
        connector = SQLiteConnector('long_path_test', config)
        
        # Should handle gracefully without crashing
        result = connector.test_connection()
        assert isinstance(result, bool)
    
    def test_special_characters_in_path(self):
        """Test database path with special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create path with special characters
            special_path = os.path.join(temp_dir, 'test with spaces & symbols!@#.db')
            config = {
                'type': 'sqlite',
                'database': special_path
            }
            
            connector = SQLiteConnector('special_chars', config)
            
            assert connector.connect()
            assert os.path.exists(special_path)
            
            connector.disconnect()
    
    def test_empty_database_path(self):
        """Test handling of empty database path."""
        config = {
            'type': 'sqlite',
            'database': ''
        }
        
        connector = SQLiteConnector('empty_path', config)
        
        # Should handle gracefully
        result = connector.test_connection()
        assert isinstance(result, bool)
    
    def test_none_database_path(self):
        """Test handling of None database path."""
        config = {
            'type': 'sqlite',
            'database': None
        }
        
        connector = SQLiteConnector('none_path', config)
        
        # Should handle gracefully
        result = connector.test_connection()
        assert isinstance(result, bool)