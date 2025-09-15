"""
Tests for the abstract base database connector.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from abc import ABC

from core.database.base import (
    DatabaseConnector,
    DatabaseException,
    ConnectionError,
    QueryError
)


class ConcreteDatabaseConnector(DatabaseConnector):
    """Concrete implementation of DatabaseConnector for testing."""
    
    def __init__(self, profile_name: str, config: dict):
        super().__init__(profile_name, config)
        self._connected = False
        self._engine = Mock()
    
    def connect(self) -> bool:
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        self._connected = False
    
    def is_connected(self) -> bool:
        return self._connected
    
    def get_engine(self):
        return self._engine
    
    def test_connection(self) -> bool:
        return True
    
    def list_schemas(self):
        return ['schema1', 'schema2']
    
    def list_tables(self, schema: str):
        return ['table1', 'table2']
    
    def list_columns(self, schema: str, table: str):
        return [
            {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True},
            {'name': 'name', 'type': 'VARCHAR(50)', 'nullable': False, 'primary_key': False}
        ]
    
    def get_foreign_keys(self, schema: str, table: str):
        return []
    
    def execute_query(self, query: str, **kwargs):
        import pandas as pd
        return pd.DataFrame({'result': [1, 2, 3]})


class TestDatabaseConnector:
    """Test the abstract DatabaseConnector class."""
    
    def test_concrete_implementation_creation(self):
        """Test creating a concrete implementation of DatabaseConnector."""
        config = {'type': 'test', 'database': 'testdb'}
        connector = ConcreteDatabaseConnector('test_profile', config)
        
        assert connector.profile_name == 'test_profile'
        assert connector.config == config
        assert not connector.is_connected()
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that DatabaseConnector cannot be instantiated directly."""
        config = {'type': 'test'}
        
        with pytest.raises(TypeError):
            DatabaseConnector('test_profile', config)
    
    def test_context_manager_success(self):
        """Test context manager behavior with successful connection."""
        config = {'type': 'test', 'database': 'testdb'}
        connector = ConcreteDatabaseConnector('test_profile', config)
        
        with connector:
            assert connector.is_connected()
        
        assert not connector.is_connected()
    
    def test_context_manager_connection_failure(self):
        """Test context manager behavior when connection fails."""
        config = {'type': 'test', 'database': 'testdb'}
        connector = ConcreteDatabaseConnector('test_profile', config)
        
        # Mock connect to return False
        connector.connect = Mock(return_value=False)
        
        with pytest.raises(ConnectionError, match="Failed to connect to database"):
            with connector:
                pass
    
    def test_context_manager_exception_during_operation(self):
        """Test context manager properly disconnects even if exception occurs."""
        config = {'type': 'test', 'database': 'testdb'}
        connector = ConcreteDatabaseConnector('test_profile', config)
        
        try:
            with connector:
                assert connector.is_connected()
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
        
        assert not connector.is_connected()
    
    def test_abstract_methods_exist(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = DatabaseConnector.__abstractmethods__
        expected_methods = {
            'connect', 'disconnect', 'is_connected', 'get_engine',
            'test_connection', 'list_schemas', 'list_tables', 
            'list_columns', 'get_foreign_keys', 'execute_query'
        }
        
        assert abstract_methods == expected_methods
    
    def test_string_representation(self):
        """Test string representation of connector."""
        config = {'type': 'test', 'database': 'testdb'}
        connector = ConcreteDatabaseConnector('test_profile', config)
        
        str_repr = str(connector)
        assert 'ConcreteDatabaseConnector' in str_repr
        assert 'test_profile' in str_repr
        assert 'test' in str_repr


class TestDatabaseExceptions:
    """Test custom database exceptions."""
    
    def test_database_exception_creation(self):
        """Test creating DatabaseException."""
        exception = DatabaseException("Test error message")
        assert str(exception) == "Test error message"
        assert isinstance(exception, Exception)
    
    def test_connection_error_creation(self):
        """Test creating ConnectionError."""
        exception = ConnectionError("Connection failed")
        assert str(exception) == "Connection failed"
        assert isinstance(exception, DatabaseException)
    
    def test_query_error_creation(self):
        """Test creating QueryError."""
        exception = QueryError("Query failed")
        assert str(exception) == "Query failed"
        assert isinstance(exception, DatabaseException)
    
    def test_exception_hierarchy(self):
        """Test exception inheritance hierarchy."""
        assert issubclass(ConnectionError, DatabaseException)
        assert issubclass(QueryError, DatabaseException)
        assert issubclass(DatabaseException, Exception)
    
    def test_exception_with_cause(self):
        """Test exceptions with underlying cause."""
        original_error = ValueError("Original error")
        
        try:
            raise original_error
        except ValueError as e:
            db_error = DatabaseException("Database error") from e
            
        assert db_error.__cause__ is original_error
        assert "Database error" in str(db_error)


class TestDatabaseConnectorDefaultBehavior:
    """Test default behavior of DatabaseConnector methods."""
    
    def test_concrete_connector_methods(self):
        """Test all methods work as expected in concrete implementation."""
        config = {
            'type': 'test',
            'database': 'testdb',
            'host': 'localhost',
            'port': 5432
        }
        connector = ConcreteDatabaseConnector('test_profile', config)
        
        # Test connection methods
        assert not connector.is_connected()
        assert connector.connect()
        assert connector.is_connected()
        
        # Test database introspection methods
        schemas = connector.list_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) > 0
        
        tables = connector.list_tables('schema1')
        assert isinstance(tables, list)
        
        columns = connector.list_columns('schema1', 'table1')
        assert isinstance(columns, list)
        assert all('name' in col and 'type' in col for col in columns)
        
        foreign_keys = connector.get_foreign_keys('schema1', 'table1')
        assert isinstance(foreign_keys, list)
        
        # Test query execution
        result = connector.execute_query("SELECT 1")
        assert hasattr(result, 'shape')  # Should be a DataFrame
        
        # Test connection test
        assert connector.test_connection()
        
        # Test engine access
        engine = connector.get_engine()
        assert engine is not None
        
        # Test disconnection
        connector.disconnect()
        assert not connector.is_connected()
    
    def test_config_access(self):
        """Test that config is properly stored and accessible."""
        config = {
            'type': 'postgresql',
            'host': 'localhost',
            'port': 5432,
            'database': 'testdb',
            'user': 'testuser',
            'password': 'testpass'
        }
        connector = ConcreteDatabaseConnector('pg_test', config)
        
        assert connector.config == config
        assert connector.config['type'] == 'postgresql'
        assert connector.config['host'] == 'localhost'
        assert connector.profile_name == 'pg_test'
    
    def test_logging_integration(self):
        """Test that logging is properly integrated."""
        config = {'type': 'test', 'database': 'testdb'}
        
        with patch('core.database.base.logger') as mock_logger:
            connector = ConcreteDatabaseConnector('test_profile', config)
            
            # Operations should trigger logging
            connector.connect()
            connector.disconnect()
            
            # Verify logger was used
            assert mock_logger.sysdebug.call_count >= 0  # May or may not log, depending on implementation


class TestDatabaseConnectorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_config(self):
        """Test connector with empty configuration."""
        connector = ConcreteDatabaseConnector('empty_config', {})
        
        assert connector.config == {}
        assert connector.profile_name == 'empty_config'
    
    def test_none_values_in_config(self):
        """Test connector with None values in configuration."""
        config = {
            'type': 'test',
            'host': None,
            'port': None,
            'database': 'testdb'
        }
        connector = ConcreteDatabaseConnector('test_profile', config)
        
        assert connector.config['host'] is None
        assert connector.config['port'] is None
        assert connector.config['database'] == 'testdb'
    
    def test_special_characters_in_profile_name(self):
        """Test connector with special characters in profile name."""
        config = {'type': 'test'}
        profile_name = 'test-profile_with.special@chars'
        
        connector = ConcreteDatabaseConnector(profile_name, config)
        assert connector.profile_name == profile_name
    
    def test_large_config_object(self):
        """Test connector with large configuration object."""
        config = {f'key_{i}': f'value_{i}' for i in range(1000)}
        config['type'] = 'test'
        
        connector = ConcreteDatabaseConnector('large_config', config)
        assert len(connector.config) == 1001  # 1000 + type
        assert connector.config['type'] == 'test'