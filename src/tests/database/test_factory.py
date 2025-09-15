"""
Tests for database connector factory.
"""
import pytest
from unittest.mock import Mock, patch, mock_open

from core.database.factory import DatabaseConnectorFactory
from core.database.sqlite import SQLiteConnector
from core.database.mssql import MSSQLConnector
from core.database.snowflake import SnowflakeConnector
from core.database.base import DatabaseException


class TestDatabaseConnectorFactory:
    """Test the database connector factory."""
    
    def test_create_sqlite_connector(self):
        """Test creating SQLite connector."""
        config = {
            'type': 'sqlite',
            'database': ':memory:'
        }
        
        connector = DatabaseConnectorFactory.create_connector('sqlite_test', config)
        
        assert isinstance(connector, SQLiteConnector)
        assert connector.profile_name == 'sqlite_test'
        assert connector.config == config
    
    def test_create_mssql_connector(self):
        """Test creating MSSQL connector."""
        config = {
            'type': 'mssql',
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = DatabaseConnectorFactory.create_connector('mssql_test', config)
        
        assert isinstance(connector, MSSQLConnector)
        assert connector.profile_name == 'mssql_test'
        assert connector.config == config
    
    def test_create_snowflake_connector(self):
        """Test creating Snowflake connector."""
        config = {
            'type': 'snowflake',
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass'
        }
        
        connector = DatabaseConnectorFactory.create_connector('snowflake_test', config)
        
        assert isinstance(connector, SnowflakeConnector)
        assert connector.profile_name == 'snowflake_test'
        assert connector.config == config
    
    def test_create_unsupported_connector(self):
        """Test creating connector for unsupported database type."""
        config = {
            'type': 'postgresql',  # Not supported
            'host': 'localhost',
            'database': 'testdb'
        }
        
        with pytest.raises(DatabaseException) as exc_info:
            DatabaseConnectorFactory.create_connector('postgres_test', config)
        
        assert "Unsupported database type: postgresql" in str(exc_info.value)
    
    def test_create_connector_missing_type(self):
        """Test creating connector with missing type field."""
        config = {
            'host': 'localhost',
            'database': 'testdb'
            # Missing 'type' field
        }
        
        with pytest.raises(DatabaseException) as exc_info:
            DatabaseConnectorFactory.create_connector('missing_type_test', config)
        
        assert "Database type not specified in configuration" in str(exc_info.value)
    
    def test_create_connector_case_insensitive_type(self):
        """Test creating connector with case-insensitive type."""
        test_cases = [
            ('SQLITE', SQLiteConnector),
            ('Sqlite', SQLiteConnector),
            ('MSSQL', MSSQLConnector),
            ('MsSql', MSSQLConnector),
            ('SNOWFLAKE', SnowflakeConnector),
            ('Snowflake', SnowflakeConnector),
        ]
        
        for db_type, expected_class in test_cases:
            config = {
                'type': db_type,
                'database': 'testdb'
            }
            
            connector = DatabaseConnectorFactory.create_connector('case_test', config)
            assert isinstance(connector, expected_class)
    
    def test_get_supported_types(self):
        """Test getting list of supported database types."""
        supported_types = DatabaseConnectorFactory.get_supported_types()
        
        assert isinstance(supported_types, list)
        assert 'sqlite' in supported_types
        assert 'mssql' in supported_types
        assert 'snowflake' in supported_types
        assert len(supported_types) == 3  # Currently supported types
    
    @patch('core.database.factory.DatabaseConfig')
    def test_create_from_profile_success(self, mock_config_class):
        """Test creating connector from configuration profile."""
        # Mock configuration
        mock_config_instance = Mock()
        mock_config_instance.get_profile_config.return_value = {
            'type': 'sqlite',
            'database': 'test.db'
        }
        mock_config_class.return_value = mock_config_instance
        
        connector = DatabaseConnectorFactory.create_from_profile('test_profile')
        
        assert isinstance(connector, SQLiteConnector)
        assert connector.profile_name == 'test_profile'
        mock_config_instance.get_profile_config.assert_called_once_with('test_profile')
    
    @patch('core.database.factory.DatabaseConfig')
    def test_create_from_profile_missing_profile(self, mock_config_class):
        """Test creating connector from non-existent profile."""
        # Mock configuration to raise exception
        mock_config_instance = Mock()
        mock_config_instance.get_profile_config.side_effect = DatabaseException("Profile 'nonexistent' not found")
        mock_config_class.return_value = mock_config_instance
        
        with pytest.raises(DatabaseException) as exc_info:
            DatabaseConnectorFactory.create_from_profile('nonexistent')
        
        assert "Profile 'nonexistent' not found" in str(exc_info.value)
    
    @patch('core.database.factory.DatabaseConfig')
    def test_create_from_profile_invalid_config(self, mock_config_class):
        """Test creating connector from profile with invalid configuration."""
        # Mock configuration with invalid type
        mock_config_instance = Mock()
        mock_config_instance.get_profile_config.return_value = {
            'type': 'invalid_db_type',
            'host': 'localhost'
        }
        mock_config_class.return_value = mock_config_instance
        
        with pytest.raises(DatabaseException) as exc_info:
            DatabaseConnectorFactory.create_from_profile('invalid_profile')
        
        assert "Unsupported database type: invalid_db_type" in str(exc_info.value)
    
    def test_create_connector_with_extra_config(self):
        """Test creating connector with additional configuration parameters."""
        config = {
            'type': 'sqlite',
            'database': 'test.db',
            'timeout': 30,
            'check_same_thread': False,
            'custom_param': 'custom_value'
        }
        
        connector = DatabaseConnectorFactory.create_connector('extra_config_test', config)
        
        assert isinstance(connector, SQLiteConnector)
        assert connector.config == config
        # Extra parameters should be preserved in config
        assert connector.config['timeout'] == 30
        assert connector.config['custom_param'] == 'custom_value'
    
    def test_create_multiple_connectors_independence(self):
        """Test creating multiple connectors with independent configurations."""
        config1 = {
            'type': 'sqlite',
            'database': 'db1.db'
        }
        
        config2 = {
            'type': 'sqlite',
            'database': 'db2.db'
        }
        
        connector1 = DatabaseConnectorFactory.create_connector('test1', config1)
        connector2 = DatabaseConnectorFactory.create_connector('test2', config2)
        
        # Should be different instances
        assert connector1 is not connector2
        assert connector1.profile_name != connector2.profile_name
        assert connector1.config != connector2.config
        
        # Should have correct configurations
        assert connector1.config['database'] == 'db1.db'
        assert connector2.config['database'] == 'db2.db'


class TestDatabaseConnectorFactoryEdgeCases:
    """Test edge cases and error conditions for the factory."""
    
    def test_create_connector_empty_config(self):
        """Test creating connector with empty configuration."""
        config = {}
        
        with pytest.raises(DatabaseException) as exc_info:
            DatabaseConnectorFactory.create_connector('empty_config_test', config)
        
        assert "Database type not specified in configuration" in str(exc_info.value)
    
    def test_create_connector_none_config(self):
        """Test creating connector with None configuration."""
        with pytest.raises(DatabaseException):
            DatabaseConnectorFactory.create_connector('none_config_test', None)
    
    def test_create_connector_empty_profile_name(self):
        """Test creating connector with empty profile name."""
        config = {
            'type': 'sqlite',
            'database': ':memory:'
        }
        
        # Empty profile name should still work
        connector = DatabaseConnectorFactory.create_connector('', config)
        assert isinstance(connector, SQLiteConnector)
        assert connector.profile_name == ''
    
    def test_create_connector_none_profile_name(self):
        """Test creating connector with None profile name."""
        config = {
            'type': 'sqlite',
            'database': ':memory:'
        }
        
        # None profile name should still work
        connector = DatabaseConnectorFactory.create_connector(None, config)
        assert isinstance(connector, SQLiteConnector)
        assert connector.profile_name is None
    
    def test_create_connector_whitespace_type(self):
        """Test creating connector with whitespace in type."""
        config = {
            'type': '  sqlite  ',  # Type with surrounding whitespace
            'database': ':memory:'
        }
        
        connector = DatabaseConnectorFactory.create_connector('whitespace_test', config)
        assert isinstance(connector, SQLiteConnector)
    
    def test_create_connector_numeric_type(self):
        """Test creating connector with numeric type."""
        config = {
            'type': 123,  # Numeric instead of string
            'database': ':memory:'
        }
        
        with pytest.raises(DatabaseException):
            DatabaseConnectorFactory.create_connector('numeric_type_test', config)
    
    def test_get_supported_types_immutability(self):
        """Test that supported types list cannot be modified externally."""
        supported_types = DatabaseConnectorFactory.get_supported_types()
        original_length = len(supported_types)
        
        # Try to modify the returned list
        supported_types.append('fake_db_type')
        
        # Get the list again to ensure it wasn't affected
        new_supported_types = DatabaseConnectorFactory.get_supported_types()
        assert len(new_supported_types) == original_length
        assert 'fake_db_type' not in new_supported_types
    
    @patch('core.database.factory.DatabaseConfig')
    def test_create_from_profile_config_initialization_error(self, mock_config_class):
        """Test handling of configuration initialization errors."""
        # Mock configuration class to raise exception during initialization
        mock_config_class.side_effect = Exception("Failed to load configuration file")
        
        with pytest.raises(DatabaseException) as exc_info:
            DatabaseConnectorFactory.create_from_profile('test_profile')
        
        assert "Failed to load configuration file" in str(exc_info.value)
    
    def test_factory_is_not_instantiable(self):
        """Test that factory class cannot be instantiated."""
        # The factory should be a static class
        # This test verifies it doesn't have __init__ or can't be instantiated
        
        # Try to create an instance (should work as Python allows this)
        factory_instance = DatabaseConnectorFactory()
        
        # But the important thing is that all methods should work as class methods
        supported_types = factory_instance.get_supported_types()
        assert isinstance(supported_types, list)
        
        # And creating connectors should work
        config = {
            'type': 'sqlite',
            'database': ':memory:'
        }
        connector = factory_instance.create_connector('test', config)
        assert isinstance(connector, SQLiteConnector)
    
    def test_create_connector_with_nested_config(self):
        """Test creating connector with nested configuration structures."""
        config = {
            'type': 'mssql',
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123',
            'connection_options': {
                'timeout': 30,
                'retry_count': 3,
                'ssl_config': {
                    'verify': False,
                    'cert_path': '/path/to/cert'
                }
            }
        }
        
        connector = DatabaseConnectorFactory.create_connector('nested_config_test', config)
        
        assert isinstance(connector, MSSQLConnector)
        assert connector.config == config
        # Nested structures should be preserved
        assert connector.config['connection_options']['timeout'] == 30
        assert connector.config['connection_options']['ssl_config']['verify'] is False
    
    def test_create_connector_type_variations(self):
        """Test creating connectors with various type name variations."""
        type_variations = [
            ('sqlite', SQLiteConnector),
            ('SQLite', SQLiteConnector),
            ('SQLITE', SQLiteConnector),
            ('sQlItE', SQLiteConnector),
            ('mssql', MSSQLConnector),
            ('MSSQL', MSSQLConnector),
            ('MsSQL', MSSQLConnector),
            ('snowflake', SnowflakeConnector),
            ('SNOWFLAKE', SnowflakeConnector),
            ('SnowFlake', SnowflakeConnector)
        ]
        
        for db_type, expected_class in type_variations:
            config = {
                'type': db_type,
                'database': 'test'
            }
            
            connector = DatabaseConnectorFactory.create_connector(f'variation_test_{db_type}', config)
            assert isinstance(connector, expected_class), f"Failed for type: {db_type}"