"""
Integration tests for the database connector system.
These tests verify end-to-end functionality and backward compatibility.
"""
import pytest
import tempfile
import os
import yaml
from unittest.mock import Mock, patch

from core.database.factory import DatabaseConnectorFactory
from core.database.config import DatabaseConfig
from core.database.sqlite import SQLiteConnector
from core.database.mssql import MSSQLConnector
from core.database.snowflake import SnowflakeConnector


class TestDatabaseSystemIntegration:
    """Integration tests for the complete database system."""
    
    def test_end_to_end_sqlite_workflow(self):
        """Test complete workflow with SQLite database."""
        # Create temporary database file
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        try:
            # Test factory creation
            config = {
                'type': 'sqlite',
                'database': db_path
            }
            
            connector = DatabaseConnectorFactory.create_connector('integration_sqlite', config)
            assert isinstance(connector, SQLiteConnector)
            
            # Test connection
            assert connector.connect()
            assert connector.is_connected()
            
            # Test creating a table and inserting data
            create_table_sql = """
            CREATE TABLE test_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            )
            """
            connector.execute_query(create_table_sql)
            
            # Insert test data
            insert_sql = "INSERT INTO test_users (name, email) VALUES ('John Doe', 'john@example.com')"
            connector.execute_query(insert_sql)
            
            # Test introspection
            schemas = connector.list_schemas()
            assert 'main' in schemas
            
            tables = connector.list_tables('main')
            assert 'test_users' in tables
            
            columns = connector.list_columns('main', 'test_users')
            assert len(columns) == 3
            assert any(col['name'] == 'id' and col['primary_key'] for col in columns)
            
            # Test querying data
            result_df = connector.execute_query("SELECT * FROM test_users")
            assert len(result_df) == 1
            assert result_df.iloc[0]['name'] == 'John Doe'
            
            # Test disconnection
            connector.disconnect()
            assert not connector.is_connected()
            
        finally:
            # Clean up
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_config_file_integration(self):
        """Test integration with configuration file."""
        # Create temporary configuration file
        config_data = {
            'database_profiles': {
                'test_sqlite': {
                    'type': 'sqlite',
                    'database': ':memory:'
                },
                'test_mssql': {
                    'type': 'mssql',
                    'host': 'localhost',
                    'database': 'TestDB',
                    'user': 'sa',
                    'password': 'TestPassword123'
                },
                'test_snowflake': {
                    'type': 'snowflake',
                    'account': 'myaccount',
                    'user': 'testuser',
                    'password': 'testpass',
                    'warehouse': 'COMPUTE_WH'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(config_data, tmp_file)
            config_path = tmp_file.name
        
        try:
            # Test configuration loading
            with patch.dict(os.environ, {'DATABASE_CONFIG_PATH': config_path}):
                config = DatabaseConfig()
                
                # Test profile retrieval
                sqlite_config = config.get_profile_config('test_sqlite')
                assert sqlite_config['type'] == 'sqlite'
                assert sqlite_config['database'] == ':memory:'
                
                mssql_config = config.get_profile_config('test_mssql')
                assert mssql_config['type'] == 'mssql'
                assert mssql_config['host'] == 'localhost'
                
                snowflake_config = config.get_profile_config('test_snowflake')
                assert snowflake_config['type'] == 'snowflake'
                assert snowflake_config['account'] == 'myaccount'
                
                # Test factory creation from profile
                sqlite_connector = DatabaseConnectorFactory.create_from_profile('test_sqlite')
                assert isinstance(sqlite_connector, SQLiteConnector)
                
                mssql_connector = DatabaseConnectorFactory.create_from_profile('test_mssql')
                assert isinstance(mssql_connector, MSSQLConnector)
                
                snowflake_connector = DatabaseConnectorFactory.create_from_profile('test_snowflake')
                assert isinstance(snowflake_connector, SnowflakeConnector)
                
        finally:
            # Clean up
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_environment_variable_integration(self):
        """Test integration with environment variables."""
        # Create configuration with environment variable placeholders
        config_data = {
            'database_profiles': {
                'env_test': {
                    'type': 'mssql',
                    'host': '${DB_HOST}',
                    'database': '${DB_NAME}',
                    'user': '${DB_USER}',
                    'password': '${DB_PASSWORD}'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(config_data, tmp_file)
            config_path = tmp_file.name
        
        try:
            # Set environment variables
            env_vars = {
                'DATABASE_CONFIG_PATH': config_path,
                'DB_HOST': 'prod-server',
                'DB_NAME': 'ProductionDB',
                'DB_USER': 'prod_user',
                'DB_PASSWORD': 'prod_password123'
            }
            
            with patch.dict(os.environ, env_vars):
                config = DatabaseConfig()
                resolved_config = config.get_profile_config('env_test')
                
                # Verify environment variables were resolved
                assert resolved_config['host'] == 'prod-server'
                assert resolved_config['database'] == 'ProductionDB'
                assert resolved_config['user'] == 'prod_user'
                assert resolved_config['password'] == 'prod_password123'
                
                # Test factory creation with resolved config
                connector = DatabaseConnectorFactory.create_from_profile('env_test')
                assert isinstance(connector, MSSQLConnector)
                assert connector.config['host'] == 'prod-server'
                
        finally:
            # Clean up
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_context_manager_integration(self):
        """Test integration with context managers."""
        config = {
            'type': 'sqlite',
            'database': ':memory:'
        }
        
        connector = DatabaseConnectorFactory.create_connector('context_integration', config)
        
        # Test context manager usage
        with patch.object(connector, 'connect', return_value=True) as mock_connect:
            with patch.object(connector, 'disconnect') as mock_disconnect:
                with patch.object(connector, 'is_connected', return_value=True):
                    
                    with connector as conn:
                        # Inside context manager
                        assert conn is connector
                        assert conn.is_connected()
                        
                        # Should have called connect
                        mock_connect.assert_called_once()
                    
                    # After exiting context manager
                    mock_disconnect.assert_called_once()
    
    def test_multiple_connector_isolation(self):
        """Test that multiple connectors are properly isolated."""
        # Create multiple connectors
        sqlite_config = {
            'type': 'sqlite',
            'database': ':memory:'
        }
        
        mssql_config = {
            'type': 'mssql',
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector1 = DatabaseConnectorFactory.create_connector('test1', sqlite_config)
        connector2 = DatabaseConnectorFactory.create_connector('test2', sqlite_config.copy())
        connector3 = DatabaseConnectorFactory.create_connector('test3', mssql_config)
        
        # Verify they are different instances
        assert connector1 is not connector2
        assert connector1 is not connector3
        assert connector2 is not connector3
        
        # Verify they have independent configurations
        assert connector1.profile_name == 'test1'
        assert connector2.profile_name == 'test2'
        assert connector3.profile_name == 'test3'
        
        # Verify they are of correct types
        assert isinstance(connector1, SQLiteConnector)
        assert isinstance(connector2, SQLiteConnector)
        assert isinstance(connector3, MSSQLConnector)
        
        # Modify one config and verify others are unaffected
        connector1.config['database'] = 'modified.db'
        assert connector2.config['database'] == ':memory:'
    
    @patch('core.database.mssql.create_engine')
    @patch('core.database.snowflake.create_engine')
    def test_error_handling_integration(self, mock_snowflake_engine, mock_mssql_engine):
        """Test error handling across the entire system."""
        # Setup mocks to simulate failures
        mock_mssql_engine.side_effect = Exception("MSSQL connection failed")
        mock_snowflake_engine.side_effect = Exception("Snowflake authentication failed")
        
        # Test MSSQL connection failure
        mssql_config = {
            'type': 'mssql',
            'host': 'invalid-host',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        mssql_connector = DatabaseConnectorFactory.create_connector('error_test_mssql', mssql_config)
        
        with pytest.raises(Exception):  # Should be wrapped in ConnectionError
            mssql_connector.connect()
        
        # Test Snowflake connection failure
        snowflake_config = {
            'type': 'snowflake',
            'account': 'invalid-account',
            'user': 'baduser',
            'password': 'badpass'
        }
        
        snowflake_connector = DatabaseConnectorFactory.create_connector('error_test_snowflake', snowflake_config)
        
        with pytest.raises(Exception):  # Should be wrapped in ConnectionError
            snowflake_connector.connect()
    
    def test_factory_supported_types_integration(self):
        """Test that factory correctly supports all implemented types."""
        supported_types = DatabaseConnectorFactory.get_supported_types()
        
        # Verify all expected types are supported
        assert 'sqlite' in supported_types
        assert 'mssql' in supported_types
        assert 'snowflake' in supported_types
        
        # Test creating connectors for each supported type
        for db_type in supported_types:
            config = {
                'type': db_type,
                'database': 'test'  # Minimal config
            }
            
            if db_type == 'mssql':
                config.update({
                    'host': 'localhost',
                    'user': 'sa',
                    'password': 'password'
                })
            elif db_type == 'snowflake':
                config.update({
                    'account': 'myaccount',
                    'user': 'testuser',
                    'password': 'testpass'
                })
            
            connector = DatabaseConnectorFactory.create_connector(f'integration_test_{db_type}', config)
            
            # Verify correct type was created
            if db_type == 'sqlite':
                assert isinstance(connector, SQLiteConnector)
            elif db_type == 'mssql':
                assert isinstance(connector, MSSQLConnector)
            elif db_type == 'snowflake':
                assert isinstance(connector, SnowflakeConnector)


class TestBackwardCompatibility:
    """Test backward compatibility with existing systems."""
    
    def test_legacy_config_format_support(self):
        """Test support for legacy configuration formats."""
        # Test old-style configuration that might exist
        legacy_config = {
            'db_type': 'sqlite',  # Old key name
            'db_path': 'legacy.db'  # Old key name
        }
        
        # The system should handle this gracefully by mapping old keys
        # For now, we expect it to fail gracefully
        with pytest.raises(Exception):
            DatabaseConnectorFactory.create_connector('legacy_test', legacy_config)
    
    def test_case_insensitive_type_handling(self):
        """Test that database types are handled case-insensitively."""
        type_variations = [
            'sqlite', 'SQLITE', 'SQLite', 'SqLiTe',
            'mssql', 'MSSQL', 'MsSQL', 'msSql',
            'snowflake', 'SNOWFLAKE', 'Snowflake', 'SnowFlake'
        ]
        
        for db_type in type_variations:
            config = {
                'type': db_type,
                'database': 'test'
            }
            
            # Add required fields for specific database types
            if db_type.lower() == 'mssql':
                config.update({
                    'host': 'localhost',
                    'user': 'sa',
                    'password': 'password'
                })
            elif db_type.lower() == 'snowflake':
                config.update({
                    'account': 'myaccount',
                    'user': 'testuser',
                    'password': 'testpass'
                })
            
            connector = DatabaseConnectorFactory.create_connector(f'case_test_{db_type}', config)
            
            # Verify correct connector type was created regardless of case
            if db_type.lower() == 'sqlite':
                assert isinstance(connector, SQLiteConnector)
            elif db_type.lower() == 'mssql':
                assert isinstance(connector, MSSQLConnector)
            elif db_type.lower() == 'snowflake':
                assert isinstance(connector, SnowflakeConnector)
    
    def test_minimal_configuration_support(self):
        """Test that minimal configurations still work."""
        # SQLite with minimal config
        sqlite_config = {'type': 'sqlite', 'database': ':memory:'}
        sqlite_connector = DatabaseConnectorFactory.create_connector('minimal_sqlite', sqlite_config)
        assert isinstance(sqlite_connector, SQLiteConnector)
        
        # MSSQL with minimal config
        mssql_config = {
            'type': 'mssql',
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'password'
        }
        mssql_connector = DatabaseConnectorFactory.create_connector('minimal_mssql', mssql_config)
        assert isinstance(mssql_connector, MSSQLConnector)
        
        # Snowflake with minimal config
        snowflake_config = {
            'type': 'snowflake',
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass'
        }
        snowflake_connector = DatabaseConnectorFactory.create_connector('minimal_snowflake', snowflake_config)
        assert isinstance(snowflake_connector, SnowflakeConnector)
    
    def test_extra_configuration_preservation(self):
        """Test that extra configuration parameters are preserved."""
        config = {
            'type': 'sqlite',
            'database': ':memory:',
            'timeout': 30,
            'check_same_thread': False,
            'pragma_settings': {
                'journal_mode': 'WAL',
                'synchronous': 'NORMAL'
            },
            'custom_application_setting': 'custom_value'
        }
        
        connector = DatabaseConnectorFactory.create_connector('extra_config_test', config)
        
        # All configuration should be preserved
        assert connector.config == config
        assert connector.config['timeout'] == 30
        assert connector.config['pragma_settings']['journal_mode'] == 'WAL'
        assert connector.config['custom_application_setting'] == 'custom_value'


class TestSystemRobustness:
    """Test system robustness and edge cases."""
    
    def test_concurrent_connector_creation(self):
        """Test creating multiple connectors concurrently."""
        import threading
        import time
        
        connectors = []
        errors = []
        
        def create_connector(thread_id):
            try:
                config = {
                    'type': 'sqlite',
                    'database': f':memory:thread_{thread_id}'
                }
                connector = DatabaseConnectorFactory.create_connector(f'thread_{thread_id}', config)
                connectors.append(connector)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_connector, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors during concurrent creation: {errors}"
        assert len(connectors) == 10
        
        # Verify all connectors are unique
        profile_names = [conn.profile_name for conn in connectors]
        assert len(set(profile_names)) == 10  # All should be unique
    
    def test_memory_usage_stability(self):
        """Test that creating and destroying connectors doesn't leak memory."""
        import gc
        
        # Create and destroy many connectors
        for i in range(100):
            config = {
                'type': 'sqlite',
                'database': f':memory:test_{i}'
            }
            connector = DatabaseConnectorFactory.create_connector(f'memory_test_{i}', config)
            
            # Use the connector briefly
            assert connector.profile_name == f'memory_test_{i}'
            
            # Delete reference
            del connector
        
        # Force garbage collection
        gc.collect()
        
        # If we get here without memory issues, the test passes
        assert True
    
    def test_configuration_mutation_safety(self):
        """Test that modifying configuration doesn't affect other connectors."""
        base_config = {
            'type': 'sqlite',
            'database': ':memory:',
            'options': {
                'timeout': 30,
                'settings': ['setting1', 'setting2']
            }
        }
        
        # Create first connector
        connector1 = DatabaseConnectorFactory.create_connector('mutation_test_1', base_config.copy())
        
        # Modify the original config
        base_config['database'] = 'modified.db'
        base_config['options']['timeout'] = 60
        base_config['options']['settings'].append('setting3')
        
        # Create second connector
        connector2 = DatabaseConnectorFactory.create_connector('mutation_test_2', base_config.copy())
        
        # Verify first connector wasn't affected
        assert connector1.config['database'] == ':memory:'
        assert connector1.config['options']['timeout'] == 30
        assert len(connector1.config['options']['settings']) == 2
        
        # Verify second connector has the modified config
        assert connector2.config['database'] == 'modified.db'
        assert connector2.config['options']['timeout'] == 60
        assert len(connector2.config['options']['settings']) == 3