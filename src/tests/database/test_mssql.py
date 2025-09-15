"""
Tests for MSSQL database connector.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
import urllib.parse

from core.database.mssql import MSSQLConnector
from core.database.base import ConnectionError, QueryError


class TestMSSQLConnector:
    """Test the MSSQL database connector."""
    
    def test_sql_auth_initialization(self):
        """Test initialization with SQL authentication."""
        config = {
            'type': 'mssql',
            'host': 'localhost',
            'port': 1433,
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123',
            'driver': 'ODBC Driver 18 for SQL Server',
            'encrypt': 'yes',
            'trust_server_certificate': 'yes'
        }
        
        connector = MSSQLConnector('mssql_test', config)
        
        assert connector.profile_name == 'mssql_test'
        assert connector.config == config
        assert not connector.is_connected()
    
    def test_windows_auth_initialization(self):
        """Test initialization with Windows authentication."""
        config = {
            'type': 'mssql',
            'host': 'localhost',
            'database': 'AdventureWorks2022',
            'driver': 'ODBC Driver 17 for SQL Server',
            'trusted_connection': 'yes'
        }
        
        connector = MSSQLConnector('mssql_windows', config)
        
        assert connector.profile_name == 'mssql_windows'
        assert connector.config == config
        assert not connector.is_connected()
    
    def test_build_connection_string_sql_auth(self):
        """Test building connection string for SQL authentication."""
        config = {
            'host': 'testserver',
            'port': 1433,
            'database': 'TestDB',
            'user': 'testuser',
            'password': 'testpass',
            'driver': 'ODBC Driver 18 for SQL Server',
            'encrypt': 'yes',
            'trust_server_certificate': 'yes'
        }
        
        connector = MSSQLConnector('test', config)
        conn_str = connector._build_connection_string()
        
        assert 'DRIVER={ODBC Driver 18 for SQL Server}' in conn_str
        assert 'SERVER=testserver,1433' in conn_str
        assert 'DATABASE=TestDB' in conn_str
        assert 'UID=testuser' in conn_str
        assert 'PWD=testpass' in conn_str
        assert 'Encrypt=yes' in conn_str
        assert 'TrustServerCertificate=yes' in conn_str
    
    def test_build_connection_string_windows_auth(self):
        """Test building connection string for Windows authentication."""
        config = {
            'host': 'localhost',
            'database': 'AdventureWorks2022',
            'driver': 'ODBC Driver 17 for SQL Server',
            'trusted_connection': 'yes'
        }
        
        connector = MSSQLConnector('test', config)
        conn_str = connector._build_connection_string()
        
        assert 'DRIVER={ODBC Driver 17 for SQL Server}' in conn_str
        assert 'SERVER=localhost,1433' in conn_str  # Default port
        assert 'DATABASE=AdventureWorks2022' in conn_str
        assert 'Trusted_Connection=yes' in conn_str
        assert 'UID=' not in conn_str
        assert 'PWD=' not in conn_str
    
    def test_build_connection_string_default_values(self):
        """Test building connection string with default values."""
        config = {
            'host': 'myserver',
            'database': 'MyDB',
            'user': 'myuser',
            'password': 'mypass'
        }
        
        connector = MSSQLConnector('test', config)
        conn_str = connector._build_connection_string()
        
        # Should use defaults
        assert 'DRIVER={ODBC Driver 18 for SQL Server}' in conn_str
        assert 'SERVER=myserver,1433' in conn_str
        assert 'Encrypt=yes' in conn_str
        assert 'TrustServerCertificate=yes' in conn_str
    
    @patch('core.database.mssql.create_engine')
    def test_connect_success(self, mock_create_engine):
        """Test successful database connection."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        mock_connection.execute.return_value.scalar.return_value = 1
        mock_create_engine.return_value = mock_engine
        
        config = {
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        
        assert connector.connect()
        assert connector.is_connected()
        assert connector.engine is mock_engine
        
        # Verify connection string was URL-encoded
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0][0]
        assert call_args.startswith('mssql+pyodbc:///?odbc_connect=')
    
    @patch('core.database.mssql.create_engine')
    def test_connect_fallback_to_master(self, mock_create_engine):
        """Test fallback to master database when target DB fails."""
        # First call fails, second call succeeds
        mock_engine1 = Mock()
        mock_engine1.connect.side_effect = Exception("Cannot open database")
        
        mock_engine2 = Mock()
        mock_connection2 = Mock()
        mock_engine2.connect.return_value.__enter__.return_value = mock_connection2
        mock_engine2.connect.return_value.__exit__.return_value = None
        mock_connection2.execute.return_value.scalar.return_value = 1
        
        mock_create_engine.side_effect = [mock_engine1, mock_engine2]
        
        config = {
            'host': 'localhost',
            'database': 'NonExistentDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        
        with patch('core.database.mssql.logger') as mock_logger:
            assert connector.connect()
            
            # Should have logged the fallback
            mock_logger.warning.assert_called()
            mock_logger.info.assert_called()
        
        assert connector.is_connected()
        assert connector.engine is mock_engine2
        
        # Should have tried to connect twice
        assert mock_create_engine.call_count == 2
        
        # Second call should be to master database
        second_call_args = mock_create_engine.call_args_list[1][0][0]
        decoded_conn_str = urllib.parse.unquote_plus(second_call_args.split('odbc_connect=')[1])
        assert 'DATABASE=master' in decoded_conn_str
    
    @patch('core.database.mssql.create_engine')
    def test_connect_failure_both_attempts(self, mock_create_engine):
        """Test connection failure for both target and master database."""
        mock_create_engine.side_effect = [
            Exception("Cannot open database"),
            Exception("Cannot connect to master")
        ]
        
        config = {
            'host': 'invalid-server',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        
        with pytest.raises(ConnectionError):
            connector.connect()
        
        assert not connector.is_connected()
        assert connector.engine is None
    
    def test_disconnect(self):
        """Test disconnecting from database."""
        config = {
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        
        # Mock connected state
        mock_engine = Mock()
        connector.engine = mock_engine
        connector._connected = True
        
        connector.disconnect()
        
        assert not connector.is_connected()
        assert connector.engine is None
        mock_engine.dispose.assert_called_once()
    
    @patch('core.database.mssql.create_engine')
    def test_test_connection_success(self, mock_create_engine):
        """Test successful connection test."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        mock_connection.execute.return_value.scalar.return_value = 1
        mock_create_engine.return_value = mock_engine
        
        config = {
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        
        assert connector.test_connection()
    
    @patch('core.database.mssql.create_engine')
    def test_test_connection_failure(self, mock_create_engine):
        """Test connection test failure."""
        mock_create_engine.side_effect = Exception("Connection failed")
        
        config = {
            'host': 'invalid-server',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        
        assert not connector.test_connection()
    
    @patch('core.database.mssql.create_engine')
    @patch('core.database.mssql.inspect')
    def test_list_schemas_success(self, mock_inspect, mock_create_engine):
        """Test listing database schemas."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_inspector = Mock()
        mock_inspector.get_schema_names.return_value = ['dbo', 'sales', 'production']
        mock_inspect.return_value = mock_inspector
        
        config = {
            'host': 'localhost',
            'database': 'AdventureWorks2022',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        schemas = connector.list_schemas()
        
        assert schemas == ['dbo', 'sales', 'production']
        mock_inspect.assert_called_once_with(mock_engine)
        mock_inspector.get_schema_names.assert_called_once()
    
    @patch('core.database.mssql.create_engine')
    @patch('core.database.mssql.inspect')
    def test_list_schemas_fallback_adventureworks(self, mock_inspect, mock_create_engine):
        """Test schema listing fallback for AdventureWorks."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_inspector = Mock()
        mock_inspector.get_schema_names.side_effect = Exception("Access denied")
        mock_inspect.return_value = mock_inspector
        
        # Mock connection for database name check
        mock_connection = Mock()
        mock_connection.execute.return_value.scalar.return_value = "AdventureWorks2022"
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        
        config = {
            'host': 'localhost',
            'database': 'AdventureWorks2022',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        schemas = connector.list_schemas()
        
        expected_schemas = ["dbo", "HumanResources", "Person", "Production", "Purchasing", "Sales"]
        assert schemas == expected_schemas
    
    @patch('core.database.mssql.create_engine')
    @patch('core.database.mssql.inspect')
    def test_list_tables_success(self, mock_inspect, mock_create_engine):
        """Test listing tables in a schema."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ['users', 'orders', 'products']
        mock_inspect.return_value = mock_inspector
        
        config = {
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        tables = connector.list_tables('dbo')
        
        assert tables == ['users', 'orders', 'products']
        mock_inspector.get_table_names.assert_called_once_with(schema='dbo')
    
    @patch('core.database.mssql.create_engine')
    @patch('core.database.mssql.inspect')
    def test_list_tables_fallback_query(self, mock_inspect, mock_create_engine):
        """Test table listing fallback using INFORMATION_SCHEMA."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.side_effect = Exception("Access denied")
        mock_inspect.return_value = mock_inspector
        
        # Mock connection for fallback query
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.__iter__.return_value = [('table1',), ('table2',)]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        
        config = {
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        tables = connector.list_tables('dbo')
        
        assert tables == ['table1', 'table2']
    
    @patch('core.database.mssql.create_engine')
    @patch('core.database.mssql.inspect')
    def test_list_columns_success(self, mock_inspect, mock_create_engine):
        """Test listing columns in a table."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_inspector = Mock()
        mock_inspector.get_columns.return_value = [
            {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'autoincrement': True},
            {'name': 'username', 'type': 'VARCHAR(50)', 'nullable': False, 'autoincrement': False},
            {'name': 'email', 'type': 'VARCHAR(100)', 'nullable': True, 'autoincrement': False}
        ]
        mock_inspector.get_pk_constraint.return_value = {'constrained_columns': ['id']}
        mock_inspect.return_value = mock_inspector
        
        config = {
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        columns = connector.list_columns('dbo', 'users')
        
        assert len(columns) == 3
        
        # Check first column (primary key)
        id_col = columns[0]
        assert id_col['name'] == 'id'
        assert id_col['type'] == 'INTEGER'
        assert id_col['nullable'] is False
        assert id_col['primary_key'] is True
        
        # Check second column (not primary key)
        username_col = columns[1]
        assert username_col['name'] == 'username'
        assert username_col['primary_key'] is False
    
    @patch('core.database.mssql.create_engine')
    @patch('core.database.mssql.inspect')
    def test_get_foreign_keys_success(self, mock_inspect, mock_create_engine):
        """Test getting foreign keys for a table."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_inspector = Mock()
        mock_inspector.get_foreign_keys.return_value = [
            {
                'name': 'fk_orders_user_id',
                'constrained_columns': ['user_id'],
                'referred_schema': 'dbo',
                'referred_table': 'users',
                'referred_columns': ['id']
            }
        ]
        mock_inspect.return_value = mock_inspector
        
        config = {
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        foreign_keys = connector.get_foreign_keys('dbo', 'orders')
        
        assert len(foreign_keys) == 1
        
        fk = foreign_keys[0]
        assert fk['constraint_name'] == 'fk_orders_user_id'
        assert fk['column_name'] == 'user_id'
        assert fk['referenced_schema'] == 'dbo'
        assert fk['referenced_table'] == 'users'
        assert fk['referenced_column'] == 'id'
    
    @patch('core.database.mssql.create_engine')
    def test_execute_query_success(self, mock_create_engine):
        """Test successful query execution."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [(1, 'test'), (2, 'test2')]
        mock_result.keys.return_value = ['id', 'name']
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        
        config = {
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        with patch('pandas.DataFrame') as mock_df:
            mock_df_instance = Mock()
            mock_df.return_value = mock_df_instance
            
            result = connector.execute_query("SELECT * FROM users")
            
            assert result is mock_df_instance
            mock_df.assert_called_once_with([(1, 'test'), (2, 'test2')], columns=['id', 'name'])
    
    @patch('core.database.mssql.create_engine')
    def test_execute_query_failure(self, mock_create_engine):
        """Test query execution failure."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_connection = Mock()
        mock_connection.execute.side_effect = Exception("SQL syntax error")
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        
        config = {
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        with pytest.raises(QueryError):
            connector.execute_query("INVALID SQL")
    
    def test_context_manager(self):
        """Test using connector as context manager."""
        config = {
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('context_test', config)
        
        with patch.object(connector, 'connect', return_value=True):
            with patch.object(connector, 'disconnect'):
                with patch.object(connector, 'is_connected', return_value=True):
                    with connector as conn:
                        assert conn is connector
                        assert conn.is_connected()


class TestMSSQLConnectorEdgeCases:
    """Test edge cases and error conditions for MSSQL connector."""
    
    def test_missing_required_config(self):
        """Test initialization with missing required configuration."""
        config = {
            'type': 'mssql'
            # Missing host, database, etc.
        }
        
        connector = MSSQLConnector('incomplete_config', config)
        
        # Should not raise exception during initialization
        assert connector.profile_name == 'incomplete_config'
        assert connector.config == config
    
    def test_special_characters_in_password(self):
        """Test connection string building with special characters in password."""
        config = {
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'P@ssw0rd!@#$%^&*()',
            'port': 1433
        }
        
        connector = MSSQLConnector('special_pass', config)
        conn_str = connector._build_connection_string()
        
        # Password should be properly included
        assert 'PWD=P@ssw0rd!@#$%^&*()' in conn_str
    
    def test_unicode_characters_in_config(self):
        """Test connection string building with Unicode characters."""
        config = {
            'host': 'localhost',
            'database': 'TestDB_Ñoñó',
            'user': 'Usér',
            'password': 'Pässwörd'
        }
        
        connector = MSSQLConnector('unicode_test', config)
        conn_str = connector._build_connection_string()
        
        # Unicode characters should be preserved
        assert 'DATABASE=TestDB_Ñoñó' in conn_str
        assert 'UID=Usér' in conn_str
        assert 'PWD=Pässwörd' in conn_str
    
    def test_empty_string_values(self):
        """Test handling of empty string values in configuration."""
        config = {
            'host': '',
            'database': '',
            'user': '',
            'password': ''
        }
        
        connector = MSSQLConnector('empty_strings', config)
        conn_str = connector._build_connection_string()
        
        # Empty values should be handled
        assert 'SERVER=,1433' in conn_str
        assert 'DATABASE=' in conn_str
        assert 'UID=' in conn_str
        assert 'PWD=' in conn_str
    
    def test_very_long_connection_string(self):
        """Test handling of very long connection strings."""
        config = {
            'host': 'very-long-hostname-' + 'x' * 100,
            'database': 'very-long-database-name-' + 'x' * 100,
            'user': 'very-long-username-' + 'x' * 100,
            'password': 'very-long-password-' + 'x' * 100
        }
        
        connector = MSSQLConnector('long_strings', config)
        conn_str = connector._build_connection_string()
        
        # Should handle long strings without crashing
        assert isinstance(conn_str, str)
        assert len(conn_str) > 500  # Should be quite long
    
    @patch('core.database.mssql.create_engine')
    def test_connection_timeout_handling(self, mock_create_engine):
        """Test handling of connection timeouts."""
        mock_create_engine.side_effect = Exception("Timeout expired")
        
        config = {
            'host': 'slow-server',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('timeout_test', config)
        
        with pytest.raises(ConnectionError):
            connector.connect()
    
    def test_port_as_string(self):
        """Test handling of port specified as string."""
        config = {
            'host': 'localhost',
            'port': '1433',  # String instead of int
            'database': 'TestDB',
            'user': 'sa',
            'password': 'TestPassword123'
        }
        
        connector = MSSQLConnector('string_port', config)
        conn_str = connector._build_connection_string()
        
        assert 'SERVER=localhost,1433' in conn_str