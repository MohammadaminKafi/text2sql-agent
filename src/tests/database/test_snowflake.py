"""
Tests for Snowflake database connector.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import urllib.parse

from core.database.snowflake import SnowflakeConnector
from core.database.base import ConnectionError, QueryError


class TestSnowflakeConnector:
    """Test the Snowflake database connector."""
    
    def test_account_auth_initialization(self):
        """Test initialization with account-based authentication."""
        config = {
            'type': 'snowflake',
            'account': 'myaccount.region',
            'user': 'myuser',
            'password': 'mypassword',
            'warehouse': 'COMPUTE_WH',
            'database': 'MYDATABASE',
            'schema': 'PUBLIC',
            'role': 'MYROLE'
        }
        
        connector = SnowflakeConnector('snowflake_test', config)
        
        assert connector.profile_name == 'snowflake_test'
        assert connector.config == config
        assert not connector.is_connected()
    
    def test_minimal_initialization(self):
        """Test initialization with minimal required configuration."""
        config = {
            'type': 'snowflake',
            'account': 'myaccount',
            'user': 'myuser',
            'password': 'mypassword'
        }
        
        connector = SnowflakeConnector('snowflake_minimal', config)
        
        assert connector.profile_name == 'snowflake_minimal'
        assert connector.config == config
        assert not connector.is_connected()
    
    def test_build_connection_string_full_config(self):
        """Test building connection string with full configuration."""
        config = {
            'account': 'myaccount.us-east-1',
            'user': 'testuser',
            'password': 'testpass',
            'warehouse': 'TEST_WH',
            'database': 'TEST_DB',
            'schema': 'TEST_SCHEMA',
            'role': 'TEST_ROLE'
        }
        
        connector = SnowflakeConnector('test', config)
        conn_str = connector._build_connection_string()
        
        # Parse the URL to check components
        parsed = urllib.parse.urlparse(conn_str)
        query_params = urllib.parse.parse_qs(parsed.query)
        
        assert parsed.scheme == 'snowflake'
        assert parsed.username == 'testuser'
        assert parsed.password == 'testpass'
        assert parsed.hostname == 'myaccount.us-east-1.snowflakecomputing.com'
        assert parsed.path == '/TEST_DB/TEST_SCHEMA'
        assert query_params['warehouse'] == ['TEST_WH']
        assert query_params['role'] == ['TEST_ROLE']
    
    def test_build_connection_string_minimal_config(self):
        """Test building connection string with minimal configuration."""
        config = {
            'account': 'simple-account',
            'user': 'simpleuser',
            'password': 'simplepass'
        }
        
        connector = SnowflakeConnector('test', config)
        conn_str = connector._build_connection_string()
        
        # Parse the URL to check components
        parsed = urllib.parse.urlparse(conn_str)
        
        assert parsed.scheme == 'snowflake'
        assert parsed.username == 'simpleuser'
        assert parsed.password == 'simplepass'
        assert parsed.hostname == 'simple-account.snowflakecomputing.com'
        assert parsed.path == '/'  # No database or schema
        assert not parsed.query  # No additional parameters
    
    def test_build_connection_string_account_formatting(self):
        """Test proper formatting of account names."""
        test_cases = [
            ('myaccount', 'myaccount.snowflakecomputing.com'),
            ('myaccount.region', 'myaccount.region.snowflakecomputing.com'),
            ('myaccount.region.aws', 'myaccount.region.aws.snowflakecomputing.com'),
        ]
        
        for account, expected_host in test_cases:
            config = {
                'account': account,
                'user': 'testuser',
                'password': 'testpass'
            }
            
            connector = SnowflakeConnector('test', config)
            conn_str = connector._build_connection_string()
            parsed = urllib.parse.urlparse(conn_str)
            
            assert parsed.hostname == expected_host
    
    @patch('core.database.snowflake.create_engine')
    def test_connect_success(self, mock_create_engine):
        """Test successful database connection."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        mock_connection.execute.return_value.scalar.return_value = 1
        mock_create_engine.return_value = mock_engine
        
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass',
            'warehouse': 'TEST_WH'
        }
        
        connector = SnowflakeConnector('test', config)
        
        assert connector.connect()
        assert connector.is_connected()
        assert connector.engine is mock_engine
        
        # Verify connection string
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0][0]
        assert call_args.startswith('snowflake://testuser:testpass@')
    
    @patch('core.database.snowflake.create_engine')
    def test_connect_failure(self, mock_create_engine):
        """Test connection failure."""
        mock_create_engine.side_effect = Exception("Authentication failed")
        
        config = {
            'account': 'invalid-account',
            'user': 'baduser',
            'password': 'badpass'
        }
        
        connector = SnowflakeConnector('test', config)
        
        with pytest.raises(ConnectionError):
            connector.connect()
        
        assert not connector.is_connected()
        assert connector.engine is None
    
    def test_disconnect(self):
        """Test disconnecting from database."""
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass'
        }
        
        connector = SnowflakeConnector('test', config)
        
        # Mock connected state
        mock_engine = Mock()
        connector.engine = mock_engine
        connector._connected = True
        
        connector.disconnect()
        
        assert not connector.is_connected()
        assert connector.engine is None
        mock_engine.dispose.assert_called_once()
    
    @patch('core.database.snowflake.create_engine')
    def test_test_connection_success(self, mock_create_engine):
        """Test successful connection test."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        mock_connection.execute.return_value.scalar.return_value = 1
        mock_create_engine.return_value = mock_engine
        
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass'
        }
        
        connector = SnowflakeConnector('test', config)
        
        assert connector.test_connection()
    
    @patch('core.database.snowflake.create_engine')
    def test_test_connection_failure(self, mock_create_engine):
        """Test connection test failure."""
        mock_create_engine.side_effect = Exception("Connection failed")
        
        config = {
            'account': 'invalid-account',
            'user': 'baduser',
            'password': 'badpass'
        }
        
        connector = SnowflakeConnector('test', config)
        
        assert not connector.test_connection()
    
    @patch('core.database.snowflake.create_engine')
    @patch('core.database.snowflake.inspect')
    def test_list_schemas_success(self, mock_inspect, mock_create_engine):
        """Test listing database schemas."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_inspector = Mock()
        mock_inspector.get_schema_names.return_value = ['PUBLIC', 'INFORMATION_SCHEMA', 'MYSCHEMA']
        mock_inspect.return_value = mock_inspector
        
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass',
            'database': 'MYDATABASE'
        }
        
        connector = SnowflakeConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        schemas = connector.list_schemas()
        
        assert schemas == ['PUBLIC', 'INFORMATION_SCHEMA', 'MYSCHEMA']
        mock_inspect.assert_called_once_with(mock_engine)
        mock_inspector.get_schema_names.assert_called_once()
    
    @patch('core.database.snowflake.create_engine')
    @patch('core.database.snowflake.inspect')
    def test_list_schemas_fallback_query(self, mock_inspect, mock_create_engine):
        """Test schema listing fallback using INFORMATION_SCHEMA."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_inspector = Mock()
        mock_inspector.get_schema_names.side_effect = Exception("Access denied")
        mock_inspect.return_value = mock_inspector
        
        # Mock connection for fallback query
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.__iter__.return_value = [('SCHEMA1',), ('SCHEMA2',)]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass',
            'database': 'MYDATABASE'
        }
        
        connector = SnowflakeConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        schemas = connector.list_schemas()
        
        assert schemas == ['SCHEMA1', 'SCHEMA2']
    
    @patch('core.database.snowflake.create_engine')
    @patch('core.database.snowflake.inspect')
    def test_list_tables_success(self, mock_inspect, mock_create_engine):
        """Test listing tables in a schema."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ['CUSTOMERS', 'ORDERS', 'PRODUCTS']
        mock_inspect.return_value = mock_inspector
        
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass',
            'database': 'MYDATABASE'
        }
        
        connector = SnowflakeConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        tables = connector.list_tables('PUBLIC')
        
        assert tables == ['CUSTOMERS', 'ORDERS', 'PRODUCTS']
        mock_inspector.get_table_names.assert_called_once_with(schema='PUBLIC')
    
    @patch('core.database.snowflake.create_engine')
    @patch('core.database.snowflake.inspect')
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
        mock_result.__iter__.return_value = [('TABLE1',), ('TABLE2',)]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass',
            'database': 'MYDATABASE'
        }
        
        connector = SnowflakeConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        tables = connector.list_tables('PUBLIC')
        
        assert tables == ['TABLE1', 'TABLE2']
    
    @patch('core.database.snowflake.create_engine')
    @patch('core.database.snowflake.inspect')
    def test_list_columns_success(self, mock_inspect, mock_create_engine):
        """Test listing columns in a table."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_inspector = Mock()
        mock_inspector.get_columns.return_value = [
            {'name': 'ID', 'type': 'NUMBER(38,0)', 'nullable': False, 'autoincrement': True},
            {'name': 'NAME', 'type': 'VARCHAR(100)', 'nullable': False, 'autoincrement': False},
            {'name': 'EMAIL', 'type': 'VARCHAR(255)', 'nullable': True, 'autoincrement': False}
        ]
        mock_inspector.get_pk_constraint.return_value = {'constrained_columns': ['ID']}
        mock_inspect.return_value = mock_inspector
        
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass',
            'database': 'MYDATABASE'
        }
        
        connector = SnowflakeConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        columns = connector.list_columns('PUBLIC', 'CUSTOMERS')
        
        assert len(columns) == 3
        
        # Check first column (primary key)
        id_col = columns[0]
        assert id_col['name'] == 'ID'
        assert id_col['type'] == 'NUMBER(38,0)'
        assert id_col['nullable'] is False
        assert id_col['primary_key'] is True
        
        # Check second column (not primary key)
        name_col = columns[1]
        assert name_col['name'] == 'NAME'
        assert name_col['primary_key'] is False
    
    @patch('core.database.snowflake.create_engine')
    @patch('core.database.snowflake.inspect')
    def test_get_foreign_keys_success(self, mock_inspect, mock_create_engine):
        """Test getting foreign keys for a table."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_inspector = Mock()
        mock_inspector.get_foreign_keys.return_value = [
            {
                'name': 'FK_ORDERS_CUSTOMER_ID',
                'constrained_columns': ['CUSTOMER_ID'],
                'referred_schema': 'PUBLIC',
                'referred_table': 'CUSTOMERS',
                'referred_columns': ['ID']
            }
        ]
        mock_inspect.return_value = mock_inspector
        
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass',
            'database': 'MYDATABASE'
        }
        
        connector = SnowflakeConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        foreign_keys = connector.get_foreign_keys('PUBLIC', 'ORDERS')
        
        assert len(foreign_keys) == 1
        
        fk = foreign_keys[0]
        assert fk['constraint_name'] == 'FK_ORDERS_CUSTOMER_ID'
        assert fk['column_name'] == 'CUSTOMER_ID'
        assert fk['referenced_schema'] == 'PUBLIC'
        assert fk['referenced_table'] == 'CUSTOMERS'
        assert fk['referenced_column'] == 'ID'
    
    @patch('core.database.snowflake.create_engine')
    def test_execute_query_success(self, mock_create_engine):
        """Test successful query execution."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [(1, 'John'), (2, 'Jane')]
        mock_result.keys.return_value = ['ID', 'NAME']
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass',
            'database': 'MYDATABASE'
        }
        
        connector = SnowflakeConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        with patch('pandas.DataFrame') as mock_df:
            mock_df_instance = Mock()
            mock_df.return_value = mock_df_instance
            
            result = connector.execute_query("SELECT * FROM CUSTOMERS")
            
            assert result is mock_df_instance
            mock_df.assert_called_once_with([(1, 'John'), (2, 'Jane')], columns=['ID', 'NAME'])
    
    @patch('core.database.snowflake.create_engine')
    def test_execute_query_failure(self, mock_create_engine):
        """Test query execution failure."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_connection = Mock()
        mock_connection.execute.side_effect = Exception("SQL compilation error")
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass',
            'database': 'MYDATABASE'
        }
        
        connector = SnowflakeConnector('test', config)
        connector.engine = mock_engine
        connector._connected = True
        
        with pytest.raises(QueryError):
            connector.execute_query("INVALID SQL")
    
    def test_context_manager(self):
        """Test using connector as context manager."""
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass'
        }
        
        connector = SnowflakeConnector('context_test', config)
        
        with patch.object(connector, 'connect', return_value=True):
            with patch.object(connector, 'disconnect'):
                with patch.object(connector, 'is_connected', return_value=True):
                    with connector as conn:
                        assert conn is connector
                        assert conn.is_connected()


class TestSnowflakeConnectorEdgeCases:
    """Test edge cases and error conditions for Snowflake connector."""
    
    def test_missing_required_config(self):
        """Test initialization with missing required configuration."""
        config = {
            'type': 'snowflake'
            # Missing account, user, password
        }
        
        connector = SnowflakeConnector('incomplete_config', config)
        
        # Should not raise exception during initialization
        assert connector.profile_name == 'incomplete_config'
        assert connector.config == config
    
    def test_special_characters_in_password(self):
        """Test connection string building with special characters in password."""
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'P@ssw0rd!@#$%^&*()'
        }
        
        connector = SnowflakeConnector('special_pass', config)
        conn_str = connector._build_connection_string()
        
        # Password should be properly URL-encoded
        parsed = urllib.parse.urlparse(conn_str)
        assert parsed.password == 'P@ssw0rd!@#$%^&*()'
    
    def test_unicode_characters_in_config(self):
        """Test connection string building with Unicode characters."""
        config = {
            'account': 'myaccount',
            'user': 'Usér',
            'password': 'Pässwörd',
            'database': 'DATÄBASE',
            'schema': 'SCHÉMÄ'
        }
        
        connector = SnowflakeConnector('unicode_test', config)
        conn_str = connector._build_connection_string()
        
        # Unicode characters should be properly URL-encoded
        parsed = urllib.parse.urlparse(conn_str)
        assert parsed.username == 'Usér'
        assert parsed.password == 'Pässwörd'
        assert 'DATÄBASE/SCHÉMÄ' in parsed.path
    
    def test_empty_string_values(self):
        """Test handling of empty string values in configuration."""
        config = {
            'account': '',
            'user': '',
            'password': '',
            'warehouse': '',
            'database': '',
            'schema': ''
        }
        
        connector = SnowflakeConnector('empty_strings', config)
        conn_str = connector._build_connection_string()
        
        # Should handle empty values gracefully
        parsed = urllib.parse.urlparse(conn_str)
        assert parsed.scheme == 'snowflake'
        assert parsed.hostname == '.snowflakecomputing.com'
    
    def test_case_sensitivity_handling(self):
        """Test proper case handling for Snowflake identifiers."""
        config = {
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass',
            'warehouse': 'my_warehouse',  # lowercase
            'database': 'My_Database',     # mixed case
            'schema': 'MY_SCHEMA',        # uppercase
            'role': 'my_role'             # lowercase
        }
        
        connector = SnowflakeConnector('case_test', config)
        conn_str = connector._build_connection_string()
        
        # Check that identifiers are preserved as-is
        parsed = urllib.parse.urlparse(conn_str)
        query_params = urllib.parse.parse_qs(parsed.query)
        
        assert query_params['warehouse'] == ['my_warehouse']
        assert query_params['role'] == ['my_role']
        assert 'My_Database/MY_SCHEMA' in parsed.path
    
    def test_very_long_connection_string(self):
        """Test handling of very long connection strings."""
        config = {
            'account': 'very-long-account-' + 'x' * 100,
            'user': 'very-long-username-' + 'x' * 100,
            'password': 'very-long-password-' + 'x' * 100,
            'warehouse': 'VERY_LONG_WAREHOUSE_' + 'X' * 100,
            'database': 'VERY_LONG_DATABASE_' + 'X' * 100,
            'schema': 'VERY_LONG_SCHEMA_' + 'X' * 100
        }
        
        connector = SnowflakeConnector('long_strings', config)
        conn_str = connector._build_connection_string()
        
        # Should handle long strings without crashing
        assert isinstance(conn_str, str)
        assert len(conn_str) > 1000  # Should be quite long
    
    @patch('core.database.snowflake.create_engine')
    def test_authentication_error_handling(self, mock_create_engine):
        """Test handling of various authentication errors."""
        error_messages = [
            "Authentication failed",
            "Incorrect username or password",
            "Account does not exist",
            "User has been locked",
            "Password has expired"
        ]
        
        config = {
            'account': 'badaccount',
            'user': 'baduser',
            'password': 'badpass'
        }
        
        for error_msg in error_messages:
            mock_create_engine.side_effect = Exception(error_msg)
            connector = SnowflakeConnector('auth_error_test', config)
            
            with pytest.raises(ConnectionError):
                connector.connect()
    
    def test_warehouse_parameter_handling(self):
        """Test various warehouse parameter configurations."""
        test_cases = [
            ({'warehouse': 'COMPUTE_WH'}, 'COMPUTE_WH'),
            ({'warehouse': 'compute_wh'}, 'compute_wh'),
            ({'warehouse': 'MY-WAREHOUSE'}, 'MY-WAREHOUSE'),
            ({'warehouse': 'WH_123'}, 'WH_123'),
        ]
        
        for config_warehouse, expected_warehouse in test_cases:
            config = {
                'account': 'myaccount',
                'user': 'testuser',
                'password': 'testpass',
                **config_warehouse
            }
            
            connector = SnowflakeConnector('warehouse_test', config)
            conn_str = connector._build_connection_string()
            
            parsed = urllib.parse.urlparse(conn_str)
            query_params = urllib.parse.parse_qs(parsed.query)
            
            assert query_params['warehouse'] == [expected_warehouse]
    
    def test_role_parameter_handling(self):
        """Test various role parameter configurations."""
        test_cases = [
            ({'role': 'ACCOUNTADMIN'}, 'ACCOUNTADMIN'),
            ({'role': 'sysadmin'}, 'sysadmin'),
            ({'role': 'MY_CUSTOM_ROLE'}, 'MY_CUSTOM_ROLE'),
            ({'role': 'role-with-dashes'}, 'role-with-dashes'),
        ]
        
        for config_role, expected_role in test_cases:
            config = {
                'account': 'myaccount',
                'user': 'testuser',
                'password': 'testpass',
                **config_role
            }
            
            connector = SnowflakeConnector('role_test', config)
            conn_str = connector._build_connection_string()
            
            parsed = urllib.parse.urlparse(conn_str)
            query_params = urllib.parse.parse_qs(parsed.query)
            
            assert query_params['role'] == [expected_role]