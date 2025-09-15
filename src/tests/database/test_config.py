"""
Tests for database configuration management.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from core.config.database_config import DatabaseConfig, get_db_config


class TestDatabaseConfig:
    """Test the DatabaseConfig class."""
    
    def test_load_valid_config(self, temp_database_config):
        """Test loading a valid configuration file."""
        config = temp_database_config
        
        assert config._config is not None
        assert 'profiles' in config._config
        assert 'default_profile' in config._config
        assert config._config['default_profile'] == 'test_sqlite'
    
    def test_get_profile_exists(self, temp_database_config):
        """Test getting an existing profile."""
        config = temp_database_config
        
        profile = config.get_profile('test_sqlite')
        assert profile['type'] == 'sqlite'
        assert profile['database'] == ':memory:'
        assert 'description' in profile
    
    def test_get_profile_not_exists(self, temp_database_config):
        """Test getting a non-existent profile raises exception."""
        config = temp_database_config
        
        with pytest.raises(ValueError, match="Profile 'nonexistent' not found"):
            config.get_profile('nonexistent')
    
    def test_get_default_profile(self, temp_database_config):
        """Test getting the default profile."""
        config = temp_database_config
        
        default_profile = config.get_default_profile()
        assert default_profile['type'] == 'sqlite'
        assert default_profile['database'] == ':memory:'
    
    def test_list_profiles(self, temp_database_config):
        """Test listing all profiles."""
        config = temp_database_config
        
        profiles = config.list_profiles()
        assert isinstance(profiles, dict)
        assert 'test_sqlite' in profiles
        assert 'test_mssql' in profiles
        assert 'test_snowflake' in profiles
        
        # Check descriptions are included
        assert profiles['test_sqlite'] == 'Test SQLite in-memory database'
    
    def test_resolve_env_vars_with_default(self, temp_database_config):
        """Test environment variable resolution with default values."""
        config = temp_database_config
        
        # Test with non-existent env var (should use default)
        result = config._resolve_env_vars("${NONEXISTENT_VAR:-default_value}")
        assert result == "default_value"
    
    def test_resolve_env_vars_existing(self, temp_database_config):
        """Test environment variable resolution with existing env var."""
        config = temp_database_config
        
        with patch.dict(os.environ, {'TEST_VAR': 'actual_value'}):
            result = config._resolve_env_vars("${TEST_VAR:-default_value}")
            assert result == "actual_value"
    
    def test_resolve_env_vars_no_default(self, temp_database_config):
        """Test environment variable resolution without default."""
        config = temp_database_config
        
        with patch.dict(os.environ, {'TEST_VAR': 'env_value'}):
            result = config._resolve_env_vars("${TEST_VAR}")
            assert result == "env_value"
    
    def test_resolve_env_vars_complex_string(self, temp_database_config):
        """Test environment variable resolution in complex strings."""
        config = temp_database_config
        
        with patch.dict(os.environ, {'DB_HOST': 'localhost', 'DB_PORT': '5432'}):
            result = config._resolve_env_vars("postgresql://${DB_HOST:-127.0.0.1}:${DB_PORT:-5432}/mydb")
            assert result == "postgresql://localhost:5432/mydb"
    
    def test_missing_config_file(self):
        """Test behavior when config file doesn't exist."""
        with patch.object(DatabaseConfig, '_get_config_path', return_value=Path('/nonexistent/config.yaml')):
            with pytest.raises(FileNotFoundError):
                DatabaseConfig()
    
    def test_invalid_yaml_config(self):
        """Test behavior with invalid YAML content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            with patch.object(DatabaseConfig, '_get_config_path', return_value=Path(config_path)):
                with pytest.raises(Exception):  # Should raise YAML parsing error
                    DatabaseConfig()
        finally:
            os.unlink(config_path)
    
    def test_config_without_default_profile(self):
        """Test configuration without default_profile specified."""
        config_data = {
            'profiles': {
                'test_db': {
                    'type': 'sqlite',
                    'database': ':memory:'
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
                # Should use first profile as default
                default = config.get_default_profile()
                assert default['type'] == 'sqlite'
        finally:
            os.unlink(config_path)


class TestDatabaseConfigGlobal:
    """Test global database configuration functions."""
    
    def test_get_db_config_singleton(self):
        """Test that get_db_config returns the same instance."""
        with patch.object(DatabaseConfig, '__init__', return_value=None):
            with patch.object(DatabaseConfig, '_config', {'test': 'config'}):
                config1 = get_db_config()
                config2 = get_db_config()
                assert config1 is config2
    
    def test_get_db_config_initialization(self):
        """Test that get_db_config initializes the configuration."""
        # Reset the global config
        import core.config.database_config
        core.config.database_config._db_config = None
        
        with patch.object(DatabaseConfig, '__init__', return_value=None) as mock_init:
            with patch.object(DatabaseConfig, '_config', {'initialized': True}):
                config = get_db_config()
                mock_init.assert_called_once()
                assert hasattr(config, '_config')


class TestEnvironmentVariableResolution:
    """Test environment variable resolution edge cases."""
    
    def test_nested_env_vars(self, temp_database_config):
        """Test nested environment variable references."""
        config = temp_database_config
        
        with patch.dict(os.environ, {'OUTER_VAR': '${INNER_VAR:-inner_default}', 'INNER_VAR': 'inner_value'}):
            result = config._resolve_env_vars("${OUTER_VAR}")
            # Should resolve to the outer var value, not recursively resolve inner
            assert result == "${INNER_VAR:-inner_default}"
    
    def test_multiple_env_vars_same_string(self, temp_database_config):
        """Test multiple environment variables in the same string."""
        config = temp_database_config
        
        with patch.dict(os.environ, {'VAR1': 'value1', 'VAR2': 'value2'}):
            result = config._resolve_env_vars("${VAR1}-${VAR2}-${VAR3:-default3}")
            assert result == "value1-value2-default3"
    
    def test_env_var_with_special_characters(self, temp_database_config):
        """Test environment variables with special characters."""
        config = temp_database_config
        
        with patch.dict(os.environ, {'SPECIAL_VAR': 'value!@#$%^&*()'}):
            result = config._resolve_env_vars("${SPECIAL_VAR}")
            assert result == "value!@#$%^&*()"
    
    def test_empty_env_var(self, temp_database_config):
        """Test empty environment variable."""
        config = temp_database_config
        
        with patch.dict(os.environ, {'EMPTY_VAR': ''}):
            result = config._resolve_env_vars("${EMPTY_VAR:-default}")
            # Empty string should be used, not the default
            assert result == ""
    
    def test_invalid_env_var_syntax(self, temp_database_config):
        """Test invalid environment variable syntax."""
        config = temp_database_config
        
        # Missing closing brace
        result = config._resolve_env_vars("${INVALID_VAR")
        assert result == "${INVALID_VAR"  # Should remain unchanged
        
        # Missing opening brace
        result = config._resolve_env_vars("INVALID_VAR}")
        assert result == "INVALID_VAR}"  # Should remain unchanged