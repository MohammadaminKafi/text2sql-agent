"""
Database configuration loader and environment variable resolver.
"""
import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Initialize smartlog if not already done
from core.smartlog import init_logging
try:
    init_logging(console=False)  # Initialize without console to avoid conflicts
except Exception:
    pass  # Already initialized

from core.smartlog import get_logger

logger = get_logger(__name__)


class DatabaseConfig:
    """Handles loading and resolving database configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the database configuration loader.
        
        Args:
            config_path: Path to the configuration file. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "database_config.yaml"
        
        self.config_path = Path(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load the configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            logger.sysdebug(f"Loaded database configuration from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Database configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing database configuration: {e}")
            raise
    
    def _resolve_env_vars(self, value: Any) -> Any:
        """
        Recursively resolve environment variables in configuration values.
        
        Supports formats:
        - ${VAR_NAME} - required variable
        - ${VAR_NAME:-default_value} - optional with default
        """
        if isinstance(value, str):
            # Pattern to match ${VAR_NAME} or ${VAR_NAME:-default}
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else None
                
                env_value = os.getenv(var_name)
                if env_value is not None:
                    return env_value
                elif default_value is not None:
                    return default_value
                else:
                    raise EnvironmentError(f"Required environment variable not set: {var_name}")
            
            return re.sub(pattern, replacer, value)
        
        elif isinstance(value, dict):
            return {k: self._resolve_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_env_vars(item) for item in value]
        else:
            return value
    
    def get_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Get a database profile configuration with environment variables resolved.
        
        Args:
            profile_name: Name of the profile or alias
            
        Returns:
            Resolved configuration dictionary
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        
        # Check if it's an alias
        aliases = self._config.get('aliases', {})
        if profile_name in aliases:
            profile_name = aliases[profile_name]
        
        # Get the profile
        profiles = self._config.get('database_profiles', {})
        if profile_name not in profiles:
            available = list(profiles.keys()) + list(aliases.keys())
            raise ValueError(f"Profile '{profile_name}' not found. Available: {available}")
        
        profile = profiles[profile_name].copy()
        
        # Resolve environment variables
        try:
            profile = self._resolve_env_vars(profile)
            logger.sysdebug(f"Resolved profile '{profile_name}': {profile.get('description', 'No description')}")
            return profile
        except EnvironmentError as e:
            logger.error(f"Failed to resolve environment variables for profile '{profile_name}': {e}")
            raise
    
    def get_default_profile(self) -> Dict[str, Any]:
        """Get the default database profile."""
        default_name = self._config.get('default_profile', 'mssql_docker')
        return self.get_profile(default_name)
    
    def list_profiles(self) -> Dict[str, str]:
        """
        List all available profiles with their descriptions.
        
        Returns:
            Dictionary mapping profile names to descriptions
        """
        if not self._config:
            return {}
        
        profiles = {}
        
        # Add main profiles
        for name, config in self._config.get('database_profiles', {}).items():
            profiles[name] = config.get('description', 'No description')
        
        # Add aliases
        for alias, target in self._config.get('aliases', {}).items():
            profiles[f"{alias} (alias)"] = f"Alias for {target}"
        
        return profiles


# Global configuration instance
_db_config = None


def get_db_config() -> DatabaseConfig:
    """Get the global database configuration instance."""
    global _db_config
    if _db_config is None:
        _db_config = DatabaseConfig()
    return _db_config


def reload_db_config(config_path: Optional[str] = None) -> DatabaseConfig:
    """Reload the database configuration."""
    global _db_config
    _db_config = DatabaseConfig(config_path)
    return _db_config