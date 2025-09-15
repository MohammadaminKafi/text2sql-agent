"""
Database connection factory and manager.
"""
from typing import Dict, Any, Optional, Type, Union, List
from core.database.base import DatabaseConnector, ConnectionError
from core.database.mssql import MSSQLConnector
from core.database.snowflake import SnowflakeConnector
from core.database.sqlite import SQLiteConnector
from core.config.database_config import get_db_config, DatabaseConfig
from core.smartlog import get_logger

logger = get_logger(__name__)


class DatabaseFactory:
    """Factory for creating database connectors based on configuration profiles."""
    
    # Registry of connector classes by database type
    _connector_registry: Dict[str, Type[DatabaseConnector]] = {
        'mssql': MSSQLConnector,
        'snowflake': SnowflakeConnector,
        'sqlite': SQLiteConnector,
    }
    
    @classmethod
    def register_connector(cls, db_type: str, connector_class: Type[DatabaseConnector]) -> None:
        """
        Register a new connector class for a database type.
        
        Args:
            db_type: Database type identifier
            connector_class: Connector class to register
        """
        cls._connector_registry[db_type] = connector_class
        logger.sysdebug(f"Registered connector for database type: {db_type}")
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available database types."""
        return list(cls._connector_registry.keys())
    
    @classmethod
    def create_connector(cls, profile_name: str, 
                        config: Optional[DatabaseConfig] = None) -> DatabaseConnector:
        """
        Create a database connector based on a configuration profile.
        
        Args:
            profile_name: Name of the configuration profile
            config: Optional database configuration instance. If None, uses global config.
            
        Returns:
            Configured database connector instance
            
        Raises:
            ConnectionError: If profile not found or unsupported database type
        """
        if config is None:
            config = get_db_config()
        
        try:
            profile_config = config.get_profile(profile_name)
        except Exception as e:
            raise ConnectionError(f"Failed to get profile '{profile_name}': {e}")
        
        db_type = profile_config.get('type')
        if not db_type:
            raise ConnectionError(f"Profile '{profile_name}' missing 'type' field")
        
        connector_class = cls._connector_registry.get(db_type)
        if not connector_class:
            available_types = list(cls._connector_registry.keys())
            raise ConnectionError(
                f"Unsupported database type: {db_type}. "
                f"Available types: {available_types}"
            )
        
        try:
            connector = connector_class(profile_name, profile_config)
            logger.sysdebug(f"Created {db_type} connector for profile '{profile_name}'")
            return connector
        except Exception as e:
            raise ConnectionError(f"Failed to create connector for profile '{profile_name}': {e}")
    
    @classmethod
    def create_default_connector(cls, config: Optional[DatabaseConfig] = None) -> DatabaseConnector:
        """
        Create a connector using the default profile.
        
        Args:
            config: Optional database configuration instance
            
        Returns:
            Configured database connector instance
        """
        if config is None:
            config = get_db_config()
        
        try:
            profile_config = config.get_default_profile()
            default_profile_name = config._config.get('default_profile', 'mssql_docker')
            
            db_type = profile_config.get('type')
            connector_class = cls._connector_registry.get(db_type)
            
            if not connector_class:
                raise ConnectionError(f"Unsupported database type in default profile: {db_type}")
            
            connector = connector_class(default_profile_name, profile_config)
            logger.sysdebug(f"Created default {db_type} connector")
            return connector
            
        except Exception as e:
            raise ConnectionError(f"Failed to create default connector: {e}")


class DatabaseManager:
    """Manages multiple database connections and provides a unified interface."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize the database manager.
        
        Args:
            config: Optional database configuration instance
        """
        self.config = config or get_db_config()
        self._connectors: Dict[str, DatabaseConnector] = {}
        self._default_connector: Optional[DatabaseConnector] = None
    
    def get_connector(self, profile_name: str) -> DatabaseConnector:
        """
        Get a connector for the specified profile, creating it if necessary.
        
        Args:
            profile_name: Name of the configuration profile
            
        Returns:
            Database connector instance
        """
        if profile_name not in self._connectors:
            self._connectors[profile_name] = DatabaseFactory.create_connector(
                profile_name, self.config
            )
        
        return self._connectors[profile_name]
    
    def get_default_connector(self) -> DatabaseConnector:
        """
        Get the default database connector.
        
        Returns:
            Default database connector instance
        """
        if self._default_connector is None:
            self._default_connector = DatabaseFactory.create_default_connector(self.config)
        
        return self._default_connector
    
    def list_profiles(self) -> Dict[str, str]:
        """List all available database profiles."""
        return self.config.list_profiles()
    
    def test_connection(self, profile_name: str) -> bool:
        """
        Test connection to a specific profile.
        
        Args:
            profile_name: Name of the configuration profile
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            connector = self.get_connector(profile_name)
            return connector.test_connection()
        except Exception as e:
            logger.error(f"Connection test failed for profile '{profile_name}': {e}")
            return False
    
    def test_all_connections(self) -> Dict[str, bool]:
        """
        Test connections to all configured profiles.
        
        Returns:
            Dictionary mapping profile names to connection status
        """
        results = {}
        profiles = self.config.list_profiles()
        
        for profile_name in profiles:
            if not profile_name.endswith(" (alias)"):  # Skip aliases in testing
                results[profile_name] = self.test_connection(profile_name)
        
        return results
    
    def close_all_connections(self) -> None:
        """Close all open database connections."""
        for profile_name, connector in self._connectors.items():
            try:
                connector.disconnect()
                logger.sysdebug(f"Closed connection for profile: {profile_name}")
            except Exception as e:
                logger.warning(f"Error closing connection for profile '{profile_name}': {e}")
        
        if self._default_connector:
            try:
                self._default_connector.disconnect()
                logger.sysdebug("Closed default connection")
            except Exception as e:
                logger.warning(f"Error closing default connection: {e}")
        
        self._connectors.clear()
        self._default_connector = None
    
    def get_connector_info(self, profile_name: str) -> Dict[str, Any]:
        """
        Get information about a connector without connecting.
        
        Args:
            profile_name: Name of the configuration profile
            
        Returns:
            Dictionary with connector information
        """
        try:
            profile_config = self.config.get_profile(profile_name)
            return {
                'profile_name': profile_name,
                'type': profile_config.get('type'),
                'description': profile_config.get('description'),
                'connected': profile_name in self._connectors and 
                           self._connectors[profile_name].is_connected()
            }
        except Exception as e:
            return {
                'profile_name': profile_name,
                'error': str(e)
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_all_connections()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def reset_db_manager() -> DatabaseManager:
    """Reset the global database manager instance."""
    global _db_manager
    if _db_manager:
        _db_manager.close_all_connections()
    _db_manager = DatabaseManager()
    return _db_manager


# Convenience functions for common operations
def create_connector(profile_name: str) -> DatabaseConnector:
    """Create a database connector for the specified profile."""
    return DatabaseFactory.create_connector(profile_name)


def get_default_connector() -> DatabaseConnector:
    """Get the default database connector."""
    return get_db_manager().get_default_connector()


def execute_query(query: str, profile_name: Optional[str] = None, **kwargs) -> Any:
    """
    Execute a query using the specified profile or default connector.
    
    Args:
        query: SQL query to execute
        profile_name: Optional profile name. If None, uses default.
        **kwargs: Additional arguments passed to execute_query
        
    Returns:
        Query results
    """
    manager = get_db_manager()
    
    if profile_name:
        connector = manager.get_connector(profile_name)
    else:
        connector = manager.get_default_connector()
    
    return connector.execute_query(query, **kwargs)