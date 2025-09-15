#!/usr/bin/env python3
"""
Simple validation script for the database connector system.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from core.database.factory import DatabaseConnectorFactory
        print("✓ DatabaseConnectorFactory imported")
        
        from core.database.config import DatabaseConfig
        print("✓ DatabaseConfig imported")
        
        from core.database.sqlite import SQLiteConnector
        print("✓ SQLiteConnector imported")
        
        from core.database.mssql import MSSQLConnector
        print("✓ MSSQLConnector imported")
        
        from core.database.snowflake import SnowflakeConnector
        print("✓ SnowflakeConnector imported")
        
        from core.database.base import DatabaseConnector, DatabaseException, ConnectionError, QueryError
        print("✓ Base classes imported")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_factory():
    """Test the factory functionality."""
    print("\nTesting factory functionality...")
    
    try:
        from core.database.factory import DatabaseConnectorFactory
        from core.database.sqlite import SQLiteConnector
        from core.database.mssql import MSSQLConnector
        from core.database.snowflake import SnowflakeConnector
        
        # Test supported types
        supported_types = DatabaseConnectorFactory.get_supported_types()
        print(f"✓ Supported types: {supported_types}")
        
        # Test SQLite connector creation
        sqlite_config = {
            'type': 'sqlite',
            'database': ':memory:'
        }
        
        sqlite_connector = DatabaseConnectorFactory.create_connector('test_sqlite', sqlite_config)
        if isinstance(sqlite_connector, SQLiteConnector):
            print("✓ SQLite connector created successfully")
        else:
            print("✗ SQLite connector has wrong type")
            return False
        
        # Test MSSQL connector creation
        mssql_config = {
            'type': 'mssql',
            'host': 'localhost',
            'database': 'TestDB',
            'user': 'sa',
            'password': 'password'
        }
        
        mssql_connector = DatabaseConnectorFactory.create_connector('test_mssql', mssql_config)
        if isinstance(mssql_connector, MSSQLConnector):
            print("✓ MSSQL connector created successfully")
        else:
            print("✗ MSSQL connector has wrong type")
            return False
        
        # Test Snowflake connector creation
        snowflake_config = {
            'type': 'snowflake',
            'account': 'myaccount',
            'user': 'testuser',
            'password': 'testpass'
        }
        
        snowflake_connector = DatabaseConnectorFactory.create_connector('test_snowflake', snowflake_config)
        if isinstance(snowflake_connector, SnowflakeConnector):
            print("✓ Snowflake connector created successfully")
        else:
            print("✗ Snowflake connector has wrong type")
            return False
        
        # Test unsupported type
        try:
            DatabaseConnectorFactory.create_connector('test_unsupported', {'type': 'postgresql'})
            print("✗ Should have failed for unsupported type")
            return False
        except Exception:
            print("✓ Correctly rejected unsupported type")
        
        return True
        
    except Exception as e:
        print(f"✗ Factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sqlite_basic():
    """Test basic SQLite functionality."""
    print("\nTesting SQLite basic functionality...")
    
    try:
        from core.database.factory import DatabaseConnectorFactory
        
        config = {
            'type': 'sqlite',
            'database': ':memory:'
        }
        
        connector = DatabaseConnectorFactory.create_connector('test_sqlite', config)
        
        # Test connection
        if connector.connect():
            print("✓ SQLite connection successful")
        else:
            print("✗ SQLite connection failed")
            return False
        
        # Test basic query
        connector.execute_query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        connector.execute_query("INSERT INTO test (name) VALUES ('Test Name')")
        result = connector.execute_query("SELECT * FROM test")
        
        if len(result) == 1 and result.iloc[0]['name'] == 'Test Name':
            print("✓ SQLite basic operations successful")
        else:
            print("✗ SQLite basic operations failed")
            return False
        
        # Test introspection
        schemas = connector.list_schemas()
        tables = connector.list_tables('main')
        columns = connector.list_columns('main', 'test')
        
        if 'main' in schemas and 'test' in tables and len(columns) >= 2:
            print("✓ SQLite introspection successful")
        else:
            print("✗ SQLite introspection failed")
            return False
        
        connector.disconnect()
        print("✓ SQLite disconnection successful")
        
        return True
        
    except Exception as e:
        print(f"✗ SQLite test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("Database Connector System Validation")
    print("=" * 40)
    
    tests = [
        ("Module Imports", test_imports),
        ("Factory Functionality", test_factory),
        ("SQLite Basic Operations", test_sqlite_basic),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 40)
    print("Validation Results:")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All validations passed! Database system is working correctly.")
        return 0
    else:
        print("\n❌ Some validations failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)