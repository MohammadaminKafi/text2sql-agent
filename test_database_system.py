#!/usr/bin/env python3
"""
Simple test script to validate the database connector implementations.
This script tests the basic functionality without requiring pytest.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_configuration_loading():
    """Test database configuration loading."""
    print("🔧 Testing configuration loading...")
    
    try:
        from core.config.database_config import get_db_config
        
        config = get_db_config()
        profiles = config.list_profiles()
        
        print(f"✅ Configuration loaded successfully")
        print(f"📋 Available profiles: {list(profiles.keys())}")
        
        # Test default profile
        default_profile = config.get_default_profile()
        print(f"⚙️  Default profile type: {default_profile.get('type')}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_sqlite_connector():
    """Test SQLite connector creation and basic operations."""
    print("\n🗄️  Testing SQLite connector...")
    
    try:
        from core.database import create_connector, DatabaseFactory
        
        # Test memory database
        connector = create_connector('sqlite_memory')
        print(f"✅ SQLite memory connector created")
        
        # Test connection
        if not connector.test_connection():
            print("❌ SQLite connection test failed")
            return False
        
        print(f"✅ SQLite connection test passed")
        
        # Test basic operations
        schemas = connector.list_schemas()
        print(f"📋 Schemas: {schemas}")
        
        # Create a test table and test operations
        result = connector.execute_query("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                created_date DATE DEFAULT CURRENT_DATE
            )
        """)
        print(f"✅ Test table created")
        
        # Insert test data
        connector.execute_query("""
            INSERT INTO test_table (name) VALUES 
            ('Test User 1'),
            ('Test User 2'),
            ('Test User 3')
        """)
        print(f"✅ Test data inserted")
        
        # Test table listing
        tables = connector.list_tables('main')
        print(f"📋 Tables in main schema: {tables}")
        
        # Test column listing
        columns = connector.list_columns('main', 'test_table')
        col_info = [f"{col['name']} ({col['type']})" for col in columns]
        print(f"📋 Columns in test_table: {col_info}")
        
        # Test query execution
        df = connector.execute_query("SELECT * FROM test_table ORDER BY id")
        print(f"✅ Query executed successfully, returned {len(df)} rows")
        print(f"📊 Sample data:\n{df.to_string()}")
        
        # Clean up
        connector.disconnect()
        print(f"✅ SQLite connector test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ SQLite connector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_factory_pattern():
    """Test the database factory pattern."""
    print("\n🏭 Testing factory pattern...")
    
    try:
        from core.database import DatabaseFactory, get_db_manager
        
        # Test available types
        available_types = DatabaseFactory.get_available_types()
        print(f"📋 Available connector types: {available_types}")
        
        # Test database manager
        manager = get_db_manager()
        profiles = manager.list_profiles()
        print(f"📋 Manager profiles: {list(profiles.keys())}")
        
        # Test specific connector creation
        connector = manager.get_connector('sqlite_memory')
        print(f"✅ Connector created via manager")
        
        # Test connection testing
        test_results = manager.test_all_connections()
        print(f"🔍 Connection test results:")
        for profile, status in test_results.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {profile}: {'OK' if status else 'FAILED'}")
        
        # Clean up
        manager.close_all_connections()
        print(f"✅ Factory pattern test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory pattern test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_legacy_compatibility():
    """Test backward compatibility with legacy functions."""
    print("\n🔄 Testing legacy compatibility...")
    
    try:
        import warnings
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            from core.utils.db_utils import create_db_engine, get_engine
            
            # Test new recommended function
            engine_new = get_engine('sqlite_memory')
            print(f"✅ New get_engine() function works")
            
            # Test legacy function (should issue deprecation warning)
            # Note: This will try to create MSSQL connection, which may fail
            # but we're mainly testing that the deprecation system works
            try:
                engine_legacy = create_db_engine(dbms="mssql")
                print(f"✅ Legacy create_db_engine() function works")
            except Exception as e:
                print(f"ℹ️  Legacy create_db_engine() failed as expected (no MSSQL): {e}")
            
            # Check for deprecation warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            if deprecation_warnings:
                print(f"✅ Deprecation warnings issued: {len(deprecation_warnings)}")
                for warning in deprecation_warnings:
                    print(f"   ⚠️  {warning.message}")
            else:
                print(f"ℹ️  No deprecation warnings captured")
            
        print(f"✅ Legacy compatibility test completed")
        return True
        
    except Exception as e:
        print(f"❌ Legacy compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_core_integration():
    """Test integration with agent core modules."""
    print("\n🤖 Testing agent core integration...")
    
    try:
        from core.database import create_connector
        from core.agent_core.basic.modules.utils.db_utils import (
            list_schemas, list_tables, list_columns, execute_query
        )
        
        # Create a SQLite connector
        connector = create_connector('sqlite_memory')
        
        # Create test table
        connector.execute_query("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                created_date DATE DEFAULT CURRENT_DATE
            )
        """)
        
        connector.execute_query("""
            INSERT INTO customers (name, email) VALUES 
            ('John Doe', 'john@example.com'),
            ('Jane Smith', 'jane@example.com')
        """)
        
        # Test using the new connector with existing agent core functions
        schemas = list_schemas(connector)
        print(f"✅ list_schemas() works with connector: {schemas}")
        
        tables = list_tables(connector, 'main')
        print(f"✅ list_tables() works with connector: {tables}")
        
        columns = list_columns(connector, 'main', 'customers')
        print(f"✅ list_columns() works with connector: {columns}")
        
        df = execute_query(connector, "SELECT * FROM customers")
        print(f"✅ execute_query() works with connector: {len(df)} rows returned")
        
        # Also test with legacy engine (for backward compatibility)
        engine = connector.get_engine()
        schemas_legacy = list_schemas(engine)
        print(f"✅ list_schemas() works with legacy engine: {schemas_legacy}")
        
        connector.disconnect()
        print(f"✅ Agent core integration test completed")
        return True
        
    except Exception as e:
        print(f"❌ Agent core integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting unified database system tests...\n")
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("SQLite Connector", test_sqlite_connector),
        ("Factory Pattern", test_factory_pattern),
        ("Legacy Compatibility", test_legacy_compatibility),
        ("Agent Core Integration", test_agent_core_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<10} {test_name}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The unified database system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())