import logging
from typing import Optional

from core.database import get_db_manager, create_connector
from core.utils.llm_utils import get_llm
from core.agent_core.basic.top_flow import BasicText2SQLFlow

_engine = None
_flow = None
_db_connector = None

def init_components():
    global _engine, _flow, _db_connector
    logging.getLogger(__name__).debug("Initializing engine/flow...")

    try:
        # Use new unified database system
        _db_connector = create_connector("mssql_docker")  # or "mssql_adventureworks" for Windows auth
        _engine = _db_connector.get_engine()
        
        # Test the connection
        if not _db_connector.test_connection():
            logging.getLogger(__name__).warning("Database connection test failed, but proceeding...")
        
        lm = get_llm("avalai")
        _flow = BasicText2SQLFlow(engine=_engine, lm=lm)
        logging.getLogger(__name__).debug("Initialized engine/flow with unified database system.")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to initialize with unified system: {e}")
        logging.getLogger(__name__).info("Falling back to legacy database initialization...")
        
        # Fallback to legacy system
        from core.utils.db_utils import create_db_engine
        
        _engine = create_db_engine(dbms="mssql", database="AdventureWorks2022")
        lm = get_llm("avalai")
        _flow = BasicText2SQLFlow(engine=_engine, lm=lm)
        logging.getLogger(__name__).debug("Initialized engine/flow with legacy system.")

def shutdown_components():
    global _engine, _flow, _db_connector
    log = logging.getLogger(__name__)
    log.debug("Shutting down components...")
    
    try:
        if _flow and hasattr(_flow, "close"):
            _flow.close()
    except Exception:
        log.exception("Error closing flow")
    
    try:
        if _db_connector and hasattr(_db_connector, "disconnect"):
            _db_connector.disconnect()
    except Exception:
        log.exception("Error disconnecting database connector")
    
    try:
        if _engine is not None and hasattr(_engine, "dispose"):
            _engine.dispose()
    except Exception:
        log.exception("Error disposing engine")
    
    _flow = None
    _engine = None
    _db_connector = None

def get_flow():
    if _flow is None:
        raise RuntimeError("Flow not initialized")
    return _flow

def get_db_connector():
    """Get the current database connector instance."""
    if _db_connector is None:
        raise RuntimeError("Database connector not initialized")
    return _db_connector

def get_engine():
    """Get the current database engine."""
    if _engine is None:
        raise RuntimeError("Database engine not initialized")
    return _engine
