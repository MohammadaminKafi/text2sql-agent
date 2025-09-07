# src/backend/app/deps.py
import logging
from typing import Optional

_engine = None
_flow = None

def init_components():
    global _engine, _flow
    logging.getLogger(__name__).debug("Initializing engine/flow...")

    # ⬇️ Lazy imports happen AFTER logging is ready
    from core.utils.db_utils import create_db_engine
    from core.utils.llm_utils import get_llm
    from core.agent_core.basic.top_flow import BasicText2SQLFlow

    _engine = create_db_engine(dbms="mssql", database="ADVENTUREWORKS")
    lm = get_llm("avalai")
    _flow = BasicText2SQLFlow(engine=_engine, lm=lm)
    logging.getLogger(__name__).debug("Initialized engine/flow.")

def shutdown_components():
    global _engine, _flow
    log = logging.getLogger(__name__)
    log.debug("Shutting down components...")
    try:
        if _flow and hasattr(_flow, "close"):
            _flow.close()
    except Exception:
        log.exception("Error closing flow")
    try:
        if _engine is not None and hasattr(_engine, "dispose"):
            _engine.dispose()
    except Exception:
        log.exception("Error disposing engine")
    _flow = None
    _engine = None

def get_flow():
    if _flow is None:
        raise RuntimeError("Flow not initialized")
    return _flow
