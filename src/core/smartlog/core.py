from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from .levels import install_custom_levels
from .levels import (
    LVL_LIB_LLM_DEBUG, LVL_LIB_DEBUG,
)
from .filters import TruncatingFormatter, LevelWhitelistFilter, RemapLevelFilter
from .files import slugify, ensure_dir, rebuild_file_handlers
from .state import (
    state,
    DEFAULT_FORMAT, DEFAULT_DATEFMT, DEFAULT_BASE_DIR,
    DEFAULT_NOISY_LLM_LOGGERS, DEFAULT_NOISY_LIB_LOGGERS,
    CONSOLE_WHITELIST,
)

def init_logging(
    *,
    level: int = logging.DEBUG,
    console: bool = True,
    console_truncate_len: int = 120,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    clear_existing_handlers: bool = True,
    llm_logger_prefixes: Sequence[str] = DEFAULT_NOISY_LLM_LOGGERS,
    lib_logger_prefixes: Sequence[str] = DEFAULT_NOISY_LIB_LOGGERS,
    rotation_max_bytes: int = 10_000_000,
    rotation_backup_count: int = 5,
    console_include_warnings_errors: bool = True,
) -> None:
    """
    High-level setup:
    - Installs custom levels + logger methods (sysdebug, flowdebug, sysinfo, flowinfo, ftime, ftokens, …)
    - Console shows system/flow debug+info (and optionally warnings+).
    - Noisy libs remapped: LLM→LIB_LLM_DEBUG, other libs→LIB_DEBUG (only for < WARNING).
    - File handlers are created when `create_thread()` is called.
    """
    if state.configured:
        return

    install_custom_levels()

    state.base_dir = Path(base_dir)
    state.rotation_max_bytes = rotation_max_bytes
    state.rotation_backup_count = rotation_backup_count
    state.console_truncate_len = console_truncate_len
    state.formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger()
    if clear_existing_handlers and root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)
    root.setLevel(level)

    # Remap noisy libs
    mapping = {p: LVL_LIB_LLM_DEBUG for p in llm_logger_prefixes}
    mapping.update({p: LVL_LIB_DEBUG for p in lib_logger_prefixes})
    root.addFilter(RemapLevelFilter(mapping))

    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(TruncatingFormatter(fmt=fmt, datefmt=datefmt, max_len=console_truncate_len))
        allowed = set(CONSOLE_WHITELIST)
        if not console_include_warnings_errors:
            allowed -= {logging.WARNING, logging.ERROR, logging.CRITICAL}
        ch.addFilter(LevelWhitelistFilter(allowed))
        root.addHandler(ch)
        state.handlers.console = ch

    state.configured = True
    logging.getLogger(__name__).sysdebug("Logging initialized (root=%s)", logging.getLevelName(level))


def create_thread(input_name: str, *, when: Optional[datetime] = None) -> Path:
    """
    Create/activate a new log thread directory under base_dir:
      {slug(input_name)}-{YYYYmmdd_HHMMSS}
    Sets all file handlers to write there.
    """
    if not state.configured:
        init_logging()

    when = when or datetime.now()
    folder = f"{slugify(input_name)}-{when.strftime('%Y%m%d_%H%M%S')}"
    thread_dir = state.base_dir / folder
    ensure_dir(thread_dir)

    rebuild_file_handlers(thread_dir, state.formatter)
    state.current_thread_dir = thread_dir

    logging.getLogger(__name__).sysinfo("Activated log thread: %s", thread_dir)
    return thread_dir


def get_active_thread_dir():
    from .files import get_active_thread_dir as _get
    return _get()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else __name__)
    return logging.getLogger(name if name else __name__)
