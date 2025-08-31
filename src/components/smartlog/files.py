from __future__ import annotations
import re
from pathlib import Path
from logging.handlers import RotatingFileHandler
import logging
from typing import Optional
from .filters import LevelWhitelistFilter
from .levels import LVL_FLOW_DEBUG, LVL_FLOW_INFO, LVL_FLOW_TIME_INFO, LVL_FLOW_TOKEN_INFO
from .state import state

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "run"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def close_and_detach(handler: Optional[logging.Handler]) -> None:
    if handler is None:
        return
    root = logging.getLogger()
    try:
        root.removeHandler(handler)
    except Exception:
        pass
    try:
        handler.close()
    except Exception:
        pass

def rotating_file_handler(path: Path, formatter: logging.Formatter) -> RotatingFileHandler:
    h = RotatingFileHandler(
        path,
        maxBytes=state.rotation_max_bytes,
        backupCount=state.rotation_backup_count,
        encoding="utf-8",
    )
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    return h

def rebuild_file_handlers(thread_dir: Path, fmt: logging.Formatter) -> None:
    close_and_detach(state.handlers.all_file)
    close_and_detach(state.handlers.flow_file)
    close_and_detach(state.handlers.flow_time_file)
    close_and_detach(state.handlers.flow_tokens_file)

    root = logging.getLogger()

    all_h = rotating_file_handler(thread_dir / "all.log", fmt)

    flow_h = rotating_file_handler(thread_dir / "flow.log", fmt)
    flow_h.addFilter(LevelWhitelistFilter({LVL_FLOW_DEBUG, LVL_FLOW_INFO}))

    flow_time_h = rotating_file_handler(thread_dir / "flow_time.log", fmt)
    flow_time_h.addFilter(LevelWhitelistFilter({LVL_FLOW_TIME_INFO}))

    flow_tokens_h = rotating_file_handler(thread_dir / "flow_tokens.log", fmt)
    flow_tokens_h.addFilter(LevelWhitelistFilter({LVL_FLOW_TOKEN_INFO}))

    for h in (all_h, flow_h, flow_time_h, flow_tokens_h):
        root.addHandler(h)

    state.handlers.all_file = all_h
    state.handlers.flow_file = flow_h
    state.handlers.flow_time_file = flow_time_h
    state.handlers.flow_tokens_file = flow_tokens_h

def get_active_thread_dir() -> Optional[Path]:
    return state.current_thread_dir