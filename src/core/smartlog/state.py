from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field

from .levels import (
    LVL_SYSTEM_DEBUG, LVL_FLOW_DEBUG, LVL_SYSTEM_INFO, LVL_FLOW_INFO
)

DEFAULT_FORMAT = "%(asctime)s  [%(levelname)s]  %(name)s › %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"
DEFAULT_BASE_DIR = "./log"

# LLM-adjacent noisy loggers → LIB_LLM_DEBUG
DEFAULT_NOISY_LLM_LOGGERS: Tuple[str, ...] = (
    "LiteLLM",
    "litellm",
    "openai",
    "openai._base_client",
    "anthropic",
    "vertexai",
    "google.generativeai",
)

# Non-LLM libs → LIB_DEBUG
DEFAULT_NOISY_LIB_LOGGERS: Tuple[str, ...] = (
    "httpx",
    "urllib3",
    "httpcore",
    "PIL",
    "PIL.PngImagePlugin",
    "matplotlib",
    "matplotlib.font_manager",
)

# Console shows:
CONSOLE_WHITELIST = {
    LVL_SYSTEM_DEBUG,
    LVL_FLOW_DEBUG,
    LVL_SYSTEM_INFO,
    LVL_FLOW_INFO,
    logging.WARNING,
    logging.ERROR,
    logging.CRITICAL,
}

@dataclass
class Handlers:
    console: Optional[logging.Handler] = None
    all_file: Optional[logging.Handler] = None
    flow_file: Optional[logging.Handler] = None
    flow_time_file: Optional[logging.Handler] = None
    flow_tokens_file: Optional[logging.Handler] = None

@dataclass
class State:
    configured: bool = False
    base_dir: Path = Path(DEFAULT_BASE_DIR)
    current_thread_dir: Optional[Path] = None
    handlers: Handlers = field(default_factory=Handlers)
    formatter: Optional[logging.Formatter] = None
    console_truncate_len: int = 120
    rotation_max_bytes: int = 10_000_000
    rotation_backup_count: int = 5

state = State()