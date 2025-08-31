from __future__ import annotations
import logging
from typing import Dict, Iterable


class TruncateLongMsgs(logging.Filter):
    """Truncate long messages (console only)."""

    def __init__(self, max_len: int = 300):
        super().__init__()
        self.max_len = max_len

    def filter(self, record: logging.LogRecord) -> bool:
        if self.max_len <= 0:
            return True
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if len(msg) > self.max_len:
            record.msg = msg[: self.max_len] + " …(truncated)"
            record.args = ()
        return True


class TruncatingFormatter(logging.Formatter):
    """Console-only formatter that truncates the message without affecting other handlers."""

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, style: str = "%", max_len: int = 300):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.max_len = max_len

    def format(self, record: logging.LogRecord) -> str:
        if self.max_len <= 0:
            return super().format(record)
        # Preserve original values
        orig_msg, orig_args = record.msg, record.args
        try:
            full_msg = record.getMessage()
            if len(full_msg) > self.max_len:
                record.msg = full_msg[: self.max_len] + " …(truncated)"
                record.args = ()
            return super().format(record)
        finally:
            # Restore original values so file handlers see the full message
            record.msg, record.args = orig_msg, orig_args


class LevelWhitelistFilter(logging.Filter):
    """Allow only specific levels."""

    def __init__(self, allowed_levels: Iterable[int]):
        super().__init__()
        self.allowed = set(allowed_levels)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in self.allowed


class RemapLevelFilter(logging.Filter):
    """
    Remap levels for specific logger-name prefixes to custom levels.
    Only remaps when original level < WARNING so warnings/errors remain intact.
    """

    def __init__(self, prefix_to_level: Dict[str, int], max_original_level: int = logging.WARNING):
        super().__init__()
        self.prefix_to_level = dict(prefix_to_level)
        self.max_original_level = max_original_level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= self.max_original_level:
            return True
        name = record.name
        for prefix, new_level in self.prefix_to_level.items():
            if name == prefix or name.startswith(prefix + "."):
                record.levelno = new_level
                record.levelname = logging.getLevelName(new_level)
                break
        return True