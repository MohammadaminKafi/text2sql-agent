import logging
from logging.handlers import RotatingFileHandler
from typing import Iterable, Optional


DEFAULT_FORMAT = "%(asctime)s  [%(levelname)s]  %(name)s › %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"

# Chatty libs you may want to quiet down
NOISY_LOGGERS: tuple[str, ...] = (
    "LiteLLM",
    "litellm",
    "httpx",
    "urllib3",
    "httpcore",
    "openai",
    "openai._base_client",
)


class TruncateLongMsgs(logging.Filter):
    """Truncates very long log messages to keep console readable."""

    def __init__(self, max_len: int = 300):
        super().__init__()
        self.max_len = max_len

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            # If formatting fails, let it pass unmodified.
            return True
        if self.max_len and len(msg) > self.max_len:
            record.msg = msg[: self.max_len] + " …(truncated)"
            record.args = ()
        return True


_configured = False  # guard against double-initialisation


def setup_logging(
    *,
    level: int = logging.DEBUG,
    console: bool = True,
    console_truncate_len: int = 100,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
    log_file: Optional[str] = None,
    file_level: Optional[int] = None,
    file_max_bytes: int = 5_000_000,
    file_backup_count: int = 3,
    quiet: Iterable[str] = NOISY_LOGGERS,
) -> None:
    """
    Configure root logging once. Call this from your application's entry point.

    - Modules should *not* call this; they just use `logging.getLogger(__name__)`.
    - Adds a console handler (with optional truncation) and an optional rotating file handler.
    - Silences noisy third-party loggers.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    # If the runtime already attached handlers (e.g., Jupyter), clear them so
    # your config is authoritative. Comment this out if you prefer additive.
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        if console_truncate_len and console_truncate_len > 0:
            ch.addFilter(TruncateLongMsgs(console_truncate_len))
        root.addHandler(ch)

    if log_file:
        fh = RotatingFileHandler(
            log_file, maxBytes=file_max_bytes, backupCount=file_backup_count
        )
        fh.setLevel(file_level if file_level is not None else level)
        fh.setFormatter(formatter)
        # Usually you don't want truncation in files; keep full content for forensics.
        root.addHandler(fh)

    # Quiet down libraries that are too chatty at DEBUG/INFO.
    for name in quiet:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).debug("🚀 Logging initialised (level=%s)", logging.getLevelName(level))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Convenience accessor. Prefer `logging.getLogger(__name__)` in your modules,
    but this is here if you like importing a helper.
    """
    return logging.getLogger(name if name else __name__)
