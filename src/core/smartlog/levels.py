from __future__ import annotations
import logging

# ----- Custom level numbers -----
LVL_SYSTEM_DEBUG = 11
LVL_FLOW_DEBUG = 12
LVL_LLM_DEBUG = 13
LVL_LIB_LLM_DEBUG = 14
LVL_LIB_DEBUG = 15

LVL_SYSTEM_INFO = 21
LVL_FLOW_INFO = 22
LVL_FLOW_TIME_INFO = 23
LVL_FLOW_TOKEN_INFO = 24


def _register_level(level_no: int, level_name: str, method_name: str) -> None:
    logging.addLevelName(level_no, level_name)

    def _log_for_level(self: logging.Logger, msg, *args, **kwargs):
        if self.isEnabledFor(level_no):
            self._log(level_no, msg, args, **kwargs)

    setattr(logging.Logger, method_name, _log_for_level)


def install_custom_levels() -> None:
    # Debug-ish
    _register_level(LVL_SYSTEM_DEBUG, "SYSTEM_DEBUG", "sysdebug")
    _register_level(LVL_FLOW_DEBUG, "FLOW_DEBUG", "flowdebug")
    _register_level(LVL_LLM_DEBUG, "LLM_DEBUG", "llmdebug")
    _register_level(LVL_LIB_LLM_DEBUG, "LIB_LLM_DEBUG", "libllmdebug")
    _register_level(LVL_LIB_DEBUG, "LIB_DEBUG", "libdebug")

    # Info-ish
    _register_level(LVL_SYSTEM_INFO, "SYSTEM_INFO", "sysinfo")
    _register_level(LVL_FLOW_INFO, "FLOW_INFO", "flowinfo")
    _register_level(LVL_FLOW_TIME_INFO, "FLOW_TIME_INFO", "ftime")
    _register_level(LVL_FLOW_TOKEN_INFO, "FLOW_TOKEN_INFO", "ftokens")
