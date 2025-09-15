import os
import logging
from urllib.parse import urlparse
from typing import Optional

import dspy

from core.smartlog import get_logger

logger = get_logger(__name__)

# ───────────────────  Convenience LM creator ─────────────────── #
def create_dspy_lm(
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    api_base: str = "https://api.avalapis.ir/v1",
    temperature: float = 0.0,
    max_tokens: int = 4000,
    cache: bool = False,
    cache_in_memory: bool = False,
    num_retries: int = 1
):

    # Resolve API key and base URL from common env names, allowing overrides
    env_api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("LITELLM_API_KEY")
        or os.getenv("AVALAI_API_KEY")
    )
    env_api_base = (
        os.getenv("OPENAI_BASE")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("AVALAI_API_BASE")
        or api_base
    )

    api_key = api_key or env_api_key
    api_base = env_api_base or api_base
    urlparse(api_base)

    logger.sysdebug("🌟 Initialising dspy.LM: model=%s  api_base=%s", model, api_base)
    lm = dspy.LM(
    model=model, 
    api_key=api_key, 
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        cache_in_memory=cache_in_memory,
        #num_retries=5,
    )

    return lm


def get_llm(option: str):
    """
    Return a configured LLM object based on the given option name.
    Supported: 'avalai', 'ollama_chat', 'gemma3:27b', 'avalai_gemma', 'lmstudio'.
    """
    opt = option.lower().strip()

    if opt == "avalai":
        return create_dspy_lm(
            model="openai/gpt-4o-mini",
            api_base="https://api.avalai.ir/v1",
            temperature=0.3,
        )

    elif opt == "ollama":
        return create_dspy_lm(
            model="ollama_chat",
            api_key="none",
            api_base="http://199.168.172.141:11434/v1",
            temperature=0.3,
        )

    elif opt == "lmstudio":
        return create_dspy_lm(
            model="openai/",   # adjust model name if needed
            api_key="none",
            api_base="http://172.21.54.32:1234/v1",
            temperature=0.3,
        )

    else:
        raise ValueError(f"Unknown LLM option: {option}")
