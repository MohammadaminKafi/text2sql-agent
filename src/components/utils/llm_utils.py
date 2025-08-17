import os
import logging
from urllib.parse import urlparse
from typing import Optional

import dspy

logger = logging.getLogger(__name__)

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

    api_key = api_key or os.getenv("AvalAI_API_KEY")
    urlparse(api_base)

    logger.debug("🌟 Initialising dspy.LM: model=%s  api_base=%s", model, api_base)
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
