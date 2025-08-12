import os
import logging
from urllib.parse import urlparse

import dspy

logger = logging.getLogger(__name__)

# ───────────────────  Convenience LM creator ─────────────────── #
def create_dspy_lm(
    model: str = "openai/gpt-4o-mini",
    api_key: str | None = None,
    api_base: str = "https://api.avalapis.ir/v1",
):

    api_key = api_key or os.getenv("AvalAI_API_KEY")
    urlparse(api_base)

    logger.debug("🌟 Initialising dspy.LM: model=%s  api_base=%s", model, api_base)
    lm = dspy.LM(
        model=model, 
        api_key=api_key, 
        api_base=api_base,
        temperature=0.3,
        #max_tokens=8000,
        cache=False,
        #cache_in_memory=False,
        #num_retries=5,
    )

    return lm
