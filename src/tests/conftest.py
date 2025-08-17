import pytest
import urllib
from sqlalchemy import create_engine
from components.utils.llm_utils import create_dspy_lm


def pytest_addoption(parser):
    parser.addoption(
        "--llm-model",
        action="store",
        default="openai/gpt-4o-mini",
        help="Name of the LLM model to use (default: openai/gpt-4o-mini)",
    )
    parser.addoption(
        "--llm-api-base",
        action="store",
        default="https://api.avalai.ir/v1",
        help="Base URL for the LLM API endpoint",
    )
    parser.addoption(
        "--llm-api-key",
        action="store",
        default=None,
        help="API key for authenticating with the LLM provider",
    )
    parser.addoption(
        "--llm-cache-enabled",
        action="store",
        default="false",
        help="Enable or disable caching (true/false, default: true)",
    )
    parser.addoption(
        "--llm-max-tokens",
        action="store",
        type=int,
        default=4000,
        help="Maximum number of tokens to generate (default: 512)",
    )
    parser.addoption(
        "--llm-temperature",
        action="store",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0, lower = more deterministic)",
    )


@pytest.fixture(scope="session")
def llm(request):
    """Fixture that builds the DSPy LLM client with CLI args."""
    model = request.config.getoption("--llm-model")
    api_base = request.config.getoption("--llm-api-base")
    api_key = request.config.getoption("--llm-api-key")
    cache_enabled = request.config.getoption("--llm-cache-enabled").lower() == "true"
    max_tokens = request.config.getoption("--llm-max-tokens")
    temperature = request.config.getoption("--llm-temperature")

    lm = create_dspy_lm(
        model=model,
        api_base=api_base,
        api_key=api_key,
        cache=cache_enabled,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return lm


@pytest.fixture(scope="session")
def db_engine():
    engine_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=AdventureWorks2022;"
        "Trusted_Connection=yes;"
    )
    engine = create_engine(engine_url)
    yield engine
    engine.dispose()


# Optional: cursor/session fixture
@pytest.fixture
def db_session(db_engine):
    with db_engine.connect() as conn:
        yield conn