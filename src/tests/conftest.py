import os
import pytest
import dspy

def pytest_addoption(parser):
    parser.addoption("--llm", action="store_true", default=False,
                     help="Run tests that call a live LLM (network).")
    parser.addoption("--model", action="store", default=None,
                     help="Model name for LLM (overrides default from main.py).")
    parser.addoption("--api-base", action="store", default=None,
                     help="API base URL for LLM (overrides default from main.py).")
    parser.addoption("--api-key", action="store", default=None,
                     help="API key for LLM (overrides default from main.py).")

def pytest_configure(config):
    # register the custom mark to silence warnings
    config.addinivalue_line("markers", "llm: tests that call a live LLM")

def pytest_runtest_setup(item):
    if "llm" in item.keywords and not item.config.getoption("--llm"):
        pytest.skip("Use --llm to run live LLM tests.")

@pytest.fixture(scope="session")
def lm(pytestconfig):
    if not pytestconfig.getoption("--llm"):
        return None  # ← plain return, no yield

    # Defaults exactly as in main.py
    model    = pytestconfig.getoption("--model")    or "openai/gpt-4o-mini"
    api_base = pytestconfig.getoption("--api-base") or "https://api.avalai.ir/v1"
    api_key  = pytestconfig.getoption("--api-key")  or os.getenv("AvalAI_API_KEY")

    # Build an OpenAI-compatible client via DSPy
    lm = dspy.LM(model=model, api_key=api_key, api_base=api_base, temperature=0.0)

    try:
        dspy.settings.configure(lm=lm)
    except Exception:
        dspy.configure(lm=lm)

    return lm