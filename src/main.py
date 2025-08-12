import urllib
from logging import DEBUG as LOG_LEVEL_DEBUG
from sqlalchemy import create_engine

from components.logging_setup import setup_logging

from components.top_flows import Text2SQLFlow
from components.utils.llm_utils import create_dspy_lm

def main() -> None:
    setup_logging(
        level=LOG_LEVEL_DEBUG,
        console=True,
        console_truncate_len=100,
        log_file=None,
    )

    engine_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=AdventureWorks2022;"
        "Trusted_Connection=yes;"
    )

    engine = create_engine(engine_url)

    # ollama_chat/              ->  http://199.168.172.141:11434/v1 (api_key='none')
    # gemma3:27b                ->  192.168.172.141:11434/v1
    # openai/gemma-3-27b-it     ->  https://api.avalapis.ir/v1, https://api.avalai.ir/v1
    lm = create_dspy_lm(
        model="openai/gpt-4o-mini",
        api_base="https://api.avalai.ir/v1",
    )

    flow = Text2SQLFlow(engine=engine, lm=lm)

    while True:
        try:
            prompt = input("\nAsk> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        try:
            df, readable_sql, summary = flow(prompt)
            print("\n— Final report SQL —")
            print(readable_sql)
            print("\n— Preview —")
            print(df.head())
            print("\n— Summary —")
            print(summary)
        except Exception as exc:
            print(f"💥 Failed to satisfy prompt: {exec}")


if __name__ == "__main__":
    main()
