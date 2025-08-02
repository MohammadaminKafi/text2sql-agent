"""
Entry point for the Text-to-SQL demo powered by
• Vanna + ChromaDB for vector-RAG
• AvalAI (OpenAI-compatible) LLM backend
• SQL Server hosting AdventureWorks2022
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from vanna.src.vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.src.vanna.openai.openai_chat import OpenAI_Chat

# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────

ENV_PATH = Path(__file__).with_suffix(".env")          # resolved inside container
load_dotenv(dotenv_path=ENV_PATH, override=False)


def _get_env(key: str, *, default: Optional[str] = None, required: bool = True) -> str:
    """Fetch an environment variable or raise a helpful error."""
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    force=True,
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Persistent training-flag (runs training exactly once)
#   • default path: ./cache/.vanna_trained
#   • override via TRAIN_FLAG_PATH env-var
#   • mount ./cache as a Docker volume if you want the flag to persist
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_FLAG = Path(os.getenv("TRAIN_FLAG_PATH", "./cache/.vanna_trained"))

# ──────────────────────────────────────────────────────────────────────────────
# Vanna client (ChromaDB + AvalAI)
# ──────────────────────────────────────────────────────────────────────────────


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    """Simple multiple-inheritance wrapper that glues ChromaDB + AvalAI chat-LLM."""

    def __init__(
        self,
        openai_config: Dict[str, str],
        llm_config: Dict = {},
        vdb_config: Dict = {},
    ):
        client = OpenAI(
            api_key=openai_config["api_key"],
            base_url=openai_config["base_url"],
        )
        ChromaDB_VectorStore.__init__(self, config=vdb_config)
        OpenAI_Chat.__init__(self, config=llm_config, client=client)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def build_connection_string() -> str:
    """Build a SQL-Server ODBC connection string from env-vars."""
    driver = "{ODBC Driver 18 for SQL Server}"
    server = _get_env("DB_SERVER")
    database = _get_env("DB_NAME")
    username = _get_env("DB_USER")
    password = _get_env("DB_PASSWORD")
    extras = "TrustServerCertificate=yes"
    return (
        f"DRIVER={driver};SERVER={server};DATABASE={database};"
        f"UID={username};PWD={password};{extras};"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # -------- Vanna / LLM configuration --------------------------------------
    avalai_cfg = {
        "api_key": _get_env("OPENAI_API_KEY"),
        "base_url": _get_env("OPENAI_BASE_API"),
    }
    llm_cfg = {"model": _get_env("MODEL")}
    vn = MyVanna(openai_config=avalai_cfg, llm_config=llm_cfg)
    log.info("✅  Vanna client configured (AvalAI backend)")

    if vn.test_llm_connection():
        log.info("✅  Connected to AvalAI LLM")

    # -------- Database connection -------------------------------------------
    vn.connect_to_mssql(odbc_conn_str=build_connection_string())
    log.info("✅  Connected to database")

    # -------- Agent set-up ---------------------------------------------------
    vn.create_agent(
        model=llm_cfg["model"],
        api_base=avalai_cfg["base_url"],
        api_key=avalai_cfg["api_key"],
        agent_toolkit=["run_sql", "query_rag"],
    )
    log.info("✅  Agent created")

    # -------- One-off training ----------------------------------------------
    if TRAIN_FLAG.exists():
        log.info("🏁  Training flag found at %s – skipping training.", TRAIN_FLAG)
    else:
        df_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
        plan = vn.get_training_plan_generic(df_schema)
        log.info("Training plan:\n%s", plan)
        vn.train(plan=plan)
        TRAIN_FLAG.parent.mkdir(parents=True, exist_ok=True)
        TRAIN_FLAG.touch()
        log.info("✅  Training complete (flag written to %s)", TRAIN_FLAG)

    # -------- Flask server ---------------------------------------------------
    from vanna.src.vanna.flask import VannaFlaskApp

    app = VannaFlaskApp(vn)
    log.info("🚀  Starting Flask on 0.0.0.0:44488")
    app.run(host="0.0.0.0", port=44488)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        log.exception("❌  Fatal error: %s", exc)
        raise