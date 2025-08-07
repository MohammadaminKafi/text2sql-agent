# ------------ vanna + chroma + AvalAI LLM ------------
import os

from no_commit_utils.credentials_utils import read_credentials
from openai import OpenAI

from vanna.src.vanna.base.base import VannaBase
from vanna.src.vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.src.vanna.openai.openai_chat import OpenAI_Chat

os.environ["CHROMA_CACHE_DIR"] = "./cache/chroma_cache"

MODEL = "gpt-4o-mini"  # "gemma-3-27b-it"
BASE_API = "https://api.avalapis.ir/v1"  # "https://api.avalai.ir/v1"
API_KEY = read_credentials("avalai.key")

USE_AGENT = True


# ---- Compose with ChromaDB for the full Vanna client ---------------------
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(
        self, openai_config: dict, llm_config: dict = {}, vdb_config: dict = {}
    ):
        client = OpenAI(
            api_key=openai_config["api_key"],
            base_url=openai_config.get("base_url", BASE_API),
        )
        ChromaDB_VectorStore.__init__(self, config=vdb_config)
        OpenAI_Chat.__init__(self, config=llm_config, client=client)


# ---- 3. Demo / entry-point ---------------------------------------------------
def main() -> None:
    avalai_cfg = {
        "api_key": API_KEY,
        "base_url": BASE_API,
    }

    llm_cfg = {"model": MODEL}

    vn = MyVanna(openai_config=avalai_cfg, llm_config=llm_cfg)
    print("\n✅  Vanna configured successfully (AvalAI backend)\n")

    if input("Test connection to LLM provider? [y/N] ").lower().startswith("y"):
        connection_test = vn.test_llm_connection()
        if connection_test == True:
            print("\n✅  Vanna is connected to an LLM\n")

    # ---------- DB connection (Windows Auth) ----------
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=AdventureWorks2022;"
        "Trusted_Connection=yes;"
    )
    vn.connect_to_mssql(odbc_conn_str=conn_str)
    print("✅  Connected to database\n")

    # ---------- Training (only done once) ------------
    if input("Plan for training? [y/N] ").lower().startswith("y"):
        df_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
        plan = vn.get_training_plan_generic(df_schema)
        print(f"Training plan:\n{plan}\n")

        if input("Train on this plan? [y/N] ").lower().startswith("y"):
            vn.train(plan=plan)
            print("✅  Training complete\n")

    if (
        input("Open web app or continue in command-line? [w/C] ")
        .lower()
        .startswith("w")
    ):
        from vanna.src.vanna.flask.__init__ import VannaFlaskApp

        app = VannaFlaskApp(vn)
        app.run()

    # ------------- Interactive loop ------------------
    if USE_AGENT:
        vn.create_agent(model=MODEL, api_base=BASE_API, api_key=API_KEY)
    print("Type 'exit' or 'quit' to leave.")
    while True:
        q = input("\nAsk: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not USE_AGENT:
            answer = vn.ask(question=q, visualize=False)
        elif USE_AGENT:
            answer = vn.ask_agent(question=q)
        print(answer)


if __name__ == "__main__":
    main()
