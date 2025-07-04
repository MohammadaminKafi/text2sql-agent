# ------------ vanna + chroma + AvalAI LLM ------------
import os
from openai import OpenAI
from vanna.src.vanna.openai.openai_chat import OpenAI_Chat
from vanna.src.vanna.base.base import VannaBase
from vanna.src.vanna.base.agent_base import AVannaBase
from vanna.src.vanna.chromadb.chromadb_vector import ChromaDB_VectorStore


# ---- Compose with ChromaDB for the full Vanna client ---------------------
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat, AVannaBase):
    def __init__(self, openai_config : dict, llm_config: dict = {}, vdb_config: dict = {}):
        client = OpenAI(
            api_key=openai_config["api_key"],
            base_url=openai_config.get("base_url", "https://api.avalai.ir/v1"),
        )
        ChromaDB_VectorStore.__init__(self, config=vdb_config)
        OpenAI_Chat.__init__(self, config=llm_config, client=client)


# ---- 3. Demo / entry-point ---------------------------------------------------
def main() -> None:
    avalai_cfg = {
        "api_key":  "",
        "base_url": "https://api.avalapis.ir/v1",
    }

    llm_cfg = {
        "model": "gpt-4o"
    }

    vn = MyVanna(openai_config=avalai_cfg, llm_config=llm_cfg)
    print("\n✅  Vanna configured successfully (AvalAI backend)\n")

    # ---------- DB connection (Windows Auth) ----------
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=AdventureWorks2022;"
        "Trusted_Connection=yes;"
    )
    vn.connect_to_mssql(odbc_conn_str=conn_str)
    print("✅  Connected to database\n")

    vn.ask_react_agent()
    
    while True:
        pass

    # ---------- Training (only done once) ------------
    df_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
    plan = vn.get_training_plan_generic(df_schema)
    print(f"Training plan:\n{plan}\n")

    if input("Train on this plan? [y/N] ").lower().startswith("y"):
        vn.train(plan=plan)
        print("✅  Training complete\n")

    if input("Open web app or continue in command-line? [w/C] ").lower().startswith("w"):
        from vanna.flask import VannaFlaskApp
        app = VannaFlaskApp(vn)
        app.run()


    # ------------- Interactive loop ------------------
    print("Type 'exit' or 'quit' to leave.")
    while True:
        q = input("\nAsk: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        answer = vn.ask(question=q, visualize=False)
        print(answer)


if __name__ == "__main__":
    main()
