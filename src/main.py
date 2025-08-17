import os
import urllib
import matplotlib.pyplot as plt
from logging import DEBUG as LOG_LEVEL_DEBUG
from sqlalchemy import create_engine

from components.logging_setup import setup_logging
from components.top_flows import Text2SQLFlow
from components.utils.llm_utils import create_dspy_lm

def show_viz_plots(viz: dict) -> None:
    """
    Display (or save) all matplotlib figures returned in `viz`.
    Expected structure: {"figures": List[Figure], "labels": List[str], ...}
    """
    figures = []
    labels = []
    if isinstance(viz, dict):
        figures = viz.get("figures", []) or []
        labels = viz.get("labels", []) or []

    if not figures:
        print("\n— Plots — none —")
        return

    print(f"\n— Plots ({len(figures)}) — (close the windows to continue) —")
    for i, fig in enumerate(figures):
        title = labels[i] if i < len(labels) else f"Plot {i+1}"
        # Try to set a friendly window title and suptitle (best-effort)
        try:
            if hasattr(fig, "canvas") and hasattr(fig.canvas, "manager") and fig.canvas.manager:
                try:
                    fig.canvas.manager.set_window_title(str(title))
                except Exception:
                    pass
            if title:
                try:
                    fig.suptitle(str(title))
                except Exception:
                    pass
        except Exception:
            pass

        # Try interactive show; if backend cannot display, save as PNG fallback
        try:
            fig.show()
        except Exception:
            os.makedirs("plots", exist_ok=True)
            path = os.path.join("plots", f"plot_{i+1}.png")
            try:
                fig.savefig(path, bbox_inches="tight")
                print(f"Saved plot to {path}")
            except Exception as save_exc:
                print(f"⚠️ Could not display or save plot {i+1}: {save_exc}")

    # Block until user closes plot windows; then free memory
    try:
        plt.show()
    finally:
        plt.close("all")

def main() -> None:
    setup_logging(
        level=LOG_LEVEL_DEBUG,
        console=True,
        console_truncate_len=1000,
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
        temperature=0.3
    )

    flow = Text2SQLFlow(engine=engine, lm=lm)

    while True:
        try:
            prompt = input("\nAsk> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        try:
            df, sql, summary, viz = flow(prompt)
            print("\n— Final report SQL —")
            print(sql)
            print("\n— Preview —")
            print(df.head())
            print("\n— Summary —")
            print(summary)
            show_viz_plots(viz)
        except Exception as exc:
            print(f"💥 Failed to satisfy prompt: {exc}")

if __name__ == "__main__":
    main()
