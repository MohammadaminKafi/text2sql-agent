import os
import urllib
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from components.top_flows import BasicText2SQLFlow
from components.utils.llm_utils import create_dspy_lm, get_llm
from components.utils.db_utils import create_db_engine
from components.smartlog import init_logging, create_thread

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
    load_dotenv()

    init_logging()
    create_thread("system-init")

    engine = create_db_engine(dbms="snowflake", database="ADVENTUREWORKS")

    lm = get_llm("avalai")

    flow = BasicText2SQLFlow(engine=engine, lm=lm)

    while True:
        try:
            prompt = input("\nAsk> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        try:
            create_thread("system-run")
            df, sql, summary, viz = flow(prompt)
            print("\n— Final report SQL —")
            print(sql)
            print("\n— Preview —")
            print(df.head())
            print("\n— Summary —")
            print(summary)
            # show_viz_plots(viz)
        except Exception as exc:
            print(f"💥 Failed to satisfy prompt: {exc}")

if __name__ == "__main__":
    main()
