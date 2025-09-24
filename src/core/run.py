import os
import sys
import urllib
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Add the src directory to Python path to enable imports
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from core.agent_core.basic.top_flow import BasicText2SQLFlow
from core.utils.llm_utils import create_dspy_lm, get_llm
from core.database import create_connector
from core.smartlog import init_logging, create_thread

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

    # Connect to SQL Server with Windows Authentication
    connector = create_connector("mssql_adventureworks")
    engine = connector.get_engine()

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
