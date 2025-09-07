from __future__ import annotations
import base64, io, logging
from typing import Any, Tuple

def serialize_plots(viz_objs: list[Any] | None, cap: int = 4) -> list[dict]:
    out = []
    for v in (viz_objs or []):
        try:
            cls = v.__class__.__name__
            mod = v.__class__.__module__
            if cls == "Figure" or mod.startswith("matplotlib"):
                b64, w, h = _mpl_to_b64(v)
                out.append({"format": "png", "b64": b64, "width": w, "height": h})
            elif mod.startswith("plotly"):
                b64, w, h = _plotly_to_b64(v)
                out.append({"format": "png", "b64": b64, "width": w, "height": h})
            elif hasattr(v, "to_png"):
                png_bytes = v.to_png()
                out.append({"format": "png", "b64": _b64(png_bytes)})
        except Exception:
            logging.exception("Plot serialization failed; skipping.")
            continue
        if len(out) >= cap:
            break
    return out

def _mpl_to_b64(fig) -> Tuple[str, int | None, int | None]:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    w, h = None, None
    try:
        wi, hi = fig.get_size_inches()
        dpi = fig.get_dpi()
        w, h = int(wi * dpi), int(hi * dpi)
    except Exception:
        pass
    return _b64(buf.read()), w, h

def _plotly_to_b64(fig) -> Tuple[str, int | None, int | None]:
    # requires kaleido
    png = fig.to_image(format="png")
    w = getattr(getattr(fig, "layout", None), "width", None)
    h = getattr(getattr(fig, "layout", None), "height", None)
    return _b64(png), w, h

def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")
