"""Small, dependency-free renderings of training progress: text sparklines for terminals and
SVG sparklines plus HTML tables for notebooks (``_repr_html_``)."""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence

_BARS = "▁▂▃▄▅▆▇█"
LOSS_KEYS = ("training_loss", "train_loss", "loss", "gpt_loss", "total_loss")


def loss_series(rows: Iterable[dict[str, float]]) -> list[float]:
    """The training loss per step from ``job.metrics()`` rows (whichever key the backend uses)."""
    out = []
    for row in rows:
        for key in LOSS_KEYS:
            if key in row:
                out.append(float(row[key]))
                break
    return out


def sparkline(values: Sequence[float], width: int = 40) -> str:
    """``▇▆▄▃▂▁``: the last ``width`` values scaled between their min and max."""
    values = list(values)[-width:]
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(_BARS[min(len(_BARS) - 1, int((v - lo) / span * len(_BARS)))] for v in values)


def svg_sparkline(values: Sequence[float], width: int = 220, height: int = 40) -> str:
    values = list(values)
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = width / (len(values) - 1)
    points = " ".join(
        f"{i * step:.1f},{height - 3 - (v - lo) / span * (height - 6):.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg width="{width}" height="{height}" role="img" aria-label="loss curve">'
        f'<polyline fill="none" stroke="currentColor" stroke-width="1.5" points="{points}"/>'
        "</svg>"
    )


def html_table(rows: Iterable[tuple[str, str]], *, raw: Iterable[str] = ()) -> str:
    """Two-column key/value table; values named in ``raw`` are trusted HTML (e.g. SVG)."""
    trusted = set(raw)
    cells = "".join(
        f"<tr><th style='text-align:left;padding-right:1em'>{html.escape(k)}</th>"
        f"<td>{v if k in trusted else html.escape(v)}</td></tr>"
        for k, v in rows
    )
    return f"<table>{cells}</table>"
