"""Optional matplotlib visualization with safe Persian text support."""

from __future__ import annotations

import matplotlib
import numpy as np


def _persian(text: str) -> str:
    """Convert text to Persian bidirectional format."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def plot_truss(nodes, elements, U=None, results=None, title="2D Truss", save_path=None):
    """Plot undeformed (and deformed) truss; red=tension, blue=compression."""
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = {n.id: n for n in nodes}
    idx = {n.id: i for i, n in enumerate(nodes)}
    forces = {str(r.get("id")): float(r.get("N", 0.0)) for r in results or []}

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    scale = None
    if U is not None:
        scale = 0.15 * span / max(float(np.max(np.abs(U))), 1e-12)

    for e in elements:
        ni, nj = coords[e.node_i], coords[e.node_j]
        N = forces.get(str(e.id), 0.0)
        color = "#d62728" if N > 1e-9 else ("#1f77b4" if N < -1e-9 else "gray")
        ax.plot([ni.x, nj.x], [ni.y, nj.y], color=color, lw=2)
        if U is not None and scale is not None:
            i, j = idx[e.node_i], idx[e.node_j]
            ax.plot(
                [ni.x + U[2 * i] * scale, nj.x + U[2 * j] * scale],
                [ni.y + U[2 * i + 1] * scale, nj.y + U[2 * j + 1] * scale],
                "--",
                color=color,
                alpha=0.5,
                lw=1,
            )
    for n in nodes:
        if n.is_support:
            marker = "^" if (n.support_dx and n.support_dy) else "o"
            ax.plot(n.x, n.y, marker, color="green", ms=10)
    ax.set_title(_persian(title))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return save_path
