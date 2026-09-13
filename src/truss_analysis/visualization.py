"""Optional plotting utilities with safe Persian text rendering.

matplotlib is an optional dependency (``pip install truss-analysis[viz]``);
it is imported lazily inside :func:`plot_truss` so that importing this
module never requires matplotlib to be installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .model import Element, Node

__all__ = ["plot_truss"]


def _persian(text: str) -> str:
    """Convert text to Persian bidirectional display form.

    Falls back to the original ``text`` when the optional reshaping
    dependencies (``arabic_reshaper``, ``python-bidi``) are unavailable.
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return str(get_display(arabic_reshaper.reshape(text)))
    except Exception:
        return text


def plot_truss(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    U: npt.ArrayLike | None = None,
    results: Sequence[Mapping[str, Any]] | None = None,
    title: str = "2D Truss",
    save_path: str | Path | None = None,
) -> str | Path | None:
    """Plot the undeformed (and optionally deformed) truss.

    Members are coloured by axial force sign: red for tension, blue for
    compression, grey for (near-)zero force.  Supports are drawn as green
    markers.  When ``U`` is given, the deformed shape is overlaid as dashed
    lines, scaled automatically to 15 percent of the structure extent.

    Parameters
    ----------
    nodes : Sequence[Node]
        Truss nodes.
    elements : Sequence[Element]
        Truss elements.
    U : array_like or None, optional
        Flat nodal displacement vector ``[ux_0, uy_0, ux_1, uy_1, ...]``
        in node order.
    results : Sequence[Mapping[str, Any]] or None, optional
        Per-member results; each mapping should carry ``"id"`` and the
        axial force ``"N"``.
    title : str, optional
        Plot title; Persian text is reshaped automatically.
    save_path : str, Path or None, optional
        When given, the figure is saved to this path (Agg backend) instead
        of being shown interactively.

    Returns
    -------
    str, Path or None
        ``save_path`` when the figure was saved, otherwise ``None``.
    """
    import matplotlib

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
    scale: float | None = None
    disp: npt.NDArray[np.float64] | None = None
    if U is not None:
        disp = np.asarray(U, dtype=np.float64)
        scale = 0.15 * span / max(float(np.max(np.abs(disp))), 1e-12)

    for e in elements:
        ni, nj = coords[e.node_i], coords[e.node_j]
        n_force = forces.get(str(e.id), 0.0)
        color = (
            "#d62728" if n_force > 1e-9 else ("#1f77b4" if n_force < -1e-9 else "gray")
        )
        ax.plot([ni.x, nj.x], [ni.y, nj.y], color=color, lw=2)
        if disp is not None and scale is not None:
            i, j = idx[e.node_i], idx[e.node_j]
            ax.plot(
                [ni.x + disp[2 * i] * scale, nj.x + disp[2 * j] * scale],
                [ni.y + disp[2 * i + 1] * scale, nj.y + disp[2 * j + 1] * scale],
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
