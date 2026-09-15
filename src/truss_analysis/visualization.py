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

__all__ = ["plot_buckling_mode", "plot_truss"]


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


def plot_buckling_mode(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    mode: npt.ArrayLike,
    title: str = "Buckling Mode",
    save_path: str | Path | None = None,
) -> str | Path | None:
    """Plot a buckling mode shape on the undeformed truss geometry.

    Visualizes the critical buckling mode from a linearized stability
    analysis by overlaying the deformed shape (scaled) on the original
    geometry. The mode is shown with dashed lines, colored by the
    displacement magnitude.

    Parameters
    ----------
    nodes : Sequence[Node]
        Truss nodes in the same order as used in the stability analysis.
    elements : Sequence[Element]
        Truss elements connecting the nodes.
    mode : array_like
        Buckling mode vector of shape ``(2n,)`` containing displacements
        for each DOF. Fixed DOFs should be exactly zero.
    title : str, optional
        Plot title; Persian text is reshaped automatically. Can include
        the critical load factor, e.g. ``"Buckling Mode (λ_cr = 2.34)"``.
    save_path : str, Path or None, optional
        When given, the figure is saved to this path (Agg backend) instead
        of being shown interactively.

    Returns
    -------
    str, Path or None
        ``save_path`` when the figure was saved, otherwise ``None``.

    Notes
    -----
    Addresses Issue B6: ``plot_buckling_mode()`` missing from
    ``visualization.py`` (C2(⚠️ج)).

    Examples
    --------
    >>> from truss_analysis.stability import linearized_buckling_load_factor
    >>> from truss_analysis.visualization import plot_buckling_mode
    >>> result = linearized_buckling_load_factor(nodes, elements, forces)
    >>> plot_buckling_mode(nodes, elements, result.mode,
    ...                    title=f"Buckling Mode (λ_cr = {result.lambda_cr:.2f})")
    """
    import matplotlib

    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    coords = {n.id: i for i, n in enumerate(nodes)}
    mode_arr = np.asarray(mode, dtype=np.float64)

    # Compute scale for visualization (15% of structure extent)
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    max_disp = float(np.max(np.abs(mode_arr)))
    scale = 0.15 * span / max(max_disp, 1e-12)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot original geometry in light gray
    for e in elements:
        ni, nj = nodes[coords[e.node_i]], nodes[coords[e.node_j]]
        ax.plot([ni.x, nj.x], [ni.y, nj.y], color="lightgray", lw=1.5, alpha=0.7)

    # Plot buckling mode (deformed shape) with color based on displacement magnitude
    disp_magnitude = np.sqrt(mode_arr[::2] ** 2 + mode_arr[1::2] ** 2)
    max_node_disp = np.max(disp_magnitude)

    for i, e in enumerate(elements):
        ni, nj = nodes[coords[e.node_i]], nodes[coords[e.node_j]]
        idx_i, idx_j = coords[e.node_i], coords[e.node_j]

        # Deformed positions
        x_i_def = ni.x + mode_arr[2 * idx_i] * scale
        y_i_def = ni.y + mode_arr[2 * idx_i + 1] * scale
        x_j_def = nj.x + mode_arr[2 * idx_j] * scale
        y_j_def = nj.y + mode_arr[2 * idx_j + 1] * scale

        # Color by average displacement magnitude of element nodes
        avg_disp = (disp_magnitude[idx_i] + disp_magnitude[idx_j]) / 2
        color_intensity = avg_disp / max_node_disp if max_node_disp > 1e-12 else 0.5

        ax.plot(
            [x_i_def, x_j_def],
            [y_i_def, y_j_def],
            color=plt.cm.viridis(color_intensity),
            lw=2.5,
            alpha=0.9,
            label="Buckling mode" if i == 0 else None,
        )

    # Add node markers sized by displacement magnitude
    for i, n in enumerate(nodes):
        node_disp = disp_magnitude[i]
        size = 50 * (node_disp / max_node_disp if max_node_disp > 1e-12 else 0.5) + 30
        ax.plot(n.x, n.y, "o", color="red", ms=size, alpha=0.7)

    ax.set_title(_persian(title))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Add colorbar for displacement magnitude
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(0, max_node_disp))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Displacement magnitude (a.u.)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return save_path
