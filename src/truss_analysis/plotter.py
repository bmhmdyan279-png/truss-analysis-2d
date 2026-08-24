"""Visualization module for truss analysis."""

import matplotlib

matplotlib.use("Agg")  # Backend without GUI
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def setup_persian_font():
    """Setup Persian font (Vazirmatn) for plots."""
    font_path = (
        Path(__file__).parent.parent
        / "templates"
        / "assets"
        / "fonts"
        / "Vazirmatn-Regular.ttf"
    )
    if font_path.exists():
        from matplotlib.font_manager import FontProperties

        return FontProperties(fname=str(font_path))
    return None


def plot_truss(
    nodes: List,
    elements: List,
    displacements: Optional[Dict] = None,
    show: bool = True,
    filename: Optional[str] = None,
    scale: float = 10.0,
    font_prop=None,
):
    """Plot original and deformed truss."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Original truss
    ax1 = axes[0]
    for elem in elements:
        n1 = next(n for n in nodes if n.id == elem.node_i)
        n2 = next(n for n in nodes if n.id == elem.node_j)
        ax1.plot([n1.x, n2.x], [n1.y, n2.y], "k-", linewidth=2)

    # Draw nodes
    node_x = [n.x for n in nodes]
    node_y = [n.y for n in nodes]
    ax1.scatter(node_x, node_y, c="black", s=50, zorder=5)

    # Draw supports
    for node in nodes:
        if node.is_support:
            if node.support_dx and node.support_dy:
                ax1.plot(node.x, node.y, "s", color="red", markersize=12, zorder=6)
            elif node.support_dy:
                ax1.plot(node.x, node.y, "^", color="blue", markersize=12, zorder=6)

    title1 = "خرپای اولیه" if font_prop else "Original Truss"
    ax1.set_title(title1, fontproperties=font_prop, fontsize=14, fontweight="bold")
    ax1.set_xlabel("X (m)", fontproperties=font_prop)
    ax1.set_ylabel("Y (m)", fontproperties=font_prop)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Deformed truss
    ax2 = axes[1]
    if displacements:
        for elem in elements:
            n1 = next(n for n in nodes if n.id == elem.node_i)
            n2 = next(n for n in nodes if n.id == elem.node_j)
            u1 = displacements.get(n1.id, (0, 0))
            u2 = displacements.get(n2.id, (0, 0))
            x1_def = n1.x + scale * u1[0]
            y1_def = n1.y + scale * u1[1]
            x2_def = n2.x + scale * u2[0]
            y2_def = n2.y + scale * u2[1]
            ax2.plot([x1_def, x2_def], [y1_def, y2_def], "r-", linewidth=2)

        # Draw deformed nodes
        def_x = [n.x + scale * displacements.get(n.id, (0, 0))[0] for n in nodes]
        def_y = [n.y + scale * displacements.get(n.id, (0, 0))[1] for n in nodes]
        ax2.scatter(def_x, def_y, c="red", s=50, zorder=5)

    title2 = (
        f"خرپای تغییرشکل‌یافته (ضریب: {scale}x)"
        if font_prop
        else f"Deformed Truss (scale: {scale}x)"
    )
    ax2.set_title(title2, fontproperties=font_prop, fontsize=14, fontweight="bold")
    ax2.set_xlabel("X (m)", fontproperties=font_prop)
    ax2.set_ylabel("Y (m)", fontproperties=font_prop)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"📊 Plot saved to {filename}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_axial_force(
    nodes: List,
    elements: List,
    forces: List[Dict],
    filename: Optional[str] = None,
    font_prop=None,
):
    """Plot axial force diagram with tension (blue) and compression (red)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    force_dict = {f["element"]: f["force"] for f in forces}

    for elem in elements:
        n1 = next(n for n in nodes if n.id == elem.node_i)
        n2 = next(n for n in nodes if n.id == elem.node_j)
        force = force_dict.get(elem.id, 0)

        # Blue for tension, red for compression
        color = "blue" if force >= 0 else "red"
        linewidth = 2 + min(abs(force) * 0.001, 6)

        ax.plot([n1.x, n2.x], [n1.y, n2.y], color=color, linewidth=linewidth)

        # Add force label at midpoint
        mid_x = (n1.x + n2.x) / 2
        mid_y = (n1.y + n2.y) / 2
        label = f"{force:.1f}"
        ax.text(
            mid_x,
            mid_y,
            label,
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    title = (
        "دیاگرام نیروی محوری (آبی: کشش، قرمز: فشار)"
        if font_prop
        else "Axial Force Diagram (Blue: Tension, Red: Compression)"
    )
    ax.set_title(title, fontproperties=font_prop, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"📊 Axial force plot saved to {filename}")
    plt.close(fig)
