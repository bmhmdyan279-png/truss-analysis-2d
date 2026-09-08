"""Thermal scenarios - one documented geometric criterion (member centroid).

Every local scenario is defined on the **member geometric centroid**, which
yields an exact partition ``left | mid | right`` (half-open intervals,
pairwise disjoint, complete - zero orphan members) and covers the five
supported scenarios (``uniform``, ``local_left``, ``local_mid``,
``local_right``, ``linear_gradient``).  The criterion is family-independent
by construction: it uses only node coordinates, never family-specific
member numbering or topology.
"""

from __future__ import annotations

from ..model import Element, Node

T_AMBIENT: float = 20.0

SCENARIOS: tuple[str, ...] = (
    "uniform",
    "local_left",
    "local_mid",
    "local_right",
    "linear_gradient",
)


def span_bounds(nodes: list[Node]) -> tuple[float, float]:
    """Return ``(min_x, span)``; a zero span degrades to 1.0 (documented)."""
    xs = [n.x for n in nodes]
    min_x = min(xs)
    span = max(xs) - min_x
    return min_x, (span if span > 0.0 else 1.0)


def relative_eps(span: float) -> float:
    """Return the boundary tolerance, relative to the span.

    A fixed absolute epsilon would not scale with the geometry units; the
    relative form keeps the snapping guard meaningful for spans expressed
    in metres as well as millimetres.
    """
    return max(1e-12, 1e-9 * span)


def member_centroids(nodes: list[Node], elements: list[Element]) -> dict[str, float]:
    """Geometric centroid x-coordinate of every member."""
    node_map = {n.id: n for n in nodes}
    return {e.id: (node_map[e.node_i].x + node_map[e.node_j].x) / 2.0 for e in elements}


def scenario_partition(
    nodes: list[Node], elements: list[Element]
) -> tuple[set[str], set[str], set[str]]:
    """Exact partition of members by centroid: ``[0,L/3) | [L/3,2L/3] | (2L/3,L]``.

    Centroids within ``relative_eps`` of a boundary are snapped onto the
    boundary first (the documented IEEE-754 guard); the half-open intervals
    then make the assignment exact, pairwise disjoint and complete.
    """
    min_x, span = span_bounds(nodes)
    eps = relative_eps(span)
    left_bound = min_x + span / 3.0
    right_bound = min_x + 2.0 * span / 3.0
    left: set[str] = set()
    mid: set[str] = set()
    right: set[str] = set()
    for eid, xc in member_centroids(nodes, elements).items():
        if abs(xc - left_bound) <= eps:
            xc = left_bound
        elif abs(xc - right_bound) <= eps:
            xc = right_bound
        if xc < left_bound:
            left.add(eid)
        elif xc <= right_bound:
            mid.add(eid)
        else:
            right.add(eid)
    return left, mid, right


def get_scenario_temperatures(
    nodes: list[Node],
    elements: list[Element],
    scenario: str,
    t_target: float,
) -> dict[str, float]:
    """Member temperature field [degC] for each supported scenario.

    * ``uniform``: every member at ``t_target``.
    * ``local_left`` / ``local_mid`` / ``local_right``: hot third at
      ``t_target``, the rest at :data:`T_AMBIENT` (centroid partition).
    * ``linear_gradient``: linear in the centroid position from
      :data:`T_AMBIENT` at ``min_x`` to ``t_target`` at ``max_x``.
    """
    if scenario not in SCENARIOS:
        msg = f"Unknown thermal scenario: '{scenario}' (expected one of {SCENARIOS})"
        raise ValueError(msg)
    min_x, span = span_bounds(nodes)
    centroids = member_centroids(nodes, elements)
    hot: set[str] = set()
    if scenario in ("local_left", "local_mid", "local_right"):
        left, mid, right = scenario_partition(nodes, elements)
        hot = {"local_left": left, "local_mid": mid, "local_right": right}[scenario]
    temps: dict[str, float] = {}
    for e in elements:
        if scenario == "uniform":
            temps[e.id] = float(t_target)
        elif scenario == "linear_gradient":
            ratio = (centroids[e.id] - min_x) / span
            temps[e.id] = T_AMBIENT + (float(t_target) - T_AMBIENT) * ratio
        else:
            temps[e.id] = float(t_target) if e.id in hot else T_AMBIENT
    return temps
