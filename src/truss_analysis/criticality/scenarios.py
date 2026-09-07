"""Thermal scenarios — one documented geometric criterion (member centroid).

History (DR-002 / CONTEXT_LOCK §4.4): the pre-prompt-4 code used a "both
nodes in the third" rule for ``local_left``, a centroid rule for
``local_mid``, and the proposal text described a third "any node" rule —
three incompatible definitions, 10 orphan members and a 30 % under-coverage
of the fire zone on ``pratt_8``/``howe_8``.

Prompt 4 unifies every local scenario on the **member geometric centroid**,
which yields an exact partition ``left | mid | right`` (half-open intervals,
pairwise disjoint, complete — zero orphans) and adds the missing
``local_right`` scenario.  Family-independent by construction.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from truss_analysis.model import Element, Node

T_AMBIENT: float = 20.0

SCENARIOS: Tuple[str, ...] = (
    "uniform",
    "local_left",
    "local_mid",
    "local_right",
    "linear_gradient",
)


def span_bounds(nodes: List[Node]) -> Tuple[float, float]:
    """Return ``(min_x, span)``; a zero span degrades to 1.0 (documented)."""
    xs = [n.x for n in nodes]
    min_x = min(xs)
    span = max(xs) - min_x
    return min_x, (span if span > 0.0 else 1.0)


def relative_eps(span: float) -> float:
    """Boundary tolerance, relative to the span (prompt-04 §C13).

    Replaces the legacy absolute ``eps = 1e-9`` which did not scale with the
    geometry units (CONTEXT_LOCK §4.5 B5).
    """
    return max(1e-12, 1e-9 * span)


def member_centroids(nodes: List[Node], elements: List[Element]) -> Dict[str, float]:
    """Geometric centroid x-coordinate of every member."""
    node_map = {n.id: n for n in nodes}
    return {e.id: (node_map[e.node_i].x + node_map[e.node_j].x) / 2.0 for e in elements}


def scenario_partition(
    nodes: List[Node], elements: List[Element]
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Exact partition of members by centroid: ``[0,L/3) | [L/3,2L/3] | (2L/3,L]``.

    Centroids within ``relative_eps`` of a boundary are snapped onto the
    boundary first (the documented IEEE-754 guard); the half-open intervals
    then make the assignment exact, pairwise disjoint and complete.
    """
    min_x, span = span_bounds(nodes)
    eps = relative_eps(span)
    left_bound = min_x + span / 3.0
    right_bound = min_x + 2.0 * span / 3.0
    left: Set[str] = set()
    mid: Set[str] = set()
    right: Set[str] = set()
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
    nodes: List[Node],
    elements: List[Element],
    scenario: str,
    t_target: float,
) -> Dict[str, float]:
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
    hot: Set[str] = set()
    if scenario in ("local_left", "local_mid", "local_right"):
        left, mid, right = scenario_partition(nodes, elements)
        hot = {"local_left": left, "local_mid": mid, "local_right": right}[scenario]
    temps: Dict[str, float] = {}
    for e in elements:
        if scenario == "uniform":
            temps[e.id] = float(t_target)
        elif scenario == "linear_gradient":
            ratio = (centroids[e.id] - min_x) / span
            temps[e.id] = T_AMBIENT + (float(t_target) - T_AMBIENT) * ratio
        else:
            temps[e.id] = float(t_target) if e.id in hot else T_AMBIENT
    return temps
