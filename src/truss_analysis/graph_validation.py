"""Graph-theoretic and numerical validation for generated truss topologies.

Ensures structural validity *before* any FEM analysis is performed and
reports the numerical quality of the base stiffness matrix.

Checks / report fields
----------------------
1. **Connectivity** — the truss graph is connected (BFS).
2. **No orphan nodes** — every node is referenced by >= 1 element.
3. **No zero-length elements** — every element has L > 0.
4. **No self-loops / duplicate parallel members** — the member multiset over
   unordered node pairs is simple.
5. **Static determinacy** — ``m + r - 2j`` (0 = determinate, > 0 = degree of
   static indeterminacy, < 0 = under-braced).
6. **Numerical rank and condition of ``K_ff``** via SVD — not merely
   "singular or not": a mechanism is declared when ``rank(K_ff) < n_dof_free``
   (SVD-based detection, the same criterion the solver applies), and a
   ``cond(K_ff) > 1e12`` raises a logged warning (ill-conditioning).
7. **Geometric symmetry** — mirror bijection on nodes preserving supports,
   members and (mirrored) loads.

``validate_topology`` raises :class:`TopologyValidationError` on any hard
failure (items 1-4 and mechanisms); :func:`structural_report` returns the
full :class:`TopologyReport` for cataloguing.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .assembly import assemble_global_matrices
from .model import Element, Node
from .numerics import (
    DEFAULT_TOLERANCES,
    NumericalStatus,
    classify_conditioning,
)

__all__ = [
    "COND_WARNING_THRESHOLD",
    "TopologyReport",
    "TopologyValidationError",
    "structural_report",
    "validate_topology",
]

logger = logging.getLogger(__name__)

#: Condition number above which ``TopologyReport.cond_warning`` is set.
#: Kept as a module constant for backwards compatibility; the authoritative
#: value is :attr:`NumericalTolerances.cond_warning`.
COND_WARNING_THRESHOLD = DEFAULT_TOLERANCES.cond_warning

#: Relative singular-value cutoff for the *diagnostic* near-mechanism test.
#: Deliberately looser than the solver's hard gate
#: (:attr:`NumericalTolerances.rank_rel_cutoff`) so model validation warns
#: about a marginal structure earlier than the solver refuses to factorise
#: it. Both values come from one policy so the relationship cannot drift.
_RANK_RTOL = DEFAULT_TOLERANCES.near_singular_rel_cutoff


class TopologyValidationError(Exception):
    """Raised when a generated topology fails validation."""


@dataclass(frozen=True)
class TopologyReport:
    """Complete structural description of one topology."""

    n_nodes: int
    n_members: int
    n_reactions: int
    n_dof_free: int
    indeterminacy: int
    connected: bool
    orphan_nodes: tuple[int, ...]
    zero_length_members: tuple[int, ...]
    duplicate_members: tuple[int, ...]
    self_loops: tuple[int, ...]
    mechanism: bool
    rank_k_ff: int
    cond_k_ff: float
    cond_warning: bool
    symmetric: bool
    numerical_status: NumericalStatus = NumericalStatus.STABLE


def _to_objects(
    model: dict[str, Any],
) -> tuple[list[Node], list[Element]]:
    nodes = [
        Node(
            id=str(n["id"]),
            x=float(n["x"]),
            y=float(n["y"]),
            is_support=bool(n.get("is_support", False)),
            support_dx=bool(n.get("support_dx", False)),
            support_dy=bool(n.get("support_dy", False)),
        )
        for n in model["nodes"]
    ]
    elements = [
        Element(
            id=str(e["id"]),
            node_i=str(e["node_i"]),
            node_j=str(e["node_j"]),
            E=float(e["E"]),
            A=float(e["A"]),
        )
        for e in model["elements"]
    ]
    return nodes, elements


def _fixed_dofs(nodes: list[Node]) -> list[int]:
    fixed: list[int] = []
    for i, node in enumerate(nodes):
        if node.is_support:
            if node.support_dx:
                fixed.append(2 * i)
            if node.support_dy:
                fixed.append(2 * i + 1)
    return fixed


def _graph_checks(
    model: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], bool]:
    nodes = model["nodes"]
    elements = model["elements"]
    node_ids = {n["id"] for n in nodes}

    referenced: set[int] = set()
    for elem in elements:
        referenced.add(elem["node_i"])
        referenced.add(elem["node_j"])
    orphans = tuple(sorted(node_ids - referenced))

    coords = {n["id"]: (n["x"], n["y"]) for n in nodes}
    zero_len = tuple(
        sorted(
            e["id"]
            for e in elements
            if (coords[e["node_j"]][0] - coords[e["node_i"]][0]) ** 2
            + (coords[e["node_j"]][1] - coords[e["node_i"]][1]) ** 2
            < 1.0e-20
        )
    )
    self_loops = tuple(sorted(e["id"] for e in elements if e["node_i"] == e["node_j"]))

    seen: dict[tuple[int, int], int] = {}
    duplicates: list[int] = []
    for e in elements:
        key = tuple(sorted((e["node_i"], e["node_j"])))
        if key in seen:
            duplicates.append(e["id"])
        else:
            seen[key] = e["id"]

    adj: dict[int, set[int]] = {nid: set() for nid in node_ids}
    for elem in elements:
        adj[elem["node_i"]].add(elem["node_j"])
        adj[elem["node_j"]].add(elem["node_i"])
    start = nodes[0]["id"]
    visited = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nb in adj[current]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    connected = visited == node_ids
    return orphans, zero_len, tuple(sorted(duplicates)), self_loops, connected


def _symmetric(model: dict[str, Any]) -> bool:
    nodes = model["nodes"]
    xs = [n["x"] for n in nodes]
    total = min(xs) + max(xs)
    by_pos: dict[tuple[float, float], int] = {}
    for n in nodes:
        by_pos[(round(n["x"], 9), round(n["y"], 9))] = n["id"]
    mirror_node: dict[int, int] = {}
    for n in nodes:
        key = (round(total - n["x"], 9), round(n["y"], 9))
        if key not in by_pos:
            return False
        mirror_node[n["id"]] = by_pos[key]
    for n in nodes:
        m = next(o for o in nodes if o["id"] == mirror_node[n["id"]])
        # support *presence* must mirror; the dx/dy restraint split (pin vs
        # roller) is a statics convention, not geometry, so it is ignored.
        if bool(m.get("is_support")) != bool(n.get("is_support")):
            return False
    pairs = {tuple(sorted((e["node_i"], e["node_j"]))) for e in model["elements"]}
    mirrored = {
        tuple(sorted((mirror_node[e["node_i"]], mirror_node[e["node_j"]])))
        for e in model["elements"]
    }
    if pairs != mirrored:
        return False
    loads = {ld["node_id"]: ld for ld in model.get("loads", [])}
    for nid, ld in loads.items():
        other = loads.get(mirror_node[nid])
        if other is None:
            return False
        if abs(other.get("Fx", 0.0) + ld.get("Fx", 0.0)) > 1e-12:
            return False
        if abs(other.get("Fy", 0.0) - ld.get("Fy", 0.0)) > 1e-12:
            return False
    return True


def structural_report(model: dict[str, Any]) -> TopologyReport:
    """Full structural report including SVD rank and condition of K_ff."""
    orphans, zero_len, dups, loops, connected = _graph_checks(model)
    n_nodes = len(model["nodes"])
    n_members = len(model["elements"])
    n_reactions = sum(
        (2 if (n.get("support_dx") and n.get("support_dy")) else 1)
        for n in model["nodes"]
        if n.get("is_support")
    )
    n_dof_free = 2 * n_nodes - n_reactions
    indeterminacy = n_members + n_reactions - 2 * n_nodes

    if orphans or zero_len or dups or loops or not connected:
        # graph-level defects: object conversion / stiffness assembly are not
        # meaningful yet (self-loops and zero lengths are rejected here, before
        # the Element constructor or the assembler can raise their own errors)
        return TopologyReport(
            n_nodes=n_nodes,
            n_members=n_members,
            n_reactions=n_reactions,
            n_dof_free=n_dof_free,
            indeterminacy=indeterminacy,
            connected=connected,
            orphan_nodes=orphans,
            zero_length_members=zero_len,
            duplicate_members=dups,
            self_loops=loops,
            mechanism=False,
            rank_k_ff=-1,
            cond_k_ff=float("nan"),
            cond_warning=False,
            symmetric=_symmetric(model),
            numerical_status=NumericalStatus.STABLE,
        )

    nodes, elements = _to_objects(model)
    k_full, _, _, fixed_back = assemble_global_matrices(nodes, elements)
    free = [d for d in range(2 * n_nodes) if d not in set(fixed_back)]
    k_ff = k_full[np.ix_(free, free)]
    sing = np.linalg.svd(k_ff, compute_uv=False)
    s_max = float(sing[0]) if sing.size else 0.0
    rank = int(np.count_nonzero(sing > s_max * _RANK_RTOL)) if s_max > 0.0 else 0
    mechanism = rank < n_dof_free
    cond = float(s_max / sing[-1]) if (sing.size and sing[-1] > 0.0) else float("inf")
    cond_warning = cond > COND_WARNING_THRESHOLD
    # Structural classification is deliberately kept separate from numerical
    # verdicts: determinacy is algebraic, stability is rank(K_ff), and
    # conditioning is a statement about round-off amplification. Conflating
    # them makes a report unreadable, so the status is recorded explicitly.
    numerical_status = classify_conditioning(cond, mechanism, DEFAULT_TOLERANCES)
    return TopologyReport(
        n_nodes=n_nodes,
        n_members=n_members,
        n_reactions=n_reactions,
        n_dof_free=n_dof_free,
        indeterminacy=indeterminacy,
        connected=connected,
        orphan_nodes=orphans,
        zero_length_members=zero_len,
        duplicate_members=dups,
        self_loops=loops,
        mechanism=mechanism,
        rank_k_ff=rank,
        cond_k_ff=cond,
        cond_warning=cond_warning,
        symmetric=_symmetric(model),
        numerical_status=numerical_status,
    )


def validate_topology(model: dict[str, Any]) -> None:
    """Validate a truss model dictionary; raise on any hard failure.

    Parameters
    ----------
    model:
        Model dictionary (output of ``TopologyGenerator.generate`` or
        ``TopologyGenerator.generate_determinate_control``).

    Raises
    ------
    TopologyValidationError
        If connectivity, orphan, zero-length, simplicity or mechanism checks
        fail.  Ill-conditioning (``cond > 1e12``) only logs a warning.
    """
    report = structural_report(model)
    if report.orphan_nodes:
        raise TopologyValidationError(f"Orphan nodes found: {report.orphan_nodes}")
    if report.zero_length_members:
        raise TopologyValidationError(
            f"Zero-length elements: {report.zero_length_members}"
        )
    if report.self_loops:
        raise TopologyValidationError(f"Self-loop elements: {report.self_loops}")
    if report.duplicate_members:
        raise TopologyValidationError(
            f"Duplicate parallel elements: {report.duplicate_members}"
        )
    if not report.connected:
        raise TopologyValidationError("Disconnected truss graph")
    if report.mechanism:
        raise TopologyValidationError(
            f"Kinematic mechanism: rank(K_ff)={report.rank_k_ff} < "
            f"n_dof_free={report.n_dof_free}"
        )
    if report.cond_warning:
        logger.warning(
            "ill-conditioned base stiffness: cond(K_ff)=%.3e (topology accepted)",
            report.cond_k_ff,
        )
