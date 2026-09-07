"""Graph-theoretic validation for generated truss topologies.

Ensures structural validity *before* any FEM analysis is performed.
All three checks must pass for a topology to be accepted into the
Phase 1 study set (18 indeterminate + 3 determinate = 21 models).

Checks
------
1. **Connectivity** — the truss graph is connected (BFS).
2. **No orphan nodes** — every node is referenced by ≥ 1 element.
3. **No zero-length elements** — every element has L > 0.
"""

from __future__ import annotations

from typing import Any

__all__ = ["TopologyValidationError", "validate_topology"]


class TopologyValidationError(Exception):
    """Raised when a generated topology fails validation."""


def validate_topology(model: dict[str, Any]) -> None:
    """Validate a truss model dictionary.

    Parameters
    ----------
    model:
        Truss model dictionary (output of ``TopologyGenerator.generate``).

    Raises
    ------
    TopologyValidationError
        If any validation check fails.
    """
    nodes: list[dict[str, Any]] = model["nodes"]
    elements: list[dict[str, Any]] = model["elements"]

    _check_orphan_nodes(nodes, elements)
    _check_zero_length_elements(nodes, elements)
    _check_connectivity(nodes, elements)


def _check_orphan_nodes(
    nodes: list[dict[str, Any]], elements: list[dict[str, Any]]
) -> None:
    """Ensure every node is referenced by at least one element."""
    node_ids = {n["id"] for n in nodes}
    referenced: set[int] = set()
    for elem in elements:
        referenced.add(elem["node_i"])
        referenced.add(elem["node_j"])

    orphans = node_ids - referenced
    if orphans:
        raise TopologyValidationError(f"Orphan nodes found: {sorted(orphans)}")


def _check_zero_length_elements(
    nodes: list[dict[str, Any]], elements: list[dict[str, Any]]
) -> None:
    """Ensure no element has zero (or near-zero) length."""
    node_map = {n["id"]: (n["x"], n["y"]) for n in nodes}
    for elem in elements:
        xi, yi = node_map[elem["node_i"]]
        xj, yj = node_map[elem["node_j"]]
        length_sq = (xj - xi) ** 2 + (yj - yi) ** 2
        if length_sq < 1.0e-20:
            raise TopologyValidationError(
                f"Zero-length element id={elem['id']} "
                f"(nodes {elem['node_i']} → {elem['node_j']})"
            )


def _check_connectivity(
    nodes: list[dict[str, Any]], elements: list[dict[str, Any]]
) -> None:
    """Ensure the truss graph is connected via BFS."""
    if not nodes:
        raise TopologyValidationError("Empty node list")

    node_ids = {n["id"] for n in nodes}

    # Build adjacency list
    adj: dict[int, set[int]] = {nid: set() for nid in node_ids}
    for elem in elements:
        ni, nj = elem["node_i"], elem["node_j"]
        adj[ni].add(nj)
        adj[nj].add(ni)

    # BFS from first node
    start = nodes[0]["id"]
    visited: set[int] = {start}
    queue: list[int] = [start]
    while queue:
        current = queue.pop(0)
        for neighbour in adj[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    if visited != node_ids:
        unreachable = node_ids - visited
        raise TopologyValidationError(
            f"Disconnected graph — unreachable nodes: {sorted(unreachable)}"
        )
