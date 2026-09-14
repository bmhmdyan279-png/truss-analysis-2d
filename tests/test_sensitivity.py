"""Tests for the independent validator (sensitivity checks)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from truss_analysis.model import Element, Node
from truss_analysis.reliability_adapter import NodalLoad
from truss_analysis.sensitivity import IndependentValidator


def _create_simple_truss() -> tuple[list[Node], list[Element], list[NodalLoad]]:
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=3.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="3", x=1.5, y=2.0, is_support=False),
    ]
    elements = [
        Element(id="1", node_i="1", node_j="3", E=200e9, A=0.01),
        Element(id="2", node_i="2", node_j="3", E=200e9, A=0.01),
        # bottom chord: without it the two-member "truss" is a kinematic
        # mechanism (rank(K_ff)=2 < 3 free DOFs); the legacy solver solved it
        # silently; the rank guard raises instead.
        Element(id="3", node_i="1", node_j="2", E=200e9, A=0.01),
    ]
    loads = [NodalLoad(node_id="3", fx=1000.0, fy=-2000.0)]
    return nodes, elements, loads


def test_independent_validator_initialization() -> None:
    nodes, elements, loads = _create_simple_truss()
    validator = IndependentValidator(nodes, elements, loads)
    assert len(validator.nodes) == 3
    assert len(validator.elements) == 3
    assert validator.node_map["1"] == 0


def test_independent_validator_compute_all() -> None:
    nodes, elements, loads = _create_simple_truss()
    validator = IndependentValidator(nodes, elements, loads)
    results = validator.compute_all()

    assert len(results) == 3
    for res in results:
        assert res.member_id in ["1", "2", "3"]
        assert isinstance(res.ddm_sensitivity, float)
        assert isinstance(res.strain_energy, float)
        assert res.strain_energy >= 0.0


def test_independent_validator_zero_displacement() -> None:
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=1.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
    ]
    elements = [Element(id="1", node_i="1", node_j="2", E=200e9, A=0.01)]
    loads: list[NodalLoad] = []

    validator = IndependentValidator(nodes, elements, loads)
    results = validator.compute_all()

    assert len(results) == 1
    assert results[0].ddm_sensitivity == 0.0
    assert results[0].strain_energy == 0.0


def test_independent_validator_with_reference_problem() -> None:
    ref_path = Path("examples/reference_problem.json")
    if not ref_path.exists():
        pytest.skip("reference_problem.json not found")

    with open(ref_path, encoding="utf-8") as f:
        data = json.load(f)

    nodes = [
        Node(
            id=str(n["id"]),
            x=float(n["x"]),
            y=float(n["y"]),
            is_support=bool(n.get("is_support", False)),
            support_dx=bool(n.get("support_dx", False)),
            support_dy=bool(n.get("support_dy", False)),
        )
        for n in data["nodes"]
    ]
    elements = [
        Element(
            id=str(e["id"]),
            node_i=str(e["node_i"]),
            node_j=str(e["node_j"]),
            E=float(e["E"]),
            A=float(e["A"]),
        )
        for e in data["elements"]
    ]
    loads = [
        NodalLoad(
            node_id=str(ld["node_id"]),
            fx=float(ld["Fx"]),
            fy=float(ld["Fy"]),
        )
        for ld in data.get("loads", [])
    ]

    validator = IndependentValidator(nodes, elements, loads)
    results = validator.compute_all()

    assert len(results) == 7
    for res in results:
        assert isinstance(res.ddm_sensitivity, float)
        assert res.strain_energy >= 0.0


def _three_bar_indeterminate(alpha: float, delta_T: float):
    """Statically indeterminate three-bar truss (both supports fully fixed).

    With ``alpha > 0`` on member 1 the baseline carries restrained thermal
    expansion, so the DDM numerator must be the MECHANICAL elongation
    ``b_i^T U_f - dL_pre,i``: the area enters both ``K`` and the imposed
    force ``k_i dL_pre,i b_i``.
    """
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="3", x=2.0, y=1.5, is_support=False),
    ]
    elements = [
        Element(
            id="1",
            node_i="1",
            node_j="3",
            E=210e9,
            A=0.005,
            alpha=alpha,
            delta_T=delta_T,
        ),
        Element(id="2", node_i="2", node_j="3", E=210e9, A=0.008),
        Element(id="3", node_i="1", node_j="2", E=210e9, A=0.006),
    ]
    loads = [NodalLoad(node_id="3", fx=20e3, fy=-50e3)]
    return nodes, elements, loads


def _max_nodal_magnitude(nodes, elements, loads) -> float:
    validator = IndependentValidator(nodes, elements, loads)
    U, _ = validator.compute_baseline()
    return float(np.max(np.hypot(U[0::2], U[1::2])))


def _fd_sensitivity(nodes, elements, loads, i: int, d_area: float) -> float:
    """Central difference of the max nodal magnitude w.r.t. member area i."""
    plus = [replace(e, A=e.A + d_area) if j == i else e for j, e in enumerate(elements)]
    minus = [
        replace(e, A=e.A - d_area) if j == i else e for j, e in enumerate(elements)
    ]
    return (
        _max_nodal_magnitude(nodes, plus, loads)
        - _max_nodal_magnitude(nodes, minus, loads)
    ) / (2 * d_area)


@pytest.mark.parametrize(("alpha", "delta_T"), [(1.2e-5, 400.0), (0.0, 0.0)])
def test_ddm_matches_central_finite_difference(alpha: float, delta_T: float) -> None:
    """DDM == central difference of max nodal magnitude, prestress or not.

    The heated case pins the mechanical-elongation numerator: with the total
    elongation the member-1 sensitivity came out with the wrong sign and two
    orders of magnitude too large (round-4 audit, critic 5 finding 1).
    """
    nodes, elements, loads = _three_bar_indeterminate(alpha, delta_T)
    results = IndependentValidator(nodes, elements, loads).compute_all()
    for i, res in enumerate(results):
        d_area = 1e-5 * elements[i].A
        fd = _fd_sensitivity(nodes, elements, loads, i, d_area)
        scale = max(abs(fd), 1e-12)
        assert abs(res.ddm_sensitivity - fd) <= 1e-5 * scale + 1e-12, (
            f"member {res.member_id}: DDM={res.ddm_sensitivity!r} FD={fd!r}"
        )
