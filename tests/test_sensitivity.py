"""Tests for Phase 5 IndependentValidator."""

import json
from pathlib import Path

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
    ]
    loads = [NodalLoad(node_id="3", fx=1000.0, fy=-2000.0)]
    return nodes, elements, loads


def test_independent_validator_initialization() -> None:
    nodes, elements, loads = _create_simple_truss()
    validator = IndependentValidator(nodes, elements, loads)
    assert len(validator.nodes) == 3
    assert len(validator.elements) == 2
    assert validator.node_map["1"] == 0


def test_independent_validator_compute_all() -> None:
    nodes, elements, loads = _create_simple_truss()
    validator = IndependentValidator(nodes, elements, loads)
    results = validator.compute_all()

    assert len(results) == 2
    for res in results:
        assert res.member_id in ["1", "2"]
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

    with open(ref_path, "r", encoding="utf-8") as f:
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
