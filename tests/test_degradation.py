"""Tests for Phase 4 degradation operator."""

from __future__ import annotations

import math

import pytest
from truss_analysis.degradation import DamageOperator
from truss_analysis.model import Element, Node
from truss_analysis.reliability_adapter import NodalLoad


@pytest.fixture
def simple_determinate_truss() -> tuple[list[Node], list[Element], list[NodalLoad]]:
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="3", x=2.0, y=3.0, is_support=False),
    ]
    elements = [
        Element(id="A", node_i="1", node_j="3", E=200e9, A=0.01, I_sec=1e-4),
        Element(id="B", node_i="2", node_j="3", E=200e9, A=0.01, I_sec=1e-4),
        Element(id="C", node_i="1", node_j="2", E=200e9, A=0.01, I_sec=1e-4),
    ]
    loads = [NodalLoad(node_id="3", fx=10000.0, fy=-20000.0)]
    return nodes, elements, loads


@pytest.fixture
def stable_indeterminate_truss() -> tuple[list[Node], list[Element], list[NodalLoad]]:
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=False),
        Node(id="3", x=8.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="4", x=2.0, y=3.0, is_support=False),
        Node(id="5", x=6.0, y=3.0, is_support=False),
    ]
    elements = [
        Element(id="1", node_i="1", node_j="2", E=210e9, A=0.01, I_sec=8e-10),
        Element(id="2", node_i="2", node_j="3", E=210e9, A=0.01, I_sec=8e-10),
        Element(id="3", node_i="1", node_j="4", E=210e9, A=0.015, I_sec=1.2e-9),
        Element(id="4", node_i="4", node_j="2", E=210e9, A=0.015, I_sec=1.2e-9),
        Element(id="5", node_i="2", node_j="5", E=210e9, A=0.015, I_sec=1.2e-9),
        Element(id="6", node_i="5", node_j="3", E=210e9, A=0.015, I_sec=1.2e-9),
        Element(id="7", node_i="4", node_j="5", E=210e9, A=0.02, I_sec=1.6e-9),
    ]
    loads = [NodalLoad(node_id="4", fx=20000.0, fy=-30000.0)]
    return nodes, elements, loads


def test_baseline_alpha_equals_one(simple_determinate_truss):
    nodes, elements, loads = simple_determinate_truss
    op = DamageOperator(nodes, elements, loads)
    prof = op.analyze_member("A", alphas=[1.0])
    assert len(prof.points) == 1
    assert not prof.points[0].is_singular
    scf = prof.points[0].max_disp / prof.baseline_max_disp
    assert math.isclose(scf, 1.0, rel_tol=1e-9)


def test_geometric_scaling_rule(simple_determinate_truss):
    nodes, elements, loads = simple_determinate_truss
    op = DamageOperator(nodes, elements, loads)
    alpha = 0.5
    degraded = op._apply_geometric_scaling(elements, "A", alpha)
    elem_a = next(e for e in degraded if e.id == "A")
    original_a = next(e for e in elements if e.id == "A")
    assert math.isclose(elem_a.A, original_a.A * alpha)
    assert math.isclose(elem_a.I_sec, original_a.I_sec * (alpha**2))


def test_stable_indeterminate_run(stable_indeterminate_truss):
    nodes, elements, loads = stable_indeterminate_truss
    op = DamageOperator(nodes, elements, loads)
    prof = op.analyze_member("1", alphas=[1.0, 0.9, 0.8, 0.7], probe_near_zero=True)
    assert isinstance(prof.is_key_element, bool)
    assert not any(p.is_singular for p in prof.points)
    assert len(prof.points) == 4
