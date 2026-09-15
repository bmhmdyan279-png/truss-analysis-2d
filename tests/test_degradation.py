"""Tests for the degradation operator."""

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


# --------------------------------------------------------------------------
# Round-5 audit pins
# --------------------------------------------------------------------------


def test_scf_alpha_min_is_order_invariant(simple_determinate_truss):
    """``scf_alpha_min`` must select by VALUE, not by list position.

    The pre-fix code took ``scfs[-1]``, silently changing meaning for any
    non-descending ``alphas`` sequence (round-5 audit, finding F4).
    """
    nodes, elements, loads = simple_determinate_truss
    op = DamageOperator(nodes, elements, loads)
    descending = op.analyze_member("A", alphas=[1.0, 0.9, 0.8, 0.7])
    shuffled = op.analyze_member("A", alphas=[0.8, 1.0, 0.7, 0.9])
    assert math.isclose(descending.scf_alpha_min, shuffled.scf_alpha_min, rel_tol=1e-12)
    # And the value really is the SCF at the smallest non-singular alpha.
    smallest = min(
        (p for p in shuffled.points if not p.is_singular), key=lambda p: p.alpha
    )
    assert math.isclose(
        shuffled.scf_alpha_min,
        smallest.max_disp / shuffled.baseline_max_disp,
        rel_tol=1e-12,
    )


def test_key_element_detection_matches_mechanism_semantics(
    simple_determinate_truss, stable_indeterminate_truss
):
    """The Cholesky+dpocon probe must classify exactly like a rank test.

    Determinate truss: every member is kinematically essential. The
    indeterminate fixture: bottom-chord members 1-2 are redundant, every
    diagonal/top member is essential (pinned against the pre-fix full-SVD
    implementation, which agreed on all members -- round-5 finding F5).
    """
    nodes, elements, loads = simple_determinate_truss
    op = DamageOperator(nodes, elements, loads)
    for elem in elements:
        assert op._check_mechanism(
            nodes, op._apply_geometric_scaling(elements, elem.id, 1e-6)
        ), f"member {elem.id} of a determinate truss must be key"

    nodes2, elements2, loads2 = stable_indeterminate_truss
    op2 = DamageOperator(nodes2, elements2, loads2)
    expected_key = {
        "1": False,
        "2": False,
        "3": True,
        "4": True,
        "5": True,
        "6": True,
        "7": True,
    }
    for elem in elements2:
        got = op2._check_mechanism(
            nodes2, op2._apply_geometric_scaling(elements2, elem.id, 1e-6)
        )
        assert got == expected_key[elem.id], elem.id


def test_solve_forces_match_postprocess(simple_determinate_truss):
    """``_solve`` must recover forces through the canonical implementation.

    Guards the delegation to ``postprocess.calculate_element_forces`` --
    the "two implementations of one physics" pattern is what produced the
    2.6.0 demand-chain bugs (round-5 audit, finding F5).
    """
    from truss_analysis.postprocess import calculate_element_forces

    nodes, elements, loads = simple_determinate_truss
    # Give one member a thermal + fabrication strain so the prestress
    # convention is exercised, not just the bare mechanical elongation.
    elements[0].delta_T = 100.0
    elements[0].alpha = 1.2e-5
    elements[0].delta_L_free = 1e-4
    op = DamageOperator(nodes, elements, loads)
    U, forces = op._solve(nodes, elements)
    results, _, _ = calculate_element_forces(nodes, elements, U)
    for r in results:
        assert math.isclose(
            forces[str(r["id"])], float(r["N"]), rel_tol=1e-12, abs_tol=1e-6
        )


def test_thermal_degradation_missing_temperature_names_members():
    """A missing temperature entry must raise ValueError naming the member.

    Pre-fix this surfaced as a bare ``KeyError`` (round-5 audit, F7).
    """
    from truss_analysis.degradation import ThermalDegradation

    elements = [Element(id="bar", node_i="1", node_j="2", E=210e9, A=1e-3)]
    with pytest.raises(ValueError, match="bar"):
        ThermalDegradation(temps={}).apply(elements)
    # Complete mapping still works.
    out = ThermalDegradation(temps={"bar": 600.0}).apply(elements)
    assert out[0].E < elements[0].E
