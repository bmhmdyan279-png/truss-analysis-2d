"""Limit-state tests: hand solutions, DCR, theta_cr, governing switch."""

from __future__ import annotations

import math

import pytest

from truss_analysis.limitstates import (
    ci_two_component,
    dcr_field,
    member_axial_forces,
    member_critical_temperature,
    system_critical_temperature,
    yield_capacity,
)
from truss_analysis.model import Element, Node

F_Y = 235.0e6
E = 210.0e9
P_CR_HAND = math.pi**2 * E * 1e-6 / 2.0**2  # 518154.2310571913 N


def _column(i_sec=1e-6, load=100.0e3):
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        # guided support: horizontal DOF restrained, vertical free ->
        # single compression member with a non-singular K_ff
        Node(id="2", x=0.0, y=2.0, is_support=True, support_dx=True, support_dy=False),
    ]
    elements = [
        Element(id="c", node_i="1", node_j="2", E=E, A=0.01, I_sec=i_sec),
    ]
    loads = {"2": {"Fx": 0.0, "Fy": -load}}
    return nodes, elements, loads


def _tie():
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=3.0, y=0.0, is_support=True, support_dy=True),
    ]
    elements = [Element(id="t", node_i="1", node_j="2", E=E, A=0.01, I_sec=1e-6)]
    loads = {"2": {"Fx": 80.0e3, "Fy": 0.0}}
    return nodes, elements, loads


def _uniform(elements, t):
    return {e.id: float(t) for e in elements}


def test_column_buckling_hand_solution() -> None:
    nodes, elements, loads = _column()
    states = dcr_field(nodes, elements, loads, _uniform(elements, 20.0), F_Y)
    st = states["c"]
    assert st.compression
    assert st.p_cr == pytest.approx(P_CR_HAND, rel=1e-9)
    assert st.n_rd == pytest.approx(F_Y * 0.01, rel=1e-12)
    assert st.capacity_governing.value == "buckling"
    assert st.dcr == pytest.approx(100.0e3 / P_CR_HAND, rel=1e-9)


def test_tie_yield_hand_solution() -> None:
    nodes, elements, loads = _tie()
    f_y_small = 10.0e6
    states = dcr_field(nodes, elements, loads, _uniform(elements, 20.0), f_y_small)
    st = states["t"]
    assert not st.compression
    assert st.p_cr is None
    assert st.capacity_governing.value == "yield"
    assert st.dcr == pytest.approx(80.0e3 / (f_y_small * 0.01), rel=1e-9)


def test_yield_capacity_uses_k_y() -> None:
    cap20 = yield_capacity(0.01, F_Y, 20.0)
    cap600 = yield_capacity(0.01, F_Y, 600.0)
    assert cap20 == pytest.approx(F_Y * 0.01, rel=1e-12)
    # k_y(600) = 0.470 from the material fixture
    assert cap600 == pytest.approx(0.470 * F_Y * 0.01, rel=1e-9)


def test_axial_forces_sign_convention() -> None:
    nodes, elements, loads = _tie()
    forces = member_axial_forces(nodes, elements, loads, _uniform(elements, 20.0))
    assert forces["t"] == pytest.approx(80.0e3, rel=1e-9)  # tension positive


def test_governing_switches_displacement_to_buckling() -> None:
    nodes, elements, loads = _column()
    cold = ci_two_component(nodes, elements, loads, _uniform(elements, 20.0), 0.7, F_Y)
    hot = ci_two_component(nodes, elements, loads, _uniform(elements, 800.0), 0.7, F_Y)
    assert cold.governing["c"] == "displacement"
    assert hot.governing["c"] == "buckling"
    assert hot.components["c"].dcr_component > hot.components["c"].u_component


def test_member_and_system_critical_temperatures_consistent() -> None:
    nodes, elements, loads = _column()
    theta_cr = member_critical_temperature(nodes, elements, loads, "c", F_Y)
    theta_sys = system_critical_temperature(nodes, elements, loads, F_Y)
    assert theta_cr is not None
    assert 640.0 < theta_cr < 690.0  # k_E crossing of DCR=1 (grid-interpolated)
    assert theta_sys <= theta_cr
    assert theta_sys >= theta_cr - 25.0  # grid resolution of theta*_sys
    # below theta_sys nothing fails, above it something does
    below = dcr_field(nodes, elements, loads, _uniform(elements, theta_sys), F_Y)
    assert all(s.dcr < 1.0 for s in below.values())


def test_dcr_field_on_campaign_truss(campaign) -> None:
    cm = next(c for c in campaign if c.name == "warren_4_shallow")
    temps = {e.id: 600.0 for e in cm.elements}
    states = dcr_field(cm.nodes, cm.elements, cm.loads, temps, F_Y)
    assert set(states) == {e.id for e in cm.elements}
    for st in states.values():
        assert st.dcr >= 0.0
        if not st.compression:
            assert st.p_cr is None
