"""Limit-state tests: hand solutions, DCR, theta_cr, governing switch."""

from __future__ import annotations

import math

import pytest

from truss_analysis.limitstates import (
    BucklingModel,
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


# Hand-derived EN 1993-1-2 4.2.3.1 capacity for _column() at 20 degC.
# At ambient k_E = k_y = 1, so f_y,theta = f_y and:
#   lambda_bar = sqrt(A * f_y / P_cr) = sqrt(0.01 * 235e6 / P_CR_HAND)
#   alpha_fire = 0.65 * 0.49                (curve c, per 4.2.3.1(3))
#   Phi        = 0.5 [1 + alpha_fire (lambda_bar - 0.2) + lambda_bar^2]
#   chi        = 1 / (Phi + sqrt(Phi^2 - lambda_bar^2))
#   N_b,Rd     = chi * A * f_y / gamma_M,fi = chi * N_Rd
_LAMBDA_HAND = math.sqrt(0.01 * F_Y / P_CR_HAND)  # 2.129631229...
_ALPHA_FIRE = 0.65 * 0.49
_PHI_HAND = 0.5 * (1.0 + _ALPHA_FIRE * (_LAMBDA_HAND - 0.2) + _LAMBDA_HAND**2)
_CHI_HAND = 1.0 / (_PHI_HAND + math.sqrt(_PHI_HAND**2 - _LAMBDA_HAND**2))
_CAPACITY_HAND = _CHI_HAND * F_Y * 0.01


def test_column_buckling_hand_solution() -> None:
    """Compression capacity follows the Eurocode buckling curve, not bare Euler."""
    nodes, elements, loads = _column()
    states = dcr_field(nodes, elements, loads, _uniform(elements, 20.0), F_Y)
    st = states["c"]
    assert st.compression
    # P_cr is an *input* to the slenderness, not the capacity itself.
    assert st.p_cr == pytest.approx(P_CR_HAND, rel=1e-9)
    assert st.n_rd == pytest.approx(F_Y * 0.01, rel=1e-12)
    assert st.lambda_bar == pytest.approx(_LAMBDA_HAND, rel=1e-9)
    assert st.chi == pytest.approx(_CHI_HAND, rel=1e-9)
    assert st.capacity == pytest.approx(_CAPACITY_HAND, rel=1e-9)
    assert st.capacity_governing.value == "buckling"
    assert st.dcr == pytest.approx(100.0e3 / _CAPACITY_HAND, rel=1e-9)
    # chi < 1 means buckling genuinely reduces the capacity below yield.
    assert st.chi < 1.0
    assert st.capacity < st.n_rd


def test_column_legacy_euler_only_model() -> None:
    """The historical min(P_cr, N_Rd) model stays selectable and reproducible."""
    nodes, elements, loads = _column()
    states = dcr_field(
        nodes,
        elements,
        loads,
        _uniform(elements, 20.0),
        F_Y,
        buckling_model=BucklingModel.EULER_ONLY,
    )
    st = states["c"]
    # P_cr (518 kN) < N_Rd (2350 kN), so Euler governs in the legacy model.
    assert st.capacity == pytest.approx(P_CR_HAND, rel=1e-9)
    assert st.dcr == pytest.approx(100.0e3 / P_CR_HAND, rel=1e-9)
    assert st.chi is None
    assert st.capacity_governing.value == "buckling"


def test_chi_model_is_more_conservative_than_euler_only() -> None:
    """chi must never give a larger capacity than min(P_cr, N_Rd).

    The buckling curve lies below both of its asymptotes for finite
    slenderness, so switching models can only reduce capacity. If this ever
    fails, the reduction factor has been wired up backwards.
    """
    for i_sec in (1e-7, 1e-6, 1e-5, 1e-4):
        nodes, elements, loads = _column(i_sec=i_sec)
        temps = _uniform(elements, 20.0)
        chi_cap = dcr_field(nodes, elements, loads, temps, F_Y)["c"].capacity
        legacy = dcr_field(
            nodes, elements, loads, temps, F_Y, buckling_model=BucklingModel.EULER_ONLY
        )["c"]
        assert chi_cap <= legacy.capacity * (1 + 1e-12), i_sec


def test_stocky_member_is_yield_governed() -> None:
    """Below lambda_bar = 0.2 buckling need not be considered (6.3.1(4))."""
    # A very stocky stub: huge I drives lambda_bar towards zero.
    nodes, elements, loads = _column(i_sec=1.0, load=100.0e3)
    st = dcr_field(nodes, elements, loads, _uniform(elements, 20.0), F_Y)["c"]
    assert st.lambda_bar is not None
    assert st.lambda_bar < 0.2
    assert st.capacity_governing.value == "yield"
    # chi saturates at 1, so the capacity is the plain yield resistance.
    assert st.chi == pytest.approx(1.0, rel=1e-12)
    assert st.capacity == pytest.approx(st.n_rd, rel=1e-12)


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
