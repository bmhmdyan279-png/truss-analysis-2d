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

# Two diagnostics this module exercises deliberately.
#
# LegacyBucklingModelWarning: EULER_ONLY is kept so that results computed with it
#   stay reproducible bit for bit, and these tests are that reproduction.
# BucklingCheckWarning: the campaign fixtures carry members with I_sec <= 0, so
#   their DCR is reported as +inf with a warning -- the documented behaviour for
#   missing data, asserted in test_dcr_field_on_campaign_truss.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.LegacyBucklingModelWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.BucklingCheckWarning"
    ),
]

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


def test_alpha_one_exposes_no_damage_criticality_at_any_temperature() -> None:
    """The round-4 audit's exposing test: alpha = 1 perturbed NOTHING.

    An unperturbed member must report zero perturbation criticality at any
    temperature. Pre-2.7 the DCR component referenced the COLD state, so at
    600 degC this returned dcr_component ~ +2.1 and governing = buckling for
    a member that was not perturbed at all -- fire degradation masquerading
    as damage criticality.
    """
    nodes, elements, loads = _column()
    for t in (20.0, 600.0, 800.0):
        res = ci_two_component(nodes, elements, loads, _uniform(elements, t), 1.0, F_Y)
        comp = res.components["c"]
        assert comp.u_component == pytest.approx(0.0, abs=1e-12)
        assert comp.dcr_component == pytest.approx(0.0, abs=1e-12)
        assert comp.ci == pytest.approx(0.0, abs=1e-12)
        assert res.ci_values["c"] == pytest.approx(0.0, abs=1e-12)


def test_fire_component_isolated_from_damage_component() -> None:
    """Fire severity is explicit; the damage component stays same-temperature.

    The determinate column's force cannot redistribute, so softening it
    leaves the DCR ratio at 1 (dcr_component = 0) at every temperature; all
    temperature response lives in fire_component. ``governing`` follows the
    damage components only -- the pre-2.7 "switch to buckling at 800 degC"
    was the fire term leaking into the perturbation criticality.
    """
    nodes, elements, loads = _column()
    cold = ci_two_component(nodes, elements, loads, _uniform(elements, 20.0), 0.7, F_Y)
    hot = ci_two_component(nodes, elements, loads, _uniform(elements, 800.0), 0.7, F_Y)
    cc, hc = cold.components["c"], hot.components["c"]
    # determinate: force ratio 1 -> no damage-driven DCR change, ever
    assert cc.dcr_component == pytest.approx(0.0, abs=1e-12)
    assert hc.dcr_component == pytest.approx(0.0, abs=1e-12)
    # displacement component: temperature-invariant under a uniform field
    assert hc.u_component == pytest.approx(cc.u_component, rel=1e-9)
    assert hc.u_component == pytest.approx(1.0 / 0.7 - 1.0, rel=1e-9)
    assert cold.governing["c"] == "displacement"
    assert hot.governing["c"] == "displacement"
    # the fire severity is where the temperature response lives
    assert cc.fire_component == pytest.approx(0.0, abs=1e-12)
    assert hc.fire_component > 1.0  # capacity collapses toward 800 degC
    # combined == the explicit product == the legacy cold-referenced ratio
    assert hc.dcr_combined == pytest.approx(
        (1.0 + hc.dcr_component) * (1.0 + hc.fire_component) - 1.0, rel=1e-12
    )


def test_dcr_component_same_temperature_baseline_on_redundant_frame() -> None:
    """On a redundant heated frame the damage component is a genuine ratio.

    alpha = 1 must still give exactly zero (base state == perturbed state),
    while alpha < 1 measures the redistribution AT temperature T, not the
    fire degradation -- the two effects are separately reported.
    """
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="3", x=2.0, y=1.5, is_support=False),
    ]
    elements = [
        Element(
            id="1", node_i="1", node_j="3", E=210e9, A=0.005, I_sec=1e-6, alpha=1.2e-5
        ),
        Element(
            id="2", node_i="2", node_j="3", E=210e9, A=0.008, I_sec=1e-6, alpha=1.2e-5
        ),
        Element(
            id="3", node_i="1", node_j="2", E=210e9, A=0.006, I_sec=1e-6, alpha=1.2e-5
        ),
    ]
    loads = {"3": {"Fx": 20e3, "Fy": -50e3}}
    temps = _uniform(elements, 600.0)

    untouched = ci_two_component(nodes, elements, loads, temps, 1.0, F_Y)
    for comp in untouched.components.values():
        assert comp.dcr_component == pytest.approx(0.0, abs=1e-12)
        assert comp.u_component == pytest.approx(0.0, abs=1e-12)
    # the fire severity is nonzero and reported where it belongs -- except
    # for member 3, which is stress-free when cold (it spans two fixed
    # supports): a 0 -> X ratio is undefined and follows the same
    # near-zero-baseline convention as the legacy cold-referenced component
    assert untouched.components["1"].fire_component > 0.0
    assert untouched.components["2"].fire_component > 0.0
    assert untouched.components["3"].fire_component == 0.0

    damaged = ci_two_component(nodes, elements, loads, temps, 0.7, F_Y)
    # redundant structure: softening member 1 must redistribute forces and
    # show up as a nonzero same-temperature DCR ratio somewhere. (Member 1
    # itself happens to be the sole load path in its own direction here --
    # k_1 d_1 ~ 1 -- so ITS force ratio stays at 1 like a determinate bar;
    # the redistribution lands on the other members.)
    assert max(abs(comp.dcr_component) for comp in damaged.components.values()) > 1e-9
    # and the composite never mixes the fire term in
    for comp in damaged.components.values():
        assert comp.ci == pytest.approx(
            max(comp.u_component, comp.dcr_component), rel=1e-12, abs=1e-15
        )
        assert comp.dcr_combined == pytest.approx(
            (1.0 + comp.dcr_component) * (1.0 + comp.fire_component) - 1.0,
            rel=1e-12,
        )


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


def test_member_critical_temperature_brackets_the_crossing_tightly() -> None:
    """Bisection, not linear interpolation: a two-sided pin on the root.

    The returned temperature must sit ON the failure side (DCR >= 1, by the
    conservative upper-end convention) and one tolerance-step below it must
    still be safe -- verified through ``dcr_field``, i.e. through the
    independent per-point engine path, not the UniformForceScan the root
    finder itself uses. Pre-2.7 the crossing was linearly interpolated
    across a 25 degC cell of a curved DCR(T), with no guarantee on which
    side of the true root the answer landed.
    """
    nodes, elements, loads = _column()
    theta = member_critical_temperature(nodes, elements, loads, "c", F_Y)
    assert theta is not None
    at_theta = dcr_field(nodes, elements, loads, _uniform(elements, theta), F_Y)["c"]
    assert at_theta.dcr >= 1.0 - 1e-9
    below = dcr_field(nodes, elements, loads, _uniform(elements, theta - 0.05), F_Y)[
        "c"
    ]
    assert below.dcr < 1.0
    # and the answer stays inside the grid cell the coarse scan bracketed
    grid = tuple(range(20, 1201, 25))
    coarse = next(
        t
        for t in grid
        if dcr_field(nodes, elements, loads, _uniform(elements, float(t)), F_Y)["c"].dcr
        >= 1.0
    )
    assert coarse - 25.0 < theta <= coarse


def test_zero_stiffness_endpoint_counts_as_collapse(monkeypatch) -> None:
    """A grid point with k_E(T) <= 0 is failure-by-collapse, not a crash.

    With the real Eurocode law the scan always fails through k_y = 0 first
    (capacity gone at 1100 degC), so the zero-stiffness endpoint at
    1200 degC is only reachable with a capacity law that survives it; the
    monkeypatched k_y isolates exactly that MechanismError path. Pre-2.7 the
    error escaped from the middle of the scan.
    """
    import truss_analysis.limitstates as ls

    monkeypatch.setattr(ls, "eurocode_k_y", lambda theta: 1.0)
    # determinate tie in constant tension: force is temperature-invariant,
    # capacity (patched) never falls -> the scan survives to 1195 degC
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=3.0, y=0.0, is_support=True, support_dy=True),
    ]
    elements = [Element(id="t", node_i="1", node_j="2", E=E, A=0.01, I_sec=1e-6)]
    loads = {"2": {"Fx": 50.0e3, "Fy": 0.0}}
    grid = (20.0, 600.0, 1195.0, 1200.0)

    theta_sys = system_critical_temperature(nodes, elements, loads, F_Y, grid)
    assert theta_sys == 1195.0  # collapse AT 1200 counts as failure there

    theta_m = member_critical_temperature(
        nodes, elements, loads, "t", F_Y, temp_grid=grid
    )
    assert theta_m is not None
    assert theta_m == pytest.approx(1200.0, abs=0.01)


# --------------------------------------------------------------------------
# Round-5 audit pins: failure_mode labels (C8-5) and the loud I_sec guard
# --------------------------------------------------------------------------


def test_member_critical_temperature_failure_mode_material() -> None:
    """A genuine limit-state crossing is labelled MATERIAL."""
    from truss_analysis.limitstates import (
        FailureMode,
        member_critical_temperature_detailed,
    )

    nodes, elements, loads = _column(load=400.0e3)
    res = member_critical_temperature_detailed(nodes, elements, loads, "c", F_Y)
    assert res.failure_mode is FailureMode.MATERIAL
    assert res.theta is not None
    assert res.theta < 1200.0
    # The scalar facade is bit-for-bit the same number.
    assert member_critical_temperature(nodes, elements, loads, "c", F_Y) == res.theta


def test_member_critical_temperature_failure_mode_collapse() -> None:
    """A member that never reaches DCR=1 but dies with k_E(1200)=0 is
    labelled STIFFNESS_COLLAPSE -- it must not masquerade as a material
    critical temperature in engineering output (round-5 audit, C8-5)."""
    from truss_analysis.limitstates import (
        FailureMode,
        member_critical_temperature_detailed,
    )

    # Unloaded tie: with N = 0 the material DCR never reaches 1 (k_y hits
    # zero before k_E does, so ANY non-zero force fails as MATERIAL just
    # below 1200 degC); the only way the scan can end is the zero-stiffness
    # endpoint itself. The default grid stops at 1195, so the endpoint is
    # only reachable -- and only worth labelling -- on a grid that has it.
    nodes, elements, loads = _tie()
    loads = {"2": {"Fx": 0.0, "Fy": 0.0}}
    grid = (20.0, 600.0, 1195.0, 1200.0)
    res = member_critical_temperature_detailed(
        nodes, elements, loads, "t", F_Y, temp_grid=grid
    )
    assert res.failure_mode is FailureMode.STIFFNESS_COLLAPSE
    assert res.theta == pytest.approx(1200.0, abs=0.01)
    assert (
        member_critical_temperature(nodes, elements, loads, "t", F_Y, temp_grid=grid)
        == res.theta
    )


def test_member_critical_temperature_failure_mode_none() -> None:
    from truss_analysis.limitstates import (
        FailureMode,
        member_critical_temperature_detailed,
    )

    nodes, elements, loads = _tie()
    loads = {"2": {"Fx": 1.0e3, "Fy": 0.0}}
    res = member_critical_temperature_detailed(
        nodes, elements, loads, "t", F_Y, temp_grid=tuple(range(20, 501, 25))
    )
    assert res.theta is None
    assert res.failure_mode is FailureMode.NONE


def test_system_critical_temperature_failure_modes() -> None:
    from truss_analysis.limitstates import (
        FailureMode,
        system_critical_temperature_detailed,
    )

    # Heavily loaded column: a member limit state stops the scan.
    nodes, elements, loads = _column(load=400.0e3)
    res = system_critical_temperature_detailed(nodes, elements, loads, F_Y)
    assert res.failure_mode is FailureMode.MATERIAL
    assert res.theta == system_critical_temperature(nodes, elements, loads, F_Y)

    # Unloaded tie: nothing fails until the zero-stiffness endpoint.
    nodes_t, elements_t, loads_t = _tie()
    loads_t = {"2": {"Fx": 0.0, "Fy": 0.0}}
    grid = (20.0, 600.0, 1195.0, 1200.0)
    res_t = system_critical_temperature_detailed(
        nodes_t, elements_t, loads_t, F_Y, temp_grid=grid
    )
    assert res_t.failure_mode is FailureMode.STIFFNESS_COLLAPSE
    assert res_t.theta == 1195.0

    # Truncated grid with no failure anywhere: NONE.
    res_n = system_critical_temperature_detailed(
        nodes_t, elements_t, loads_t, F_Y, temp_grid=tuple(range(20, 301, 25))
    )
    assert res_n.failure_mode is FailureMode.NONE
    assert res_n.theta == 295.0  # range(20, 301, 25) ends at 295


def test_dcr_field_warns_for_compressed_members_without_i_sec() -> None:
    """The fire chain must be as loud as the static report path when a
    compressed member has no second moment of area (round-5 audit, C7-3)."""
    from truss_analysis.exceptions import BucklingCheckWarning

    nodes, elements, loads = _column(i_sec=0.0, load=100.0e3)
    temps = _uniform(elements, 20.0)
    with pytest.warns(BucklingCheckWarning, match="I_sec <= 0"):
        states = dcr_field(nodes, elements, loads, temps, F_Y)
    assert states["c"].dcr == float("inf")

    # Tension members and members with a real I_sec stay quiet.
    nodes_t, elements_t, loads_t = _tie()
    temps_t = _uniform(elements_t, 20.0)
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", BucklingCheckWarning)
        dcr_field(nodes_t, elements_t, loads_t, temps_t, F_Y)
