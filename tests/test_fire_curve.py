"""Fire-exposure layer: ISO 834 curve and lumped-capacitance steel heating.

Oracles, in the project's verification hierarchy order:

* closed-form solutions (constant-property convection limit: exponential
  approach, matched to machine precision),
* an INDEPENDENT integrator (scipy ``solve_ivp`` RK45 at tight tolerance),
* the first-principles energy balance (heat in through the surface ==
  enthalpy stored, ``rho * int c_a dtheta``),
* published ISO 834 table points,
* physical invariants (monotonicity, gas-temperature bound, section-factor
  and shadow-factor orderings, step-halving convergence).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from truss_analysis.material.steel_eurocode import specific_heat, unit_mass
from truss_analysis.thermal.fire_curve import (
    ALPHA_C,
    EPSILON_RES,
    STEFAN_BOLTZMANN,
    h_net,
    iso_834_temperature,
    steel_temperature,
)


def test_iso834_matches_published_table_points() -> None:
    """The standard's own rounded table values, to within 1 degC."""
    table = {5: 576, 10: 679, 15: 739, 30: 842, 60: 945, 120: 1049}
    for minutes, expected in table.items():
        assert iso_834_temperature(float(minutes)) == pytest.approx(expected, abs=1.0)
    assert iso_834_temperature(0.0) == 20.0


def test_iso834_monotone_vectorised_and_guards_negative_time() -> None:
    t = np.array([0.0, 5.0, 30.0, 120.0])
    out = iso_834_temperature(t)
    assert np.all(np.diff(out) > 0.0)
    assert isinstance(out, np.ndarray)
    with pytest.raises(ValueError, match=">= 0"):
        iso_834_temperature(-1.0)


def test_h_net_components_and_zero_at_equilibrium() -> None:
    assert h_net(500.0, 500.0) == pytest.approx(0.0, abs=1e-9)
    # convection-only: exact linear law
    conv = h_net(500.0, 100.0, emissivity=0.0)
    assert conv == pytest.approx(ALPHA_C * 400.0)
    # radiation-only: exact Stefan-Boltzmann difference
    rad = h_net(500.0, 100.0, alpha_c=0.0)
    expected = EPSILON_RES * STEFAN_BOLTZMANN * (773.0**4 - 373.0**4)
    assert rad == pytest.approx(expected, rel=1e-12)
    assert h_net(500.0, 100.0) > 0.0  # heating
    assert h_net(100.0, 500.0) < 0.0  # cooling


def test_convection_only_constant_properties_matches_closed_form() -> None:
    """Exponential approach to a constant gas temperature, machine-precise.

    d(theta)/dt = beta (theta_g - theta) with beta = alpha_c (A_m/V)/(rho c_a)
    has the exact solution theta(t) = theta_g - (theta_g - theta_0) e^{-beta t}.
    The specific heat is frozen and radiation switched off to hit that limit.
    """
    import truss_analysis.thermal.fire_curve as fc

    am_v, rho, c_fix, tg = 183.0, unit_mass(), 600.0, 1000.0
    beta = ALPHA_C * am_v / (rho * c_fix)
    exact = tg - (tg - 20.0) * np.exp(-beta * 600.0)
    original = fc.specific_heat
    try:
        fc.specific_heat = lambda _theta: c_fix
        res = fc.steel_temperature(
            10.0,
            am_v,
            fire_curve=lambda _t: tg,
            emissivity=0.0,
            max_step_s=0.5,
        )
    finally:
        fc.specific_heat = original
    assert res.theta_final == pytest.approx(exact, rel=1e-10)


def test_matches_independent_integrator_solve_ivp() -> None:
    """Cross-validation against a second, independent ODE solver."""
    am_v, rho = 183.0, unit_mass()

    def rhs(t_s: float, y: np.ndarray) -> list[float]:
        flux = float(h_net(iso_834_temperature(t_s / 60.0), y[0]))
        return [flux * am_v / (rho * float(specific_heat(y[0])))]

    sol = solve_ivp(rhs, (0.0, 1800.0), [20.0], rtol=1e-9, atol=1e-9)
    mine = steel_temperature(30.0, am_v)
    assert mine.theta_final == pytest.approx(float(sol.y[0, -1]), abs=0.01)


def test_energy_balance_heat_in_equals_enthalpy_stored() -> None:
    """First-principles closure: surface heat input == stored enthalpy."""
    am_v, rho = 183.0, unit_mass()
    res = steel_temperature(30.0, am_v)
    stored = rho * quad(lambda th: float(specific_heat(th)), 20.0, res.theta_final)[0]
    delivered = np.trapezoid(h_net(res.theta_gas, res.theta_steel), res.time_s) * am_v
    assert delivered == pytest.approx(stored, rel=2e-4)


def test_physical_invariants() -> None:
    res = steel_temperature(60.0, 200.0)
    assert res.theta_steel[0] == 20.0
    assert np.all(np.diff(res.theta_steel) > 0.0)  # monotone heating
    assert np.all(res.theta_steel[1:] < res.theta_gas[1:])  # lags the gas
    assert res.theta_final < 1200.0


def test_section_factor_and_shadow_factor_orderings() -> None:
    thin = steel_temperature(20.0, 250.0).theta_steel[1:]
    thick = steel_temperature(20.0, 100.0).theta_steel[1:]
    assert np.all(thin > thick)
    full = steel_temperature(20.0, 183.0).theta_steel[1:]
    shaded = steel_temperature(20.0, 183.0, shadow_factor=0.75).theta_steel[1:]
    assert np.all(full > shaded)


def test_step_halving_convergence() -> None:
    coarse = steel_temperature(30.0, 183.0, max_step_s=5.0).theta_final
    fine = steel_temperature(30.0, 183.0, max_step_s=0.5).theta_final
    assert abs(coarse - fine) < 0.01  # EN-capped 5 s step is already converged


def test_output_sampling_and_input_guards() -> None:
    res = steel_temperature(30.0, 183.0, n_output=7)
    assert len(res.time_s) == 7
    assert res.time_s[0] == 0.0
    assert res.time_s[-1] == pytest.approx(1800.0)
    with pytest.raises(ValueError, match="duration_min"):
        steel_temperature(0.0, 183.0)
    with pytest.raises(ValueError, match="section_factor"):
        steel_temperature(30.0, 0.0)
    with pytest.raises(ValueError, match="shadow_factor"):
        steel_temperature(30.0, 183.0, shadow_factor=1.5)
    with pytest.raises(ValueError, match="max_step_s"):
        steel_temperature(30.0, 183.0, max_step_s=0.0)


def test_custom_fire_curve_is_accepted() -> None:
    """Any gas-temperature history works: constant 500 degC oven."""
    res = steel_temperature(60.0, 183.0, fire_curve=lambda _t: 500.0)
    assert res.theta_final < 500.0
    # after an hour in a 500 degC oven the member is nearly equilibrated
    assert res.theta_final > 490.0


def test_fire_to_structure_bridge_restrained_bar() -> None:
    """End-to-end: fire exposure -> steel temperature -> structural chain.

    A fully restrained bar heated by the fire for 15 minutes must develop
    the same DCR as the prescribed-temperature analysis at the temperature
    the fire solver reports -- the two halves of the library speak through
    one number (round-5 audit: the ISO-834 gap, C1-9/C3-9/C6-3).
    """
    from truss_analysis.limitstates import dcr_field, member_axial_forces
    from truss_analysis.model import Element, Node

    theta = steel_temperature(15.0, 183.0).theta_final
    alpha = 1.2e-5
    E0, A = 210e9, 1e-3
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=3.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [
        Element(
            id="bar",
            node_i="1",
            node_j="2",
            E=E0,
            A=A,
            I_sec=1e-7,
            alpha=alpha,
            delta_T=theta - 20.0,
        )
    ]
    forces = member_axial_forces(nodes, elements, {}, {"bar": theta})
    # Analytical oracle: a fully restrained bar with temperature-degraded
    # stiffness carries exactly N = -k_E(theta) E A alpha dT (the library's
    # closed-form golden case, at fire temperature), driven here by a
    # fire-exposure temperature instead of a hand-typed number.
    from truss_analysis.material.steel_eurocode import k_E

    expected_N = -float(k_E(theta)) * E0 * A * alpha * (theta - 20.0)
    assert forces["bar"] == pytest.approx(expected_N, rel=1e-12)
    # And the fire limit-state chain consumes the same temperature: the
    # restrained bar at ~650 degC is well past yield, a real verdict.
    states = dcr_field(nodes, elements, {}, {"bar": theta}, 355e6)
    assert states["bar"].axial_force == pytest.approx(expected_N, rel=1e-9)
    assert states["bar"].dcr > 1.0


def test_negative_density_rejected() -> None:
    with pytest.raises(ValueError, match="rho_a"):
        steel_temperature(5.0, 183.0, rho_a=-1.0)
