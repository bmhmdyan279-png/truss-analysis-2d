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

import itertools
import warnings

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from truss_analysis.exceptions import LumpedCapacityWarning
from truss_analysis.material.steel_eurocode import (
    specific_heat,
    thermal_conductivity,
    unit_mass,
)
from truss_analysis.thermal.fire_curve import (
    ALPHA_C,
    EPSILON_RES,
    LUMPED_SECTION_FACTOR_LIMIT,
    STEFAN_BOLTZMANN,
    T_LIM_FIRE_GROWTH_MIN,
    ParametricFire,
    biot_critical_section_factor,
    h_net,
    iso_834_temperature,
    lumped_capacity_biot,
    parametric_fire_temperature,
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


# --------------------------------------------------------------------------
# B2: EN 1991-1-2 Annex A parametric temperature-time curve
# --------------------------------------------------------------------------

# Reference values from the Access Steel worked example SX042a-EN-EU
# ("Example: Parametric fire curve for a fire compartment", Z. Sokol /
# F. Wald, CTU Prague, EN 1991-1-2:2002 Annex A): an 8.5 x 10.0 x 3.15 m
# office compartment, four openings, concrete floor/ceiling and lightweight
# concrete walls.
SX042A = ParametricFire(
    opening_factor=0.1024, q_td=181.8, b_value=1234.0, t_lim_min=20.0
)


def test_parametric_fire_matches_access_steel_worked_example() -> None:
    """The published intermediate values, to the precision they are quoted at."""
    assert SX042A.gamma == pytest.approx(5.791, rel=1e-3)
    assert SX042A.t_max_h == pytest.approx(0.355, rel=1e-3)
    assert SX042A.t_max_star_h == pytest.approx(2.056, rel=1e-3)
    assert SX042A.theta_max == pytest.approx(1052.0, abs=1.0)
    assert SX042A.cooling_rate == pytest.approx(250.0)
    assert SX042A.ventilation_controlled is True
    assert SX042A.cooling_factor_x == pytest.approx(1.0)


def test_parametric_cooling_branch_matches_the_published_line() -> None:
    """The example states the cooling line as ``theta = 1566 - 250 t*``."""
    for t_star in (2.5, 3.0, 4.0, 5.0):
        t_min = t_star / SX042A.gamma * 60.0
        # 1566 = 1052 + 250 * 2.056 with the example's rounded theta_max;
        # use the exact peak so the comparison tests the slope, not the
        # document's rounding.
        expected = SX042A.theta_max - 250.0 * (t_star - SX042A.t_max_star_h)
        assert float(SX042A(t_min)) == pytest.approx(expected, rel=1e-9)


def test_parametric_heating_branch_matches_the_closed_form() -> None:
    """Hand evaluation of A.1(1) at several scaled times."""
    for t_star in (0.05, 0.1, 0.5, 1.0, 2.056):
        t_min = t_star / SX042A.gamma * 60.0
        hand = 20.0 + 1325.0 * (
            1.0
            - 0.324 * np.exp(-0.2 * t_star)
            - 0.204 * np.exp(-1.7 * t_star)
            - 0.472 * np.exp(-19.0 * t_star)
        )
        assert float(SX042A(t_min)) == pytest.approx(hand, rel=1e-12)


def test_parametric_curve_is_continuous_at_its_peak() -> None:
    """Heating and cooling branches must meet, not jump."""
    fire = ParametricFire(opening_factor=0.04, q_td=420.0, b_value=1160.0)
    peak_min = fire.t_max_star_h / fire.gamma * 60.0
    eps = 1e-6
    left = float(fire(peak_min - eps))
    right = float(fire(peak_min + eps))
    # a continuity check needs an absolute tolerance: probing eps away from
    # the peak moves theta by O(dtheta/dt * eps), which no relative tolerance
    # on the peak value can express
    assert left == pytest.approx(fire.theta_max, abs=1e-3)
    assert right == pytest.approx(fire.theta_max, abs=1e-3)
    assert abs(left - right) < 1e-3


def test_parametric_curve_structural_invariants() -> None:
    fire = ParametricFire(opening_factor=0.04, q_td=420.0, b_value=1160.0)
    assert fire.gamma == pytest.approx(1.0, rel=1e-12)
    t = np.linspace(0.0, 600.0, 60001)
    theta = np.asarray(fire(t), dtype=float)
    assert theta[0] == pytest.approx(20.0)
    # single peak, at the predicted real time, below the 20 + 1325 asymptote
    assert theta.max() == pytest.approx(fire.theta_max, rel=1e-12)
    assert t[theta.argmax()] == pytest.approx(fire.t_max_star_h / fire.gamma * 60.0)
    assert theta.max() < 20.0 + 1325.0
    # monotone rise then monotone decay, never below ambient
    peak = int(theta.argmax())
    assert np.all(np.diff(theta[: peak + 1]) >= 0.0)
    assert np.all(np.diff(theta[peak:]) <= 1e-12)
    assert np.all(theta >= 20.0)
    # scalar and array evaluation agree
    assert np.allclose(np.asarray(fire(np.array([5.0, 50.0]))), [fire(5.0), fire(50.0)])


def test_parametric_cooling_rate_branches() -> None:
    """All three cooling rates of A.2, selected by ``t*_max``."""
    # t*_max >= 2 h  ->  250 K/h   (the worked example)
    assert SX042A.cooling_rate == 250.0
    # t*_max <= 0.5 h ->  625 K/h.  A heavily insulated boundary (large b)
    # with a small opening makes Gamma << 1, so the scaled clock barely
    # advances: t*_max = t_max * Gamma = 1.0 h * 0.0695 = 0.0695 h.
    fast = ParametricFire(
        opening_factor=0.02, q_td=100.0, b_value=2200.0, t_lim_min=15.0
    )
    assert fast.gamma < 1.0
    assert fast.t_max_star_h <= 0.5
    assert fast.cooling_rate == 625.0
    # 0.5 < t*_max < 2 h  ->  250 (3 - t*_max)
    mid = ParametricFire(opening_factor=0.04, q_td=200.0, b_value=1000.0)
    assert 0.5 < mid.t_max_star_h < 2.0
    assert mid.cooling_rate == pytest.approx(250.0 * (3.0 - mid.t_max_star_h))


def test_parametric_t_lim_bounds_a_fuel_controlled_fire() -> None:
    """``t_max = max(0.2e-3 q_td / O, t_lim)``: t_lim must be able to win."""
    light = ParametricFire(opening_factor=0.2, q_td=50.0, b_value=1234.0)
    assert light.t_ventilation_h == pytest.approx(0.2e-3 * 50.0 / 0.2)
    assert light.t_lim_h == pytest.approx(20.0 / 60.0)
    assert light.t_max_h == pytest.approx(light.t_lim_h)
    assert light.ventilation_controlled is False
    # fuel-controlled branch: x = t_lim / t_max = 1 as well
    assert light.cooling_factor_x == pytest.approx(1.0)


def test_parametric_fire_growth_rate_table() -> None:
    assert T_LIM_FIRE_GROWTH_MIN == {"slow": 25.0, "medium": 20.0, "fast": 15.0}
    for name, t_lim in T_LIM_FIRE_GROWTH_MIN.items():
        fire = ParametricFire(
            opening_factor=0.2, q_td=50.0, b_value=1234.0, t_lim_min=t_lim
        )
        assert fire.t_max_h == pytest.approx(t_lim / 60.0), name


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"opening_factor": 0.01}, r"opening_factor O must lie in"),
        ({"opening_factor": 0.3}, r"opening_factor O must lie in"),
        ({"b_value": 50.0}, r"b_value must lie in"),
        ({"b_value": 3000.0}, r"b_value must lie in"),
        ({"q_td": 0.0}, r"q_td must be > 0"),
        ({"q_td": -5.0}, r"q_td must be > 0"),
        ({"t_lim_min": 0.0}, r"t_lim_min must be > 0"),
    ],
)
def test_parametric_rejects_parameters_outside_the_code_limits(
    kwargs: dict, match: str
) -> None:
    """O and b are hard limits of validity in A.1(3)/A.2, not suggestions."""
    base = {"opening_factor": 0.1, "q_td": 300.0, "b_value": 1234.0}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ParametricFire(**base)


def test_parametric_warns_outside_the_fire_load_range_of_validity() -> None:
    """q_td outside [50, 1000] is an extrapolation, so it warns, not raises."""
    with pytest.warns(UserWarning, match="range of validity"):
        ParametricFire(opening_factor=0.1, q_td=1500.0, b_value=1234.0)
    with pytest.warns(UserWarning, match="range of validity"):
        ParametricFire(opening_factor=0.1, q_td=20.0, b_value=1234.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ParametricFire(opening_factor=0.1, q_td=420.0, b_value=1234.0)


def test_parametric_negative_time_rejected_and_function_form_agrees() -> None:
    with pytest.raises(ValueError, match="time must be >= 0"):
        SX042A(-1.0)
    assert float(
        parametric_fire_temperature(30.0, 0.1024, 181.8, 1234.0, 20.0)
    ) == pytest.approx(float(SX042A(30.0)), rel=1e-12)


def test_parametric_fire_drives_the_member_heating_solver() -> None:
    """The curve is a drop-in ``fire_curve`` for :func:`steel_temperature`.

    A parametric fire that decays must leave the member cooler at the end of
    a long exposure than the ISO 834 curve, which never cools -- the whole
    reason performance-based design uses it.
    """
    section_factor = 180.0
    param = steel_temperature(
        120.0, section_factor, fire_curve=SX042A, warn_lumped=False
    )
    standard = steel_temperature(120.0, section_factor, warn_lumped=False)
    assert param.theta_final < standard.theta_final
    # the integrator samples theta_g on its own grid, so the sampled peak
    # sits just below the analytic one -- it can never exceed it
    assert param.theta_gas.max() <= SX042A.theta_max + 1e-9
    assert param.theta_gas.max() == pytest.approx(SX042A.theta_max, abs=1.0)
    assert param.theta_steel[0] == pytest.approx(20.0)
    # The steel can never exceed the hottest gas it has seen -- but it CAN
    # exceed the *instantaneous* gas temperature, because once the parametric
    # curve decays below the member temperature the flux reverses and the
    # member cools toward it from above.  Bounding against the running
    # maximum is the invariant that is actually true.
    running_max = np.maximum.accumulate(param.theta_gas)
    assert np.all(param.theta_steel <= running_max + 1e-9)
    assert param.theta_steel.max() <= param.theta_gas.max() + 1e-9
    # ... and the reversal must actually happen: a decaying fire leaves the
    # member cooling at the end, which a nominal (never-cooling) curve cannot
    assert np.any(np.diff(param.theta_steel) < 0.0)
    assert param.theta_steel[-1] < param.theta_steel.max()
    assert param.theta_gas[-1] == pytest.approx(20.0, abs=1.0)


# --------------------------------------------------------------------------
# B7: applicability of the lumped-capacitance assumption
# --------------------------------------------------------------------------


def test_lumped_capacity_warning_for_thick_sections() -> None:
    """``A_m/V`` below the threshold must say the answer is un-conservative."""
    with pytest.warns(LumpedCapacityWarning, match="thick section"):
        res = steel_temperature(30.0, 30.0, warn_lumped=True)
    assert res.theta_final > 20.0  # still computed, just flagged


def test_lumped_capacity_warning_can_be_suppressed() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", LumpedCapacityWarning)
        res = steel_temperature(30.0, 30.0, warn_lumped=False)
    assert res.theta_final > 20.0


def test_no_lumped_warning_for_slender_sections() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", LumpedCapacityWarning)
        steel_temperature(30.0, LUMPED_SECTION_FACTOR_LIMIT, warn_lumped=True)
        steel_temperature(30.0, 250.0, warn_lumped=True)


def test_biot_number_matches_its_own_definition() -> None:
    """``Bi = h_eff (V/A_m) / lambda`` recomputed independently of the helper."""
    section_factor = 120.0
    flux = float(h_net(800.0, 20.0, EPSILON_RES, ALPHA_C))
    h_eff = flux / (800.0 - 20.0)
    lambda_a = float(thermal_conductivity(800.0))
    expected = h_eff / (section_factor * lambda_a)
    assert lumped_capacity_biot(section_factor) == pytest.approx(expected, rel=1e-12)


def test_biot_number_scales_inversely_with_section_factor() -> None:
    """Thicker (lower ``A_m/V``) means a larger Biot number, strictly."""
    factors = [20.0, 50.0, 100.0, 200.0, 400.0]
    biots = [lumped_capacity_biot(sf) for sf in factors]
    assert all(a > b for a, b in itertools.pairwise(biots))
    # doubling the section factor halves the Biot number exactly
    assert lumped_capacity_biot(100.0) == pytest.approx(
        0.5 * lumped_capacity_biot(50.0), rel=1e-12
    )


def test_biot_critical_section_factor_is_the_inverse_relation() -> None:
    """The two helpers must be exact inverses of one another."""
    sf_crit = biot_critical_section_factor(0.1)
    assert lumped_capacity_biot(sf_crit) == pytest.approx(0.1, rel=1e-12)
    assert biot_critical_section_factor(0.05) == pytest.approx(2.0 * sf_crit)
    with pytest.raises(ValueError, match="biot_limit must be > 0"):
        biot_critical_section_factor(0.0)


def test_code_threshold_is_the_conservative_one() -> None:
    """The ``A_m/V < 50`` trigger must fire before ``Bi`` reaches 0.1.

    If the classical criterion were the stricter of the two, the documented
    threshold would be letting through members the physics objects to, and
    the warning message quoting both numbers would be self-contradicting.
    """
    sf_crit = biot_critical_section_factor(0.1)
    assert sf_crit < LUMPED_SECTION_FACTOR_LIMIT
    assert lumped_capacity_biot(LUMPED_SECTION_FACTOR_LIMIT) < 0.1


def test_lumped_capacity_warning_rejects_bad_section_factor() -> None:
    with pytest.raises(ValueError, match="section_factor"):
        lumped_capacity_biot(0.0)
    with pytest.raises(ValueError, match="section_factor"):
        lumped_capacity_biot(-10.0)
