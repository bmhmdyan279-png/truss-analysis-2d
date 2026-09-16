"""Level 1 — analytical validation against independent hand solutions.

Six cases, every expected value derived by hand (symbolically, then
evaluated); the full step-by-step derivations live in the validation
report's analytical appendix.  Acceptance gate: relative error < 1e-6
(measured agreement is at machine level, ~1e-16; the gate is the published
criterion, not a tuned one).

Case list
---------
A1  symmetric two-bar truss, apex load:  delta = P L / (2 A E sin^2 theta)
    — note: cos(theta) does NOT appear in the denominator; the classic trap
    is measuring theta from the vertical instead of the horizontal.
A2  three-member statically determinate truss under an inclined load:
    member forces by joint equilibrium (exact fractions), displacements by
    the unit-load (virtual work) method.
A3a thermal free expansion: determinate truss, one heated member, no
    external load -> all forces zero, elongation = alpha_T dT L.
A3b fully restrained heated bar -> N = -E A alpha_T dT, zero displacement.
A4  the A2 truss under a linear temperature-gradient scenario: the
    hand-computed temperature field, degraded moduli and virtual-work
    displacements; determinate forces must be unchanged by temperature.
A5  Euler buckling P_cr = pi^2 E I / L^2 (cold and at 600 degC via the
    reduction factor) on the idealised square HSS, then the full
    EN 1993-1-2 4.2.3.1 compression capacity chi(lambda_bar) * N_Rd and the
    resulting DCR; plus a companion case pinning the legacy
    min(P_cr, N_Rd) model so the two cannot silently converge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.criticality import (
    base_displacement,
    build_engine,
    get_scenario_temperatures,
    load_vector,
)
from truss_analysis.limitstates import BucklingModel, Governing, dcr_field
from truss_analysis.material import steel_eurocode as ec
from truss_analysis.model import Element, Node
from truss_analysis.postprocess import calculate_element_forces, calculate_reactions
from truss_analysis.sections import euler_buckling_load, idealised_square_hss
from truss_analysis.solver import solve

# A5 reproduces the legacy EULER_ONLY capacity against a published analytic value;
# the model it exercises warns by design.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.ConstantAlphaWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.LegacyBucklingModelWarning"
    ),
]

REL_TOL = 1e-6  # acceptance gate (measured values are ~1e-16)
E = 210.0e9
A = 0.01
EA = E * A
ALPHA_T = 1.2e-5
F_Y = 235.0e6
SQRT13 = math.sqrt(13.0)


def _rel(lib: float, hand: float) -> float:
    return abs(lib - hand) / max(abs(hand), 1e-300)


def _fem(nodes, elements, nodal_loads):
    """Assemble -> solve -> element forces (mechanical loads only)."""
    K, F, _f_mech, fixed = assemble_global_matrices(nodes, elements)
    for node_id, fx, fy in nodal_loads:
        i = next(k for k, n in enumerate(nodes) if n.id == node_id)
        F[2 * i] += fx
        F[2 * i + 1] += fy
    U = solve(K, F, fixed)
    forces, _energy, _prestress = calculate_element_forces(nodes, elements, U)
    return K, F, fixed, U, forces


def _two_bar_nodes():
    return [
        Node("1", 0.0, 0.0, True, True, True),
        Node("2", 8.0, 0.0, True, True, True),
        Node("3", 4.0, 3.0),
    ]


def _two_bar_elements(**kwargs):
    return [
        Element("1", "1", "3", E, A, **kwargs),
        Element("2", "2", "3", E, A, **kwargs),
    ]


def _three_bar_nodes():
    return [
        Node("1", 0.0, 0.0, True, True, True),
        Node("2", 6.0, 0.0, True, False, True),  # roller: dy only
        Node("3", 2.0, 3.0),
    ]


def _three_bar_elements(**kwargs):
    return [
        Element("1", "1", "2", E, A, **kwargs),
        Element("2", "1", "3", E, A, **kwargs),
        Element("3", "2", "3", E, A, **kwargs),
    ]


# ---------------------------------------------------------------------
# A1 — symmetric two-bar truss, closed form delta = P L / (2 A E sin^2)
# ---------------------------------------------------------------------


def test_a1_two_bar_closed_form_displacement():
    """L = 5 m, sin(theta) = 3/5, P = -90 kN  =>  uy = -1/3360 m."""
    nodes, elements = _two_bar_nodes(), _two_bar_elements()
    _K, _F, _fixed, U, forces = _fem(nodes, elements, [("3", 0.0, -90.0e3)])
    hand_uy = -90.0e3 * 5.0 / (2.0 * A * E * 0.36)  # = -1/3360
    assert _rel(U[5], hand_uy) < REL_TOL
    assert abs(U[4]) <= 1e-15  # symmetry: zero horizontal displacement
    for f in forces:
        assert _rel(f["N"], -75.0e3) < REL_TOL  # N = P / (2 sin) compression


def test_a1_two_bar_reactions():
    nodes, elements = _two_bar_nodes(), _two_bar_elements()
    K, F, fixed, U, _forces = _fem(nodes, elements, [("3", 0.0, -90.0e3)])
    R = calculate_reactions(nodes, K, U, F, fixed)
    assert _rel(R["1"]["Fx"], 60.0e3) < REL_TOL
    assert _rel(R["1"]["Fy"], 45.0e3) < REL_TOL
    assert _rel(R["2"]["Fx"], -60.0e3) < REL_TOL
    assert _rel(R["2"]["Fy"], 45.0e3) < REL_TOL


# ---------------------------------------------------------------------
# A2 — three determinate members, inclined load, virtual work
# ---------------------------------------------------------------------


def test_a2_three_member_forces_exact_fractions():
    """N = (68/9, -7*sqrt(13)/9, -85/9) kN from joint equilibrium."""
    nodes, elements = _three_bar_nodes(), _three_bar_elements()
    _K, _F, _fixed, _U, forces = _fem(nodes, elements, [("3", 6.0e3, -8.0e3)])
    hand = [68.0e3 / 9.0, -7.0e3 * SQRT13 / 9.0, -85.0e3 / 9.0]
    for got, want in zip(forces, hand, strict=True):
        assert _rel(got["N"], want) < REL_TOL


def test_a2_three_member_displacements_virtual_work():
    """u(N3) = ((3757-91 sqrt13)/54, -(3757+182 sqrt13)/81) kN m / EA."""
    nodes, elements = _three_bar_nodes(), _three_bar_elements()
    _K, _F, _fixed, U, _forces = _fem(nodes, elements, [("3", 6.0e3, -8.0e3)])
    hand_ux = (3757.0 - 91.0 * SQRT13) / 54.0 * 1.0e3 / EA
    hand_uy = -(3757.0 + 182.0 * SQRT13) / 81.0 * 1.0e3 / EA
    assert _rel(U[4], hand_ux) < REL_TOL
    assert _rel(U[5], hand_uy) < REL_TOL


def test_a2_three_member_reactions():
    nodes, elements = _three_bar_nodes(), _three_bar_elements()
    K, F, fixed, U, _forces = _fem(nodes, elements, [("3", 6.0e3, -8.0e3)])
    R = calculate_reactions(nodes, K, U, F, fixed)
    assert _rel(R["1"]["Fx"], -6.0e3) < REL_TOL
    assert _rel(R["1"]["Fy"], 7.0e3 / 3.0) < REL_TOL
    assert abs(R["2"]["Fx"]) <= 1e-9  # roller: no horizontal reaction
    assert _rel(R["2"]["Fy"], 17.0e3 / 3.0) < REL_TOL


# ---------------------------------------------------------------------
# A3 — thermal strain: free expansion and the fully restrained bar
# ---------------------------------------------------------------------


def test_a3a_free_expansion_determinate():
    """Heated member of a determinate truss: N = 0, dl = alpha dT L."""
    nodes = _two_bar_nodes()
    elements = [
        Element("1", "1", "3", E, A, alpha=ALPHA_T, delta_T=100.0),
        Element("2", "2", "3", E, A, alpha=ALPHA_T, delta_T=0.0),
    ]
    _K, _F, _fixed, U, forces = _fem(nodes, elements, [])
    for f in forces:
        assert abs(f["N"]) <= 1e-6  # zero force up to solver round-off
    assert _rel(forces[0]["delta_L_prestress"], ALPHA_T * 100.0 * 5.0) < REL_TOL
    assert forces[1]["delta_L_prestress"] == 0.0
    # hand solution: unit-load virtual forces on the heated member are
    # n_x = 5/8 (unit +x load) and n_y = 5/6 (unit +y load), the other
    # member contributes nothing (dl_2 = 0):
    # u_x = (5/8) * 6e-3 = 3.75e-3 m,  u_y = (5/6) * 6e-3 = 5e-3 m
    assert _rel(U[4], 3.75e-3) < REL_TOL
    assert _rel(U[5], 5.0e-3) < REL_TOL


def test_a3b_restrained_bar_thermal_compression():
    """Bar between two fixed supports: N = -E A alpha dT, u = 0."""
    nodes = [
        Node("1", 0.0, 0.0, True, True, True),
        Node("2", 4.0, 0.0, True, True, True),
    ]
    elements = [Element("1", "1", "2", E, A, alpha=ALPHA_T, delta_T=100.0)]
    K, F, fixed, U, forces = _fem(nodes, elements, [])
    hand_N = -E * A * ALPHA_T * 100.0  # = -2.52e6 N
    assert _rel(forces[0]["N"], hand_N) < REL_TOL
    assert np.all(U == 0.0)  # every DOF restrained
    R = calculate_reactions(nodes, K, U, F, fixed)
    assert _rel(R["1"]["Fx"], -hand_N) < REL_TOL
    assert _rel(R["2"]["Fx"], hand_N) < REL_TOL


# ---------------------------------------------------------------------
# A4 — linear temperature-gradient scenario on the determinate truss
# ---------------------------------------------------------------------


def test_a4_linear_gradient_temperature_field():
    """T(x_c) = 20 + (800-20) x_c/6  =>  (410, 150, 540) degC by hand."""
    nodes, elements = _three_bar_nodes(), _three_bar_elements()
    temps = get_scenario_temperatures(nodes, elements, "linear_gradient", 800.0)
    assert temps == pytest.approx({"1": 410.0, "2": 150.0, "3": 540.0}, abs=1e-9)
    assert ec.k_E(temps["1"]) == pytest.approx(0.69, abs=1e-12)
    assert ec.k_E(temps["2"]) == pytest.approx(0.95, abs=1e-12)
    assert ec.k_E(temps["3"]) == pytest.approx(0.484, abs=1e-12)


def test_a4_linear_gradient_virtual_work_displacements():
    """Determinate: forces unchanged; u from virtual work with k_E(T_i) EA."""
    nodes, elements = _three_bar_nodes(), _three_bar_elements()
    temps = get_scenario_temperatures(nodes, elements, "linear_gradient", 800.0)
    loads = {"3": {"Fx": 6.0e3, "Fy": -8.0e3}}
    setup = build_engine(nodes, elements, loads, temps)
    u_free = base_displacement(setup, load_vector(nodes, loads, setup.free_dofs))
    U = np.zeros(2 * len(nodes))
    U[list(setup.free_dofs)] = u_free
    forces_hot = setup.k_axial * (setup.b_free @ u_free)

    hand_N = {"1": 68.0e3 / 9.0, "2": -7.0e3 * SQRT13 / 9.0, "3": -85.0e3 / 9.0}
    for i, eid in enumerate(setup.ids):
        assert _rel(float(forces_hot[i]), hand_N[eid]) < 1e-10

    lengths = {"1": 6.0, "2": SQRT13, "3": 5.0}
    k_es = {"1": 0.69, "2": 0.95, "3": 0.484}
    n_x = {"1": 2.0 / 3.0, "2": SQRT13 / 6.0, "3": -5.0 / 6.0}
    n_y_down = {"1": 4.0 / 9.0, "2": -2.0 * SQRT13 / 9.0, "3": -5.0 / 9.0}
    hand_ux = sum(
        hand_N[m] * n_x[m] * lengths[m] / (k_es[m] * EA) for m in ("1", "2", "3")
    )
    hand_uy = -sum(
        hand_N[m] * n_y_down[m] * lengths[m] / (k_es[m] * EA) for m in ("1", "2", "3")
    )
    assert _rel(U[4], hand_ux) < REL_TOL
    assert _rel(U[5], hand_uy) < REL_TOL


# ---------------------------------------------------------------------
# A5 — Euler buckling on the idealised square HSS
# ---------------------------------------------------------------------


def test_a5_idealised_section_and_euler_cold():
    """b = (r/2) sqrt(A/(r-1)); I = b^4 (1 - ((r-2)/r)^4) / 12; P_cr."""
    section = idealised_square_hss(0.01)
    b_hand = (25.0 / 2.0) * math.sqrt(0.01 / 24.0)
    i_hand = b_hand**4 * (1.0 - (23.0 / 25.0) ** 4) / 12.0
    assert _rel(section.b, b_hand) < REL_TOL
    assert _rel(section.i_sec, i_hand) < REL_TOL
    assert _rel(section.area, 0.01) < REL_TOL
    for length in (5.0, math.sqrt(29.0)):
        p_cr = euler_buckling_load(section.i_sec, length, E)
        hand = math.pi**2 * E * i_hand / length**2
        assert _rel(p_cr, hand) < REL_TOL


def test_a5_dcr_at_600_degrees_with_buckling_reduction():
    """Full EN 1993-1-2 4.2.3.1 hand line for a compression member at 600 degC.

    Every quantity below is derived independently of the implementation:

    .. code-block:: text

        N_cr(600) = 0.31 * pi^2 E I / L^2           (k_E(600) = 0.31)
        f_y,600   = 0.47 * f_y                      (k_y(600) = 0.47)
        N_Rd(600) = f_y,600 * A / gamma_M,fi        (gamma_M,fi = 1.0)
        lambda_bar = sqrt(A * f_y,600 / N_cr(600))
        Phi       = 0.5 [1 + 0.65*0.49*(lambda_bar - 0.2) + lambda_bar^2]
        chi       = 1 / (Phi + sqrt(Phi^2 - lambda_bar^2))
        N_b,fi,Rd = chi * N_Rd(600)
        DCR       = |N| / N_b,fi,Rd

    The imperfection factor is curve c (alpha = 0.49) scaled by the fire
    factor 0.65 required by 4.2.3.1(3).
    """
    section = idealised_square_hss(0.01)
    nodes = _two_bar_nodes()
    elements = [
        Element("1", "1", "3", E, A, I_sec=section.i_sec),
        Element("2", "2", "3", E, A, I_sec=section.i_sec),
    ]
    loads = {"3": {"Fx": 0.0, "Fy": -90.0e3}}
    temps = {"1": 600.0, "2": 600.0}
    states = dcr_field(nodes, elements, loads, temps, F_Y)
    st = states["1"]

    assert ec.k_E(600.0) == 0.31  # exact fixture values used by the hand line
    assert ec.k_y(600.0) == 0.47

    p_cr_20 = math.pi**2 * E * section.i_sec / 25.0
    hand_p_cr = 0.31 * p_cr_20
    hand_f_y_600 = 0.47 * F_Y
    hand_n_rd = hand_f_y_600 * A  # gamma_M,fi = 1.0

    hand_lambda = math.sqrt(A * hand_f_y_600 / hand_p_cr)
    alpha_fire = 0.65 * 0.49  # curve c, scaled per 4.2.3.1(3)
    phi = 0.5 * (1.0 + alpha_fire * (hand_lambda - 0.2) + hand_lambda**2)
    hand_chi = 1.0 / (phi + math.sqrt(phi**2 - hand_lambda**2))
    hand_capacity = hand_chi * hand_n_rd
    hand_dcr = 75.0e3 / hand_capacity

    assert st.compression
    assert _rel(st.p_cr or 0.0, hand_p_cr) < REL_TOL
    assert _rel(st.n_rd, hand_n_rd) < REL_TOL
    assert _rel(st.lambda_bar or 0.0, hand_lambda) < REL_TOL
    assert _rel(st.chi or 0.0, hand_chi) < REL_TOL
    assert _rel(st.capacity, hand_capacity) < REL_TOL
    assert _rel(st.dcr, hand_dcr) < REL_TOL
    # lambda_bar ~ 0.655 is well above the 0.2 threshold, so buckling does
    # reduce the capacity below the yield resistance and is the governing mode.
    assert st.capacity_governing is Governing.BUCKLING
    assert st.capacity < st.n_rd


def test_a5_legacy_euler_only_capacity_still_reproducible():
    """The historical ``min(P_cr, N_Rd)`` model must remain selectable.

    It is not the default any more: at ``lambda_bar ~ 0.65`` it overestimates
    the capacity by ~22% because it ignores residual stresses and initial
    out-of-straightness, which made the reported DCR and critical temperature
    optimistic. It is kept so previously published numbers stay reproducible,
    and pinned here so the two models cannot silently converge.
    """
    section = idealised_square_hss(0.01)
    nodes = _two_bar_nodes()
    elements = [
        Element("1", "1", "3", E, A, I_sec=section.i_sec),
        Element("2", "2", "3", E, A, I_sec=section.i_sec),
    ]
    loads = {"3": {"Fx": 0.0, "Fy": -90.0e3}}
    temps = {"1": 600.0, "2": 600.0}

    legacy = dcr_field(nodes, elements, loads, temps, F_Y, BucklingModel.EULER_ONLY)[
        "1"
    ]
    modern = dcr_field(nodes, elements, loads, temps, F_Y)["1"]

    p_cr_20 = math.pi**2 * E * section.i_sec / 25.0
    hand_p_cr = 0.31 * p_cr_20
    hand_n_rd = 0.47 * F_Y * A

    assert _rel(legacy.dcr, 75.0e3 / min(hand_p_cr, hand_n_rd)) < REL_TOL
    assert legacy.chi is None
    assert legacy.capacity_governing is Governing.YIELD
    # The chi model is strictly more conservative for this member.
    assert modern.dcr > legacy.dcr
    assert modern.capacity < legacy.capacity
