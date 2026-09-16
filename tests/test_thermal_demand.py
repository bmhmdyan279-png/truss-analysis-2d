"""Thermal demand in the fire chain — closed forms, cross-path equivalence.

Audit round 2, finding 1: the DCR / theta_sys / CI / retrofit chain degraded
``E(T)`` but solved against the *mechanical* right-hand side only, while
``run()`` applied the same temperature through the equivalent nodal forces of
restrained expansion.  On a redundant structure a heated member therefore
developed real compression that the demand chain never saw — non-conservative
exactly where the library claims coverage (local fire scenarios).

Every test here pins the corrected physics against an *independent* path:
hand-derived closed forms, or the assembly/solver/postprocess chain that
``run()`` uses (pre-scaled ``E`` + ``delta_T``), which shares no code with the
rank-1 engine.  Elements without ``alpha``/``delta_L_free`` must reproduce the
previous numbers bit-for-bit — that reduction is asserted too.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.criticality import (
    base_displacement,
    brute_force_ci,
    build_engine,
    ci_sweep,
    member_forces,
    multi_criteria_ci,
    prestress_lengths,
    reaction_influence,
    total_load_vector,
)
from truss_analysis.criticality.engine import MechanismError
from truss_analysis.exceptions import ConstantAlphaWarning
from truss_analysis.limitstates import (
    UniformForceScan,
    dcr_field,
    member_axial_forces,
    member_critical_temperature,
    system_critical_temperature,
)
from truss_analysis.material.steel_eurocode import k_E as eurocode_k_E
from truss_analysis.material.steel_eurocode import thermal_strain
from truss_analysis.model import Element, Node
from truss_analysis.postprocess import calculate_element_forces, calculate_reactions
from truss_analysis.solver import solve

# ConstantAlphaWarning: this module exercises the fire chain with hot
# temperature fields, which is the path that warns by design -- a member
# above 150 degC expanded with a constant alpha understates the EN
# 1993-1-2 imposed strain by 4-19% and the solver says so. These tests
# pin the *default* behaviour bit for bit, which is exactly what the
# warning says it is preserving; the warning itself, the opt-in
# `use_effective_alpha` path and the measured size of the gap are
# asserted in tests/test_thermal_demand.py.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.ConstantAlphaWarning"
    ),
]

E0 = 210e9
A0 = 0.01
I0 = 1e-6
ALPHA_STEEL = 1.2e-5
F_Y = 235e6
T_AMB = 20.0


def _restrained_bar(alpha: float = ALPHA_STEEL) -> tuple[list[Node], list[Element]]:
    """One bar between two fully fixed supports: expansion fully restrained."""
    nodes = [
        Node("1", 0.0, 0.0, True, True, True),
        Node("2", 4.0, 0.0, True, True, True),
    ]
    elements = [
        Element("b", "1", "2", E0, A0, I_sec=I0, alpha=alpha),
    ]
    return nodes, elements


def _redundant_truss(alpha: float = ALPHA_STEEL) -> tuple[list[Node], list[Element]]:
    """Two pinned supports + one free apex: m + r - 2j = 3 + 4 - 6 = 1.

    The bottom chord ``ab`` spans two fully fixed nodes, so its thermal
    expansion is restrained; heating it produces real compression.  The
    inclination of ``ac``/``bc`` makes the vertical DOF of ``c`` braced (no
    collinear mechanism).
    """
    nodes = [
        Node("a", 0.0, 0.0, True, True, True),
        Node("b", 4.0, 0.0, True, True, True),
        Node("c", 2.0, 3.0, False),
    ]
    elements = [
        Element("ab", "a", "b", E0, A0, I_sec=I0, alpha=alpha),
        Element("ac", "a", "c", E0, A0, I_sec=I0, alpha=alpha),
        Element("bc", "b", "c", E0, A0, I_sec=I0, alpha=alpha),
    ]
    return nodes, elements


def _determinate_truss(alpha: float = ALPHA_STEEL) -> tuple[list[Node], list[Element]]:
    """Pin at ``a``, vertical roller at ``b``: m + r - 2j = 3 + 3 - 6 = 0.

    Free thermal expansion is kinematically admissible (the dilation field
    about the pin slides ``b`` horizontally on its roller), so uniform heating
    produces zero force — the determinate limit of the restrained case.
    """
    nodes = [
        Node("a", 0.0, 0.0, True, True, True),
        Node("b", 4.0, 0.0, True, False, True),
        Node("c", 2.0, 3.0, False),
    ]
    elements = [
        Element("ab", "a", "b", E0, A0, I_sec=I0, alpha=alpha),
        Element("ac", "a", "c", E0, A0, I_sec=I0, alpha=alpha),
        Element("bc", "b", "c", E0, A0, I_sec=I0, alpha=alpha),
    ]
    return nodes, elements


def _assembly_reference(
    nodes: list[Node],
    elements: list[Element],
    loads: dict[str, dict[str, float]],
    temps: dict[str, float],
    extra_scale: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float], float, dict[str, dict[str, float]]]:
    """Independent full path: pre-scale E by k_E(T), set delta_T, assemble.

    Mirrors what ``run()`` computes for a model whose members already carry
    the temperature-degraded modulus — shares no code with the engine.
    Returns ``(U_full, forces, strain_energy, reactions)``.
    """
    ref = []
    for e in elements:
        scale = float(eurocode_k_E(temps[e.id]))
        if extra_scale is not None:
            scale *= float(extra_scale.get(e.id, 1.0))
        ref.append(
            replace(
                e,
                E=e.E * scale,
                delta_T=temps[e.id] - T_AMB,
            )
        )
    K, F_ext, _F_mech, fixed = assemble_global_matrices(nodes, ref)
    idx = {n.id: i for i, n in enumerate(nodes)}
    for nid, comp in loads.items():
        i = idx[nid]
        F_ext[2 * i] += comp.get("Fx", 0.0)
        F_ext[2 * i + 1] += comp.get("Fy", 0.0)
    U = solve(K, F_ext, fixed)
    results, strain_energy, _pw = calculate_element_forces(nodes, ref, U)
    forces = {str(r["id"]): float(r["N"]) for r in results}
    reactions = calculate_reactions(nodes, K, U, F_ext, fixed)
    return U, forces, strain_energy, reactions


# ---------------------------------------------------------------------
# prestress_lengths: the imposed-elongation definition itself
# ---------------------------------------------------------------------


def test_prestress_lengths_formula() -> None:
    nodes, elements = _redundant_truss()
    elements[0] = replace(elements[0], delta_L_free=5e-5)
    temps = {"ab": 620.0, "ac": 20.0, "bc": 320.0}
    dl = prestress_lengths(nodes, elements, temps)
    lab = 4.0
    lac = float(np.hypot(2.0, 3.0))
    assert dl[0] == pytest.approx(ALPHA_STEEL * 600.0 * lab + 5e-5, rel=1e-14)
    assert dl[1] == pytest.approx(0.0, abs=1e-18)  # ac at ambient: no strain
    assert dl[2] == pytest.approx(ALPHA_STEEL * 300.0 * lac, rel=1e-14)


def test_prestress_lengths_without_temps_is_fabrication_only() -> None:
    nodes, elements = _redundant_truss()
    elements[1] = replace(elements[1], delta_L_free=-2e-5)
    dl = prestress_lengths(nodes, elements, None)
    assert dl[0] == 0.0
    assert dl[1] == -2e-5
    assert dl[2] == 0.0


# ---------------------------------------------------------------------
# Closed forms: restrained bar and free expansion
# ---------------------------------------------------------------------


def test_restrained_bar_closed_form_compression() -> None:
    """N(T) = -k_E(T) E A alpha (T - 20): the fire chain sees thermal stress."""
    nodes, elements = _restrained_bar()
    for t in (220.0, 600.0, 900.0):
        forces = member_axial_forces(nodes, elements, {}, {"b": t})
        hand = -float(eurocode_k_E(t)) * E0 * A0 * ALPHA_STEEL * (t - T_AMB)
        assert forces["b"] == pytest.approx(hand, rel=1e-12)
        assert forces["b"] < 0.0  # compression


def test_restrained_bar_at_ambient_is_unstressed() -> None:
    nodes, elements = _restrained_bar()
    forces = member_axial_forces(nodes, elements, {}, {"b": T_AMB})
    assert forces["b"] == pytest.approx(0.0, abs=1e-9)


def test_alpha_zero_reduces_to_previous_behaviour() -> None:
    """No expansion coefficient -> no thermal force: bit-for-bit the old path."""
    nodes, elements = _restrained_bar(alpha=0.0)
    forces = member_axial_forces(nodes, elements, {}, {"b": 900.0})
    assert forces["b"] == 0.0  # exactly zero, not approximately


def test_determinate_free_expansion_zero_force() -> None:
    """Determinate truss, uniform heat: expansion is free, forces vanish."""
    nodes, elements = _determinate_truss()
    temps = {e.id: 700.0 for e in elements}
    forces = member_axial_forces(nodes, elements, {}, temps)
    for v in forces.values():
        assert abs(v) <= 1e-6


# ---------------------------------------------------------------------
# Engine == assembly path (the run() physics), base and perturbed
# ---------------------------------------------------------------------


def test_engine_matches_assembly_path_local_fire() -> None:
    nodes, elements = _redundant_truss()
    temps = {"ab": 650.0, "ac": 300.0, "bc": T_AMB}
    loads = {"c": {"Fx": 12e3, "Fy": -45e3}, "a": {"Fx": 8e3}}

    setup = build_engine(nodes, elements, loads, temps)
    u = base_displacement(setup, total_load_vector(nodes, loads, setup))
    engine_forces = member_forces(setup, u)

    U_ref, forces_ref, _e_ref, _r_ref = _assembly_reference(
        nodes, elements, loads, temps
    )
    u_full = np.zeros(2 * len(nodes))
    u_full[list(setup.free_dofs)] = u
    assert np.allclose(u_full, U_ref, rtol=1e-11, atol=1e-14)
    for i, eid in enumerate(setup.ids):
        assert engine_forces[i] == pytest.approx(forces_ref[eid], rel=1e-10, abs=1e-6)


def test_hot_restrained_chord_carries_compression_the_old_chain_missed() -> None:
    """The demand the fix exists for: hot bottom chord in real compression."""
    nodes, elements = _redundant_truss()
    temps = {"ab": 600.0, "ac": T_AMB, "bc": T_AMB}
    forces = member_axial_forces(nodes, elements, {}, temps)
    # ab is fully restrained between two pins: heating it alone compresses it
    # (the apex members pick up part of the thrust through redundancy).
    assert forces["ab"] < -1e3
    # ...and the DCR must reflect that demand.
    states_hot = dcr_field(nodes, elements, {}, temps, F_Y)
    cold = {e.id: T_AMB for e in elements}
    elements_noexp = [replace(e, alpha=0.0) for e in elements]
    states_noexp = dcr_field(nodes, elements_noexp, {}, temps, F_Y)
    assert states_hot["ab"].dcr > states_noexp["ab"].dcr
    assert states_hot["ab"].axial_force < 0.0
    del cold


def test_rank1_sweep_matches_bruteforce_under_thermal_demand() -> None:
    nodes, elements = _redundant_truss()
    temps = {"ab": 600.0, "ac": 350.0, "bc": T_AMB}
    loads = {"c": {"Fx": 0.0, "Fy": -60e3}}
    alpha = 0.7

    setup = build_engine(nodes, elements, loads, temps)
    u = base_displacement(setup, total_load_vector(nodes, loads, setup))
    sweep = ci_sweep(setup, u, alpha)
    ci_ref, u_max_base_ref, _ = brute_force_ci(nodes, elements, loads, temps, alpha)

    assert float(np.max(np.abs(u))) == pytest.approx(u_max_base_ref, rel=1e-12)
    for eid in setup.ids:
        assert sweep.ci_values[eid] == pytest.approx(ci_ref[eid], rel=1e-9, abs=1e-14)

    # Strongest check: the perturbed column against the independent assembly
    # path with the target member additionally softened by alpha.
    U_ref, _f, _e, _r = _assembly_reference(
        nodes, elements, loads, temps, extra_scale={"ac": alpha}
    )
    col = np.zeros(2 * len(nodes))
    col[list(setup.free_dofs)] = sweep.u_pert[:, setup.ids.index("ac")]
    assert np.allclose(col, U_ref, rtol=1e-10, atol=1e-14)


def test_multi_criteria_matches_assembly_path_under_thermal_demand() -> None:
    """All five indices against direct measurement, heated redundant frame.

    The load case places a mechanical load ON a support node (``a``) so the
    ``f_ext_fixed`` term of the reaction index is exercised: reactions are the
    residual ``(K u)[fixed] - F_ext[fixed]``, not ``(K u)[fixed]``.
    """
    nodes, elements = _redundant_truss()
    temps = {"ab": 600.0, "ac": 400.0, "bc": T_AMB}
    loads = {"c": {"Fx": 15e3, "Fy": -50e3}, "a": {"Fx": 9e3, "Fy": 0.0}}
    alpha = 0.5
    target = "ac"

    setup = build_engine(nodes, elements, loads, temps)
    u = base_displacement(setup, total_load_vector(nodes, loads, setup))
    sweep = ci_sweep(setup, u, alpha)
    k_scale = {e.id: float(eurocode_k_E(temps[e.id])) for e in elements}
    infl = reaction_influence(nodes, elements, k_scale, loads=loads, temps=temps)
    results = multi_criteria_ci(setup, u, sweep, alpha, infl)

    _U0, forces0, energy0, reactions0 = _assembly_reference(
        nodes, elements, loads, temps
    )
    _U1, forces1, energy1, reactions1 = _assembly_reference(
        nodes, elements, loads, temps, extra_scale={target: alpha}
    )

    disp_max0 = float(np.max(np.abs(_U0)))
    disp_max1 = float(np.max(np.abs(_U1)))
    force_max0 = max(abs(v) for v in forces0.values())
    force_max1 = max(abs(v) for v in forces1.values())
    react_max0 = max(max(abs(v["Fx"]), abs(v["Fy"])) for v in reactions0.values())
    react_max1 = max(max(abs(v["Fx"]), abs(v["Fy"])) for v in reactions1.values())

    r = results[target]
    assert r.ci_displacement == pytest.approx(disp_max1 / disp_max0 - 1.0, rel=1e-10)
    assert r.ci_force_max == pytest.approx(force_max1 / force_max0 - 1.0, rel=1e-10)
    assert r.ci_force_self == pytest.approx(
        abs(forces1[target]) / abs(forces0[target]) - 1.0, rel=1e-10
    )
    assert r.ci_energy == pytest.approx(energy1 / energy0 - 1.0, rel=1e-10)
    assert r.ci_reaction == pytest.approx(react_max1 / react_max0 - 1.0, rel=1e-10)


# ---------------------------------------------------------------------
# Determinate truss under thermal-only load: CI must be exactly zero
# ---------------------------------------------------------------------


def test_determinate_thermal_only_ci_is_zero() -> None:
    """Softening a member of a determinate truss cannot change a force-free
    thermal state: the expansion field is stiffness-independent, so every
    perturbed column equals the base and CI = 0 exactly."""
    from truss_analysis.criticality import compute_ci_for_topology

    nodes, elements = _determinate_truss()
    res = compute_ci_for_topology(nodes, elements, {}, {}, "uniform", 600.0)
    assert res.u_max_base > 0.0  # the expansion is visible
    for v in res.ci_values.values():
        assert v == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------
# Critical temperatures: direction of the correction
# ---------------------------------------------------------------------


def test_theta_sys_drops_when_expansion_is_restrained() -> None:
    """Restrained thermal expansion ADDS demand: theta_sys must fall.

    The pre-fix chain degraded capacity with temperature while ignoring the
    growing compression of the restrained chord — systematically overstating
    the critical temperature of exactly the structures the fire scenario
    targets.
    """
    loads = {"c": {"Fx": 0.0, "Fy": -120e3}}
    nodes_exp, elements_exp = _redundant_truss(alpha=ALPHA_STEEL)
    nodes_no, elements_no = _redundant_truss(alpha=0.0)
    grid = tuple(range(20, 1201, 25))
    theta_with = system_critical_temperature(nodes_exp, elements_exp, loads, F_Y, grid)
    theta_without = system_critical_temperature(nodes_no, elements_no, loads, F_Y, grid)
    assert theta_with < theta_without
    assert theta_with >= grid[0]


def test_member_critical_temperature_sees_thermal_demand() -> None:
    """The essence of finding 1, on one member.

    Chord ``ab`` spans two fully fixed supports, so it can never carry a
    *mechanical* force: the old demand chain (stiffness degradation only)
    reported ``None`` — this member never fails at any temperature.  With the
    restrained-expansion demand it fails at a low temperature, because the
    compression ``k_E(T) k0 alpha (T-20) L`` grows while the capacity falls.
    """
    loads = {"c": {"Fx": 0.0, "Fy": -80e3}}
    nodes_exp, elements_exp = _redundant_truss(alpha=ALPHA_STEEL)
    nodes_no, elements_no = _redundant_truss(alpha=0.0)
    grid = tuple(range(20, 1201, 25))
    t_exp = member_critical_temperature(
        nodes_exp, elements_exp, loads, "ab", F_Y, temp_grid=grid
    )
    t_no = member_critical_temperature(
        nodes_no, elements_no, loads, "ab", F_Y, temp_grid=grid
    )
    assert t_exp is not None  # thermal demand makes the chord fail
    assert t_no is None or t_exp < t_no
    # Closed form for the fully restrained chord under uniform heating:
    # N(T) = -k_E(T) E A alpha (T - 20); check the crossing temperature is
    # consistent with the DCR the same formula produces.
    forces = member_axial_forces(
        nodes_exp, elements_exp, {}, {e.id: t_exp for e in elements_exp}
    )
    states = dcr_field(
        nodes_exp, elements_exp, {}, {e.id: t_exp for e in elements_exp}, F_Y
    )
    assert forces["ab"] < 0.0
    assert states["ab"].dcr == pytest.approx(1.0, rel=0.15)


# ---------------------------------------------------------------------
# UniformForceScan: exactness against the per-point path, one factorisation
# ---------------------------------------------------------------------


def test_uniform_scan_matches_member_axial_forces() -> None:
    nodes, elements = _redundant_truss()
    elements = [replace(elements[0], delta_L_free=4e-5), *elements[1:]]
    loads = {"c": {"Fx": 7e3, "Fy": -30e3}}
    scan = UniformForceScan.build(nodes, elements, loads)
    for t in (20.0, 45.0, 220.0, 600.0, 875.0, 1195.0):
        exact = member_axial_forces(nodes, elements, loads, {e.id: t for e in elements})
        fast = scan.forces_dict_at(t)
        for eid in exact:
            assert fast[eid] == pytest.approx(exact[eid], rel=1e-11, abs=1e-6), (t, eid)


def test_uniform_scan_including_fabrication_strain_at_ambient() -> None:
    nodes, elements = _redundant_truss()
    elements = [replace(elements[1], delta_L_free=-3e-5), *elements[1:]]
    loads: dict[str, dict[str, float]] = {}
    scan = UniformForceScan.build(nodes, elements, loads)
    exact = member_axial_forces(nodes, elements, loads, {e.id: T_AMB for e in elements})
    fast = scan.forces_dict_at(T_AMB)
    for eid in exact:
        assert fast[eid] == pytest.approx(exact[eid], rel=1e-12, abs=1e-6)


def test_critical_temperature_scans_factorise_once(monkeypatch) -> None:
    """The grid must not rebuild the engine: exactly ONE lu_factor per scan."""
    import truss_analysis.criticality.engine as engine_mod

    calls = {"n": 0}
    real_lu_factor = engine_mod.lu_factor

    def counting_lu_factor(*args, **kwargs):
        calls["n"] += 1
        return real_lu_factor(*args, **kwargs)

    monkeypatch.setattr(engine_mod, "lu_factor", counting_lu_factor)

    nodes, elements = _redundant_truss()
    loads = {"c": {"Fx": 0.0, "Fy": -90e3}}
    grid = tuple(range(20, 1201, 25))  # the default 48-point grid

    calls["n"] = 0
    system_critical_temperature(nodes, elements, loads, F_Y, grid)
    assert calls["n"] == 1

    calls["n"] = 0
    member_critical_temperature(nodes, elements, loads, "ab", F_Y, temp_grid=grid)
    assert calls["n"] == 1


def test_scan_raises_mechanism_when_stiffness_vanishes() -> None:
    """k_E(1200 degC) = 0: the structure is a mechanism, reported as such."""
    nodes, elements = _redundant_truss()
    scan = UniformForceScan.build(nodes, elements, {})
    with pytest.raises(MechanismError):
        scan.forces_at(1200.0)


# ---------------------------------------------------------------------
# Honest invariance: alpha = 0 exact, restrained expansion temperature-
# dependent (the refined statement of docs/theory.md 5.4)
# ---------------------------------------------------------------------


def test_uniform_ci_invariance_holds_only_without_imposed_strain() -> None:
    from truss_analysis.criticality import compute_ci_for_topology

    loads = {"c": {"Fx": 5e3, "Fy": -40e3}}

    nodes0, elements0 = _redundant_truss(alpha=0.0)
    base0 = compute_ci_for_topology(nodes0, elements0, loads, {}, "uniform", 20.0)
    hot0 = compute_ci_for_topology(nodes0, elements0, loads, {}, "uniform", 700.0)
    drift0 = max(abs(hot0.ci_values[k] - base0.ci_values[k]) for k in base0.ci_values)
    assert drift0 < 1e-12  # no imposed strain -> exact scale invariance
    assert hot0.tau_vs_base == pytest.approx(1.0, abs=1e-10)

    nodes1, elements1 = _redundant_truss(alpha=ALPHA_STEEL)
    base1 = compute_ci_for_topology(nodes1, elements1, loads, {}, "uniform", 20.0)
    hot1 = compute_ci_for_topology(nodes1, elements1, loads, {}, "uniform", 700.0)
    drift1 = max(abs(hot1.ci_values[k] - base1.ci_values[k]) for k in base1.ci_values)
    # Restrained expansion makes the base state genuinely temperature-
    # dependent: the uniform-field invariance theorem no longer applies and
    # the engine must NOT pretend it does.
    assert drift1 > 1e-10


# --------------------------------------------------------------------------
# round-7 audit, item 6: alpha(T) was exposed but never wired into the demand
# chain, so the default path understated the imposed strain with no notice.
# --------------------------------------------------------------------------


def _restrained_pair(theta: float, alpha: float = 1.2e-5, length: float = 3.0):
    """A fully restrained bar, the simplest state with a nonzero thermal force."""
    nodes = [
        Node(id="L", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(
            id="R",
            x=length,
            y=0.0,
            is_support=True,
            support_dx=True,
            support_dy=True,
        ),
    ]
    elements = [
        Element(
            id="bar",
            node_i="L",
            node_j="R",
            E=210e9,
            A=0.01,
            I_sec=1e-6,
            alpha=alpha,
        )
    ]
    return nodes, elements, {"bar": theta}, length


@pytest.mark.parametrize(
    ("theta", "understated_pct"),
    [(100.0, 3.9), (200.0, 6.8), (400.0, 12.3), (600.0, 17.1), (700.0, 19.4)],
)
def test_effective_alpha_reproduces_the_code_curve_exactly(
    theta: float, understated_pct: float
) -> None:
    """The opt-in path is not an approximation of eps_th(T); it *is* eps_th(T).

    ``effective_alpha`` is the secant coefficient ``eps_th / (theta - theta_ref)``,
    so multiplying it back by the same temperature rise must return the curve
    exactly. This is what makes the flag worth having: it is not a correction
    factor tuned to one temperature, it is the standard's own elongation curve
    reached through the framework's existing ``alpha * dT * L`` term.
    """
    nodes, elements, temps, length = _restrained_pair(theta)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantAlphaWarning)
        constant = prestress_lengths(nodes, elements, temps)
    effective = prestress_lengths(nodes, elements, temps, use_effective_alpha=True)

    curve = float(thermal_strain(theta)) * length
    assert effective[0] == pytest.approx(curve, rel=1e-14)
    # and the constant path really is low by the documented amount
    shortfall = (1.0 - constant[0] / effective[0]) * 100.0
    assert shortfall == pytest.approx(understated_pct, abs=0.15)


def test_effective_alpha_is_a_no_op_at_ambient() -> None:
    """No temperature rise means no elongation either way."""
    nodes, elements, temps, _ = _restrained_pair(20.0)
    constant = prestress_lengths(nodes, elements, temps)
    effective = prestress_lengths(nodes, elements, temps, use_effective_alpha=True)
    assert constant[0] == pytest.approx(0.0, abs=1e-18)
    assert effective[0] == pytest.approx(0.0, abs=1e-18)


def test_the_default_is_unchanged_bit_for_bit() -> None:
    """The flag must not have moved anything that was already published.

    Changing the default would silently revise every fire result ever computed
    with this library, which is not a change to make quietly. The warning exists
    so the choice is deliberate; the default exists so the history stays
    reproducible.
    """
    nodes, elements, temps, length = _restrained_pair(600.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantAlphaWarning)
        got = prestress_lengths(nodes, elements, temps)
    # 1.2e-5 * (600 - 20) * 3.0, the constant-alpha value this always returned
    assert got[0] == pytest.approx(1.2e-5 * 580.0 * length, rel=1e-15)


def test_fabrication_term_is_untouched_by_the_flag() -> None:
    """``delta_L_free`` is not a thermal quantity and must not be scaled."""
    nodes = [
        Node(id="L", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=2.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [
        Element(
            id="bar",
            node_i="L",
            node_j="R",
            E=210e9,
            A=0.01,
            alpha=1.2e-5,
            delta_L_free=5e-4,
        )
    ]
    temps = {"bar": 500.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantAlphaWarning)
        constant = prestress_lengths(nodes, elements, temps)
    effective = prestress_lengths(nodes, elements, temps, use_effective_alpha=True)
    assert effective[0] - constant[0] == pytest.approx(
        (effective[0] - 5e-4) - (constant[0] - 5e-4), rel=1e-15
    )
    assert constant[0] > 5e-4
    assert effective[0] > 5e-4


def test_cold_fields_do_not_warn() -> None:
    """The threshold is a measured 5% strain error, not a round number."""
    from truss_analysis.criticality.engine import ALPHA_CONSTANCY_LIMIT

    assert ALPHA_CONSTANCY_LIMIT == 150.0
    for theta in (20.0, 80.0, 120.0, 150.0):
        nodes, elements, temps, _ = _restrained_pair(theta)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConstantAlphaWarning)
            prestress_lengths(nodes, elements, temps)


def test_no_temperature_field_does_not_warn() -> None:
    """An ambient analysis has no thermal strain to understate."""
    nodes, elements, _, _ = _restrained_pair(600.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConstantAlphaWarning)
        got = prestress_lengths(nodes, elements, None)
    assert float(np.max(np.abs(got))) == 0.0


def test_hot_field_warns_and_quantifies_the_gap() -> None:
    """The warning must say by how much, not merely that something is off."""
    nodes, elements, temps, _ = _restrained_pair(600.0)
    with pytest.warns(ConstantAlphaWarning) as caught:
        prestress_lengths(nodes, elements, temps)
    message = str(caught[0].message)
    assert "600 degC" in message
    assert "17.1%" in message
    assert "use_effective_alpha=True" in message
    assert "un-conservative DCR" in message


def test_the_warning_quotes_the_callers_own_alpha() -> None:
    """A user who already supplied a secant coefficient must not be misinformed.

    The percentage is computed from the ``alpha`` values the constant path
    actually used, not from a nominal 1.2e-5, so a model that already carries
    ``effective_alpha(T)`` in ``Element.alpha`` is told the truth about its own
    input rather than warned about somebody else's.
    """
    from truss_analysis.material.steel_eurocode import effective_alpha

    theta = 600.0
    nodes, elements, temps, _ = _restrained_pair(theta, alpha=1.2e-5)
    with pytest.warns(ConstantAlphaWarning) as low:
        prestress_lengths(nodes, elements, temps)

    warm = [
        Element(
            id="bar",
            node_i="L",
            node_j="R",
            E=210e9,
            A=0.01,
            alpha=float(effective_alpha(theta)),
        )
    ]
    with pytest.warns(ConstantAlphaWarning) as high:
        prestress_lengths(nodes, warm, temps)

    pct_low = float(str(low[0].message).split("about ")[1].split("%")[0])
    pct_high = float(str(high[0].message).split("about ")[1].split("%")[0])
    assert pct_low > 15.0
    assert pct_high < 1.0, "a secant alpha is already on the curve"


def test_build_engine_threads_the_flag_and_nothing_else() -> None:
    """The flag changes the imposed strain and must not touch the stiffness.

    ``k_E(T) E`` degradation is a separate question from how much strain the
    elongation carries; conflating them would make the flag do two things and
    neither of them checkable.
    """
    nodes = [
        Node(id="L", x=-1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=0.0, y=-1.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=0.01, is_support=False),
    ]
    elements = [
        Element(id="r1", node_i="L", node_j="A", E=210e9, A=1e-3, alpha=1.2e-5),
        Element(id="r2", node_i="R", node_j="A", E=210e9, A=1e-3, alpha=1.2e-5),
        Element(id="post", node_i="A", node_j="B", E=210e9, A=1e-6),
    ]
    temps = {"r1": 600.0, "r2": 600.0, "post": 600.0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantAlphaWarning)
        base = build_engine(nodes, elements, {}, temps)
        warm = build_engine(nodes, elements, {}, temps, use_effective_alpha=True)

    assert np.array_equal(base.k_axial, warm.k_axial), "stiffness must not move"
    assert base.free_dofs == warm.free_dofs
    assert not np.allclose(base.dl_pre, warm.dl_pre)
    assert float(np.min(warm.dl_pre[:2] / base.dl_pre[:2])) > 1.15
    # the fabrication-only member is unaffected by the flag
    assert base.dl_pre[2] == pytest.approx(warm.dl_pre[2], rel=1e-15)


def _hot_fan(theta: float):
    """Redundant shallow fan: two expanding rafters and one ``alpha = 0`` post."""
    nodes = [
        Node(id="L", x=-1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=0.0, y=-1.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=0.01, is_support=False),
    ]
    elements = [
        Element(id="r1", node_i="L", node_j="A", E=210e9, A=1e-3, alpha=1.2e-5),
        Element(id="r2", node_i="R", node_j="A", E=210e9, A=1e-3, alpha=1.2e-5),
        # alpha = 0 is an explicit "this member does not expand", and the flag
        # must respect it rather than overwrite it with the Eurocode curve
        Element(id="post", node_i="A", node_j="B", E=210e9, A=1e-6, alpha=0.0),
    ]
    temps = {"r1": theta, "r2": theta, "post": theta}
    return nodes, elements, temps


def test_effective_alpha_changes_a_redundant_force_by_the_strain_ratio() -> None:
    """End to end: the flag must reach the member force, not just the elongation.

    A fully restrained two-node bar has no free DOF at all, so ``build_engine``
    cannot factorise it and the end-to-end check has to use a redundant state.
    That is also the more honest test: here the thermal force depends on the
    assembly, not only on the imposed strain, so the flag has to survive the
    solve rather than merely the elongation.
    """
    from truss_analysis.criticality.engine import total_load_vector

    nodes, elements, temps = _hot_fan(600.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantAlphaWarning)
        base = build_engine(nodes, elements, {}, temps)
        warm = build_engine(nodes, elements, {}, temps, use_effective_alpha=True)
        n_base = member_forces(
            base, base_displacement(base, total_load_vector(nodes, {}, base))
        )
        n_warm = member_forces(
            warm, base_displacement(warm, total_load_vector(nodes, {}, warm))
        )

    # the two expanding rafters scale by exactly the strain ratio
    expected = float(thermal_strain(600.0)) / (1.2e-5 * 580.0)
    assert expected > 1.15
    for i in (0, 1):
        assert float(n_warm[i] / n_base[i]) == pytest.approx(expected, rel=1e-6)
    # The alpha = 0 post carries no imposed elongation under either setting --
    # but its *force* still changes, and that is the correct physics rather than
    # a leak in the flag. The fan is redundant, so when the rafters expand more
    # the apex moves further and equilibrium redistributes into the post. A test
    # that asserted the post force was unchanged would be asserting that a
    # redundant structure does not redistribute load, which is the wrong thing
    # to pin. What must not change is the post's own imposed strain.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantAlphaWarning)
        dl_base = prestress_lengths(nodes, elements, temps)
    dl_warm = prestress_lengths(nodes, elements, temps, use_effective_alpha=True)
    assert dl_base[2] == 0.0
    assert dl_warm[2] == 0.0
    assert abs(float(n_base[2])) > 0.0
    assert float(n_warm[2]) != pytest.approx(float(n_base[2]), rel=1e-6)


def test_zero_alpha_member_is_not_overwritten_by_the_flag() -> None:
    """The rule the end-to-end test above depends on, stated on its own."""
    nodes, elements, temps = _hot_fan(600.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantAlphaWarning)
        base = prestress_lengths(nodes, elements, temps)
    warm = prestress_lengths(nodes, elements, temps, use_effective_alpha=True)

    assert base[2] == 0.0
    assert warm[2] == 0.0, "alpha = 0 means the member does not expand"
    assert warm[0] > base[0]
    assert warm[1] > base[1]
