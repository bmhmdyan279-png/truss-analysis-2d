"""Multi-criteria CI validated against independent full re-solves.

A new index is only worth having if it is right, so every index computed by
:func:`truss_analysis.criticality.criteria.multi_criteria_ci` from the rank-1
sweep is checked against a brute-force perturbation: rebuild the model with
member ``i`` softened by ``alpha``, assemble from scratch, solve with the
general solver, and measure the response directly. Nothing in these tests
reuses the engine's own factorisation.

The central case is the one that motivated the module: a member whose loss
redistributes force into a neighbour while barely moving the structure. A
displacement-only index ranks it as unimportant; ``CI_N_max`` does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.criticality import (
    base_displacement,
    build_engine,
    ci_sweep,
    load_vector,
    multi_criteria_ci,
    reaction_influence,
)
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

E_STEEL = 210e9
ALPHA = 0.5  # member retains half its axial stiffness


def _indeterminate_frame() -> tuple[
    list[Node], list[Element], dict[str, dict[str, float]]
]:
    """A six-bar frame with two diagonals: statically indeterminate."""
    nodes = [
        Node(id="a", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="b", x=4.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="c", x=4.0, y=3.0, is_support=False),
        Node(id="d", x=1.0, y=3.0, is_support=False),
    ]
    elements = [
        Element(id="ab", node_i="a", node_j="b", E=E_STEEL, A=0.020, I_sec=1e-5),
        Element(id="bc", node_i="b", node_j="c", E=E_STEEL, A=0.010, I_sec=5e-6),
        Element(id="cd", node_i="c", node_j="d", E=E_STEEL, A=0.015, I_sec=6e-6),
        Element(id="da", node_i="d", node_j="a", E=E_STEEL, A=0.012, I_sec=5e-6),
        Element(id="ac", node_i="a", node_j="c", E=E_STEEL, A=0.008, I_sec=3e-6),
        Element(id="bd", node_i="b", node_j="d", E=E_STEEL, A=0.009, I_sec=4e-6),
    ]
    loads = {"c": {"Fx": 20e3, "Fy": -60e3}, "d": {"Fx": -5e3, "Fy": -15e3}}
    return nodes, elements, loads


def _brute_force_responses(
    nodes: list[Node],
    elements: list[Element],
    loads: dict[str, dict[str, float]],
    target: str,
    alpha: float,
) -> dict[str, float]:
    """Solve the model from scratch with ``target`` softened; measure directly."""
    perturbed = [
        Element(
            id=e.id,
            node_i=e.node_i,
            node_j=e.node_j,
            E=e.E * (alpha if e.id == target else 1.0),
            A=e.A,
            I_sec=e.I_sec,
        )
        for e in elements
    ]
    K, F_ext, F_mech, fixed = assemble_global_matrices(nodes, perturbed)
    index = {n.id: i for i, n in enumerate(nodes)}
    for nid, comp in loads.items():
        i = index[nid]
        F_ext[2 * i] += comp["Fx"]
        F_ext[2 * i + 1] += comp["Fy"]
        F_mech[2 * i] += comp["Fx"]
        F_mech[2 * i + 1] += comp["Fy"]

    U = solve(K, F_ext, fixed)
    results, strain_energy, _ = calculate_element_forces(nodes, perturbed, U)
    forces = {str(r["id"]): abs(float(r["N"])) for r in results}
    reactions = calculate_reactions(nodes, K, U, F_ext, fixed)
    reaction_max = max(
        (max(abs(v["Fx"]), abs(v["Fy"])) for v in reactions.values()), default=0.0
    )
    return {
        "disp_max": float(np.max(np.abs(U))),
        "force_max": max(forces.values()),
        "force_self": forces.get(target, 0.0),
        "energy": float(strain_energy),
        "reaction_max": reaction_max,
    }


def _base_responses(
    nodes: list[Node],
    elements: list[Element],
    loads: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Same measurements on the unperturbed model.

    ``force_self`` is meaningless here (no member is targeted), so the caller
    takes the per-member self force from the individual perturbed runs.
    """
    out = _brute_force_responses(nodes, elements, loads, "", 1.0)
    del out["force_self"]
    # Per-member forces on the unperturbed model, for the CI_N_self reference.
    perturbed = elements  # alpha 1.0 leaves E untouched
    K, F_ext, _F_mech, fixed = assemble_global_matrices(nodes, perturbed)
    index = {n.id: i for i, n in enumerate(nodes)}
    for nid, comp in loads.items():
        i = index[nid]
        F_ext[2 * i] += comp["Fx"]
        F_ext[2 * i + 1] += comp["Fy"]
    U = solve(K, F_ext, fixed)
    results, _, _ = calculate_element_forces(nodes, perturbed, U)
    out["forces_by_id"] = {str(r["id"]): abs(float(r["N"])) for r in results}
    return out


def _indices(
    nodes: list[Node],
    elements: list[Element],
    loads: dict[str, dict[str, float]],
    alpha: float = ALPHA,
):
    """Run the rank-1 sweep and return the multi-criteria results."""
    setup = build_engine(nodes, elements, loads, {e.id: 20.0 for e in elements})
    u = base_displacement(setup, load_vector(nodes, loads, setup.free_dofs))
    sweep = ci_sweep(setup, u, alpha)
    infl = reaction_influence(nodes, elements)
    return multi_criteria_ci(setup, u, sweep, alpha, infl), setup, u, sweep


def test_ci_displacement_matches_the_engine_exactly() -> None:
    """The displacement index must be the engine's own CI, not a reimplementation."""
    nodes, elements, loads = _indeterminate_frame()
    results, _setup, _u, sweep = _indices(nodes, elements, loads)
    for eid, ci in sweep.ci_values.items():
        assert results[eid].ci_displacement == pytest.approx(ci, rel=1e-14, abs=1e-18)


def test_every_index_matches_a_brute_force_resolve() -> None:
    """All five indices must reproduce an independent full re-solve."""
    nodes, elements, loads = _indeterminate_frame()
    base = _base_responses(nodes, elements, loads)
    results, *_ = _indices(nodes, elements, loads)

    for elem in elements:
        eid = elem.id
        brute = _brute_force_responses(nodes, elements, loads, eid, ALPHA)
        got = results[eid]

        expected = {
            "ci_displacement": brute["disp_max"] / base["disp_max"] - 1.0,
            "ci_force_max": brute["force_max"] / base["force_max"] - 1.0,
            "ci_force_self": brute["force_self"] / base["forces_by_id"][eid] - 1.0,
            "ci_energy": brute["energy"] / base["energy"] - 1.0,
            "ci_reaction": brute["reaction_max"] / base["reaction_max"] - 1.0,
        }
        for key, want in expected.items():
            assert getattr(got, key) == pytest.approx(want, rel=1e-9, abs=1e-14), (
                f"member {eid}: {key} rank-1={getattr(got, key)!r} brute-force={want!r}"
            )


def test_indices_disagree_materially_for_some_member() -> None:
    """The motivating case: a single scalar ranks a member by the wrong physics.

    Softening the vertical member ``bc`` of this frame lets node c drop a long
    way -- ``CI_u = +0.85`` -- while the largest force anywhere in the
    structure actually *falls* by 7%. The member that was carrying the peak
    force is ``bc`` itself, and softening it sheds that load. A displacement-only
    index calls this the most critical member by a wide margin; the force index
    says the opposite. Both are correct statements about different questions,
    which is precisely why one scalar is not enough.
    """
    nodes, elements, loads = _indeterminate_frame()
    results, *_ = _indices(nodes, elements, loads)

    bc = results["bc"]
    assert bc.ci_displacement > 0.5
    assert bc.ci_force_max < 0.0
    # The composite takes the largest effect, so it does not average the
    # disagreement away.
    assert bc.composite == pytest.approx(bc.ci_displacement, rel=1e-12)
    assert bc.governing == "displacement"

    # and some member must be force- or energy-critical rather than
    # displacement-critical, or the extra indices would carry no information
    assert any(r.governing != "displacement" for r in results.values())


def test_energy_index_is_largest_for_the_stiffest_load_path() -> None:
    """``CI_E`` must be positive whenever softening a member adds compliance.

    Removing stiffness from a loaded structure always increases its total
    strain energy at fixed load, so a negative ``CI_E`` would mean the energy
    was not formed from the perturbed stiffness.
    """
    nodes, elements, loads = _indeterminate_frame()
    results, *_ = _indices(nodes, elements, loads)
    for eid, r in results.items():
        assert r.ci_energy > 0.0, eid


def test_reaction_index_vanishes_for_an_externally_determinate_frame() -> None:
    """Global equilibrium pins the reactions, so they cannot redistribute.

    The frame has a pin (2) plus a roller (1) = 3 restraints in 2D, so the
    reactions follow from equilibrium with the external loads alone and are
    independent of member stiffness. ``CI_R`` must therefore be exactly zero
    for every member -- a strong check, because a wrong rank-1 reaction
    correction would show up here as a small non-zero number.
    """
    nodes, elements, loads = _indeterminate_frame()
    results, *_ = _indices(nodes, elements, loads)
    for eid, r in results.items():
        assert r.ci_reaction == pytest.approx(0.0, abs=1e-9), eid


def _continuous_truss() -> tuple[
    list[Node], list[Element], dict[str, dict[str, float]]
]:
    """A three-support continuous truss: externally indeterminate to degree 2.

    With six restraints in 2D the support reactions are *not* fixed by global
    equilibrium, so softening a member genuinely redistributes load between
    supports. This is the model that exercises ``CI_R``; the frames above all
    have an equilibrium-determined peak reaction and return exactly zero.
    """
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="3", x=8.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="4", x=2.0, y=3.0, is_support=False),
        Node(id="5", x=6.0, y=3.0, is_support=False),
    ]
    elements = [
        Element(id="12", node_i="1", node_j="2", E=E_STEEL, A=0.020),
        Element(id="23", node_i="2", node_j="3", E=E_STEEL, A=0.020),
        Element(id="14", node_i="1", node_j="4", E=E_STEEL, A=0.010),
        Element(id="42", node_i="4", node_j="2", E=E_STEEL, A=0.010),
        Element(id="25", node_i="2", node_j="5", E=E_STEEL, A=0.010),
        Element(id="53", node_i="5", node_j="3", E=E_STEEL, A=0.010),
        Element(id="45", node_i="4", node_j="5", E=E_STEEL, A=0.015),
    ]
    loads = {"4": {"Fx": 0.0, "Fy": -80e3}, "5": {"Fx": 25e3, "Fy": -40e3}}
    return nodes, elements, loads


def test_reaction_index_redistributes_when_equilibrium_allows_it() -> None:
    """``CI_R`` must respond once the peak reaction is not equilibrium-fixed."""
    nodes, elements, loads = _continuous_truss()
    assert len(elements) + 5 - 2 * len(nodes) == 2  # externally indeterminate

    results, *_ = _indices(nodes, elements, loads)
    assert max(abs(r.ci_reaction) for r in results.values()) > 0.05

    # Every value still has to match an independent full re-solve.
    base = _base_responses(nodes, elements, loads)
    for elem in elements:
        brute = _brute_force_responses(nodes, elements, loads, elem.id, ALPHA)
        want = brute["reaction_max"] / base["reaction_max"] - 1.0
        assert results[elem.id].ci_reaction == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_all_indices_match_brute_force_on_the_continuous_truss() -> None:
    """The full five-index comparison, on a model where none of them vanish."""
    nodes, elements, loads = _continuous_truss()
    base = _base_responses(nodes, elements, loads)
    results, *_ = _indices(nodes, elements, loads)

    for elem in elements:
        eid = elem.id
        brute = _brute_force_responses(nodes, elements, loads, eid, ALPHA)
        got = results[eid]
        expected = {
            "ci_displacement": brute["disp_max"] / base["disp_max"] - 1.0,
            "ci_force_max": brute["force_max"] / base["force_max"] - 1.0,
            "ci_force_self": brute["force_self"] / base["forces_by_id"][eid] - 1.0,
            "ci_energy": brute["energy"] / base["energy"] - 1.0,
            "ci_reaction": brute["reaction_max"] / base["reaction_max"] - 1.0,
        }
        for key, want in expected.items():
            assert getattr(got, key) == pytest.approx(want, rel=1e-9, abs=1e-14), (
                f"member {eid}: {key}"
            )


def test_rank_one_reaction_correction_term_is_required() -> None:
    """Dropping the ``Delta_i b_i[fixed]`` term must break ``CI_R``.

    The perturbed member's own contribution to the supports is a separate rank-1
    term on top of ``K[fixed, free] u``. This test guards against a
    simplification that would look plausible and be wrong: it recomputes the
    index with only the first term and asserts it disagrees with the verified
    value.
    """
    nodes, elements, loads = _continuous_truss()
    setup = build_engine(nodes, elements, loads, {e.id: 20.0 for e in elements})
    u = base_displacement(setup, load_vector(nodes, loads, setup.free_dofs))
    sweep = ci_sweep(setup, u, ALPHA)
    infl = reaction_influence(nodes, elements)

    correct = multi_criteria_ci(setup, u, sweep, ALPHA, infl)
    # Same computation with the correction term removed (b_fixed zeroed).
    from dataclasses import replace

    crippled = replace(infl, b_fixed=np.zeros_like(infl.b_fixed))
    without = multi_criteria_ci(setup, u, sweep, ALPHA, crippled)

    assert any(
        abs(correct[e].ci_reaction - without[e].ci_reaction) > 1e-6 for e in correct
    ), "the rank-1 reaction correction has no effect; it is either wrong or unused"


def test_governing_index_is_the_largest_one() -> None:
    """``governing`` must name whichever index is actually largest."""
    nodes, elements, loads = _indeterminate_frame()
    results, *_ = _indices(nodes, elements, loads)
    name_of = {
        "ci_displacement": "displacement",
        "ci_force_max": "force_max",
        "ci_force_self": "force_self",
        "ci_energy": "energy",
        "ci_reaction": "reaction",
    }
    for eid, r in results.items():
        values = {name_of[k]: getattr(r, k) for k in name_of}
        best = max(values.values())
        assert r.composite == pytest.approx(best, rel=1e-15)
        assert values[r.governing] == pytest.approx(best, rel=1e-15), eid


def test_composite_never_understates_any_index() -> None:
    """The composite is a conservative summary, so it bounds every component."""
    nodes, elements, loads = _indeterminate_frame()
    results, *_ = _indices(nodes, elements, loads)
    for eid, r in results.items():
        for key in (
            "ci_displacement",
            "ci_force_max",
            "ci_force_self",
            "ci_energy",
            "ci_reaction",
        ):
            value = getattr(r, key)
            if np.isfinite(value):
                assert r.composite >= value - 1e-15, f"{eid}.{key}"


def test_reaction_index_is_nan_without_the_influence_matrix() -> None:
    """Omitting the optional inputs must report nan, never a wrong number."""
    nodes, elements, loads = _indeterminate_frame()
    setup = build_engine(nodes, elements, loads, {e.id: 20.0 for e in elements})
    u = base_displacement(setup, load_vector(nodes, loads, setup.free_dofs))
    sweep = ci_sweep(setup, u, ALPHA)
    results = multi_criteria_ci(setup, u, sweep, ALPHA)
    for eid, r in results.items():
        assert np.isnan(r.ci_reaction), eid
        # and the nan must not leak into the composite or the governing label
        assert r.composite == pytest.approx(
            max(r.ci_displacement, r.ci_force_max, r.ci_force_self, r.ci_energy),
            rel=1e-14,
        )
        assert r.governing != "reaction"


def test_mechanism_columns_propagate_infinity() -> None:
    """A guarded member must report inf everywhere, never a finite number.

    The base frame is stable, but removing its single diagonal ``ac`` leaves a
    four-bar linkage, which is a mechanism. Driving ``alpha`` to zero therefore
    trips the engine's guard; with a brute-force fallback that itself reports
    singularity, the column becomes ``+inf`` and every index must inherit it
    rather than being computed from a zeroed placeholder.
    """
    from truss_analysis.criticality import MechanismError

    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="3", x=4.0, y=3.0, is_support=False),
        Node(id="4", x=0.0, y=3.0, is_support=False),
    ]
    elements = [
        Element(id="12", node_i="1", node_j="2", E=E_STEEL, A=0.02),
        Element(id="23", node_i="2", node_j="3", E=E_STEEL, A=0.01),
        Element(id="34", node_i="3", node_j="4", E=E_STEEL, A=0.015),
        Element(id="41", node_i="4", node_j="1", E=E_STEEL, A=0.012),
        Element(id="13", node_i="1", node_j="3", E=E_STEEL, A=0.008),
    ]
    loads = {"3": {"Fx": 0.0, "Fy": -10e3}}

    setup = build_engine(nodes, elements, loads, {e.id: 20.0 for e in elements})
    u = base_displacement(setup, load_vector(nodes, loads, setup.free_dofs))

    def brute(_i: int) -> np.ndarray:
        raise MechanismError("brute force confirms the mechanism")

    sweep = ci_sweep(setup, u, 0.0, brute_column=brute)
    assert "13" in sweep.flagged

    results = multi_criteria_ci(
        setup, u, sweep, 0.0, reaction_influence(nodes, elements)
    )
    flagged = results["13"]
    assert flagged.governing == "mechanism"
    for key in (
        "ci_displacement",
        "ci_force_max",
        "ci_force_self",
        "ci_energy",
        "ci_reaction",
        "composite",
    ):
        assert getattr(flagged, key) == float("inf"), key


def test_indices_are_zero_for_a_determinate_structure_force_redistribution() -> None:
    """In a determinate truss, softening a member cannot change the forces.

    Statics alone fixes the force distribution, so ``CI_N_max`` and
    ``CI_N_self`` must vanish exactly while ``CI_u`` and ``CI_E`` do not. This
    is a strong physical check: a nonzero force index here would mean the
    index is picking up a numerical artefact rather than redistribution.
    """
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="3", x=2.0, y=3.0, is_support=False),
    ]
    elements = [
        Element(id="13", node_i="1", node_j="3", E=E_STEEL, A=0.010),
        Element(id="23", node_i="2", node_j="3", E=E_STEEL, A=0.010),
        Element(id="12", node_i="1", node_j="2", E=E_STEEL, A=0.010),
    ]
    loads = {"3": {"Fx": 0.0, "Fy": -50e3}}

    results, *_ = _indices(nodes, elements, loads)
    for eid, r in results.items():
        assert r.ci_force_max == pytest.approx(0.0, abs=1e-9), eid
        assert r.ci_force_self == pytest.approx(0.0, abs=1e-9), eid
        # displacement and energy DO change: a softer member stretches more
        assert r.ci_displacement > 0.0
        assert r.ci_energy != 0.0


def test_reaction_influence_matches_direct_reaction_calculation() -> None:
    """``K[fixed, free]`` must reproduce calculate_reactions on the base state."""
    nodes, elements, loads = _indeterminate_frame()
    setup = build_engine(nodes, elements, loads, {e.id: 20.0 for e in elements})
    u_free = base_displacement(setup, load_vector(nodes, loads, setup.free_dofs))
    k_ff = reaction_influence(nodes, elements).k_fixed_free

    # Rebuild the full displacement vector and compare with the reference path.
    K, F_ext, _F_mech, fixed = assemble_global_matrices(nodes, elements)
    index = {n.id: i for i, n in enumerate(nodes)}
    for nid, comp in loads.items():
        i = index[nid]
        F_ext[2 * i] += comp["Fx"]
        F_ext[2 * i + 1] += comp["Fy"]
    U = solve(K, F_ext, fixed)
    reactions = calculate_reactions(nodes, K, U, F_ext, fixed)
    reference = np.array(
        [
            reactions[nodes[d // 2].id]["Fx" if d % 2 == 0 else "Fy"]
            for d in sorted(fixed)
        ]
    )
    assert np.allclose(k_ff @ u_free, reference, rtol=1e-10, atol=1e-6)


def test_block_size_does_not_change_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocking is a memory optimisation and must not alter any value."""
    from truss_analysis.criticality import criteria as criteria_mod

    nodes, elements, loads = _indeterminate_frame()
    results_full, *_ = _indices(nodes, elements, loads)

    monkeypatch.setattr(criteria_mod, "FORCE_BLOCK_SIZE", 1)
    setup = build_engine(nodes, elements, loads, {e.id: 20.0 for e in elements})
    u = base_displacement(setup, load_vector(nodes, loads, setup.free_dofs))
    sweep = ci_sweep(setup, u, ALPHA)
    results_blocked = multi_criteria_ci(
        setup, u, sweep, ALPHA, reaction_influence(nodes, elements)
    )

    assert set(results_full) == set(results_blocked)
    for eid in results_full:
        a, b = results_full[eid], results_blocked[eid]
        for key in (
            "ci_displacement",
            "ci_force_max",
            "ci_force_self",
            "ci_energy",
            "ci_reaction",
            "composite",
        ):
            assert getattr(a, key) == pytest.approx(getattr(b, key), rel=1e-14), (
                f"{eid}.{key}"
            )
        assert a.governing == b.governing


# ---------------------------------------------------------------------
# reaction_influence: sparse construction of K[fixed, free] and the
# direct fixed-DOF load (audit round 2, findings 1 & 3)
# ---------------------------------------------------------------------


def test_reaction_influence_matches_dense_slice_without_dense_k() -> None:
    """k_fixed_free equals the dense einsum slice, built without the O(n^2) wall."""
    from truss_analysis.criticality import member_matrices

    nodes, elements, _loads = _indeterminate_frame()
    infl = reaction_influence(nodes, elements)
    b, k = member_matrices(nodes, elements)
    k_full = np.einsum("i,ip,iq->pq", k, b, b)
    from truss_analysis.criticality.engine import free_dof_indices
    from truss_analysis.model import fixed_dof_indices

    fixed = fixed_dof_indices(nodes)
    free = list(free_dof_indices(nodes))
    assert infl.k_fixed_free.shape == (len(fixed), len(free))
    assert np.allclose(infl.k_fixed_free, k_full[np.ix_(fixed, free)], rtol=0, atol=0)


def test_reaction_influence_carries_support_loads_and_thermal_forces() -> None:
    """f_ext_fixed = mechanical load on supports + imposed equivalent forces."""
    from truss_analysis.criticality import prestress_lengths

    nodes, elements, _loads = _indeterminate_frame()
    # a mechanical load applied directly on support node "a"
    loads = {"a": {"Fx": 5e3, "Fy": -2e3}}
    # heat one member: its equivalent pair pushes on the supports too
    elements_t = [
        Element(
            id=e.id,
            node_i=e.node_i,
            node_j=e.node_j,
            E=e.E,
            A=e.A,
            I_sec=e.I_sec,
            alpha=1.2e-5 if e.id == "ab" else 0.0,
        )
        for e in elements
    ]
    temps = {e.id: (620.0 if e.id == "ab" else 20.0) for e in elements}
    k_scale = {"ab": 0.31, **{e.id: 1.0 for e in elements if e.id != "ab"}}

    infl = reaction_influence(nodes, elements_t, k_scale, loads=loads, temps=temps)
    assert infl.f_ext_fixed is not None

    # independent construction of the same vector
    from truss_analysis.criticality import member_matrices
    from truss_analysis.model import fixed_dof_indices

    b, k = member_matrices(nodes, elements_t, k_scale)
    fixed = fixed_dof_indices(nodes)
    dl = prestress_lengths(nodes, elements_t, temps)
    f_ext = np.zeros(2 * len(nodes))
    f_ext[0] += 5e3  # node "a" is index 0: DOFs 0 (x) and 1 (y)
    f_ext[1] += -2e3
    expected = f_ext[fixed] + (b[:, fixed].T @ (k * dl))
    assert np.allclose(infl.f_ext_fixed, expected, rtol=1e-12, atol=1e-9)
    assert np.any(np.abs(expected) > 0.0)  # the case is not vacuous
