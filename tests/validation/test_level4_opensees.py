"""Level 4 (part 2) — reference validation against OpenSeesPy.

Skipped as a whole when the optional ``openseespy`` dependency is absent
(there is NO mock mode: fabricated reference numbers are worse than none).

For each protocol case (Pratt 4-panel, Warren 6-panel, Howe 8-panel from the
campaign suite, depth variant H1) and each checked state:

* node-by-node displacement comparison and member-by-member axial-force
  comparison against an independent solver fed the SAME explicit modulus
  field ``E_i(T) = k_E(T_i) E_i`` (OpenSees ``Truss`` elements carry no
  temperature dependence of their own — this is documented in the bridge
  module so the comparison cannot be dismissed as circular: it validates
  the structural path, while the reduction curves are validated separately
  at level 2);
* the criticality sweep reproduced in OpenSees as **n+1 fully independent
  models** and compared against the internal rank-1 (Sherman–Morrison)
  sweep — the equivalence witness for the exact-update engine;
* the rank correlation classified through the pre-coded three-branch
  decision tree (>=0.9 full validation, 0.7..0.9 cause analysis, <0.7 model
  review, NaN model review).

Acceptance: rho > 0.9 on every case (measured: rho == 1.0 after the
documented 1e-10 tie-noise quantisation; the raw unquantised rho is asserted
> 0.9 as well).  Displacements, forces and CI values agree to < 1e-8
relative (measured: machine level).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openseespy")

from truss_analysis.criticality import get_scenario_temperatures  # noqa: E402
from truss_analysis.model import Element, Node  # noqa: E402
from truss_analysis.validation import (  # noqa: E402
    RHO_FULL,
    OpenseesSolveError,
    RhoVerdict,
    ci_sweep_in_opensees,
    compare_ci_ranking,
    compare_state,
    opensees_version,
    solve_truss_in_opensees,
)

CASES = ("pratt_4_H1", "warren_6_H1", "howe_8_H1")
STATE_TOL = 1e-8  # relative agreement gate for u and N (measured ~1e-14)
CI_TOL = 1e-8  # relative agreement gate for CI values (measured ~1e-13)
ALPHA = 0.7
HOT = 600.0


def _model(campaign, name):
    return next(c for c in campaign if c.name == name)


def test_opensees_reference_is_real_not_mocked():
    """Guard: the evidence must come from the actual solver binary."""
    assert opensees_version() not in ("absent", "unknown")


def test_unavailable_guard_paths(monkeypatch):
    """The 'openseespy absent' branch must refuse, never fabricate."""
    import truss_analysis.validation.opensees_reference as oref

    monkeypatch.setattr(oref, "_ops", None)
    monkeypatch.setattr(oref, "_HAS_OPENSEES", False)
    assert not oref.opensees_available()
    assert oref.opensees_version() == "absent"
    with pytest.raises(oref.OpenseesUnavailableError):
        oref.solve_truss_in_opensees([], [], {})


def test_singular_model_raises_opensees_solve_error():
    """One bar + one completely free node = a mechanism: OpenSees reports a
    nonzero analyze() status and the bridge must raise instead of returning
    garbage displacements.  (Measured quirk: a single zero pivot can slip
    through LAPACK silently — the two-free-DOF mechanism is the reliable
    singular probe.)"""
    nodes = [
        Node("1", 0.0, 0.0, True, True, True),
        Node("2", 1.0, 0.0),  # completely free: x and y are mechanisms
    ]
    elements = [Element("1", "1", "2", 210.0e9, 0.01)]
    with pytest.raises(OpenseesSolveError):
        solve_truss_in_opensees(nodes, elements, {"2": {"Fx": 1.0e3, "Fy": 0.0}})


def test_zero_baseline_displacement_is_refused(campaign):
    cm = _model(campaign, "pratt_4_H1")
    temps = {e.id: 20.0 for e in cm.elements}
    with pytest.raises(ValueError):
        ci_sweep_in_opensees(cm.nodes, cm.elements, {}, temps, 0.7)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("scenario,temperature", [("uniform", 20.0), ("uniform", HOT)])
def test_state_matches_opensees(campaign, name, scenario, temperature):
    cm = _model(campaign, name)
    temps = get_scenario_temperatures(cm.nodes, cm.elements, scenario, temperature)
    comp = compare_state(cm.nodes, cm.elements, cm.loads, temps)
    assert comp.rel_err_u_max < STATE_TOL
    assert comp.max_rel_node_err < STATE_TOL
    assert comp.max_rel_force_err < STATE_TOL
    # every node and every member was actually compared
    assert set(comp.nodal_disp_internal) == {n.id for n in cm.nodes}
    assert set(comp.forces_internal) == {e.id for e in cm.elements}
    assert set(comp.forces_opensees) == set(comp.forces_internal)


@pytest.mark.parametrize("name", CASES)
def test_hot_local_state_matches_opensees(campaign, name):
    """A heterogeneous modulus field (local_mid at 600 degC) also matches."""
    cm = _model(campaign, name)
    temps = get_scenario_temperatures(cm.nodes, cm.elements, "local_mid", HOT)
    comp = compare_state(cm.nodes, cm.elements, cm.loads, temps)
    assert comp.rel_err_u_max < STATE_TOL
    assert comp.max_rel_node_err < STATE_TOL
    assert comp.max_rel_force_err < STATE_TOL
    n_hot = sum(1 for v in temps.values() if v > 20.0)
    assert 0 < n_hot < len(temps)  # the field really is non-uniform


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("scenario", ["uniform", "local_mid"])
def test_ci_ranking_matches_n_plus_one_opensees_models(campaign, name, scenario):
    cm = _model(campaign, name)
    temps = get_scenario_temperatures(cm.nodes, cm.elements, scenario, HOT)
    comp = compare_ci_ranking(cm.nodes, cm.elements, cm.loads, temps, ALPHA)
    # the decision tree, pre-coded: this run must land in FULL_VALIDATION
    assert comp.branch.verdict is RhoVerdict.FULL_VALIDATION
    assert comp.rho > RHO_FULL
    # After the documented 1e-10 tie quantisation rho is 1.0 up to bucket-
    # boundary collisions of near-tied pairs and spearman's own fp rounding
    # (measured: exactly 1.0 on 4/6 comparisons, 1-2e-16 on the other two).
    assert comp.rho >= 1.0 - 1e-9
    assert comp.extra["rho_raw_unquantised"] > RHO_FULL
    # Sherman–Morrison vs n+1 independent models: value-level equivalence
    assert comp.max_rel_ci_diff < CI_TOL
    assert comp.n_opensees_solves == len(cm.elements) + 1
    assert comp.u_max_base_internal == pytest.approx(
        comp.u_max_base_opensees, rel=STATE_TOL
    )
