"""Equivalence of the rank-1 engine against brute force.

The most important test of the project: on >=30 seeded random trusses and on
all 21 campaign topologies, the Sherman-Morrison CI sweep must agree with a
full refactorisation per member to < 1e-12.  Also covers the K = sum k bb^T
identity, the Woodbury rank-r path, the mechanism guard and the absence of
deepcopy in the package.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.criticality import (
    MechanismError,
    brute_force_ci,
    build_engine,
    ci_sweep,
    member_matrices,
    perturb_multi,
)
from truss_analysis.criticality.engine import (
    base_displacement,
    load_vector,
    total_load_vector,
)
from truss_analysis.criticality.scenarios import get_scenario_temperatures
from truss_analysis.material.steel_eurocode import k_E as eurocode_k_E
from truss_analysis.model import Element, Node
from truss_analysis.topology_generator import generate_topology

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

PACKAGE_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "truss_analysis" / "criticality"
)


def _objects(model):
    nodes = [
        Node(
            id=str(n["id"]),
            x=float(n["x"]),
            y=float(n["y"]),
            is_support=bool(n.get("is_support", False)),
            support_dx=bool(n.get("support_dx", False)),
            support_dy=bool(n.get("support_dy", False)),
        )
        for n in model["nodes"]
    ]
    elements = [
        Element(
            id=str(e["id"]),
            node_i=str(e["node_i"]),
            node_j=str(e["node_j"]),
            E=float(e["E"]),
            A=float(e["A"]),
        )
        for e in model["elements"]
    ]
    loads = {
        ld["node_id"]: {"Fx": ld["Fx"], "Fy": ld["Fy"]} for ld in model.get("loads", [])
    }
    return nodes, elements, loads


def _engine_ci(nodes, elements, loads, temps, alpha=0.7):
    setup = build_engine(nodes, elements, loads, temps)
    u = base_displacement(setup, load_vector(nodes, loads, setup.free_dofs))
    sweep = ci_sweep(setup, u, alpha)
    return sweep.ci_values


def test_stiffness_identity_against_assembler(campaign) -> None:
    """K_ff built from sum k_i b_i b_i^T equals the library assembler's K_ff."""
    for cm in campaign[:6]:
        b, k = member_matrices(cm.nodes, cm.elements)
        from truss_analysis.criticality.engine import free_dof_indices

        free = free_dof_indices(cm.nodes)
        k_mine = np.einsum("i,ip,iq->pq", k, b, b)[np.ix_(free, free)]
        k_lib, _, _, fixed = assemble_global_matrices(cm.nodes, cm.elements)
        k_lib_ff = k_lib[np.ix_(free, free)]
        assert np.allclose(k_mine, k_lib_ff, rtol=1e-10, atol=1e-6)
        assert len(fixed) == len(cm.nodes) * 2 - len(free)


def _random_case(case: int):
    rng = np.random.default_rng(42 + case)
    family = ("warren", "pratt", "howe")[case % 3]
    n_panels = int(rng.integers(3, 9))
    span = float(rng.uniform(10.0, 40.0))
    height = float(rng.uniform(2.0, 8.0))
    model = generate_topology(family, n_panels=n_panels, span=span, height=height)
    nodes, elements, loads = _objects(model)
    scenario = ("uniform", "local_mid", "local_left", "local_right", "linear_gradient")[
        case % 5
    ]
    theta = float((300.0, 600.0, 800.0)[case % 3])
    return nodes, elements, loads, scenario, theta


def _stable(nodes, elements, loads, scenario, theta) -> bool:
    """The topology generator yields mechanisms for some (family, n_panels);
    a singular base state is a legitimate MechanismError, not an equivalence
    counterexample, so such draws are filtered out of the random suite."""
    temps = get_scenario_temperatures(nodes, elements, scenario, theta)
    try:
        build_engine(nodes, elements, loads, temps)
    except MechanismError:
        return False
    return True


def test_equivalence_random_trusses_at_least_30() -> None:
    checked = 0
    for case in range(120):
        nodes, elements, loads, scenario, theta = _random_case(case)
        if not _stable(nodes, elements, loads, scenario, theta):
            continue
        temps = get_scenario_temperatures(nodes, elements, scenario, theta)
        engine = _engine_ci(nodes, elements, loads, temps)
        brute, _, _ = brute_force_ci(nodes, elements, loads, temps, 0.7)
        diff = max(abs(engine[e] - brute[e]) for e in engine)
        assert diff < 1e-12, (case, scenario, theta, diff)
        checked += 1
        if checked >= 36:
            break
    assert checked >= 30, checked


@pytest.mark.parametrize("case", [0, 1, 3, 4, 6, 9, 10, 12])
def test_equivalence_random_trusses_fixed_seeds(case: int) -> None:
    nodes, elements, loads, scenario, theta = _random_case(case)
    temps = get_scenario_temperatures(nodes, elements, scenario, theta)
    engine = _engine_ci(nodes, elements, loads, temps)
    brute, _, _ = brute_force_ci(nodes, elements, loads, temps, 0.7)
    diff = max(abs(engine[e] - brute[e]) for e in engine)
    assert diff < 1e-12, (case, scenario, theta, diff)


def test_equivalence_campaign_21(campaign) -> None:
    for cm in campaign:
        temps = get_scenario_temperatures(cm.nodes, cm.elements, "local_mid", 600.0)
        engine = _engine_ci(cm.nodes, cm.elements, cm.loads, temps)
        brute, _, _ = brute_force_ci(cm.nodes, cm.elements, cm.loads, temps, 0.7)
        diff = max(abs(engine[e] - brute[e]) for e in engine)
        assert diff < 1e-12, (cm.name, diff)


def test_guard_routes_near_mechanism_and_flags_it(campaign) -> None:
    """alpha -> 0 on a determinate member: the guard fires, the mechanism
    is flagged, and no silent finite number is emitted."""
    cm = next(c for c in campaign if c.name == "control_1")
    temps = get_scenario_temperatures(cm.nodes, cm.elements, "uniform", 20.0)
    setup = build_engine(cm.nodes, cm.elements, cm.loads, temps)
    u = base_displacement(setup, load_vector(cm.nodes, cm.loads, setup.free_dofs))

    def singular_brute(_i: int):
        raise MechanismError("singular")

    with pytest.raises(MechanismError):
        ci_sweep(setup, u, 0.0, brute_column=None)  # guarded member, no fallback

    sweep = ci_sweep(setup, u, 0.0, brute_column=singular_brute)
    assert set(sweep.flagged.values()) == {"mechanism(singular)"}
    assert all(np.isinf(v) for v in sweep.ci_values.values())
    assert sweep.u_max_perturbed_max == np.inf


def test_woodbury_rank_r_matches_full_resolve(campaign) -> None:
    cm = next(c for c in campaign if c.name == "warren_6_shallow")
    temps = get_scenario_temperatures(cm.nodes, cm.elements, "local_mid", 600.0)
    setup = build_engine(cm.nodes, cm.elements, cm.loads, temps)
    u = base_displacement(setup, load_vector(cm.nodes, cm.loads, setup.free_dofs))
    idx = [0, 5, 9]
    alphas = [0.5, 0.7, 0.9]
    u_multi = perturb_multi(setup, u, idx, alphas)

    k_scale = {e.id: float(eurocode_k_E(temps[e.id])) for e in cm.elements}
    for i, a in zip(idx, alphas, strict=True):
        k_scale[cm.elements[i].id] *= a
    b, k = member_matrices(cm.nodes, cm.elements, k_scale)
    from scipy.linalg import lu_factor, lu_solve

    from truss_analysis.criticality.engine import free_dof_indices

    free = free_dof_indices(cm.nodes)
    lu = lu_factor(np.einsum("i,ip,iq->pq", k, b, b)[np.ix_(free, free)])
    u_full = lu_solve(lu, load_vector(cm.nodes, cm.loads, free))
    assert np.allclose(u_multi, u_full, rtol=1e-10, atol=1e-12)


def test_no_deepcopy_in_criticality_package() -> None:
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        assert "deepcopy" not in path.read_text(encoding="utf-8"), path.name


def test_engine_module_documents_validity_limit() -> None:
    src = (PACKAGE_DIR / "engine.py").read_text(encoding="utf-8")
    assert "single-member" in src
    assert "Woodbury" in src
    assert "perturb_multi" in inspect.getsource(perturb_multi)


def test_perturb_multi_raises_mechanism_error_not_linalg(campaign) -> None:
    """Multi-member damage that creates a mechanism must raise the library's
    contract error -- including the numerically-near-singular core that
    ``np.linalg.solve`` would happily invert into garbage (round-5 audit).

    On the determinate control model, deleting ANY member already removes a
    load path; deleting two leaves the Woodbury core mathematically singular
    but numerically solvable (~1e-17 pivot), which pre-fix returned a finite
    displacement vector instead of failing.
    """
    cm = next(c for c in campaign if c.name == "control_1")
    temps = get_scenario_temperatures(cm.nodes, cm.elements, "uniform", 20.0)
    setup = build_engine(cm.nodes, cm.elements, cm.loads, temps)
    u = base_displacement(setup, total_load_vector(cm.nodes, cm.loads, setup))

    for indices in ([0], [0, 1], list(range(len(cm.elements)))):
        alphas = [0.0] * len(indices)
        with pytest.raises(MechanismError):
            perturb_multi(setup, u, indices, alphas)

    # A healthy simultaneous perturbation still returns finite displacements.
    u2 = perturb_multi(setup, u, [0, 1], [0.5, 0.7])
    assert np.all(np.isfinite(u2))
