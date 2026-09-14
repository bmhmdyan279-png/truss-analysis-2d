"""Core-hardening tests: solver guards, energy-check tolerance, DDM FD."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.exceptions import (
    EnergyValidationError,
    IllConditionedWarning,
    SingularMatrixError,
)
from truss_analysis.model import Element, Node
from truss_analysis.reliability_adapter import NodalLoad
from truss_analysis.sensitivity import IndependentValidator
from truss_analysis.solver import check_energy, solve


def _two_node():
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=3.0, y=0.0, is_support=True, support_dy=True),
    ]
    elements = [Element(id="t", node_i="1", node_j="2", E=210e9, A=0.01, I_sec=1e-6)]
    return nodes, elements


def _quad_mechanism():
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=1.0, y=0.0, is_support=True, support_dy=True),
        Node(id="3", x=1.0, y=1.0),
        Node(id="4", x=0.0, y=1.0),
    ]
    elements = [
        Element(id="1", node_i="1", node_j="2", E=210e9, A=0.01),
        Element(id="2", node_i="2", node_j="3", E=210e9, A=0.01),
        Element(id="3", node_i="3", node_j="4", E=210e9, A=0.01),
        Element(id="4", node_i="4", node_j="1", E=210e9, A=0.01),
    ]
    return nodes, elements


def test_mechanism_detected_before_solve_d012() -> None:
    nodes, elements = _quad_mechanism()
    K, F, _, fixed = assemble_global_matrices(nodes, elements)
    F[4] = -1.0
    with pytest.raises(SingularMatrixError, match="mechanism detected before solve"):
        solve(K, F, fixed)


def test_ill_conditioned_warning_above_threshold() -> None:
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=True, support_dy=True),
        Node(id="3", x=2.0, y=3.0),
    ]
    elements = [
        Element(id="1", node_i="1", node_j="2", E=210e9, A=0.01),
        Element(id="2", node_i="1", node_j="3", E=210e9, A=0.01),
        # soft member: cond(K_ff) ~ 2e12 > 1e12 while rank stays full
        Element(id="3", node_i="2", node_j="3", E=210e9 * 5e-13, A=0.01),
    ]
    K, F, _, fixed = assemble_global_matrices(nodes, elements)
    F[4] = 10e3
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        solve(K, F, fixed)
    assert any(issubclass(w.category, IllConditionedWarning) for w in rec)
    # and can be silenced explicitly
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        solve(K, F, fixed, check_condition=False)
    assert not any(issubclass(w.category, IllConditionedWarning) for w in rec2)


def test_check_energy_default_tol_is_tight() -> None:
    nodes, elements = _two_node()
    K, F, _, fixed = assemble_global_matrices(nodes, elements)
    F[2] = 50e3
    U = solve(K, F, fixed)
    # exact elastic balance closes far below 1e-8 relative
    k = elements[0].E * elements[0].A / 3.0
    delta = U[2]
    strain = 0.5 * k * delta**2
    work = 0.0
    assert check_energy(U, F, strain, work) is True
    # a 1% energy error must now FAIL by default (legacy tol was 1%)
    with pytest.raises(EnergyValidationError, match="Energy balance failed"):
        check_energy(U, F, strain * 1.01, work)


def test_ddm_matches_central_finite_difference() -> None:
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0),
        Node(id="3", x=8.0, y=0.0, is_support=True, support_dy=True),
        Node(id="4", x=4.0, y=3.0),
    ]
    elements = [
        Element(id="1", node_i="1", node_j="2", E=210e9, A=0.01),
        Element(id="2", node_i="2", node_j="3", E=210e9, A=0.01),
        Element(id="3", node_i="2", node_j="4", E=210e9, A=0.01),
        Element(id="4", node_i="4", node_j="1", E=210e9, A=0.01),
        Element(id="5", node_i="4", node_j="3", E=210e9, A=0.01),
    ]
    loads = [NodalLoad(node_id="2", fx=0.0, fy=-50e3)]
    val = IndependentValidator(nodes, elements, loads)
    results = {
        r.element_id if hasattr(r, "element_id") else r.member_id: r
        for r in val.compute_all()
    }

    def u_max_magnitude(area_scale: float, member_idx: int) -> float:
        els = [
            Element(
                id=e.id,
                node_i=e.node_i,
                node_j=e.node_j,
                E=e.E,
                A=e.A * (area_scale if i == member_idx else 1.0),
            )
            for i, e in enumerate(elements)
        ]
        K, F, _, fixed = assemble_global_matrices(nodes, els)
        for ld in loads:
            idx = {n.id: i for i, n in enumerate(nodes)}[ld.node_id]
            F[2 * idx] += ld.fx
            F[2 * idx + 1] += ld.fy
        U = solve(K, F, fixed)
        return float(np.max(np.hypot(U[0::2], U[1::2])))

    h = 1e-6
    for i, e in enumerate(elements):
        fd = (u_max_magnitude(1.0 + h, i) - u_max_magnitude(1.0 - h, i)) / (2 * h * e.A)
        key = next(k for k in results if str(k) == e.id)
        ana = results[key].ddm_sensitivity
        assert ana is not None
        assert abs(ana - fd) < 1e-6 * max(1.0, abs(fd))


# ---------------------------------------------------------------------
# Sparse path: the library's own error contract (audit round 2, finding 4)
# ---------------------------------------------------------------------


def _singular_sparse_system():
    """A sparse K_ff that is exactly singular (one free DOF, zero stiffness)."""
    import scipy.sparse as sp

    # Node 1 fully fixed, node 2 free but connected by a zero-stiffness
    # member: K_ff = [[0]] -> SuperLU raises RuntimeError internally.
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=1.0, y=0.0),
    ]
    K = sp.csr_matrix((2, 2))  # identically zero
    F = np.array([0.0, 1.0])
    return nodes, K, F


def test_sparse_singular_unscreened_raises_library_error() -> None:
    """check_condition=False: SuperLU failure must surface as SingularMatrixError."""
    import scipy.sparse as sp

    _nodes, K, F = _singular_sparse_system()
    with pytest.raises(SingularMatrixError):
        solve(K, F, [], check_condition=False)
    assert sp.issparse(K)


def test_sparse_singular_after_screen_raises_library_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even when the SVD screen passed, a SuperLU failure is translated.

    An earlier revision re-raised the raw SciPy RuntimeError on the screened
    path, leaking a foreign exception type through the library's documented
    error contract.  Force the exact branch: stub the screen to report a
    healthy matrix, then hand SuperLU a singular one.
    """
    import truss_analysis.solver as solver_mod
    from truss_analysis.numerics import NumericalStatus

    monkeypatch.setattr(
        solver_mod,
        "singular_value_screen",
        lambda _k, _t: (1, 1.0, 1.0, NumericalStatus.STABLE),
    )
    _nodes, K, F = _singular_sparse_system()
    with pytest.raises(SingularMatrixError) as excinfo:
        solve(K, F, [], check_condition=True)
    assert "SuperLU" in str(excinfo.value)
    assert "SVD rank screen" in str(excinfo.value)


def test_run_exposes_check_condition_flag(tmp_path) -> None:
    """run(check_condition=False) skips the O(n^3) screen, same answer."""
    import shutil
    from pathlib import Path

    from truss_analysis.main import run

    src = Path(__file__).resolve().parents[1] / "examples" / "example1.json"
    model = tmp_path / "model.json"
    shutil.copy(src, model)

    with_screen = run(model, quiet=True, check_condition=True)
    without_screen = run(model, quiet=True, check_condition=False)
    assert with_screen.status == without_screen.status == "SUCCESS"
    for nid, rec in with_screen.displacements.items():
        assert rec["ux"] == pytest.approx(
            without_screen.displacements[nid]["ux"], rel=1e-12
        )
        assert rec["uy"] == pytest.approx(
            without_screen.displacements[nid]["uy"], rel=1e-12
        )
