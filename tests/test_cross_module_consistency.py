"""Cross-module consistency: one formulation, several numerical backends.

The library solves the same physics through more than one code path -- the
assembler plus :func:`truss_analysis.solver.solve`, the rank-1 criticality
engine, and the independent DDM/energy validator. Duplication is only
defensible if the paths are *provably* equal, so these tests compare them
against each other and against closed forms rather than trusting any single
implementation.

They cover the three identities the whole library rests on:

``K = B^T diag(k) B``
    the assembled matrix equals the compatibility-form assembly;

``K = sum_e k_e b_e b_e^T``
    and equals the sum of rank-1 member dyads;

``U_sensitivity.strain_energy == postprocess`` mechanical strain energy
    the validator and the post-processor agree member by member, including
    when imposed (thermal / fabrication) strain is present.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from truss_analysis.assembly import (
    assemble_global_matrices,
    assemble_global_matrices_sparse,
    member_geometry,
)
from truss_analysis.model import Element, Node
from truss_analysis.postprocess import calculate_element_forces
from truss_analysis.reliability_adapter import NodalLoad
from truss_analysis.sensitivity import IndependentValidator
from truss_analysis.solver import solve

#: Relative slack for identities that sum in a different order. The matrix
#: entries here are ~1e9, so a 1e-10 relative band is ~1e-1 -- still six
#: orders of magnitude tighter than any engineering tolerance, while allowing
#: for floating-point cancellation between summation orders.
_RTOL = 1e-10


def _pratt_panel() -> tuple[list[Node], list[Element], list[NodalLoad]]:
    """A small statically indeterminate truss with a diagonal."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="3", x=4.0, y=3.0, is_support=False),
        Node(id="4", x=0.0, y=3.0, is_support=False),
    ]
    elements = [
        Element(id="b1", node_i="1", node_j="2", E=210e9, A=0.02),
        Element(id="v1", node_i="2", node_j="3", E=210e9, A=0.01),
        Element(id="t1", node_i="3", node_j="4", E=210e9, A=0.015),
        Element(id="v2", node_i="4", node_j="1", E=210e9, A=0.01),
        Element(id="d1", node_i="1", node_j="3", E=210e9, A=0.008),
    ]
    loads = [NodalLoad(node_id="3", fx=20e3, fy=-60e3)]
    return nodes, elements, loads


def _thermal_model() -> tuple[list[Node], list[Element], list[NodalLoad]]:
    """Indeterminate frame with imposed strain on two members."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=3.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="3", x=1.5, y=2.0, is_support=False),
    ]
    elements = [
        Element(
            id="h",
            node_i="1",
            node_j="2",
            E=200e9,
            A=0.01,
            alpha=1.2e-5,
            delta_T=120.0,
        ),
        Element(id="l", node_i="1", node_j="3", E=200e9, A=0.012),
        Element(
            id="r",
            node_i="2",
            node_j="3",
            E=200e9,
            A=0.009,
            delta_L_free=-2e-4,
        ),
    ]
    loads = [NodalLoad(node_id="3", fx=0.0, fy=-25e3)]
    return nodes, elements, loads


def test_k_equals_btdiagkb() -> None:
    """The assembler reproduces the compatibility-form identity exactly."""
    nodes, elements, _ = _pratt_panel()
    K, _, _, _ = assemble_global_matrices(nodes, elements)
    geom = member_geometry(nodes, elements)
    B = geom.compatibility()

    K_ref = B.T @ np.diag(geom.k_axial) @ B
    scale = float(np.abs(K).max())
    assert np.abs(K - K_ref).max() <= _RTOL * scale


def test_k_equals_sum_of_rank_one_dyads() -> None:
    """K is the sum of member dyads k_e * b_e b_e^T over scattered DOFs."""
    nodes, elements, _ = _pratt_panel()
    K, _, _, _ = assemble_global_matrices(nodes, elements)
    geom = member_geometry(nodes, elements)

    K_ref = np.zeros_like(K)
    for e in range(geom.n_members):
        b = np.array([-geom.cosines[e], -geom.sines[e], geom.cosines[e], geom.sines[e]])
        K_ref[np.ix_(geom.dofs[e], geom.dofs[e])] += geom.k_axial[e] * np.outer(b, b)

    scale = float(np.abs(K).max())
    assert np.abs(K - K_ref).max() <= _RTOL * scale


def test_k_is_symmetric_and_positive_semidefinite() -> None:
    """Before boundary conditions K must be symmetric PSD with rigid-body modes."""
    nodes, elements, _ = _pratt_panel()
    K, _, _, _ = assemble_global_matrices(nodes, elements)

    assert np.allclose(K, K.T, rtol=0.0, atol=1e-6 * np.abs(K).max())
    eigvals = np.linalg.eigvalsh(K)
    # PSD: no negative eigenvalue beyond round-off.
    assert eigvals.min() >= -1e-6 * float(np.abs(eigvals).max())
    # A free 2D body has exactly three rigid-body modes (2 translations,
    # 1 rotation), so K on the unconstrained model must have rank 2n - 3.
    n = len(nodes)
    rank = int(np.sum(eigvals > 1e-10 * float(eigvals.max())))
    assert rank == 2 * n - 3


def test_sparse_assembly_matches_dense() -> None:
    """Both storage layouts must produce bit-comparable systems."""
    for builder in (_pratt_panel, _thermal_model):
        nodes, elements, _ = builder()
        K_d, F_d, Fm_d, fixed_d = assemble_global_matrices(nodes, elements)
        K_s, F_s, Fm_s, fixed_s = assemble_global_matrices_sparse(nodes, elements)

        assert sp.issparse(K_s)
        assert np.allclose(K_d, K_s.toarray(), rtol=_RTOL, atol=0.0)
        assert np.array_equal(F_d, F_s)
        assert np.array_equal(Fm_d, Fm_s)
        assert fixed_d == fixed_s


def test_sparse_flag_matches_explicit_sparse_call() -> None:
    """``assemble_global_matrices(sparse=True)`` delegates to the sparse path."""
    nodes, elements, _ = _pratt_panel()
    K_flag, *_ = assemble_global_matrices(nodes, elements, sparse=True)
    K_direct, *_ = assemble_global_matrices_sparse(nodes, elements)
    assert np.allclose(K_flag.toarray(), K_direct.toarray(), rtol=0.0, atol=0.0)


def test_sparse_and_dense_solvers_agree() -> None:
    """The displacement field must not depend on the storage layout."""
    nodes, elements, loads = _pratt_panel()
    K_d, F_d, _, fixed = assemble_global_matrices(nodes, elements)
    K_s, F_s, _, _ = assemble_global_matrices_sparse(nodes, elements)
    for load in loads:
        i = next(k for k, nd in enumerate(nodes) if nd.id == load.node_id)
        F_d[2 * i] += load.fx
        F_d[2 * i + 1] += load.fy
        F_s[2 * i] += load.fx
        F_s[2 * i + 1] += load.fy

    U_d = solve(K_d, F_d, fixed)
    U_s = solve(K_s, F_s, fixed)
    scale = float(np.abs(U_d).max())
    assert np.abs(U_d - U_s).max() <= 1e-9 * scale


def test_validator_energy_matches_postprocess_mechanical() -> None:
    """The validator and the post-processor must agree member by member.

    This is the check that catches the imposed-strain error: with ``delta_T``
    or ``delta_L_free`` present, ``0.5 u_e^T k_e u_e`` is the energy of the
    *total* elongation, not of the stress-producing mechanical part.
    """
    for builder in (_pratt_panel, _thermal_model):
        nodes, elements, loads = builder()
        validator = IndependentValidator(nodes, elements, loads)
        sens = {r.member_id: r for r in validator.compute_all()}

        U, _fixed = validator.compute_baseline()
        results, _, _ = calculate_element_forces(nodes, elements, U)
        post = {str(r["id"]): r for r in results}

        for mid, res in sens.items():
            assert res.strain_energy == pytest.approx(
                _member_strain_energy(post[mid], elements, nodes), rel=1e-9, abs=1e-12
            ), f"member {mid}: validator/postprocess strain energy disagree"


def _member_strain_energy(
    post_entry: dict[str, float],
    elements: list[Element],
    nodes: list[Node],
) -> float:
    """Recompute ``0.5 k delta_L_mech^2`` from the post-processor's own output."""
    elem = next(e for e in elements if str(e.id) == str(post_entry["id"]))
    ni = next(k for k, nd in enumerate(nodes) if nd.id == elem.node_i)
    nj = next(k for k, nd in enumerate(nodes) if nd.id == elem.node_j)
    L = float(np.hypot(nodes[nj].x - nodes[ni].x, nodes[nj].y - nodes[ni].y))
    k = elem.E * elem.A / L
    return 0.5 * k * float(post_entry["delta_L_mech"]) ** 2


def test_total_and_mechanical_energy_differ_only_by_imposed_strain() -> None:
    """Free expansion stores no energy; restrained expansion stores plenty."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=2.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="3", x=2.0, y=2.0, is_support=True, support_dx=True, support_dy=True),
    ]
    alpha, dT = 1.2e-5, 100.0
    elements = [
        Element(
            id="free",
            node_i="1",
            node_j="2",
            E=200e9,
            A=0.01,
            alpha=alpha,
            delta_T=dT,
        ),
        Element(
            id="braced",
            node_i="2",
            node_j="3",
            E=200e9,
            A=0.01,
            alpha=alpha,
            delta_T=dT,
        ),
    ]
    validator = IndependentValidator(nodes, elements, [])
    sens = {r.member_id: r for r in validator.compute_all()}

    # The horizontal member can expand: total elongation is non-zero, stored
    # mechanical energy is (essentially) zero.
    free = sens["free"]
    assert free.strain_energy_total > 0.0
    assert free.strain_energy == pytest.approx(0.0, abs=1e-9)

    # The vertical member is held at both ends: it cannot expand, so its total
    # elongation is zero while the stored energy is large.
    braced = sens["braced"]
    assert braced.strain_energy_total == pytest.approx(0.0, abs=1e-9)
    assert braced.strain_energy > 0.0


def test_fixed_dofs_single_definition() -> None:
    """Assembler and criticality engine must agree on the constrained DOFs."""
    from truss_analysis.criticality.engine import free_dof_indices
    from truss_analysis.model import fixed_dof_indices

    nodes, elements, _ = _pratt_panel()
    _, _, _, fixed = assemble_global_matrices(nodes, elements)
    free = set(free_dof_indices(nodes))

    assert fixed == fixed_dof_indices(nodes)
    assert free == set(range(2 * len(nodes))) - set(fixed)
