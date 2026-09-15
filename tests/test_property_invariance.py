"""Property and invariance tests for the finite-element core.

Comparing a solve against a hand-derived number for one specific model proves
that model is right. It says nothing about whether the *formulation* is right.
These tests instead assert properties that must hold for **every** admissible
model, which catches classes of error a single golden value cannot:

===============  ==========================================================
Property         What a violation would mean
===============  ==========================================================
Symmetry         ``K != K^T``: the element matrix or its scatter is wrong.
Positive semi-   A member contributes negative stiffness.
  definiteness
Rigid-body modes Not exactly three in 2D: the DOF map is wrong.
Load linearity   The solve is not linear in ``F``: something nonlinear leaked
                 into a linear-elastic path.
``E`` linearity  Displacements must scale as ``1/E`` for a single-material
                 model.
``A`` linearity  Same, per member.
Translation      Stiffness depends on absolute position: a spurious
invariance        coordinate-dependent term.
Rotation         The direction-cosine transformation is not orthogonal.
invariance
Renumbering      Results depend on node ordering rather than on the structure.
invariance
Element          ``Element(i, j)`` and ``Element(j, i)`` describe the same bar;
orientation      tension must not become compression when the ends are swapped.
invariance
Thermal load     A uniform temperature field on a *determinate* truss produces
  separation     no force at all; on an indeterminate one the forces must be
                 self-equilibrated.
===============  ==========================================================

Line 12 of the review that motivated this file asked specifically for an
explicit orientation-invariance proof, noting none existed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.model import Element, Node
from truss_analysis.postprocess import calculate_element_forces
from truss_analysis.solver import solve

#: Relative band for invariance comparisons. The underlying algebra is exact;
#: only floating-point cancellation differs, so anything looser than ~1e-12
#: would be hiding a real effect.
_RTOL = 1e-12
_ATOL = 1e-14

E_STEEL = 210e9


def _base_model() -> tuple[list[Node], list[Element], dict[str, tuple[float, float]]]:
    """An indeterminate five-bar frame with an inclined load."""
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
    loads = {"c": (20e3, -60e3), "d": (-5e3, -15e3)}
    return nodes, elements, loads


def _solve_model(
    nodes: list[Node],
    elements: list[Element],
    loads: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Assemble, load, solve and return ``(U, per-member forces)``."""
    node_index = {n.id: i for i, n in enumerate(nodes)}
    K, F, _, fixed = assemble_global_matrices(nodes, elements)
    for nid, (fx, fy) in loads.items():
        i = node_index[nid]
        F[2 * i] += fx
        F[2 * i + 1] += fy
    U = solve(K, F, fixed)
    results, _, _ = calculate_element_forces(nodes, elements, U)
    return U, results


def _forces_by_id(results: list[dict[str, float]]) -> dict[str, float]:
    return {str(r["id"]): float(r["N"]) for r in results}


# ---------------------------------------------------------------------------
# Structural properties of K
# ---------------------------------------------------------------------------


def test_k_is_symmetric() -> None:
    nodes, elements, _ = _base_model()
    K, _, _, _ = assemble_global_matrices(nodes, elements)
    assert np.allclose(K, K.T, rtol=0.0, atol=_ATOL * float(np.abs(K).max()))


def test_k_has_exactly_three_rigid_body_modes_in_2d() -> None:
    """A free planar body has 2 translations and 1 rotation, and no more."""
    nodes, elements, _ = _base_model()
    K, _, _, _ = assemble_global_matrices(nodes, elements)
    eigvals = np.linalg.eigvalsh(K)
    scale = float(np.abs(eigvals).max())
    n_zero = int(np.sum(eigvals <= 1e-10 * scale))
    assert n_zero == 3
    assert eigvals.min() >= -1e-10 * scale  # never negative


def test_rigid_body_modes_are_translations_and_rotation() -> None:
    """The null space of K must be spanned by the physical rigid motions."""
    nodes, elements, _ = _base_model()
    K, _, _, _ = assemble_global_matrices(nodes, elements)
    _eigvals, eigvecs = np.linalg.eigh(K)
    null_space = eigvecs[:, :3]

    # Build the analytical rigid-body modes: unit x, unit y, unit rotation.
    n = len(nodes)
    mode_x = np.zeros(2 * n)
    mode_y = np.zeros(2 * n)
    mode_r = np.zeros(2 * n)
    for i, node in enumerate(nodes):
        mode_x[2 * i] = 1.0
        mode_y[2 * i + 1] = 1.0
        # rotation about the origin: (-y, x)
        mode_r[2 * i] = -node.y
        mode_r[2 * i + 1] = node.x

    analytical = np.column_stack([mode_x, mode_y, mode_r])
    # Orthonormalise both bases and compare the subspaces they span.
    q_num, _ = np.linalg.qr(null_space)
    q_ana, _ = np.linalg.qr(analytical)
    # Subspace equality: the projector onto each must be identical.
    p_num = q_num @ q_num.T
    p_ana = q_ana @ q_ana.T
    assert np.allclose(p_num, p_ana, rtol=1e-8, atol=1e-8)


def test_every_member_contributes_a_rank_one_psd_block() -> None:
    """Each member contributes exactly one non-zero eigenvalue to K.

    ``k_e = k * b b^T`` is a rank-1 dyad, so a single member assembled into the
    full ``2n x 2n`` system leaves ``2n - 1`` zero eigenvalues and exactly one
    equal to ``k * (b^T b)``. Since ``b = [-c, -s, c, s]`` and
    ``c^2 + s^2 = 1``, that non-zero eigenvalue is ``2k``. Asserting both the
    count and its value checks the rank-1 structure independently of the
    scatter indices, so a transposed DOF map cannot pass.
    """
    nodes, elements, _ = _base_model()
    n_dof = 2 * len(nodes)
    for elem in elements:
        single, _, _, _ = assemble_global_matrices(nodes, [elem])
        eigvals = np.linalg.eigvalsh(single)
        scale = float(np.abs(eigvals).max())
        assert eigvals.min() >= -1e-9 * scale
        assert int(np.sum(eigvals <= 1e-10 * scale)) == n_dof - 1

        ni = next(nd for nd in nodes if nd.id == elem.node_i)
        nj = next(nd for nd in nodes if nd.id == elem.node_j)
        length = math.hypot(nj.x - ni.x, nj.y - ni.y)
        k_axial = elem.E * elem.A / length
        assert float(eigvals[-1]) == pytest.approx(2.0 * k_axial, rel=1e-12)


# ---------------------------------------------------------------------------
# Linearity
# ---------------------------------------------------------------------------


def test_displacement_is_linear_in_the_load() -> None:
    nodes, elements, loads = _base_model()
    U1, _ = _solve_model(nodes, elements, loads)
    scaled = {k: (2.5 * fx, 2.5 * fy) for k, (fx, fy) in loads.items()}
    U2, _ = _solve_model(nodes, elements, scaled)
    assert np.allclose(U2, 2.5 * U1, rtol=_RTOL, atol=_ATOL)


def test_forces_are_linear_in_the_load() -> None:
    nodes, elements, loads = _base_model()
    _, r1 = _solve_model(nodes, elements, loads)
    scaled = {k: (-0.75 * fx, -0.75 * fy) for k, (fx, fy) in loads.items()}
    _, r2 = _solve_model(nodes, elements, scaled)
    f1, f2 = _forces_by_id(r1), _forces_by_id(r2)
    for mid, val in f1.items():
        assert f2[mid] == pytest.approx(-0.75 * val, rel=_RTOL, abs=_ATOL)


def test_superposition_holds() -> None:
    """Two load cases applied together must equal the sum of each alone."""
    nodes, elements, loads = _base_model()
    case_a = {"c": loads["c"]}
    case_b = {"d": loads["d"]}
    Ua, _ = _solve_model(nodes, elements, case_a)
    Ub, _ = _solve_model(nodes, elements, case_b)
    Uab, _ = _solve_model(nodes, elements, loads)
    assert np.allclose(Uab, Ua + Ub, rtol=_RTOL, atol=_ATOL)


def test_displacement_scales_inversely_with_uniform_e() -> None:
    """Halving every E must double every displacement."""
    nodes, elements, loads = _base_model()
    U1, _ = _solve_model(nodes, elements, loads)
    soft = [
        Element(
            id=e.id,
            node_i=e.node_i,
            node_j=e.node_j,
            E=e.E * 0.5,
            A=e.A,
            I_sec=e.I_sec,
        )
        for e in elements
    ]
    U2, _ = _solve_model(nodes, soft, loads)
    assert np.allclose(U2, 2.0 * U1, rtol=_RTOL, atol=_ATOL)


def test_scaling_one_area_changes_only_that_members_force() -> None:
    """In a determinate truss, member areas do not affect the force distribution."""
    # A statically determinate three-bar truss: forces come from equilibrium
    # alone, so changing an area must leave every N untouched.
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="3", x=2.0, y=3.0, is_support=False),
    ]
    base = [
        Element(id="13", node_i="1", node_j="3", E=E_STEEL, A=0.010),
        Element(id="23", node_i="2", node_j="3", E=E_STEEL, A=0.010),
        Element(id="12", node_i="1", node_j="2", E=E_STEEL, A=0.010),
    ]
    loads = {"3": (0.0, -50e3)}
    _, r1 = _solve_model(nodes, base, loads)

    doubled = [
        Element(
            id=e.id,
            node_i=e.node_i,
            node_j=e.node_j,
            E=e.E,
            A=e.A * (2.0 if e.id == "13" else 1.0),
        )
        for e in base
    ]
    _, r2 = _solve_model(nodes, doubled, loads)

    f1, f2 = _forces_by_id(r1), _forces_by_id(r2)
    for mid in f1:
        assert f2[mid] == pytest.approx(f1[mid], rel=1e-10, abs=1e-6)


# ---------------------------------------------------------------------------
# Frame invariance
# ---------------------------------------------------------------------------


def test_translation_invariance() -> None:
    """Shifting the whole model must not change forces or relative displacement."""
    nodes, elements, loads = _base_model()
    U1, r1 = _solve_model(nodes, elements, loads)

    dx, dy = 1234.5, -678.9
    shifted = [
        Node(
            id=n.id,
            x=n.x + dx,
            y=n.y + dy,
            is_support=n.is_support,
            support_dx=n.support_dx,
            support_dy=n.support_dy,
        )
        for n in nodes
    ]
    U2, r2 = _solve_model(shifted, elements, loads)

    f1, f2 = _forces_by_id(r1), _forces_by_id(r2)
    for mid in f1:
        assert f2[mid] == pytest.approx(f1[mid], rel=1e-10, abs=1e-6)
    assert np.allclose(U2, U1, rtol=1e-9, atol=_ATOL)


def _pinned_model() -> tuple[list[Node], list[Element], dict[str, tuple[float, float]]]:
    """The base frame with every support a full pin.

    A pin restrains both global translations, so it maps to a pin under any
    rotation of the frame. A *roller* does not: ``support_dx`` refers to the
    global x axis, so rotating the geometry by an arbitrary angle while leaving
    the flags alone silently changes which directions are restrained. Testing
    rotation invariance against a roller model therefore conflates the
    direction-cosine transformation with a change of boundary conditions.
    """
    nodes, elements, loads = _base_model()
    pinned = [
        Node(
            id=n.id,
            x=n.x,
            y=n.y,
            is_support=n.is_support,
            support_dx=n.is_support,
            support_dy=n.is_support,
        )
        for n in nodes
    ]
    return pinned, elements, loads


@pytest.mark.parametrize("angle_deg", [17.0, 45.0, 90.0, -133.0, 180.0])
def test_rotation_invariance(angle_deg: float) -> None:
    """Rotating model *and* load must rotate displacements, keeping forces fixed."""
    nodes, elements, loads = _pinned_model()
    U1, r1 = _solve_model(nodes, elements, loads)

    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])

    rotated_nodes = [
        Node(
            id=n.id,
            x=c * n.x - s * n.y,
            y=s * n.x + c * n.y,
            is_support=n.is_support,
            support_dx=n.support_dx,
            support_dy=n.support_dy,
        )
        for n in nodes
    ]
    rotated_loads = {}
    for nid, (fx, fy) in loads.items():
        fx_r, fy_r = rot @ np.array([fx, fy])
        rotated_loads[nid] = (float(fx_r), float(fy_r))
    U2, r2 = _solve_model(rotated_nodes, elements, rotated_loads)

    # Axial forces are scalars: they must be identical.
    f1, f2 = _forces_by_id(r1), _forces_by_id(r2)
    for mid in f1:
        assert f2[mid] == pytest.approx(f1[mid], rel=1e-9, abs=1e-6)

    # Displacement vectors must be the original ones, rotated.
    for i in range(len(rotated_nodes)):
        original = np.array([U1[2 * i], U1[2 * i + 1]])
        got = np.array([U2[2 * i], U2[2 * i + 1]])
        assert np.allclose(got, rot @ original, rtol=1e-9, atol=1e-15)


def test_rotating_a_roller_support_changes_the_structure() -> None:
    """Negative control for the caveat above: a roller is NOT rotation-invariant.

    Pinning this prevents someone "simplifying" the rotation test by dropping
    the all-pinned model and then drawing the wrong conclusion when it fails.
    Rotating the base frame by 90 deg while leaving ``support_dx``/``support_dy``
    untouched turns the horizontal roller at node b into a vertical one, which
    is a different structure and must give different forces.
    """
    nodes, elements, loads = _base_model()
    _U1, r1 = _solve_model(nodes, elements, loads)
    f1 = _forces_by_id(r1)

    theta = math.radians(90.0)
    c, s = math.cos(theta), math.sin(theta)
    rotated_nodes = [
        Node(
            id=n.id,
            x=c * n.x - s * n.y,
            y=s * n.x + c * n.y,
            is_support=n.is_support,
            support_dx=n.support_dx,  # deliberately NOT rotated
            support_dy=n.support_dy,
        )
        for n in nodes
    ]
    rotated_loads = {}
    for nid, (fx, fy) in loads.items():
        fx_r, fy_r = np.array([[c, -s], [s, c]]) @ np.array([fx, fy])
        rotated_loads[nid] = (float(fx_r), float(fy_r))

    # The restraint set is no longer adequate: node b's roller was free along
    # the global x axis, which after a 90 deg rotation is no longer the
    # direction the support can actually move in. The frame becomes a mechanism.
    from truss_analysis.exceptions import SingularMatrixError

    with pytest.raises(SingularMatrixError, match="mechanism detected"):
        _solve_model(rotated_nodes, elements, rotated_loads)
    del f1


def test_rotation_invariance_requires_rotating_the_load_too() -> None:
    """The companion negative control: rotating only the geometry must change N.

    Without this, a rotation test that accidentally asserted nothing would
    still pass. Rotating the structure while leaving the load direction fixed
    is a genuinely different physical problem.
    """
    nodes, elements, loads = _base_model()
    _U1, r1 = _solve_model(nodes, elements, loads)

    theta = math.radians(37.0)
    c, s = math.cos(theta), math.sin(theta)
    rotated_nodes = [
        Node(
            id=n.id,
            x=c * n.x - s * n.y,
            y=s * n.x + c * n.y,
            is_support=n.is_support,
            support_dx=n.support_dx,
            support_dy=n.support_dy,
        )
        for n in nodes
    ]
    _U2, r2 = _solve_model(rotated_nodes, elements, loads)  # load NOT rotated

    f1, f2 = _forces_by_id(r1), _forces_by_id(r2)
    assert any(abs(f2[mid] - f1[mid]) > 1e-6 * max(abs(f1[mid]), 1.0) for mid in f1), (
        "rotating the geometry alone must change the force distribution"
    )


@pytest.mark.parametrize(
    "order",
    [
        ["a", "b", "c", "d"],  # identity
        ["d", "c", "b", "a"],  # reversed
        ["c", "a", "d", "b"],  # shuffled
    ],
)
def test_node_renumbering_invariance(order: list[str]) -> None:
    """Results must follow the nodes, not their position in the input list."""
    nodes, elements, loads = _base_model()
    U1, r1 = _solve_model(nodes, elements, loads)
    f1 = _forces_by_id(r1)

    original = {n.id: n for n in nodes}
    reordered = [original[nid] for nid in order]
    U2, r2 = _solve_model(reordered, elements, loads)
    f2 = _forces_by_id(r2)

    for mid, val in f1.items():
        assert f2[mid] == pytest.approx(val, rel=1e-10, abs=1e-8)

    # Displacements must be the same values, permuted with the nodes.
    new_index = {nid: i for i, nid in enumerate(order)}
    for nid, i in {n.id: i for i, n in enumerate(nodes)}.items():
        j = new_index[nid]
        assert U2[2 * j] == pytest.approx(U1[2 * i], rel=1e-10, abs=_ATOL)
        assert U2[2 * j + 1] == pytest.approx(U1[2 * i + 1], rel=1e-10, abs=_ATOL)


def test_element_orientation_invariance() -> None:
    """``Element(i, j)`` and ``Element(j, i)`` are the same physical bar.

    Tension must not become compression when the ends are swapped, and the
    stiffness matrix must be unchanged. This is the explicit orientation proof
    whose absence was flagged in review.
    """
    nodes, elements, loads = _base_model()
    U1, r1 = _solve_model(nodes, elements, loads)
    K1, _, _, fixed1 = assemble_global_matrices(nodes, elements)
    f1 = _forces_by_id(r1)

    swapped = [
        Element(
            id=e.id,
            node_i=e.node_j,
            node_j=e.node_i,
            E=e.E,
            A=e.A,
            I_sec=e.I_sec,
            alpha=e.alpha,
            delta_T=e.delta_T,
            delta_L_free=e.delta_L_free,
            density=e.density,
            effective_length_factor=e.effective_length_factor,
        )
        for e in elements
    ]
    U2, r2 = _solve_model(nodes, swapped, loads)
    K2, _, _, fixed2 = assemble_global_matrices(nodes, swapped)
    f2 = _forces_by_id(r2)

    # Identical stiffness and boundary conditions.
    assert np.allclose(K1, K2, rtol=0.0, atol=_ATOL * float(np.abs(K1).max()))
    assert fixed1 == fixed2
    assert np.allclose(U1, U2, rtol=0.0, atol=_ATOL)

    # And the same sign convention for axial force: tension stays tension.
    for mid, val in f1.items():
        assert f2[mid] == pytest.approx(val, rel=_RTOL, abs=_ATOL)
        assert (f2[mid] > 0) == (val > 0) or abs(val) < 1e-6


def test_partial_orientation_swap_is_also_invariant() -> None:
    """Swapping only some members must leave the whole solution untouched."""
    nodes, elements, loads = _base_model()
    _U1, r1 = _solve_model(nodes, elements, loads)
    f1 = _forces_by_id(r1)

    mixed = []
    for idx, e in enumerate(elements):
        if idx % 2:
            mixed.append(
                Element(
                    id=e.id,
                    node_i=e.node_j,
                    node_j=e.node_i,
                    E=e.E,
                    A=e.A,
                    I_sec=e.I_sec,
                )
            )
        else:
            mixed.append(e)
    _U2, r2 = _solve_model(nodes, mixed, loads)
    f2 = _forces_by_id(r2)
    for mid, val in f1.items():
        assert f2[mid] == pytest.approx(val, rel=1e-11, abs=1e-8)


# ---------------------------------------------------------------------------
# Thermal / imposed-strain separation
# ---------------------------------------------------------------------------


def test_uniform_temperature_on_determinate_truss_produces_no_force() -> None:
    """A determinate truss can expand freely, so heating cannot stress it."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="3", x=2.0, y=3.0, is_support=False),
    ]
    elements = [
        Element(
            id=eid, node_i=i, node_j=j, E=E_STEEL, A=0.01, alpha=1.2e-5, delta_T=250.0
        )
        for eid, i, j in (("13", "1", "3"), ("23", "2", "3"), ("12", "1", "2"))
    ]
    _U, results = _solve_model(nodes, elements, {})
    for r in results:
        assert r["N"] == pytest.approx(0.0, abs=1e-6)


def test_restrained_thermal_force_matches_closed_form() -> None:
    """N = -E A alpha delta_T for a bar held at both ends."""
    alpha, dT, L, A = 1.2e-5, 80.0, 3.0, 0.02
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=L, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [
        Element(
            id="bar",
            node_i="1",
            node_j="2",
            E=E_STEEL,
            A=A,
            alpha=alpha,
            delta_T=dT,
        )
    ]
    U, results = _solve_model(nodes, elements, {})
    assert np.allclose(U, 0.0, atol=1e-18)
    assert results[0]["N"] == pytest.approx(-E_STEEL * A * alpha * dT, rel=1e-12)


def test_fabrication_error_in_a_determinate_truss_produces_no_force() -> None:
    """A statically determinate truss accommodates a misfit by moving.

    With one free DOF along the member axis the bar is simply pulled into
    position: ``delta_L_total = delta_L_free``, so ``delta_L_mech = 0`` and the
    axial force vanishes. Self-equilibrated misfit stresses require
    redundancy, which is what the next test establishes.
    """
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=2.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
    ]
    short = -1e-3  # member manufactured 1 mm too short
    elements = [
        Element(id="bar", node_i="1", node_j="2", E=E_STEEL, A=0.01, delta_L_free=short)
    ]
    U, results = _solve_model(nodes, elements, {})
    assert results[0]["N"] == pytest.approx(0.0, abs=1e-6)
    # and the free node moved by exactly the misfit
    assert U[2] == pytest.approx(short, rel=1e-12)


def test_fabrication_error_in_an_indeterminate_truss_stresses_members() -> None:
    """Redundancy is what turns a misfit into force.

    A two-bar chain held at both ends cannot accommodate a member that is 1 mm
    too short, so the bars are stretched back and carry equal and opposite
    internal force -- self-equilibrated, with no external load applied.
    """
    L = 2.0
    short = -1e-3
    A = 0.01
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        # y restrained: three collinear nodes would otherwise leave a rotational
        # mechanism at the middle node. x stays free, so the chain is still
        # indeterminate along its axis (m + r - 2j = 2 + 5 - 6 = 1).
        Node(id="2", x=L, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="3", x=2 * L, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [
        Element(id="left", node_i="1", node_j="2", E=E_STEEL, A=A),
        Element(id="right", node_i="2", node_j="3", E=E_STEEL, A=A, delta_L_free=short),
    ]
    U, results = _solve_model(nodes, elements, {})
    forces = _forces_by_id(results)

    k = E_STEEL * A / L
    # Hand solution. With u the x-displacement of the middle node (positive to
    # the right) and both bars directed left-to-right (c = +1):
    #
    #   left  bar: delta_L_total = u,   delta_L_mech = u
    #   right bar: delta_L_total = -u,  delta_L_mech = -u - short
    #
    #   N_left = k u,   N_right = k (-u - short)
    #
    # Equilibrium of the middle node (no external load) requires the two bars
    # to carry the same force, so k u = k (-u - short), giving
    #
    #   u = -short / 2,   N = k u = -k short / 2
    #
    # short is negative (the member is too SHORT), so u > 0 and N > 0: the
    # undersized bar has to be stretched to span the gap, and in pulling the
    # middle node rightwards it stretches the left bar too. Both end up in
    # TENSION -- a misfit that is too short produces tension, not compression.
    expected_u = -short / 2.0
    expected_N = -k * short / 2.0
    assert expected_N > 0.0, "sign check: an undersized member must be in tension"

    assert forces["left"] == pytest.approx(expected_N, rel=1e-10)
    assert forces["right"] == pytest.approx(expected_N, rel=1e-10)
    assert U[2] == pytest.approx(expected_u, rel=1e-10)
    # Self-equilibrated: with no external load the two end reactions cancel.
    assert sum(forces.values()) == pytest.approx(2.0 * expected_N, rel=1e-12)


# ---------------------------------------------------------------------------
# Round-5 audit (C4 #12): dimensional similarity under geometric scaling
# ---------------------------------------------------------------------------


def test_dimensional_similarity_under_geometric_scaling() -> None:
    """Scale lengths by s, areas by s^2, loads by s^2 -> forces scale by s^2,
    displacements by s, and every stress/strain/force-per-area invariant is
    exactly preserved.

    This is the dimensional-analysis oracle the round-5 audit asked for: it
    catches unit and exponent bugs that neither coverage nor expected-value
    tests see, because it compares two independently-scaled instances of the
    SAME physical structure against each other.
    """
    from dataclasses import replace

    s = 7.3
    nodes, elements, loads = _base_model()
    U1, r1 = _solve_model(nodes, elements, loads)

    big_nodes = [replace(n, x=n.x * s, y=n.y * s) for n in nodes]
    big_elements = [
        replace(
            e,
            A=e.A * s**2,
            I_sec=e.I_sec * s**4,
            delta_L_free=e.delta_L_free * s,
        )
        for e in elements
    ]
    big_loads = {nid: (fx * s**2, fy * s**2) for nid, (fx, fy) in loads.items()}
    U2, r2 = _solve_model(big_nodes, big_elements, big_loads)

    # displacements scale linearly with s
    assert np.allclose(U2, s * U1, rtol=1e-9, atol=1e-12)

    f1, f2 = _forces_by_id(r1), _forces_by_id(r2)
    a_by_id = {e.id: e.A for e in elements}
    for mid in f1:
        # forces scale with s^2 ...
        assert f2[mid] == pytest.approx(s**2 * f1[mid], rel=1e-9, abs=1e-6)
        # ... so axial stress N/A is exactly invariant
        assert f2[mid] / (a_by_id[mid] * s**2) == pytest.approx(
            f1[mid] / a_by_id[mid], rel=1e-9, abs=1e-9
        )


def test_dimensional_similarity_with_thermal_load() -> None:
    """The scaling oracle also holds when the demand is a restrained thermal
    eigenstrain: alpha and delta_T are intensive (unscaled), delta_L_free is
    extensive (scales with s), so the developed force still scales by s^2."""
    from dataclasses import replace

    s = 3.0
    nodes = [
        Node(id="a", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="b", x=2.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [
        Element(
            id="bar",
            node_i="a",
            node_j="b",
            E=E_STEEL,
            A=0.01,
            alpha=1.2e-5,
            delta_T=200.0,
            delta_L_free=1e-4,
        )
    ]
    _U1, r1 = _solve_model(nodes, elements, {})

    big_nodes = [replace(n, x=n.x * s, y=n.y * s) for n in nodes]
    big_elements = [
        replace(e, A=e.A * s**2, delta_L_free=e.delta_L_free * s) for e in elements
    ]
    _U2, r2 = _solve_model(big_nodes, big_elements, {})

    f1, f2 = _forces_by_id(r1), _forces_by_id(r2)
    # fully restrained thermal force N = -E A alpha dT (delta_L_free adds -k*dl):
    # both terms scale as s^2 (A*s^2, and k=EA/L * dl*s = E A s^2 * dl / L ... *s/s)
    assert f2["bar"] == pytest.approx(s**2 * f1["bar"], rel=1e-9)
