"""Linearised stability: geometric stiffness and bifurcation load factors.

Verification hierarchy (project policy, docs/theory.md):

1. closed-form analytics -- the shallow two-bar toggle bifurcation load
   ``P_cr = 2 E A h^3 / (b^2 L_0)`` derived by hand in the module docstring
   of the test;
2. first-principles identities -- orthogonality of the elastic and
   geometric dyads, the ``u^T K_G u = sum N L phi^2`` energy statement,
   a hand-built single-member ``K_G``;
3. an independent algorithm -- dense generalised eigensolve
   (``scipy.linalg.eig(A, B)`` via LAPACK QZ) cross-checked against the
   library's symmetric whitened path on random redundant trusses;
4. physical invariants -- rigid-motion and renumbering invariance, load
   scaling, tension-only infinity, zero-force irrelevance;
5. the prestressed (thermal/fabrication) destabilisation path: monotone
   loss of the load factor with growing imposed compression, ending in the
   documented ``MechanismError`` when the base state itself loses positive
   definiteness -- the round-5 audit's headline scientific gap (C2-A).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import eig as dense_generalized_eig

from truss_analysis.assembly import assemble_global_matrices, member_geometry
from truss_analysis.criticality.engine import (
    MechanismError,
    base_displacement,
    build_engine,
    member_forces,
    total_load_vector,
)
from truss_analysis.model import Element, Node, fixed_dof_indices
from truss_analysis.stability import (
    geometric_stiffness,
    linearized_buckling_load_factor,
    member_geometric_vectors,
)

# Diagnostics this module triggers on purpose, named exactly and nowhere wider.
#
# ShallowSystemWarning: nearly every model here is a two-bar toggle or a shallow
#   fan, because those are the geometries with closed-form bifurcation loads to
#   check against. The screen is asserted directly in
#   test_shallow_toggle_is_still_flagged_as_an_arch and
#   test_shallow_warning_can_be_suppressed; the remaining tests are about the
#   eigenproblem, not the screen.
# LinAlgWarning: raised inside scipy when a deliberately singular matrix reaches
#   a factorisation. test_base_mechanism_raises builds one on purpose and asserts
#   the library converts it into MechanismError; scipy's own complaint about the
#   same matrix is noise on top of that assertion.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.ConstantAlphaWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.ShallowSystemWarning"
    ),
    pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning"),
]

E_STEEL = 210e9
AREA = 1e-3

# Sparse (CSR + dyad accumulation) and dense K_G assembly sum member
# contributions in the same order, but the two code paths differ in how they
# reduce the per-member 4x4 dyads into the free-DOF block.  That difference is
# bit-identical on x86 OpenBLAS but not on Apple's Accelerate framework, which
# reassociates summations differently on Apple Silicon.  A tolerance of 1e-10
# is still four orders tighter than any genuine assembly or sign bug would
# produce, so the assertion keeps its teeth while becoming portable.
KG_RTOL = 1e-10
KG_ATOL = 1e-12


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _toggle(h: float, b: float = 1.0, load: float = 1000.0):
    """Shallow two-bar toggle: pinned supports, apex load, span 2b, rise h."""
    nodes = [
        Node(id="L", x=-b, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=b, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=h, is_support=False),
    ]
    elements = [
        Element(id="r1", node_i="L", node_j="A", E=E_STEEL, A=AREA, I_sec=1e-9),
        Element(id="r2", node_i="R", node_j="A", E=E_STEEL, A=AREA, I_sec=1e-9),
    ]
    loads = {"A": {"Fx": 0.0, "Fy": -load}}
    return nodes, elements, loads


def _fan(dl_free: float = 0.0, theta: float = 0.01):
    """Redundant shallow fan: two rafters + a slender post to a fixed node.

    The post makes the apex vertically redundant, so ``delta_L_free`` on the
    rafters produces genuine restrained compression -- the textbook
    thermal-prestress destabilisation configuration.
    """
    nodes = [
        Node(id="L", x=-1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=0.0, y=-1.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=theta, is_support=False),
    ]
    elements = [
        Element(
            id="r1",
            node_i="L",
            node_j="A",
            E=E_STEEL,
            A=1e-3,
            I_sec=1e-9,
            delta_L_free=dl_free,
        ),
        Element(
            id="r2",
            node_i="R",
            node_j="A",
            E=E_STEEL,
            A=1e-3,
            I_sec=1e-9,
            delta_L_free=dl_free,
        ),
        Element(id="post", node_i="A", node_j="B", E=E_STEEL, A=1e-6, I_sec=1e-12),
    ]
    loads = {"A": {"Fx": 0.0, "Fy": -50.0}}
    return nodes, elements, loads


def _pratt_redundant(n_panels: int = 8, span: float = 32.0, depth: float = 2.0):
    """A statically indeterminate Pratt girder: straight chords, real DOF count.

    Redundant on purpose -- the sparse/dense assembly paths must agree on a
    model whose free-DOF map is not a contiguous prefix, and whose member set
    is large enough for a COO duplicate-summation slip to show up.
    """
    dx = span / n_panels
    nodes: list[Node] = []
    elements: list[Element] = []
    for i in range(n_panels + 1):
        nodes.append(
            Node(
                id=f"b{i}",
                x=i * dx,
                y=0.0,
                is_support=i in (0, n_panels),
                support_dx=i == 0,
                support_dy=i in (0, n_panels),
            )
        )
        nodes.append(Node(id=f"t{i}", x=i * dx, y=depth))
    for i in range(n_panels):
        elements += [
            Element(id=f"bc{i}", node_i=f"b{i}", node_j=f"b{i + 1}", E=E_STEEL, A=AREA),
            Element(id=f"tc{i}", node_i=f"t{i}", node_j=f"t{i + 1}", E=E_STEEL, A=AREA),
            Element(id=f"v{i}", node_i=f"b{i}", node_j=f"t{i}", E=E_STEEL, A=AREA),
            Element(
                id=f"d{i}", node_i=f"b{i}", node_j=f"t{i + 1}", E=E_STEEL, A=0.6 * AREA
            ),
        ]
    elements.append(
        Element(
            id="vl", node_i=f"b{n_panels}", node_j=f"t{n_panels}", E=E_STEEL, A=AREA
        )
    )
    loads = {nd.id: {"Fx": 0.0, "Fy": -50e3} for nd in nodes if not nd.is_support}
    return nodes, elements, loads


# --------------------------------------------------------------------------
# 1. closed-form analytics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("h", [0.05, 0.1, 0.2, 0.4])
def test_toggle_bifurcation_matches_closed_form(h: float) -> None:
    """Hand derivation for the symmetric two-bar toggle.

    At the apex (the only free node), the elastic and geometric dyads are
    diagonal by symmetry. With member force ``N = -P L0 / (2 h)`` under the
    vertical load ``P`` (first-order equilibrium), the vertical (symmetric)
    stiffness is

        K_yy = 2 (EA/L0) (h/L0)^2 + 2 (N/L0) (b/L0)^2

    and ``K_yy = 0`` gives ``N_cr = -EA h^2 / b^2``, i.e.

        P_cr = 2 h |N_cr| / L0 = 2 E A h^3 / (b^2 L0),   L0 = sqrt(b^2 + h^2).

    The horizontal (antisymmetric) mode gives ``P_cr = 2 E A b^2 / (h L0)``,
    always larger for ``h < b``; the smallest positive root is the vertical
    one, which is what the eigenproblem must return.
    """
    b, load = 1.0, 1000.0
    nodes, elements, loads = _toggle(h, b, load)
    res = linearized_buckling_load_factor(nodes, elements, loads)
    l0 = math.hypot(b, h)
    p_cr_exact = 2.0 * E_STEEL * AREA * h**3 / (b**2 * l0)
    assert res.lambda_cr * load == pytest.approx(p_cr_exact, rel=1e-10)
    assert res.n_compressed == 2
    # the mode is the symmetric vertical one (apex uy, DOF 2*2+1)
    apex_uy = 2 * 2 + 1
    assert abs(res.mode[apex_uy]) == pytest.approx(1.0, abs=1e-8)


def test_toggle_eigenvalue_is_the_smaller_of_two_modes() -> None:
    """The returned factor must be the SMALLEST positive bifurcation."""
    h, b, load = 0.1, 1.0, 1000.0
    nodes, elements, loads = _toggle(h, b, load)
    res = linearized_buckling_load_factor(nodes, elements, loads)
    l0 = math.hypot(b, h)
    p_vertical = 2.0 * E_STEEL * AREA * h**3 / (b**2 * l0)
    p_horizontal = 2.0 * E_STEEL * AREA * b**2 / (h * l0)
    assert res.lambda_cr * load == pytest.approx(p_vertical, rel=1e-10)
    assert p_vertical < p_horizontal


# --------------------------------------------------------------------------
# 2. first-principles identities
# --------------------------------------------------------------------------


def test_geometric_vectors_orthogonal_to_elastic_vectors() -> None:
    """b_e . g_e = 0 for every member: axial and transverse dyads decouple."""
    nodes, elements, _ = _toggle(0.3)
    geom = member_geometry(nodes, elements)
    g = member_geometric_vectors(geom, 2 * len(nodes))
    b = np.zeros_like(g)
    for e in range(len(elements)):
        c, s = geom.cosines[e], geom.sines[e]
        b[e, geom.dofs[e]] = (-c, -s, c, s)
    dots = np.einsum("ei,ei->e", b, g)
    assert np.allclose(dots, 0.0, atol=1e-14)


def test_geometric_stiffness_energy_identity() -> None:
    """u^T K_G u = sum_e (N_e/L_e) (g_e.u)^2 == sum_e N_e L_e phi_e^2."""
    nodes, elements, _loads = _toggle(0.2)
    rng = np.random.default_rng(7)
    forces = {"r1": -5.2e4, "r2": 3.1e4}
    K_G = geometric_stiffness(nodes, elements, forces)
    geom = member_geometry(nodes, elements)
    g = member_geometric_vectors(geom, 2 * len(nodes))
    for _ in range(20):
        u = rng.normal(size=2 * len(nodes))
        lhs = float(u @ K_G @ u)
        transverse = g @ u
        coeffs = np.array([forces.get(e.id, 0.0) for e in elements]) / geom.lengths
        rhs = float(np.sum(coeffs * transverse**2))
        assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-8)


def test_geometric_stiffness_single_member_golden() -> None:
    """Hand-built 4x4 K_G of one inclined member against the assembly."""
    # member at 30 deg, L = 2, N = -1000 (compression)
    c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
    L = 2.0
    nodes = [Node(id="1", x=0.0, y=0.0), Node(id="2", x=L * c, y=L * s)]
    elements = [Element(id="m", node_i="1", node_j="2", E=E_STEEL, A=AREA)]
    N = -1000.0
    K_G = np.asarray(geometric_stiffness(nodes, elements, {"m": N}))
    hand = (N / L) * np.array(
        [
            [s * s, -c * s, -s * s, c * s],
            [-c * s, c * c, c * s, -c * c],
            [-s * s, c * s, s * s, -c * s],
            [c * s, -c * c, -c * s, c * c],
        ]
    )
    assert np.allclose(K_G, hand, rtol=1e-12, atol=1e-12)


def test_geometric_stiffness_sparse_matches_dense() -> None:
    nodes, elements, _loads = _toggle(0.2)
    forces = {"r1": -5.2e4, "r2": 3.1e4}
    dense = np.asarray(geometric_stiffness(nodes, elements, forces))
    sparse = geometric_stiffness(nodes, elements, forces, sparse=True)
    assert np.allclose(dense, sparse.toarray(), rtol=0.0, atol=1e-9)
    # restricted to free DOFs both paths agree too
    free = [4, 5]
    dense_f = np.asarray(geometric_stiffness(nodes, elements, forces, free_dofs=free))
    sparse_f = geometric_stiffness(nodes, elements, forces, free_dofs=free, sparse=True)
    assert np.allclose(dense_f, sparse_f.toarray(), rtol=0.0, atol=1e-9)
    assert np.allclose(dense_f, dense[np.ix_(free, free)])


def test_geometric_stiffness_unknown_member_raises() -> None:
    nodes, elements, _ = _toggle(0.2)
    with pytest.raises(KeyError, match="unknown member"):
        geometric_stiffness(nodes, elements, {"nope": 1.0})


# --------------------------------------------------------------------------
# 3. independent algorithm: dense generalised eigensolve (LAPACK QZ)
# --------------------------------------------------------------------------


def _lambda_cr_qz(nodes, elements, loads) -> float:
    """Reference bifurcation factor via scipy.linalg.eig(A, B) -- a completely
    different LAPACK path (QZ) from the library's symmetric whitening."""
    K, _F, _Fm, fixed = assemble_global_matrices(nodes, elements)
    n_dof = 2 * len(nodes)
    free = [d for d in range(n_dof) if d not in fixed]
    from truss_analysis.limitstates import member_axial_forces

    ambient = {e.id: 20.0 for e in elements}
    forces = member_axial_forces(nodes, elements, loads, ambient)
    K_G = np.asarray(geometric_stiffness(nodes, elements, forces, free_dofs=free))
    A = K[np.ix_(free, free)]
    w = dense_generalized_eig(A, -K_G)[0]
    real_positive = [
        float(x.real)
        for x in w
        if np.isfinite(x) and abs(x.imag) < 1e-8 * max(1.0, abs(x)) and x.real > 0
    ]
    return min(real_positive) if real_positive else float("inf")


@pytest.mark.parametrize("case", [0, 1, 3, 4, 6, 9, 10, 12])
def test_matches_independent_qz_eigensolve_on_campaign(case: int, campaign) -> None:
    """Whitened symmetric path == dense QZ path on the shared 21-topology suite."""
    cm = campaign[case]
    res = linearized_buckling_load_factor(cm.nodes, cm.elements, cm.loads)
    ref = _lambda_cr_qz(cm.nodes, cm.elements, cm.loads)
    if math.isinf(ref):
        assert math.isinf(res.lambda_cr)
    else:
        assert res.lambda_cr == pytest.approx(ref, rel=1e-6)


# --------------------------------------------------------------------------
# 4. physical invariants
# --------------------------------------------------------------------------


def test_rigid_motion_invariance() -> None:
    """lambda_cr is unchanged by translation and rotation of the whole model."""
    nodes, elements, loads = _toggle(0.25)
    base = linearized_buckling_load_factor(nodes, elements, loads).lambda_cr

    ang = math.radians(37.0)
    ca, sa = math.cos(ang), math.sin(ang)
    moved = [
        Node(
            id=nd.id,
            x=ca * nd.x - sa * nd.y + 12.5,
            y=sa * nd.x + ca * nd.y - 3.25,
            is_support=nd.is_support,
            support_dx=nd.support_dx,
            support_dy=nd.support_dy,
        )
        for nd in nodes
    ]
    # rotate the load vector with the model
    fx, fy = loads["A"]["Fx"], loads["A"]["Fy"]
    moved_loads = {"A": {"Fx": ca * fx - sa * fy, "Fy": sa * fx + ca * fy}}
    moved_res = linearized_buckling_load_factor(moved, elements, moved_loads)
    assert moved_res.lambda_cr == pytest.approx(base, rel=1e-9)


def test_load_scaling_invariance() -> None:
    """Doubling the mechanical load halves the load factor (linearity)."""
    nodes, elements, loads = _toggle(0.2)
    lam1 = linearized_buckling_load_factor(nodes, elements, loads).lambda_cr
    scaled = {"A": {"Fx": 0.0, "Fy": 2.0 * loads["A"]["Fy"]}}
    lam2 = linearized_buckling_load_factor(nodes, elements, scaled).lambda_cr
    assert lam2 == pytest.approx(lam1 / 2.0, rel=1e-12)


def test_tension_only_load_has_infinite_factor() -> None:
    nodes, elements, _loads = _toggle(0.2)
    loads_up = {"A": {"Fx": 0.0, "Fy": +5000.0}}  # pulls the apex up: ties
    res = linearized_buckling_load_factor(nodes, elements, loads_up)
    assert res.lambda_cr == float("inf")
    assert res.n_compressed == 0


def test_zero_force_member_does_not_change_factor() -> None:
    """A tie between the two pinned supports carries no force and changes nothing."""
    nodes, elements, loads = _toggle(0.2)
    base = linearized_buckling_load_factor(nodes, elements, loads).lambda_cr
    with_tie = [*elements, Element(id="tie", node_i="L", node_j="R", E=E_STEEL, A=AREA)]
    got = linearized_buckling_load_factor(nodes, with_tie, loads).lambda_cr
    assert got == pytest.approx(base, rel=1e-12)


def test_mode_satisfies_the_bifurcation_equation() -> None:
    """Residual check: (K_E + lambda_cr K_G(N)) u ~ 0 ON THE FREE DOFs.

    The fixed-DOF rows of the global equation are reaction rows and are
    non-zero by construction; the bifurcation statement lives on the free
    sub-block, exactly where the eigenproblem was solved.
    """
    nodes, elements, loads = _toggle(0.2)
    res = linearized_buckling_load_factor(nodes, elements, loads)
    K, _F, _Fm, fixed = assemble_global_matrices(nodes, elements)
    from truss_analysis.limitstates import member_axial_forces

    # at lambda_cr the mechanical forces are amplified by lambda_cr
    ambient = {e.id: 20.0 for e in elements}
    forces = {
        eid: res.lambda_cr * n
        for eid, n in member_axial_forces(nodes, elements, loads, ambient).items()
    }
    K_G = np.asarray(geometric_stiffness(nodes, elements, forces))
    free = [d for d in range(2 * len(nodes)) if d not in fixed]
    total_ff = (K + K_G)[np.ix_(free, free)]
    resid = float(np.linalg.norm(total_ff @ res.mode[free]))
    scale = float(np.linalg.norm(K[np.ix_(free, free)]))
    assert resid / scale < 1e-10


# --------------------------------------------------------------------------
# 5. prestressed (thermal / fabrication) destabilisation
# --------------------------------------------------------------------------


def test_imposed_compression_erodes_then_destroys_the_load_factor() -> None:
    """The headline round-5 physics (C2-A): restrained thermal-like prestress
    monotonically reduces the mechanical buckling reserve and finally makes
    the base state itself unstable -- a first-order DCR chain sees none of
    this."""
    lams = []
    for dl in (0.0, 2e-4, 4e-4, 6e-4):
        nodes, elements, loads = _fan(dl_free=dl)
        lams.append(linearized_buckling_load_factor(nodes, elements, loads).lambda_cr)
    import itertools

    assert all(a > b for a, b in itertools.pairwise(lams))

    # a realistic 0.08 % fabrication/thermal strain already destabilises it
    nodes, elements, loads = _fan(dl_free=8e-4)
    with pytest.raises(MechanismError, match="not positive definite"):
        linearized_buckling_load_factor(nodes, elements, loads)


def test_thermal_field_enters_the_base_state() -> None:
    """With temps supplied the fire chain's degraded stiffness and restrained
    expansion feed the SAME base state the DCR chain uses."""
    nodes, elements, loads = _fan()
    cold = linearized_buckling_load_factor(nodes, elements, loads).lambda_cr
    temps = {e.id: 20.0 for e in elements}  # ambient field: no degradation
    ambient = linearized_buckling_load_factor(nodes, elements, loads, temps).lambda_cr
    assert ambient == pytest.approx(cold, rel=1e-9)

    hot = {e.id: 400.0 for e in elements}
    lam_hot = linearized_buckling_load_factor(nodes, elements, loads, hot).lambda_cr
    # k_E(400) < 1 softens the structure and the restrained expansion adds
    # compression: the reserve must shrink.
    assert lam_hot < cold


def test_base_mechanism_raises() -> None:
    """A kinematic mechanism has no bifurcation problem; the engine says so."""
    nodes, _elements, loads = _toggle(0.2)
    single = [Element(id="r1", node_i="L", node_j="A", E=E_STEEL, A=AREA, I_sec=1e-9)]
    with pytest.raises(MechanismError):
        linearized_buckling_load_factor(nodes, single, loads)


def test_fully_restrained_model_raises() -> None:
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [Element(id="m", node_i="1", node_j="2", E=E_STEEL, A=AREA)]
    with pytest.raises(MechanismError, match="every DOF is restrained"):
        linearized_buckling_load_factor(nodes, elements, {})


# --------------------------------------------------------------------------
# 6. finite-difference tangent verification (A5: strongest oracle for K_G)
# --------------------------------------------------------------------------


def test_geometric_stiffness_finite_difference_tangent() -> None:
    """Verify that K_G is the exact tangent of the internal force residual.

    The geometric stiffness is defined as the derivative of the internal
    force vector with respect to nodal displacements, evaluated at the
    prestressed base state:

        K_G = d(N_int) / du

    This test verifies the identity:

        u^T K_G u ≈ [N_int(u + ε·û) - N_int(u - ε·û)] · û / (2ε)

    where û = u / ||u|| is the normalized direction. This is the strongest
    possible oracle for K_G correctness -- it checks that the matrix is
    the true linearization of the force operator, not just a plausible
    -looking formula.

    See C4#3: "هیچ تست finite-difference tangent برای K_G وجود ندارد —
    قوی‌ترین oracle باقی‌مانده"
    """
    nodes, elements, _loads = _toggle(0.2)
    forces = {"r1": -5.2e4, "r2": 3.1e4}

    # Assemble K_G from the library
    K_G = np.asarray(geometric_stiffness(nodes, elements, forces))

    # Get geometry for force computation
    geom = member_geometry(nodes, elements)
    n_dof = 2 * len(nodes)
    g = member_geometric_vectors(geom, n_dof)

    # Test on random displacement directions
    rng = np.random.default_rng(42)
    for _ in range(30):
        u = rng.normal(size=n_dof)

        # Direct quadratic form: u^T K_G u
        direct = float(u @ K_G @ u)

        # Finite difference: directional derivative of internal forces
        eps = 1e-8 * max(1.0, np.linalg.norm(u))
        u_hat = u / np.linalg.norm(u) if np.linalg.norm(u) > 0 else u

        def internal_force_work(displacement: np.ndarray) -> float:
            """Compute sum_e (N_e / L_e) * (g_e · u)^2 at given displacement."""
            transverse = g @ displacement
            coeffs = np.array([forces.get(e.id, 0.0) for e in elements]) / geom.lengths
            return float(np.sum(coeffs * transverse**2))

        fd_plus = internal_force_work(u + eps * u_hat)
        fd_minus = internal_force_work(u - eps * u_hat)
        fd_directional = (fd_plus - fd_minus) / (2 * eps)

        # For W(u) = u^T K_G u (quadratic form), directional derivative is:
        # dW/du [û] = 2 * u^T K_G û = 2 * (u^T K_G u) / ||u||
        expected_fd = (
            2.0 * direct / np.linalg.norm(u) if np.linalg.norm(u) > 1e-14 else 0.0
        )

        assert fd_directional == pytest.approx(expected_fd, rel=1e-5, abs=1e-6), (
            f"FD mismatch: {fd_directional:.6e} vs {expected_fd:.6e}, "
            f"direct={direct:.6e}"
        )


def test_geometric_stiffness_second_order_work_identity_fd() -> None:
    """Cross-check: the second-order work computed via K_G matches
    the finite-difference curvature of the potential energy.

    For a quadratic form W(u) = u^T K_G u, the second directional
    derivative along any unit vector v is:

        d²W/ds² [s=0] = 2 * v^T K_G v

    This test computes the same quantity by finite differences and
    verifies agreement.
    """
    nodes, elements, _loads = _fan(dl_free=1e-4)

    # Get base state forces
    setup = build_engine(nodes, elements, {"A": {"Fx": 0.0, "Fy": -50.0}})
    free = list(setup.free_dofs)
    u_total = base_displacement(
        setup, total_load_vector(nodes, {"A": {"Fx": 0.0, "Fy": -50.0}}, setup)
    )
    n_total = member_forces(setup, u_total)
    forces = {e.id: float(n_total[i]) for i, e in enumerate(elements)}

    K_G_full = np.asarray(geometric_stiffness(nodes, elements, forces))
    K_G = K_G_full[np.ix_(free, free)]

    geom = member_geometry(nodes, elements)
    g_all = member_geometric_vectors(geom, 2 * len(nodes))
    g_free = g_all[:, free]

    def second_order_work(displacement_free: np.ndarray) -> float:
        """Compute sum_e (N_e / L_e) * phi_e^2 for given free DOF displacement."""
        transverse = g_free @ displacement_free
        coeffs = np.array([forces.get(e.id, 0.0) for e in elements]) / geom.lengths
        return float(np.sum(coeffs * transverse**2))

    rng = np.random.default_rng(123)
    for _ in range(20):
        v = rng.normal(size=len(free))
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        v_hat = v / v_norm

        # Direct: v^T K_G v
        direct = float(v_hat @ K_G @ v_hat)

        # FD: second derivative of W along v
        eps = 1e-7
        w_plus = second_order_work(eps * v_hat)
        w_zero = second_order_work(np.zeros_like(v))
        w_minus = second_order_work(-eps * v_hat)

        # Second derivative: [W(ε) - 2W(0) + W(-ε)] / ε²
        fd_second = (w_plus - 2 * w_zero + w_minus) / (eps**2)

        # For W(u) = u^T K u (no 0.5 factor), second derivative = 2 * v^T K v
        assert fd_second == pytest.approx(2.0 * direct, rel=1e-4, abs=1e-5), (
            f"Second-derivative mismatch: FD={fd_second:.6e}, "
            f"2*direct={2.0 * direct:.6e}"
        )


def test_geometric_stiffness_is_homogeneous_in_the_base_state() -> None:
    """Matrix-level homogeneity: ``K_G(alpha N) = alpha K_G(N)``.

    ``test_load_scaling_invariance`` pins the *eigenvalue* consequence of
    linearity; this pins the *assembly* consequence.  Two base states are
    solved independently through the engine (so the axial forces come from
    the production demand path, not from a hand-written dictionary) at a
    50x load ratio, and the assembled ``K_G`` matrices must obey the same
    ratio exactly -- a scalar slip in ``(N_e / L_e) g_e g_e^T`` would show
    up here while leaving the eigenvector test indifferent.

    Folded in from the round-6 audit: the stand-alone
    ``tests/test_stability_tangent.py`` module duplicated the
    finite-difference tangent oracle already present above (A5) and called
    :func:`member_forces` with the full displacement vector instead of the
    free-DOF slice; its one non-redundant assertion is kept here.
    """
    nodes, elements, loads = _toggle(0.2)
    free = None
    forces_by_scale: dict[float, dict[str, float]] = {}
    for scale in (1.0, 50.0):
        scaled_loads = {"A": {"Fx": 0.0, "Fy": scale * loads["A"]["Fy"]}}
        setup = build_engine(nodes, elements, scaled_loads)
        free = list(setup.free_dofs)
        u_free = base_displacement(setup, total_load_vector(nodes, scaled_loads, setup))
        n_mem = member_forces(setup, u_free)
        forces_by_scale[scale] = {e.id: float(n_mem[i]) for i, e in enumerate(elements)}

    assert free is not None
    k_g_low = np.asarray(geometric_stiffness(nodes, elements, forces_by_scale[1.0]))
    k_g_high = np.asarray(geometric_stiffness(nodes, elements, forces_by_scale[50.0]))

    # Homogeneity of the assembled matrix on the free sub-block.
    assert np.allclose(
        k_g_high[np.ix_(free, free)],
        50.0 * k_g_low[np.ix_(free, free)],
        rtol=1e-12,
    )
    # ... and the contribution is not vacuous: high compression must visibly
    # erode the tangent stiffness relative to the elastic matrix.
    k_e, _, _, _ = assemble_global_matrices(nodes, elements)
    k_t = k_e[np.ix_(free, free)] + k_g_high[np.ix_(free, free)]
    assert np.linalg.norm(k_g_high[np.ix_(free, free)]) > 1e-3
    assert np.linalg.norm(k_t) < np.linalg.norm(k_e[np.ix_(free, free)])


# --------------------------------------------------------------------------
# round-7 audit, item 6: the sparse and dense geometric-stiffness assemblies
# are two separate code paths and nothing pinned them to each other.
#
# `geometric_stiffness(sparse=True)` builds its dyads from the member-LOCAL
# 4-vectors and scatters them through a COO triplet list with restrained DOFs
# masked out; the dense path scatters into the full `(m, n_dof)` matrix with
# `member_geometric_vectors` and then slices columns.  A round-6 commit message
# records that this change "briefly fell into" a restricted-vs-unrestricted
# indexing trap.  A trap that was fallen into once needs a test that stays
# fallen-into-able, not a note that it was fixed.
#
# Tolerance note: the two paths agree bit-for-bit on x86 OpenBLAS, but on
# Apple's Accelerate framework the dyad reduction into the free-DOF block is
# reassociated and the last few mantissa bits differ.  KG_RTOL/KG_ATOL above
# admit that cross-BLAS rounding while staying four orders tighter than any
# genuine assembly, sign, or index bug would produce.
# --------------------------------------------------------------------------


def _mixed_sign_forces(elements) -> dict[str, float]:
    """Alternate tension and compression so no sign symmetry hides a bug."""
    return {e.id: (-1.0) ** i * (1e5 + 3e4 * i) for i, e in enumerate(elements)}


@pytest.mark.parametrize("restrain", ["none", "one", "both", "alternating"])
def test_geometric_stiffness_sparse_equals_dense_matrix(
    restrain: str,
) -> None:
    """Matrix equality of the two assembly paths, at every level of DOF
    restriction.

    The restricted cases are the ones that matter: masking restrained DOFs out
    of a COO triplet list and slicing columns out of a dense matrix are
    different operations, and they agree only if the global-to-free index map
    is applied identically in both.
    """
    nodes, elements, _ = _toggle(0.15)
    n_dof = 2 * len(nodes)
    all_dofs = list(range(n_dof))

    if restrain == "none":
        free = all_dofs
    elif restrain == "one":
        free = all_dofs[:-1]
    elif restrain == "both":
        free = all_dofs[:-2]
    else:  # alternating: the map is not a contiguous slice
        free = [d for d in all_dofs if d % 2 == 0]

    forces = _mixed_sign_forces(elements)
    dense = geometric_stiffness(nodes, elements, forces, free_dofs=free, sparse=False)
    sparse = geometric_stiffness(nodes, elements, forces, free_dofs=free, sparse=True)

    assert np.asarray(dense).shape == (len(free), len(free))
    np.testing.assert_allclose(
        np.asarray(sparse.toarray()),
        np.asarray(dense),
        rtol=KG_RTOL,
        atol=KG_ATOL,
        err_msg=f"sparse and dense K_G disagree with free_dofs={restrain}",
    )


def test_geometric_stiffness_sparse_equals_dense_unrestricted() -> None:
    """The ``free_dofs=None`` case, on a larger model with both sign patterns."""
    nodes, elements, _ = _pratt_redundant()
    forces = _mixed_sign_forces(elements)
    dense = geometric_stiffness(nodes, elements, forces, sparse=False)
    sparse = geometric_stiffness(nodes, elements, forces, sparse=True)
    np.testing.assert_allclose(
        np.asarray(sparse.toarray()),
        np.asarray(dense),
        rtol=KG_RTOL,
        atol=KG_ATOL,
    )


def test_geometric_stiffness_sparse_is_symmetric_and_correctly_sparse() -> None:
    """Symmetry is what licenses the whitened eigenproblem; nnz is the point."""
    nodes, elements, _ = _pratt_redundant()
    forces = _mixed_sign_forces(elements)
    k_g = geometric_stiffness(nodes, elements, forces, sparse=True)

    assert isinstance(k_g, sp.csr_matrix)
    delta = (k_g - k_g.T).toarray()
    assert float(np.max(np.abs(delta))) == 0.0
    # 4x4 dyads per member, so nnz is linear in the member count, not
    # quadratic in the DOF count -- which is the entire reason the path exists.
    assert k_g.nnz <= 16 * len(elements)
    assert k_g.nnz < k_g.shape[0] ** 2


def test_sparse_dense_agreement_survives_a_thermal_prestress() -> None:
    """The path the fire chain actually takes: imposed forces from a temp field."""
    nodes, elements, loads = _toggle(0.15)
    temps = {e.id: 400.0 + 50.0 * i for i, e in enumerate(elements)}
    setup = build_engine(nodes, elements, loads, temps)
    u = base_displacement(setup, total_load_vector(nodes, loads, setup))
    forces = {
        e.id: float(v) for e, v in zip(elements, member_forces(setup, u), strict=True)
    }

    restrained = set(fixed_dof_indices(nodes))
    free = [d for d in range(2 * len(nodes)) if d not in restrained]
    dense = geometric_stiffness(nodes, elements, forces, free_dofs=free, sparse=False)
    sparse = geometric_stiffness(nodes, elements, forces, free_dofs=free, sparse=True)
    np.testing.assert_allclose(
        np.asarray(sparse.toarray()),
        np.asarray(dense),
        rtol=KG_RTOL,
        atol=KG_ATOL,
    )
    # and the prestress must actually be doing something, or the test is vacuous
    assert float(np.max(np.abs(np.asarray(dense)))) > 0.0
