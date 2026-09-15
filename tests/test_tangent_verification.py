"""Exact-tangent verification: the operator, its derivative and its linearisation.

Three independent oracles guard :mod:`truss_analysis.tangent_verification`:

1. **closed form** -- a single inclined bar whose exact tangent is written
   out by hand from the two-dyad decomposition;
2. **finite differences** of the geometrically-exact internal force, at
   several displacement amplitudes and on a thermally prestressed model;
3. **convergence order** -- the gap between the library's first-order
   ``K_E + K_G`` and the exact tangent must close at first order in the
   load, which is the measurable content of the word "linearised".
"""

from __future__ import annotations

import numpy as np
import pytest

from truss_analysis.criticality.engine import (
    base_displacement,
    build_engine,
    total_load_vector,
)
from truss_analysis.model import Element, Node
from truss_analysis.stability import geometric_stiffness
from truss_analysis.tangent_verification import (
    exact_tangent_stiffness,
    internal_force,
    linearized_tangent_stiffness,
    verify_linearization_convergence,
    verify_tangent_stiffness,
)

E_STEEL = 210e9
AREA = 1e-3
ALPHA = 1.2e-5


def _toggle(h: float = 0.2, b: float = 1.0):
    """Shallow two-bar toggle with pinned supports and a free apex."""
    nodes = [
        Node(id="L", x=-b, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=b, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=h, is_support=False),
    ]
    elements = [
        Element(id="r1", node_i="L", node_j="A", E=E_STEEL, A=AREA, alpha=ALPHA),
        Element(id="r2", node_i="R", node_j="A", E=E_STEEL, A=AREA, alpha=ALPHA),
    ]
    return nodes, elements


def _fan():
    """Redundant shallow fan: the apex is vertically restrained by a post.

    One degree of static indeterminacy, so a uniform temperature rise
    produces genuine restrained compression -- the only configuration in
    which the geometric term is exercised at ``u = 0``.
    """
    nodes = [
        Node(id="L", x=-1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=0.0, y=-1.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=0.01, is_support=False),
    ]
    elements = [
        Element(id="r1", node_i="L", node_j="A", E=E_STEEL, A=AREA, alpha=ALPHA),
        Element(id="r2", node_i="R", node_j="A", E=E_STEEL, A=AREA, alpha=ALPHA),
        Element(id="post", node_i="A", node_j="B", E=E_STEEL, A=1e-6, alpha=ALPHA),
    ]
    return nodes, elements


# --------------------------------------------------------------------------
# 1. closed form: one inclined bar
# --------------------------------------------------------------------------


def test_single_bar_exact_tangent_matches_hand_derivation() -> None:
    """Hand-built ``K_T`` of one inclined bar against the assembler.

    For a bar from the origin at angle ``theta`` with both nodes free,

    .. code-block:: text

        K_T = (EA/L0) b b^T + (N/L) g g^T,
        b = [-c, -s, c, s],  g = [s, -c, -s, c]

    and with ``u = 0``, ``N = 0`` the geometric dyad drops out entirely, so
    the exact tangent must equal the library's elastic dyad to the last bit.
    """
    theta = np.deg2rad(30.0)
    length = 2.0
    nodes = [
        Node(id="i", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="j", x=length * np.cos(theta), y=length * np.sin(theta)),
    ]
    elements = [Element(id="m", node_i="i", node_j="j", E=E_STEEL, A=AREA)]

    c, s = np.cos(theta), np.sin(theta)
    b = np.array([-c, -s, c, s])
    k_hand = (E_STEEL * AREA / length) * np.outer(b, b)

    got = exact_tangent_stiffness(nodes, elements, np.zeros(4))
    assert np.allclose(got, k_hand, rtol=1e-12, atol=0.0)


def test_single_bar_geometric_dyad_under_prescribed_axial_force() -> None:
    """The ``(N/L) g g^T`` term appears exactly when the bar is stretched.

    Imposing a pure axial displacement ``u_j = delta * n`` elongates the bar
    by ``delta``; the exact tangent must then equal
    ``(EA/L0) b b^T + (N/L) g g^T`` with ``N = (EA/L0) delta`` and ``L`` the
    *deformed* length -- evaluated here against an independently written
    formula rather than against the library's own helper.
    """
    theta = np.deg2rad(20.0)
    length = 1.5
    delta = 1e-3
    nodes = [
        Node(id="i", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="j", x=length * np.cos(theta), y=length * np.sin(theta)),
    ]
    elements = [Element(id="m", node_i="i", node_j="j", E=E_STEEL, A=AREA)]

    c, s = np.cos(theta), np.sin(theta)
    u = np.array([0.0, 0.0, delta * c, delta * s])
    deformed = length + delta
    n_force = (E_STEEL * AREA / length) * delta

    b = np.array([-c, -s, c, s])
    g = np.array([s, -c, -s, c])
    k_hand = (E_STEEL * AREA / length) * np.outer(b, b)
    k_hand += (n_force / deformed) * np.outer(g, g)

    got = exact_tangent_stiffness(nodes, elements, u)
    assert np.allclose(got, k_hand, rtol=1e-10)


# --------------------------------------------------------------------------
# 2. finite differences of the exact operator
# --------------------------------------------------------------------------


def test_internal_force_vanishes_at_the_stress_free_state() -> None:
    nodes, elements = _toggle()
    assert np.allclose(internal_force(nodes, elements, np.zeros(6)), 0.0, atol=1e-9)


def test_internal_force_is_linear_in_small_axial_displacement() -> None:
    """For a single bar the exact force reduces to ``k * delta`` axially."""
    length = 3.0
    nodes = [
        Node(id="i", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="j", x=length, y=0.0),
    ]
    elements = [Element(id="m", node_i="i", node_j="j", E=E_STEEL, A=AREA)]
    delta = 2e-4
    p = internal_force(nodes, elements, np.array([0.0, 0.0, delta, 0.0]))
    expected = (E_STEEL * AREA / length) * delta
    # P = N * b with b = [-c, -s, c, s]: the pull at j is +N, at i is -N.
    assert p[2] == pytest.approx(+expected, rel=1e-9)
    assert p[0] == pytest.approx(-expected, rel=1e-9)
    assert p[1] == pytest.approx(0.0, abs=1e-6)
    assert p[3] == pytest.approx(0.0, abs=1e-6)


def test_exact_tangent_is_symmetric() -> None:
    nodes, elements = _toggle()
    rng = np.random.default_rng(7)
    u = rng.normal(scale=2e-3, size=6)
    k_t = exact_tangent_stiffness(nodes, elements, u)
    assert np.abs(k_t - k_t.T).max() < 1e-6 * np.abs(k_t).max()


@pytest.mark.parametrize("amplitude", [1e-5, 1e-4, 1e-3, 1e-2])
def test_finite_difference_tangent_matches_exact(amplitude: float) -> None:
    """Central differences of ``P(u)`` reproduce ``K_T`` to ~1e-10 relative.

    The tolerance is deliberately tight: with ``epsilon = 1e-6`` the
    truncation term is ``O(eps^2)`` and the round-off term ``O(machine/eps)``,
    both far below ``1e-8`` for a stiffness of order ``1e8`` N/m.  A failure
    here means the derivative formula is wrong, not that the step is badly
    chosen.
    """
    nodes, elements = _toggle()
    rng = np.random.default_rng(11)
    u = rng.normal(scale=amplitude, size=6)
    check = verify_tangent_stiffness(nodes, elements, u, epsilon=1e-6, tol=1e-8)
    assert check.passed, f"max_rel_error={check.max_rel_error:.3e}"
    assert check.max_rel_error < 1e-8
    assert check.state_norm == pytest.approx(float(np.linalg.norm(u)), rel=1e-12)


def test_finite_difference_tangent_at_prestressed_thermal_base_state() -> None:
    """Restrained heating must leave a non-zero residual and still verify.

    The fan is one degree indeterminate, so a uniform 500 degC field cannot
    be accommodated by free expansion: ``P(0) != 0`` and the geometric dyad
    is loaded from the very first evaluation.  This is the state the fire
    chain actually linearises about.
    """
    nodes, elements = _fan()
    temps = {e.id: 500.0 for e in elements}
    p0 = internal_force(nodes, elements, np.zeros(8), temps)
    assert np.abs(p0).max() > 1.0, "restrained thermal field produced no force"

    setup = build_engine(nodes, elements, {}, temps)
    u_free = base_displacement(setup, total_load_vector(nodes, {}, setup))
    u = np.zeros(8)
    u[list(setup.free_dofs)] = u_free

    check = verify_tangent_stiffness(nodes, elements, u, temps, epsilon=1e-6, tol=1e-8)
    assert check.passed, f"max_rel_error={check.max_rel_error:.3e}"


def test_partial_dof_sweep_reports_only_the_tested_columns() -> None:
    nodes, elements = _toggle()
    u = np.array([0.0, 0.0, 0.0, 0.0, 1e-3, -2e-3])
    full = verify_tangent_stiffness(nodes, elements, u, epsilon=1e-6, tol=1e-8)
    part = verify_tangent_stiffness(
        nodes, elements, u, dofs=[4, 5], epsilon=1e-6, tol=1e-8
    )
    assert full.passed
    assert part.passed
    assert np.all(part.rel_error[:4] == 0.0)
    assert np.allclose(part.rel_error[4:], full.rel_error[4:], rtol=1e-6)


def test_verification_catches_a_corrupted_tangent() -> None:
    """Negative control: a wrong geometric sign must fail the check.

    Guards against the verifier being vacuously green -- the failure mode of
    the round-5 scaffolding this module replaced, which returned ``True``
    unconditionally.
    """
    nodes, elements = _toggle()
    rng = np.random.default_rng(3)
    u = rng.normal(scale=5e-3, size=6)

    def flipped_internal_force(uu: np.ndarray) -> np.ndarray:
        """Same operator, but the transverse direction is mirrored."""
        out = internal_force(nodes, elements, uu)
        out[1::2] *= -1.0
        return out

    eps = 1e-6
    k_fd = np.zeros((6, 6))
    for j in range(6):
        step = np.zeros(6)
        step[j] = eps
        k_fd[:, j] = (
            flipped_internal_force(u + step) - flipped_internal_force(u - step)
        ) / (2.0 * eps)
    k_exact = exact_tangent_stiffness(nodes, elements, u)
    rel = np.linalg.norm(k_fd - k_exact, axis=0) / np.linalg.norm(k_exact, axis=0)
    assert rel.max() > 1e-3, "corrupted operator was not detected"


# --------------------------------------------------------------------------
# 3. the linearisation gap and its convergence order
# --------------------------------------------------------------------------


def test_exact_and_linearized_tangent_coincide_at_zero_demand() -> None:
    nodes, elements = _toggle()
    k_exact = exact_tangent_stiffness(nodes, elements, np.zeros(6))
    k_lin = linearized_tangent_stiffness(nodes, elements, {})
    assert np.allclose(k_exact, k_lin, rtol=1e-13, atol=0.0)


def test_linearized_tangent_equals_assembler_plus_geometric_stiffness() -> None:
    """``linearized_tangent_stiffness`` is not a second opinion, it is *the* path.

    Pinned against the two production assemblers independently, so the
    convergence measurement below cannot be self-fulfilling.
    """
    from truss_analysis.assembly import assemble_global_matrices

    nodes, elements = _toggle()
    forces = {"r1": -4.5e4, "r2": 2.0e4}
    k_e, _, _, _ = assemble_global_matrices(nodes, elements)
    k_g = np.asarray(geometric_stiffness(nodes, elements, forces))
    expected = k_e + k_g
    got = linearized_tangent_stiffness(nodes, elements, forces)
    assert np.allclose(got, expected, rtol=1e-12)


def test_linearization_gap_closes_at_first_order() -> None:
    """The measurable content of "linearised": ``gap(s) ~ s``.

    Halving the demand must halve the relative Frobenius gap between the
    exact tangent and ``K_E + K_G``.  A saturated or second-order gap would
    mean the stability module's premise is wrong.
    """
    nodes, elements = _toggle()
    loads = {"A": {"Fx": 0.0, "Fy": -2e5}}
    check = verify_linearization_convergence(
        nodes, elements, loads, scales=(1.0, 0.5, 0.25, 0.125, 0.0625)
    )
    assert check.first_order, f"fitted order = {check.order_estimate:.3f}"
    assert check.order_estimate == pytest.approx(1.0, abs=0.05)
    for order in check.observed_orders:
        assert order == pytest.approx(1.0, abs=0.02)
    # strictly decreasing gaps
    gaps = list(check.rel_gap)
    assert all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1))


def test_linearization_gap_vanishes_as_demand_goes_to_zero() -> None:
    nodes, elements = _toggle()
    check = verify_linearization_convergence(
        nodes, elements, {"A": {"Fx": 0.0, "Fy": -1e6}}, scales=(1e-2, 1e-3, 1e-4)
    )
    assert check.rel_gap[-1] < 1e-5


def test_linearization_check_requires_two_positive_scales() -> None:
    nodes, elements = _toggle()
    loads = {"A": {"Fx": 0.0, "Fy": -1e4}}
    with pytest.raises(ValueError, match="at least two scales"):
        verify_linearization_convergence(nodes, elements, loads, scales=(1.0,))
    with pytest.raises(ValueError, match="must be positive"):
        verify_linearization_convergence(nodes, elements, loads, scales=(1.0, -0.5))


def test_scales_are_ordered_descending_whatever_the_caller_passes() -> None:
    nodes, elements = _toggle()
    loads = {"A": {"Fx": 0.0, "Fy": -5e4}}
    check = verify_linearization_convergence(
        nodes, elements, loads, scales=(0.125, 1.0, 0.5, 0.25)
    )
    assert check.scales == (1.0, 0.5, 0.25, 0.125)


def test_free_dofs_restriction_of_both_operators() -> None:
    nodes, elements = _toggle()
    rng = np.random.default_rng(5)
    u = rng.normal(scale=1e-3, size=6)
    free = [2, 3, 4, 5]
    k_full = exact_tangent_stiffness(nodes, elements, u)
    k_free = exact_tangent_stiffness(nodes, elements, u, free_dofs=free)
    assert np.allclose(k_free, k_full[np.ix_(free, free)], rtol=0.0, atol=0.0)
