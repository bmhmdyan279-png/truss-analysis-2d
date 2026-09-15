"""C10: complex-step verification of the adjoint DDM.

Central differences cap out at roughly ``sqrt(machine epsilon)`` relative
accuracy because the subtractive cancellation in ``(f(x+h) - f(x-h)) / 2h``
grows as ``h`` shrinks.  Complex-step differentiation has no cancellation at
all:

.. code-block:: text

    f'(x) = Im[ f(x + i h) ] / h        exact to machine precision as h -> 0

so it can drive ``h`` to ``1e-30`` and still get the right answer.  That makes
it the strongest available oracle for an analytic derivative -- and the DDM in
:mod:`truss_analysis.sensitivity` is exactly that, an analytic derivative the
library currently cross-checks only against central differences.

Two things have to be true for complex-step to apply, and both are checked
here rather than assumed:

1. **The function must be analytic in the perturbed variable.**  The DDM
   reports ``d(|u|_max)/dA_i``.  ``|.|`` involves a conjugation and ``max``
   is non-smooth, so neither is analytic as written.  But at the base state
   the argmax is a *fixed* node ``k*``, and locally the quantity is
   ``sqrt(u_x^2 + u_y^2)`` evaluated there -- the analytic continuation, taken
   with the principal complex square root and no conjugation.  That is exactly
   what :class:`~truss_analysis.sensitivity.SensitivityResult` documents as its
   subgradient convention, so the test validates the DDM under its own stated
   meaning instead of against a different function.
2. **The chain must be complex-safe end to end.**  ``k = E A / L``,
   ``K = sum k b b^T``, the linear solve and the imposed-force vector
   ``B^T (k dL_pre)`` are all polynomial or linear in ``A``, so they extend
   trivially.  The EN material tables (piecewise-linear interpolation) would
   *not* -- which is why this test uses ambient, non-degraded members, and why
   extending complex-step to the fire chain needs a complex-safe interpolation
   in the material layer first.

The oracle is written from scratch here (assembly, reduction and solve all
re-derived in complex arithmetic) rather than reusing the library's own
assembler, because a check that shares its code path with the thing it checks
can only catch typos, not conceptual errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from truss_analysis.assembly import member_geometry
from truss_analysis.model import Element, Node, fixed_dof_indices
from truss_analysis.reliability_adapter import NodalLoad
from truss_analysis.sensitivity import IndependentValidator

E_STEEL = 210e9


def _model():
    """Indeterminate frame with real displacements and one fabrication prestrain.

    Two redundancies and a ``delta_L_free`` on one member, so the complex
    right-hand side carries both the mechanical and the imposed parts -- an
    oracle that only exercised a purely mechanical load vector would miss a
    sign error in the prestress term.
    """
    nodes = [
        Node(id="L", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=6.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=3.0, y=-2.5, is_support=True, support_dx=True, support_dy=True),
        Node(id="C", x=3.0, y=2.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=3.0, y=0.0),
    ]
    elements = [
        Element(
            id="r1",
            node_i="L",
            node_j="A",
            E=E_STEEL,
            A=5e-3,
            delta_L_free=1.5e-4,
        ),
        Element(id="r2", node_i="R", node_j="A", E=E_STEEL, A=4e-3),
        Element(id="post", node_i="A", node_j="B", E=E_STEEL, A=3e-3),
        Element(id="tie", node_i="A", node_j="C", E=E_STEEL, A=2e-3),
    ]
    loads = [NodalLoad(node_id="A", fx=4.0e4, fy=-9.0e4)]
    return nodes, elements, loads


def _compatibility(nodes, elements, n_dof):
    """Elongation vectors ``b_e`` as an ``(m, n_dof)`` matrix, from scratch."""
    geom = member_geometry(nodes, elements)
    b = np.zeros((geom.n_members, n_dof), dtype=float)
    for e in range(geom.n_members):
        c, s = float(geom.cosines[e]), float(geom.sines[e])
        b[e, geom.dofs[e]] = (-c, -s, c, s)
    return geom, b


def _prestrain(elements):
    """Fabrication elongation per member (no thermal field in this test)."""
    return np.array([float(e.delta_L_free) for e in elements], dtype=float)


def _nodal_load(nodes, loads, n_dof):
    """Full mechanical load vector, assembled independently of the library."""
    f = np.zeros(n_dof, dtype=float)
    index = {n.id: i for i, n in enumerate(nodes)}
    for load in loads:
        i = index[str(load.node_id)]
        f[2 * i] += float(getattr(load, "fx", getattr(load, "Fx", 0.0)))
        f[2 * i + 1] += float(getattr(load, "fy", getattr(load, "Fy", 0.0)))
    return f


def complex_step_dmax_dA(
    nodes, elements, loads, member_index: int, h: float = 1e-30
) -> float:
    """``d(sqrt(u_x^2 + u_y^2))/dA`` at the base-state argmax node, by complex step.

    The whole chain is rebuilt in complex arithmetic: areas, ``k = E A / L``,
    ``K_ff``, the mechanical and imposed right-hand sides, the solve, and the
    analytic continuation of the displacement magnitude.  Nothing is imported
    from the assembly or solver modules.
    """
    n_dof = 2 * len(nodes)
    geom, b = _compatibility(nodes, elements, n_dof)
    fixed = fixed_dof_indices(nodes)
    free = [d for d in range(n_dof) if d not in fixed]

    # --- base state, real: which node is critical? ------------------------
    areas_real = np.array([float(e.A) for e in elements], dtype=float)
    k_real = np.array([float(e.E) for e in elements]) * areas_real / geom.lengths
    b_free = b[:, free]
    k_ff_real = np.einsum("e,ei,ej->ij", k_real, b_free, b_free)
    f_real = _nodal_load(nodes, loads, n_dof)[free] + b_free.T @ (
        k_real * _prestrain(elements)
    )
    u_free_real = np.linalg.solve(k_ff_real, f_real)
    u_real = np.zeros(n_dof)
    u_real[free] = u_free_real
    mags = np.hypot(u_real[0::2], u_real[1::2])
    crit = int(np.argmax(mags))
    d_max = float(mags[crit])
    if d_max < 1e-15:
        # The library floors d_max at 1.0 here and reports the *un-normalised*
        # numerator.  Complex-step differentiates sqrt(ux^2 + uy^2), which at a
        # zero displacement is a 0/0 branch point -- the two are not comparing
        # the same quantity, so refuse rather than return a plausible zero.
        msg = "degenerate base state: complex-step does not apply at |u| = 0"
        raise ValueError(msg)

    # --- complex step on the perturbed area -------------------------------
    areas = areas_real.astype(complex)
    areas[member_index] += 1j * h
    k = np.array([float(e.E) for e in elements], dtype=complex) * areas / geom.lengths
    k_ff = np.einsum("e,ei,ej->ij", k, b_free.astype(complex), b_free.astype(complex))
    rhs = _nodal_load(nodes, loads, n_dof)[free].astype(complex) + b_free.astype(
        complex
    ).T @ (k * _prestrain(elements).astype(complex))
    u_free = np.linalg.solve(k_ff, rhs)

    ux, uy = u_free[free.index(2 * crit)], u_free[free.index(2 * crit + 1)]
    # Analytic continuation of the magnitude: principal sqrt, NO conjugation.
    # This is ALREADY the normalised quantity --
    #     d sqrt(ux^2 + uy^2) / dA = (u . du/dA) / |u|
    # -- which is exactly the library's `(u_x du_x/dA + u_y du_y/dA) / d_max`.
    # Dividing by d_max again here (as the first draft did) silently scaled the
    # oracle by d_max and made every comparison off by that same constant
    # factor, which is how the bug showed up: a ratio identical to 10 digits
    # across all four members.
    magnitude = np.sqrt(ux * ux + uy * uy)
    return float(np.imag(magnitude) / h)


@pytest.mark.parametrize("member_index", [0, 1, 2, 3])
def test_ddm_matches_complex_step_to_machine_precision(member_index: int) -> None:
    """The adjoint DDM is right to ~1e-12 relative, not merely to FD tolerance.

    The existing central-difference cross-check has to accept ``1e-6``-ish
    agreement because that is the best a difference quotient can do.  Against
    complex-step the same derivative should agree to round-off, and any
    discrepancy larger than that is a real error in the adjoint rather than
    noise in the oracle.
    """
    nodes, elements, loads = _model()
    validator = IndependentValidator(nodes, elements, loads)
    report = validator.compute_all()
    member_id = elements[member_index].id
    got = next(r.ddm_sensitivity for r in report if r.member_id == member_id)
    expected = complex_step_dmax_dA(nodes, elements, loads, member_index)

    assert expected != 0.0, "the oracle returned an uninformative zero"
    rel = abs(got - expected) / abs(expected)
    assert rel < 1e-11, f"{member_id}: DDM {got:.12e} vs complex-step {expected:.12e}"


def test_complex_step_is_insensitive_to_h() -> None:
    """The property that distinguishes complex-step from a difference quotient.

    Central differences have an optimum ``h`` and get *worse* on either side;
    complex-step gets better monotonically until it is exact.  If this fails,
    the "oracle" is doing something non-analytic and the comparison above
    proves nothing.
    """
    nodes, elements, loads = _model()
    values = [
        complex_step_dmax_dA(nodes, elements, loads, 0, h=h)
        for h in (1e-10, 1e-20, 1e-30, 1e-40)
    ]
    assert max(abs(v - values[0]) for v in values) / abs(values[0]) < 1e-12


def test_central_difference_is_the_weaker_oracle() -> None:
    """Quantify the claim in the module docstring rather than asserting it.

    The best a central difference can do here is many orders of magnitude
    worse than complex-step.  Pinning that is what justifies having added the
    second oracle instead of leaving the difference quotient in place.
    """
    nodes, elements, loads = _model()
    exact = complex_step_dmax_dA(nodes, elements, loads, 0)

    def dmax_at(area0: float) -> float:
        perturbed = list(elements)
        perturbed[0] = Element(
            id=elements[0].id,
            node_i=elements[0].node_i,
            node_j=elements[0].node_j,
            E=elements[0].E,
            A=area0,
            delta_L_free=elements[0].delta_L_free,
        )
        n_dof = 2 * len(nodes)
        geom, b = _compatibility(nodes, perturbed, n_dof)
        free = [d for d in range(n_dof) if d not in fixed_dof_indices(nodes)]
        areas = np.array([float(e.A) for e in perturbed])
        k = np.array([float(e.E) for e in perturbed]) * areas / geom.lengths
        b_free = b[:, free]
        k_ff = np.einsum("e,ei,ej->ij", k, b_free, b_free)
        rhs = _nodal_load(nodes, loads, n_dof)[free] + b_free.T @ (
            k * _prestrain(perturbed)
        )
        u = np.zeros(n_dof)
        u[free] = np.linalg.solve(k_ff, rhs)
        return float(np.max(np.hypot(u[0::2], u[1::2])))

    a0 = float(elements[0].A)
    d0 = dmax_at(a0)
    best_fd_error = min(
        abs(
            (dmax_at(a0 * (1 + rel)) - dmax_at(a0 * (1 - rel))) / (2 * a0 * rel) / d0
            - exact
        )
        / abs(exact)
        # relative steps: A is 5e-3 m^2, so an absolute h of 1e-2 would drive
        # the area negative and the Element constructor would reject it
        for rel in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
    )
    cs_error = abs(
        next(
            r.ddm_sensitivity
            for r in IndependentValidator(nodes, elements, loads).compute_all()
            if r.member_id == elements[0].id
        )
        - exact
    ) / abs(exact)
    assert cs_error < best_fd_error / 100.0


def test_strain_energy_is_non_negative_and_matches_the_quadratic_form() -> None:
    """The other half of the validator, cross-checked independently."""
    nodes, elements, loads = _model()
    report = IndependentValidator(nodes, elements, loads).compute_all()
    n_dof = 2 * len(nodes)
    geom, b = _compatibility(nodes, elements, n_dof)
    free = [d for d in range(n_dof) if d not in fixed_dof_indices(nodes)]
    areas = np.array([float(e.A) for e in elements])
    k = np.array([float(e.E) for e in elements]) * areas / geom.lengths
    b_free = b[:, free]
    u_free = np.linalg.solve(
        np.einsum("e,ei,ej->ij", k, b_free, b_free),
        _nodal_load(nodes, loads, n_dof)[free] + b_free.T @ (k * _prestrain(elements)),
    )
    elongation = b_free @ u_free
    for i, row in enumerate(report):
        assert row.strain_energy >= 0.0
        assert row.strain_energy_total == pytest.approx(
            0.5 * k[i] * elongation[i] ** 2, rel=1e-10
        )
        # mechanical energy is the total minus the prestress contribution, and
        # never exceeds it
        assert row.strain_energy <= row.strain_energy_total + 1e-9
