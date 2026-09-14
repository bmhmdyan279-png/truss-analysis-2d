"""Condition-scaled Sherman-Morrison guard tolerance.

Round-4 audit (critic 1, finding 4): the rank-1 denominator guard used a
fixed ``GUARD_TOL = 1e-8`` while the round-off level of
``denom = 1 + Delta_i d_i`` is ``~ cond(K_ff) * eps`` — at ``cond = 1e8``
that is ``2e-8``, already above the fixed threshold. A healthy but
ill-conditioned structure could then emit a silently inaccurate rank-1 CI
instead of routing the member to the exact brute-force solve. The tolerance
is now measured from the LU factors (LAPACK ``dgecon``) at build time:

    guard_tol = min(max(GUARD_TOL, GUARD_COND_COEF * cond * eps), GUARD_TOL_MAX)
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from truss_analysis.criticality.engine import (
    GUARD_COND_COEF,
    GUARD_TOL,
    GUARD_TOL_MAX,
    MechanismError,
    _solve_perturbed_full,
    base_displacement,
    build_engine,
    ci_sweep,
    total_load_vector,
)
from truss_analysis.model import Element, Node

E = 210e9


def _stiffness_spread_model(x_ratio: float, y_ratio: float):
    """One free node held by three orthogonal-ish bars.

    ``a`` (stiff, +x) and ``b`` (``x_ratio`` softer, -x) are collinear, so
    removing ``a`` leaves the x direction guarded only by ``b``:
    ``denom_a(alpha=0) = k_b / (k_a + k_b) ~ x_ratio``. ``c`` (``y_ratio``
    softer) sets the vertical stiffness and therefore ``cond(K_ff) ~
    1 / y_ratio``.
    """
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=2.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="4", x=1.0, y=1.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="3", x=1.0, y=0.0, is_support=False),
    ]
    elements = [
        Element(id="a", node_i="1", node_j="3", E=E, A=0.01),
        Element(id="b", node_i="2", node_j="3", E=E, A=0.01 * x_ratio),
        Element(id="c", node_i="4", node_j="3", E=E, A=0.01 * y_ratio),
    ]
    loads = {"3": {"Fx": 1.0e3, "Fy": 0.0}}
    temps = {e.id: 20.0 for e in elements}
    return nodes, elements, loads, temps


def _setup(x_ratio: float, y_ratio: float):
    nodes, elements, loads, temps = _stiffness_spread_model(x_ratio, y_ratio)
    setup = build_engine(nodes, elements, loads, temps)
    u = base_displacement(setup, total_load_vector(nodes, loads, setup))
    return nodes, elements, loads, temps, setup, u


def test_well_conditioned_setup_keeps_the_floor() -> None:
    """A normally conditioned model must not pay for the adaptive guard."""
    *_, setup, _u = _setup(x_ratio=0.5, y_ratio=0.5)
    assert setup.guard_tol == GUARD_TOL


def test_guard_scales_with_conditioning_and_is_capped() -> None:
    """cond ~ 1e10 -> tol grows past the floor but never past the cap."""
    *_, setup, _u = _setup(x_ratio=1e-3, y_ratio=1e-10)
    eps = float(np.finfo(float).eps)
    expected_uncapped = GUARD_COND_COEF * 1e10 * eps  # ~2.2e-4
    assert setup.guard_tol > GUARD_TOL
    assert setup.guard_tol <= GUARD_TOL_MAX
    # the cap is what binds at this conditioning, and it binds only because
    # the uncapped value would over-route
    assert expected_uncapped > GUARD_TOL_MAX


def test_ill_conditioned_healthy_member_routes_to_brute_force() -> None:
    """The critic's scenario: denom is real but below the round-off band.

    ``denom_a(alpha=0) ~ 1e-7`` (member ``a`` is *not* a mechanism: ``b``
    still holds the x direction), while ``cond(K_ff) ~ 1e8`` puts the
    round-off of the computed denom at ``~2e-8`` — a 20% relative error the
    fixed ``1e-8`` threshold waved through. The condition-scaled tolerance
    (``~2e-6``) routes the member to the exact solve instead.
    """
    nodes, elements, loads, temps, setup, u = _setup(x_ratio=1e-7, y_ratio=1e-8)
    assert setup.guard_tol > 1e-7 > GUARD_TOL

    brute = partial(_solve_perturbed_full, nodes, elements, loads, temps, 0.0)
    sweep = ci_sweep(setup, u, 0.0, brute_column=brute)

    assert sweep.flagged.get("a") == "guard:brute-force"
    # the routed column IS the exact solve, so the emitted CI is trustworthy
    i_a = [e.id for e in elements].index("a")
    assert np.array_equal(sweep.u_pert[:, i_a], brute(i_a))
    assert np.isfinite(sweep.ci_values["a"])


def test_fixed_tol_override_keeps_the_old_routing() -> None:
    """An explicit ``guard_tol`` still overrides the adaptive value.

    With the pre-2.7 fixed threshold the same member is NOT flagged: the
    rank-1 formula divides by a denominator whose relative round-off is
    O(20%) — silently inaccurate, which is exactly what the adaptive tol
    exists to prevent.
    """
    nodes, elements, loads, temps, setup, u = _setup(x_ratio=1e-7, y_ratio=1e-8)
    brute = partial(_solve_perturbed_full, nodes, elements, loads, temps, 0.0)
    sweep_old = ci_sweep(setup, u, 0.0, guard_tol=GUARD_TOL, brute_column=brute)
    assert "a" not in sweep_old.flagged
    sweep_new = ci_sweep(setup, u, 0.0, brute_column=brute)
    assert "a" in sweep_new.flagged


def test_no_fallback_on_guarded_member_still_raises() -> None:
    *_, setup, u = _setup(x_ratio=1e-7, y_ratio=1e-8)
    with pytest.raises(MechanismError):
        ci_sweep(setup, u, 0.0, brute_column=None)
