"""Uniform-field invariance of the criticality index — measured, not assumed.

In the linear-elastic model with pure stiffness degradation, a uniform
temperature field scales the stiffness matrix by ``k_E(theta)``: the
perturbation ratios (CI) must therefore be temperature-invariant.  This
module verifies that invariance as an *outcome* of the numerical engine on
all 21 suite topologies x 5 temperatures, with Kendall's tau measured at
the quantisation tolerance (see the ``ranking.tau_b`` tie-noise
convention).

Also carries the anti-fake tests: providers that hard-code the uniform CI
(constant across members, or temperature-drifting) must be REJECTED by the
same checks the real engine passes.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from truss_analysis.criticality import compute_ci_for_topology, tau_b

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

UNIFORM_TOL = 1e-10
THETAS = (200.0, 400.0, 600.0, 800.0, 1000.0)


def _ci_field(nodes, elements, loads, theta: float) -> dict[str, float]:
    res = compute_ci_for_topology(nodes, elements, loads, {}, "uniform", theta)
    return res.ci_values


def _invariance_violations(
    provider: Callable[[float], dict[str, float]],
) -> list[str]:
    """Run the invariance checks against any CI-field provider (real or fake)."""
    violations: list[str] = []
    base = provider(20.0)
    for theta in THETAS:
        field = provider(theta)
        drift = max(abs(field[k] - base[k]) for k in base)
        if drift > UNIFORM_TOL:
            violations.append(f"ci drift {drift:.3e} at theta={theta}")
        res = tau_b(field, base, quantize=UNIFORM_TOL)
        if res.is_degenerate or res.tau is None or abs(res.tau - 1.0) > UNIFORM_TOL:
            violations.append(
                f"tau={res.tau} degenerate={res.is_degenerate} at {theta}"
            )
    return violations


def test_uniform_invariance_campaign(campaign) -> None:
    for cm in campaign:
        base = _ci_field(cm.nodes, cm.elements, cm.loads, 20.0)
        for theta in THETAS:
            res = compute_ci_for_topology(
                cm.nodes, cm.elements, cm.loads, {}, "uniform", theta
            )
            drift = max(abs(res.ci_values[k] - base[k]) for k in base)
            assert drift < UNIFORM_TOL, (cm.name, theta, drift)
            assert res.tau_vs_base is not None, (cm.name, theta)
            assert abs(res.tau_vs_base - 1.0) < UNIFORM_TOL, (
                cm.name,
                theta,
                res.tau_vs_base,
            )


def test_warren4_range_reproduces_reference_values(warren4_uniform) -> None:
    """Measured CI range for the Warren-4 reference geometry (depth ratio 0.1875)."""
    nodes, elements, loads = (
        warren4_uniform.nodes,
        warren4_uniform.elements,
        warren4_uniform.loads,
    )
    for theta in (20.0, 600.0):
        res = compute_ci_for_topology(nodes, elements, loads, {}, "uniform", theta)
        vals = res.ci_values.values()
        assert max(vals) - min(vals) == pytest.approx(0.084498937, abs=1e-6)
        assert min(vals) == pytest.approx(0.004053357, abs=1e-6)


def test_antifake_constant_provider_is_rejected(campaign) -> None:
    """A provider returning a constant CI (1/alpha - 1) must fail the checks."""
    cm = next(c for c in campaign if c.name == "warren_6_shallow")

    def fake(_theta: float) -> dict[str, float]:
        return {e.id: 1.0 / 0.7 - 1.0 for e in cm.elements}

    assert _invariance_violations(fake), "constant fake CI must be rejected"


def test_antifake_temperature_drifting_provider_is_rejected(campaign) -> None:
    cm = next(c for c in campaign if c.name == "warren_6_shallow")
    real_base = _ci_field(cm.nodes, cm.elements, cm.loads, 20.0)

    def fake(theta: float) -> dict[str, float]:
        scale = 1.0 + 1e-6 * (theta - 20.0)  # drifts above the tolerance
        return {k: v * scale for k, v in real_base.items()}

    assert _invariance_violations(fake), "drifting fake CI must be rejected"


def test_real_engine_passes_the_same_checks(campaign) -> None:
    cm = next(c for c in campaign if c.name == "warren_6_shallow")
    violations = _invariance_violations(
        lambda theta: _ci_field(cm.nodes, cm.elements, cm.loads, theta)
    )
    assert violations == []
