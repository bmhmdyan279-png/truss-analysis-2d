"""Lemma 1 (uniform invariance) — measured, not assumed (prompt-04 §B7/B8).

In the linear-elastic model with pure stiffness degradation, a uniform
temperature field scales K by k_E(theta): the perturbation ratios (CI) must
therefore be temperature-invariant.  The pre-prompt-4 code *hard-coded* this
("ci_const = 1/alpha - 1"), so the lemma was never tested; here it is an
outcome of the numerical engine on all 21 campaign topologies x 5
temperatures, with tau measured at the lemma tolerance (see
``ranking.tau_b`` quantisation convention).

Also carries the anti-fake tests: providers that hard-code the uniform CI
(constant across members, or temperature-drifting) must be REJECTED by the
same checks the real engine passes.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import pytest
from truss_analysis.criticality import compute_ci_for_topology, tau_b

LEMMAS_TOL = 1e-10
THETAS = (200.0, 400.0, 600.0, 800.0, 1000.0)


def _ci_field(nodes, elements, loads, theta: float) -> Dict[str, float]:
    res = compute_ci_for_topology(nodes, elements, loads, {}, "uniform", theta)
    return res.ci_values


def _lemma_violations(provider: Callable[[float], Dict[str, float]]) -> List[str]:
    """Run the lemma checks against any CI-field provider (real or fake)."""
    violations: List[str] = []
    base = provider(20.0)
    for theta in THETAS:
        field = provider(theta)
        drift = max(abs(field[k] - base[k]) for k in base)
        if drift > LEMMAS_TOL:
            violations.append(f"ci drift {drift:.3e} at theta={theta}")
        res = tau_b(field, base, quantize=LEMMAS_TOL)
        if res.is_degenerate or res.tau is None or abs(res.tau - 1.0) > LEMMAS_TOL:
            violations.append(
                f"tau={res.tau} degenerate={res.is_degenerate} at {theta}"
            )
    return violations


def test_lemma1_uniform_invariance_campaign(campaign) -> None:
    for cm in campaign:
        base = _ci_field(cm.nodes, cm.elements, cm.loads, 20.0)
        for theta in THETAS:
            res = compute_ci_for_topology(
                cm.nodes, cm.elements, cm.loads, {}, "uniform", theta
            )
            drift = max(abs(res.ci_values[k] - base[k]) for k in base)
            assert drift < LEMMAS_TOL, (cm.name, theta, drift)
            assert res.tau_vs_base is not None, (cm.name, theta)
            assert abs(res.tau_vs_base - 1.0) < LEMMAS_TOL, (
                cm.name,
                theta,
                res.tau_vs_base,
            )


def test_lemma1_warren4_range_reproduces_context_lock(warren4_lemma) -> None:
    """CONTEXT_LOCK §4.3 measured CI range 0.086323 (depth ratio 0.1875)."""
    nodes, elements, loads = (
        warren4_lemma.nodes,
        warren4_lemma.elements,
        warren4_lemma.loads,
    )
    for theta in (20.0, 600.0):
        res = compute_ci_for_topology(nodes, elements, loads, {}, "uniform", theta)
        vals = res.ci_values.values()
        assert max(vals) - min(vals) == pytest.approx(0.084498937, abs=1e-6)
        assert min(vals) == pytest.approx(0.004053357, abs=1e-6)


def test_antifake_constant_provider_is_rejected(campaign) -> None:
    """The legacy hard-code (ci = 1/alpha - 1 for every member) must fail."""
    cm = next(c for c in campaign if c.name == "warren_6_H1")

    def fake(_theta: float) -> Dict[str, float]:
        return {e.id: 1.0 / 0.7 - 1.0 for e in cm.elements}

    assert _lemma_violations(fake), "constant fake CI must be rejected"


def test_antifake_temperature_drifting_provider_is_rejected(campaign) -> None:
    cm = next(c for c in campaign if c.name == "warren_6_H1")
    real_base = _ci_field(cm.nodes, cm.elements, cm.loads, 20.0)

    def fake(theta: float) -> Dict[str, float]:
        scale = 1.0 + 1e-6 * (theta - 20.0)  # drifts above the lemma tolerance
        return {k: v * scale for k, v in real_base.items()}

    assert _lemma_violations(fake), "drifting fake CI must be rejected"


def test_real_engine_passes_the_same_checks(campaign) -> None:
    cm = next(c for c in campaign if c.name == "warren_6_H1")
    violations = _lemma_violations(
        lambda theta: _ci_field(cm.nodes, cm.elements, cm.loads, theta)
    )
    assert violations == []
