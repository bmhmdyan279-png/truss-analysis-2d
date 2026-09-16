"""Level 3 — physical validation: uniform-field invariance measured on the full grid.

Uniform-temperature invariance of the criticality field, measured (never
assumed, never hard-coded) on **all 21 suite topologies x 5 temperatures**
(200/400/600/800/1000 degC), with three nested criteria of increasing
strength:

1. Kendall tau-b between the CI ranking at theta and at 20 degC equals
   1.000 within 1e-8 (the acceptance tolerance of the protocol).
2. Element-wise CI drift below 1e-10 (the tie-noise quantisation convention).
3. Element-wise ``allclose`` at rtol=1e-8, atol=1e-10 — the strongest
   practical form: the whole CI VECTOR, not just its ranks, is invariant.

Plus the scaling law behind the invariance: under a uniform field the
stiffness matrix scales by k_E(theta), hence
``u_max(theta) / u_max(20) == 1 / k_E(theta)`` to machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from truss_analysis.criticality import T_AMBIENT, compute_ci_for_topology
from truss_analysis.material.steel_eurocode import k_E

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

THETAS = (200.0, 400.0, 600.0, 800.0, 1000.0)
TAU_TOL = 1e-8  # protocol acceptance: tau = 1.000 +/- 1e-8
DRIFT_TOL = 1e-10  # tie-noise quantisation convention
N_TOPOLOGIES = 21


def _uniform_state(cm, theta):
    return compute_ci_for_topology(
        cm.nodes, cm.elements, cm.loads, {}, "uniform", theta
    )


def test_uniform_field_full_grid_tau_drift_and_vector_invariance(campaign) -> None:
    assert len(campaign) == N_TOPOLOGIES
    measured = 0
    for cm in campaign:
        base = _uniform_state(cm, T_AMBIENT)
        for theta in THETAS:
            res = _uniform_state(cm, theta)
            measured += 1
            # 1. rank invariance
            assert res.tau_vs_base is not None, (cm.name, theta)
            assert abs(res.tau_vs_base - 1.0) <= TAU_TOL, (
                cm.name,
                theta,
                res.tau_vs_base,
            )
            # 2. element-wise drift
            drift = max(
                abs(res.ci_values[k] - base.ci_values[k]) for k in base.ci_values
            )
            assert drift < DRIFT_TOL, (cm.name, theta, drift)
            # 3. full-vector invariance (strongest form)
            assert np.allclose(
                np.asarray([res.ci_values[k] for k in sorted(base.ci_values)]),
                np.asarray([base.ci_values[k] for k in sorted(base.ci_values)]),
                rtol=1e-8,
                atol=1e-10,
            ), (cm.name, theta)
            # scaling law: u_max(theta) = u_max(20) / k_E(theta)
            assert res.u_max_base == pytest.approx(
                base.u_max_base / k_E(theta), rel=1e-12
            ), (cm.name, theta)
    assert measured == N_TOPOLOGIES * len(THETAS)


def test_uniform_field_is_not_vacuous_cold_equals_hot_ranking_content(campaign) -> None:
    """Guard against a vacuous pass: the CI fields being compared must be
    informative (non-constant) — a constant field would make tau degenerate
    and the engine reports that as tau=None, never as 1.0."""
    cm = next(c for c in campaign if c.name == "warren_8_deep")
    res = _uniform_state(cm, 600.0)
    values = list(res.ci_values.values())
    assert max(values) - min(values) > 1e-6  # real spread, not all-equal
    assert res.tau_vs_base == pytest.approx(1.0, abs=TAU_TOL)
