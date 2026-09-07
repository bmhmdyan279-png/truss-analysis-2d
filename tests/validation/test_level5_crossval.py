"""Level 5 — cross-family transfer of a member-level CI surrogate.

Operational definition (fixed BEFORE measurement)
-------------------------------------------------
The screening framework is parameter-free mechanics — there is no fitted
model inside it — so "train on family A, test on families B and C" is
defined as the transfer of a *surrogate*: a ridge regression mapping
statics+geometry features of a member to its exact criticality index.

* samples  = (member, state) pairs; a state = (topology, local scenario,
  temperature); target = exact displacement CI (alpha = 0.7);
* features = one baseline solve (force ratio, sign) + length, orientation,
  centroid position, connectivity, heated flag, member temperature.  NO
  information from the perturbation sweep enters the features;
* train    = every Pratt-family sample (all spans, depths, scenarios,
  temperatures); test = every Warren and Howe sample;
* metric   = per-state Spearman rho between predicted and exact CI ranking;
  aggregate = mean over the 144 held-out states; gate: mean rho > 0.85.

Uniform states are excluded by design: Lemma 1 makes their CI field equal
to the ambient one, so they would add duplicated targets under conflicting
feature values instead of information.

Controls carried in this file: the force-ratio-only baseline (surrogate must
beat a single static feature), a permutation control (shuffled training
targets must collapse to rho ~ 0), and determinism (refitting reproduces the
result bit-for-bit).
"""

from __future__ import annotations

import numpy as np
import pytest
from truss_analysis.validation import (
    CV_RHO_GATE,
    FEATURES,
    RidgeSurrogate,
    cross_family_cv,
)

PERMUTATION_SEED = 20260908


def _topology_tuples(campaign):
    out = []
    for cm in campaign:
        family = cm.name.split("_")[0]
        out.append((cm.name, family, cm.nodes, cm.elements, cm.loads))
    return out


@pytest.fixture(scope="module")
def cv_result(campaign):
    return cross_family_cv(_topology_tuples(campaign))


def test_cv_gate_mean_rho(cv_result):
    """Primary acceptance: mean per-state rank rho > 0.85 on held-out families."""
    assert CV_RHO_GATE == 0.85
    assert cv_result.passed
    assert cv_result.rho_mean > CV_RHO_GATE
    assert len(cv_result.rows) == 144  # 12 topologies x 3 scenarios x 4 temps
    assert cv_result.n_train_samples > 1000
    assert cv_result.features == FEATURES


def test_cv_per_family_and_dispersion(cv_result):
    for family in ("warren", "howe"):
        assert cv_result.rho_by_family[family] > CV_RHO_GATE
    assert cv_result.rho_median > CV_RHO_GATE
    assert cv_result.rho_min > 0.6  # reported dispersion, worst state
    assert all(np.isfinite(r.rho) for r in cv_result.rows)


def test_cv_beats_single_feature_baseline(cv_result):
    """The fitted surrogate must add value over force-ratio-only ranking."""
    assert cv_result.rho_mean_force_baseline < cv_result.rho_mean
    assert cv_result.rho_mean_force_baseline < CV_RHO_GATE  # baseline alone fails


def test_cv_permutation_control_collapses(campaign):
    """Shuffled training targets: mean rho must fall to ~0 (metric sanity)."""
    res = cross_family_cv(
        _topology_tuples(campaign), shuffle_targets_seed=PERMUTATION_SEED
    )
    assert res.rho_mean < 0.2
    assert not res.passed


def test_cv_is_deterministic(campaign):
    a = cross_family_cv(_topology_tuples(campaign))
    b = cross_family_cv(_topology_tuples(campaign))
    assert a.rho_mean == b.rho_mean
    assert [r.rho for r in a.rows] == [r.rho for r in b.rows]


def test_ridge_recovers_exact_linear_relation():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200, 4))
    w = np.asarray([1.5, -2.0, 0.25, 4.0])
    y = x @ w + 0.5
    model = RidgeSurrogate(alpha=1e-8).fit(x, y)
    pred = model.predict(x)
    assert np.max(np.abs(pred - y)) < 1e-8


def test_cv_requires_both_train_and_test_families(campaign):
    tops = _topology_tuples(campaign)
    with pytest.raises(ValueError):  # no training family present
        cross_family_cv([t for t in tops if t[1] == "warren"])
    with pytest.raises(ValueError):  # no test family present
        cross_family_cv([t for t in tops if t[1] == "pratt"])


def test_ridge_input_guards():
    model = RidgeSurrogate()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        model.fit(np.zeros((3, 2)), np.zeros(4))  # shape mismatch
    with pytest.raises(ValueError):
        model.fit(np.zeros((1, 2)), np.zeros(1))  # too few samples
