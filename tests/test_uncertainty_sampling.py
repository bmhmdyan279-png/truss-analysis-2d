"""Uncertainty layer tests: LHS, copula, seeds, streaming, specs."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from truss_analysis.uncertainty import (
    DeterministicRV,
    GumbelRV,
    LognormalRV,
    RunningStat,
    TruncatedNormalRV,
    default_rv_specs,
    gaussian_copula_correlate,
    latin_hypercube,
    sample_spec_matrix,
)


def test_lhs_stratification_exact() -> None:
    u = latin_hypercube(100, 4, seed=7)
    assert u.shape == (100, 4)
    for d in range(4):
        strata = np.floor(u[:, d] * 100).astype(int)
        assert sorted(strata.tolist()) == list(range(100))
    assert np.all(u >= 0.0)
    assert np.all(u < 1.0)


def test_lhs_deterministic_and_order_independent() -> None:
    a = latin_hypercube(50, 3, seed=11)
    b = latin_hypercube(50, 3, seed=11)
    assert np.array_equal(a, b)
    c = latin_hypercube(50, 3, seed=12)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("target", [0.3, 0.7, 0.9, -0.6])
def test_gaussian_copula_realises_target_rank_correlation(target: float) -> None:
    """The realised Spearman rho reproduces the target, not its Kruskal image.

    Pre-2.7 the target matrix went straight into the Cholesky, so the output
    rank correlation was ``(6/pi) arcsin(target/2)`` -- 0.682 instead of 0.700
    at target 0.7. The tolerance is now tight enough that the distortion
    cannot come back: at 0.7 the old behaviour missed by ~0.018.
    """
    corr = np.array([[1.0, target], [target, 1.0]])
    u = latin_hypercube(40000, 2, seed=3)
    uc = gaussian_copula_correlate(u, corr)
    rho = float(spearmanr(uc[:, 0], uc[:, 1]).statistic)
    assert abs(rho - target) < 0.005, (target, rho)


def test_gaussian_copula_dimension_mismatch_and_indefinite_target() -> None:
    corr = np.array([[1.0, 0.7], [0.7, 1.0]])
    with pytest.raises(ValueError, match="columns"):
        gaussian_copula_correlate(np.zeros((10, 3)), corr)
    # not positive definite -> the Kruskal-mapped Cholesky fails and is
    # re-raised as a ValueError naming the mapping
    bad = np.array([[1.0, 0.95, 0.95], [0.95, 1.0, -0.95], [0.95, -0.95, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        gaussian_copula_correlate(np.zeros((10, 3)), bad)


def test_gaussian_copula_three_by_three_rank_correlation_matrix() -> None:
    target = np.array(
        [
            [1.00, 0.60, -0.40],
            [0.60, 1.00, 0.30],
            [-0.40, 0.30, 1.00],
        ]
    )
    u = latin_hypercube(60000, 3, seed=7)
    uc = gaussian_copula_correlate(u, target)
    measured = spearmanr(uc).statistic
    assert measured.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            assert abs(float(measured[i, j]) - target[i, j]) < 0.006, (i, j)


def test_gaussian_copula_preserves_uniform_marginals() -> None:
    corr = np.array([[1.0, 0.7], [0.7, 1.0]])
    u = latin_hypercube(20000, 2, seed=5)
    uc = gaussian_copula_correlate(u, corr)
    assert np.all(uc > 0.0)
    assert np.all(uc < 1.0)
    for d in range(2):
        assert abs(float(uc[:, d].mean()) - 0.5) < 0.01


def test_truncated_normal_respects_bounds() -> None:
    rv = TruncatedNormalRV(mean=600.0, std=50.0, low=20.0, high=1000.0, seed=5)
    x = rv.sample(5000)
    assert np.all(x >= 20.0)
    assert np.all(x <= 1000.0)
    assert abs(float(x.mean()) - 600.0) < 5.0
    with pytest.raises(ValueError, match="low < high"):
        TruncatedNormalRV(mean=1.0, std=1.0, low=5.0, high=1.0)


def test_deterministic_rv_constant() -> None:
    rv = DeterministicRV(210.0e9)
    assert np.all(rv.sample(10) == 210.0e9)


def test_legacy_gumbel_moments() -> None:
    rv = GumbelRV(mean=1.0, cov=0.2, seed=42)
    x = rv.sample(200000)
    assert abs(float(x.mean()) - 1.0) < 0.01
    assert abs(float(x.std()) - 0.2) < 0.01


def test_legacy_lognormal_moments() -> None:
    rv = LognormalRV(mean=235.0e6, cov=0.05, seed=43)
    x = rv.sample(200000)
    assert abs(float(x.mean()) - 235.0e6) / 235.0e6 < 0.01


def test_specs_citation_statuses_recorded() -> None:
    specs = default_rv_specs()
    names = [s.name for s in specs]
    assert names == ["live_load", "f_y", "fire_intensity", "E"]
    for s in specs:
        assert s.citation_status  # non-empty status string
        assert s.citation  # citation text present (verified or downgraded)
    e = next(s for s in specs if s.name == "E")
    assert "210 000" in e.citation  # corrected value, verified verbatim
    live = next(s for s in specs if s.name == "live_load")
    assert "NOT verbatim" in live.citation  # honest downgrade


def test_sample_spec_matrix_deterministic() -> None:
    specs = default_rv_specs(fire_scenario_temperature=600.0)
    means = {"live_load": 1.0, "f_y": 235.0e6, "fire_intensity": 600.0, "E": 210.0e9}
    a = sample_spec_matrix(specs, means, 200, seed=99)
    b = sample_spec_matrix(specs, means, 200, seed=99)
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert np.all(a["E"] == 210.0e9)
    assert np.all(a["fire_intensity"] >= 20.0)
    assert np.all(a["fire_intensity"] <= 1000.0)


def test_running_stat_matches_batch_stats() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(3.0, 0.5, 4000)
    rs = RunningStat()
    for i in range(0, 4000, 200):
        rs.add_batch(x[i : i + 200])
    assert rs.count == 4000
    assert rs.mean == pytest.approx(float(x.mean()), rel=1e-9)
    assert rs.std == pytest.approx(float(x.std(ddof=1)), rel=1e-6)
    # different batch order -> same statistics up to fp noise
    rs2 = RunningStat()
    perm = rng.permutation(4000)
    for i in range(0, 4000, 500):
        rs2.add_batch(x[perm[i : i + 500]])
    assert rs2.mean == pytest.approx(rs.mean, rel=1e-9)
    assert rs2.std == pytest.approx(rs.std, rel=1e-6)


def test_latin_hypercube_rejects_degenerate_shapes() -> None:
    with pytest.raises(ValueError, match="require n,dim >= 1"):
        latin_hypercube(0, 2, seed=1)
    with pytest.raises(ValueError, match="require n,dim >= 1"):
        latin_hypercube(10, 0, seed=1)


def test_sample_spec_matrix_honours_correlation_and_deterministic_specs() -> None:
    """The spec-level coupling path: correlation + a deterministic member."""
    specs = default_rv_specs(fire_scenario_temperature=400.0)
    names = [s.name for s in specs]
    means = {"live_load": 1.0, "f_y": 235e6, "fire_intensity": 400.0, "E": 210e9}
    corr = np.eye(len(specs))
    i_ll, i_fy = names.index("live_load"), names.index("f_y")
    corr[i_ll, i_fy] = corr[i_fy, i_ll] = 0.6
    out = sample_spec_matrix(specs, means, 20000, seed=17, correlation=corr)
    # deterministic specs ignore the coupling and return their spec value
    det = next(s for s in specs if s.family == "deterministic")
    assert np.all(out[det.name] == det.parameters["value"])
    # the correlated pair realises its target rank correlation
    rho = float(spearmanr(out["live_load"], out["f_y"]).statistic)
    assert abs(rho - 0.6) < 0.02, rho
