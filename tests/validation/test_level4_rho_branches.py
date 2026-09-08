"""Level 4 (part 1) — the rho decision tree and rank-correlation utility.

These tests run WITHOUT OpenSeesPy: the three-branch verdict (full
validation / cause analysis / model review) and the tie-noise quantisation
convention are pure functions and must be exercised even where the optional
reference solver is absent.  The branches are pre-coded so that a low rho
has a documented path instead of an ad-hoc reaction.
"""

from __future__ import annotations

import math

from truss_analysis.validation import (
    RHO_FLOOR,
    RHO_FULL,
    RhoVerdict,
    classify_rho,
    rank_correlation,
)


def test_full_validation_branch():
    for rho in (0.90, 0.95, 1.0):
        branch = classify_rho(rho)
        assert branch.verdict is RhoVerdict.FULL_VALIDATION
        assert branch.rho == rho
        assert branch.prescribed_action


def test_cause_analysis_branch():
    for rho in (0.70, 0.80, 0.8999999):
        branch = classify_rho(rho)
        assert branch.verdict is RhoVerdict.CAUSE_ANALYSIS
        assert branch.prescribed_action


def test_model_review_branch():
    for rho in (0.6999999, 0.5, 0.0, -1.0):
        branch = classify_rho(rho)
        assert branch.verdict is RhoVerdict.MODEL_REVIEW
        assert branch.prescribed_action


def test_nan_is_never_promoted_to_a_pass():
    branch = classify_rho(float("nan"))
    assert branch.verdict is RhoVerdict.MODEL_REVIEW
    assert math.isnan(branch.rho)


def test_thresholds_are_ordered():
    assert 0.0 < RHO_FLOOR < RHO_FULL <= 1.0


def test_rank_correlation_identical_and_reversed():
    field = {"1": 0.1, "2": 0.5, "3": 0.9, "10": 0.3}
    assert rank_correlation(field, dict(field)) == 1.0
    rev = {k: -v for k, v in field.items()}
    assert rank_correlation(field, rev) == -1.0


def test_rank_correlation_degenerate_is_nan():
    const = {"1": 0.4, "2": 0.4, "3": 0.4}
    varied = {"1": 0.1, "2": 0.2, "3": 0.3}
    assert math.isnan(rank_correlation(const, varied))
    assert math.isnan(rank_correlation(varied, const))
    assert math.isnan(rank_correlation({"1": 0.1}, {"1": 0.2}))  # <2 keys


def test_quantisation_absorbs_sub_tolerance_tie_noise():
    """Two exact solvers agree to ~1e-13; symmetric near-tied pairs would
    otherwise flip ranks arbitrarily.  Quantising at 1e-10 (the tie-noise
    convention) makes 'ranks equal up to tolerance' the measured quantity."""
    a = {"1": 0.5, "2": 0.3, "3": 0.5 + 1e-13}
    b = {"1": 0.5 + 1e-13, "2": 0.3, "3": 0.5}
    assert rank_correlation(a, b, quantize=1e-10) == 1.0
    assert rank_correlation(a, b) < 1.0  # raw spearman sees the noise


def test_quantisation_keeps_real_rank_differences():
    a = {"1": 0.5, "2": 0.3, "3": 0.1}
    b = {"1": 0.1, "2": 0.3, "3": 0.5}
    assert rank_correlation(a, b, quantize=1e-10) == -1.0


def test_quantisation_can_collapse_to_degenerate():
    """If quantising makes either field constant, the correlation is
    undefined (nan) — the same never-a-silent-number policy."""
    a = {"1": 0.5, "2": 0.5 + 1e-13}  # distinct floats, equal after quantise
    b = {"1": 0.4, "2": 0.6}  # non-constant
    # raw: perfectly rank-correlated (scipy's 2-point spearman rounds to
    # 1 - 1e-16, hence approx)
    assert abs(rank_correlation(a, b) - 1.0) < 1e-12
    assert math.isnan(rank_correlation(a, b, quantize=1e-10))  # collapsed
