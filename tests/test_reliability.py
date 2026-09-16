"""Tests for the Monte Carlo reliability engine."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from unittest.mock import MagicMock

import numpy as np
import pytest

from truss_analysis.reliability import (
    AnalysisSample,
    Direction,
    LimitState,
    MarginStatistics,
    MemberResponse,
    ReliabilityEngine,
    ServiceLimit,
    _beta_mom,
    _pf_from_beta,
    _statistics,
    sample_named_variables,
)
from truss_analysis.solver import solve
from truss_analysis.uncertainty import LognormalRV

# The reliability engine is exercised against both buckling models, including the
# legacy EULER_ONLY path whose margins must stay bit-for-bit reproducible. The
# warning that path emits is asserted in the tests selecting it deliberately.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.LegacyBucklingModelWarning"
    ),
]

_E = 200.0e9
_A = 1.0e-3
_LENGTH = 1.0
_FY = 150.0e6
_LOAD_MEAN = 100.0e3
_LOAD_COV = 0.25

_CAPACITY = _FY * _A
_LOAD_STD = _LOAD_MEAN * _LOAD_COV
_MARGIN_MEAN = _CAPACITY - _LOAD_MEAN
_BETA_THEORY = _MARGIN_MEAN / _LOAD_STD


def _single_bar_analyzer(
    E: float, A: float, length: float, yield_stress: float
) -> Callable[[Mapping[str, float]], AnalysisSample]:
    def analyze(sample: Mapping[str, float]) -> AnalysisSample:
        force = float(sample["F"])
        K = np.zeros((4, 4), dtype=np.float64)
        axial_stiffness = E * A / length
        K[2, 2] = axial_stiffness

        F = np.zeros(4, dtype=np.float64)
        F[2] = force

        U = solve(K, F, [0, 1, 3])
        axial_force = float(axial_stiffness * U[2])

        member = MemberResponse(
            axial_force=axial_force,
            E=E,
            A=A,
            I_sec=1.0,
            length=length,
            effective_length_factor=1.0,
            yield_stress=yield_stress,
        )

        return AnalysisSample(
            member_responses={1: member},
            nodal_displacements={2: (float(U[2]), 0.0)},
        )

    return analyze


def _empty_analysis(_: Mapping[str, float]) -> AnalysisSample:
    return AnalysisSample(member_responses={}, nodal_displacements={})


def test_single_member_analytical_yield_beta() -> None:
    variables = {"F": LognormalRV(mean=_LOAD_MEAN, cov=_LOAD_COV, seed=1234)}
    engine = ReliabilityEngine(
        variables=variables,
        analyze_fn=_single_bar_analyzer(_E, _A, _LENGTH, _FY),
    )

    report = engine.run(n_samples=10_000)
    stat = report.get(LimitState.YIELD, 1)

    assert stat is not None
    assert stat.sample_size == 10_000
    assert stat.valid_samples == 10_000
    assert np.isfinite(stat.beta_mom)

    assert abs(stat.mean - _MARGIN_MEAN) <= 0.10 * abs(_MARGIN_MEAN)
    assert abs(stat.std - _LOAD_STD) <= 0.10 * _LOAD_STD
    assert abs(stat.beta_mom - _BETA_THEORY) <= 0.20 * abs(_BETA_THEORY)
    assert 0.0 <= stat.pf_approx <= 1.0


def test_buckling_is_not_counted_for_tension() -> None:
    variables = {"F": LognormalRV(mean=_LOAD_MEAN, cov=_LOAD_COV, seed=2233)}
    engine = ReliabilityEngine(
        variables=variables,
        analyze_fn=_single_bar_analyzer(_E, _A, _LENGTH, _FY),
    )

    report = engine.run(n_samples=500)
    stat = report.get(LimitState.BUCKLING, 1)

    assert stat is not None
    assert stat.valid_samples == 0
    assert np.isnan(stat.beta_mom)
    assert np.isnan(stat.pf_approx)


def test_convergence_reports_use_requested_sample_sizes() -> None:
    variables = {"F": LognormalRV(mean=_LOAD_MEAN, cov=_LOAD_COV, seed=999)}
    engine = ReliabilityEngine(
        variables=variables,
        analyze_fn=_single_bar_analyzer(_E, _A, _LENGTH, _FY),
    )

    sizes = (200, 1_000, 5_000)
    reports = engine.run_convergence(sizes)

    assert set(reports.keys()) == set(sizes)
    for size in sizes:
        stat = reports[size].get(LimitState.YIELD, 1)
        assert stat is not None
        assert stat.sample_size == size

    largest = reports[max(sizes)].get(LimitState.YIELD, 1)
    assert largest is not None
    assert abs(largest.beta_mom - _BETA_THEORY) <= 0.30 * abs(_BETA_THEORY)


def test_serviceability_limit_state_is_evaluated() -> None:
    variables = {"F": LognormalRV(mean=_LOAD_MEAN, cov=_LOAD_COV, seed=777)}
    service_limits = (ServiceLimit(node_id=2, direction=Direction.X, limit=1.0e-3),)

    engine = ReliabilityEngine(
        variables=variables,
        analyze_fn=_single_bar_analyzer(_E, _A, _LENGTH, _FY),
        service_limits=service_limits,
    )

    report = engine.run(n_samples=2_000)
    stat = report.get(LimitState.SERVICEABILITY, "node_2_x")

    assert stat is not None
    assert stat.valid_samples == 2_000
    assert np.isfinite(stat.beta_mom)


def test_service_limit_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ServiceLimit(node_id=2, direction=Direction.X, limit=-1.0)


def test_engine_rejects_nonpositive_sample_size() -> None:
    engine = ReliabilityEngine(variables={}, analyze_fn=_empty_analysis)
    with pytest.raises(ValueError, match="must be positive"):
        engine.run(0)


# --- Edge Case Tests for 100% Coverage & Robustness ---


def test_run_convergence_rejects_empty_sizes() -> None:
    engine = ReliabilityEngine(variables={}, analyze_fn=_empty_analysis)
    with pytest.raises(ValueError, match="sample_sizes must not be empty"):
        engine.run_convergence([])


def test_sample_named_variables_rejects_invalid_shape() -> None:
    bad_rv = MagicMock()
    bad_rv.sample.return_value = np.array([[1.0, 2.0]])  # 2D instead of 1D
    with pytest.raises(ValueError, match="invalid sample shape"):
        sample_named_variables({"bad": bad_rv}, 2)


def test_beta_hat_is_a_deprecated_alias_of_beta_mom() -> None:
    """C8: the rename must warn, not silently change meaning.

    ``beta_hat`` read as a Hasofer-Lind reliability index; the quantity is a
    method-of-moments ``mean/std``.  The alias keeps old call sites working
    for one release while telling them the name was wrong.
    """
    stat = _stat_for_deprecation_alias()
    with pytest.warns(DeprecationWarning, match="renamed to beta_mom"):
        legacy = stat.beta_hat
    assert legacy == stat.beta_mom
    # and the new name does not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert stat.beta_mom == legacy


def _stat_for_deprecation_alias() -> MarginStatistics:
    """A minimal valid ``MarginStatistics`` for the alias test."""
    return _statistics(LimitState.YIELD, "m1", np.array([2.0, 1.0, 3.0, 0.5, -0.5]))


def test_beta_mom_and_pf_edge_cases() -> None:
    assert _beta_mom(10.0, 0.0) == float("inf")
    assert _beta_mom(-10.0, 0.0) == float("-inf")
    assert np.isnan(_beta_mom(0.0, 0.0))
    assert np.isnan(_beta_mom(float("nan"), 1.0))

    assert _pf_from_beta(float("inf")) == 0.0
    assert _pf_from_beta(float("-inf")) == 1.0
    assert np.isnan(_pf_from_beta(float("nan")))


def test_service_margin_missing_node() -> None:
    def analyze(_: Mapping[str, float]) -> AnalysisSample:
        return AnalysisSample(member_responses={}, nodal_displacements={})

    engine = ReliabilityEngine(
        variables={"dummy": LognormalRV(mean=1.0, cov=0.1, seed=1)},
        analyze_fn=analyze,
        service_limits=(ServiceLimit(node_id=99, direction=Direction.X, limit=1.0),),
    )
    report = engine.run(1)
    stat = report.get(LimitState.SERVICEABILITY, "node_99_x")

    assert stat is not None
    assert stat.valid_samples == 0
    assert np.isnan(stat.beta_mom)


def test_margin_alignment_when_member_disappears() -> None:
    call_count = 0

    def analyze(_: Mapping[str, float]) -> AnalysisSample:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            member = MemberResponse(
                axial_force=1.0,
                E=2e11,
                A=1e-3,
                I_sec=1e-9,
                length=1.0,
                effective_length_factor=1.0,
                yield_stress=1e8,
            )
            return AnalysisSample({1: member}, {})
        return AnalysisSample({}, {})

    engine = ReliabilityEngine(
        variables={"dummy": LognormalRV(mean=1.0, cov=0.1, seed=1)},
        analyze_fn=analyze,
    )
    report = engine.run(2)
    stat = report.get(LimitState.YIELD, 1)

    assert stat is not None
    assert stat.sample_size == 2
    assert stat.valid_samples == 1


# --- Round-5 audit pins: chi-aligned buckling margin, empirical pf ---------


def _member(N=-1.5e5, E=210e9, fy=355e6, temp=20.0, **kw) -> MemberResponse:
    base = dict(
        axial_force=N,
        E=E,
        A=2e-3,
        I_sec=8e-6,
        length=3.0,
        effective_length_factor=0.8,
        yield_stress=fy,
        temperature=temp,
    )
    base.update(kw)
    return MemberResponse(**base)


def _engine(**kw) -> ReliabilityEngine:
    return ReliabilityEngine(variables={}, analyze_fn=lambda s: None, **kw)


def test_buckling_margin_matches_limitstates_capacity_at_fire_temperature():
    """One capacity model for the whole library (round-5 audit, C8-1).

    The reliability buckling margin plus |N| must equal exactly the
    capacity the EN 1993-1-2 limit-state layer computes for the same
    physical member state -- pre-2.8 the margin was bare Euler while the
    DCR chain used chi, so beta_buckling was systematically optimistic
    (6x on the reference member at lambda_bar ~ 0.5).
    """
    import math as _math

    from truss_analysis.limitstates import _member_limit_state
    from truss_analysis.material.steel_eurocode import k_E, k_y

    area, inertia, length, k_fac = 2e-3, 8e-6, 3.0, 0.8
    E0, fy0, theta, N = 210e9, 355e6, 600.0, -1.5e5
    state = _member_limit_state("m", theta, N, area, inertia, length, E0, k_fac, fy0)
    member = _member(
        N=N,
        E=float(k_E(theta)) * E0,
        fy=float(k_y(theta)) * fy0,
        temp=theta,
        A=area,
        I_sec=inertia,
        length=length,
        effective_length_factor=k_fac,
    )
    margin = _engine()._buckling_margin(member)
    assert _math.isclose(margin + abs(N), state.capacity, rel_tol=1e-12)
    # And the old Euler margin really was optimistic on this member.
    from truss_analysis.limitstates import BucklingModel

    euler_margin = _engine(buckling_model=BucklingModel.EULER_ONLY)._buckling_margin(
        member
    )
    assert euler_margin > margin


def test_buckling_margin_euler_only_is_bit_for_bit_legacy():
    import math as _math

    from truss_analysis.limitstates import BucklingModel

    m = _member()
    expected = np.pi**2 * m.E * m.I_sec / (
        m.effective_length_factor * m.length
    ) ** 2 - abs(m.axial_force)
    got = _engine(buckling_model=BucklingModel.EULER_ONLY)._buckling_margin(m)
    assert _math.isclose(got, expected, rel_tol=1e-15, abs_tol=0.0)


def test_buckling_margin_ambient_uses_ambient_curve():
    """At 20 degC the EN 1993-1-1 curve applies (full imperfection alpha)."""
    import math as _math

    from truss_analysis.sections import (
        buckling_reduction_factor,
        euler_buckling_load,
        non_dimensional_slenderness,
    )

    m = _member(temp=20.0)
    p_cr = euler_buckling_load(m.I_sec, m.length, m.E, m.effective_length_factor)
    lam = non_dimensional_slenderness(m.A, m.yield_stress, p_cr)
    chi_amb = buckling_reduction_factor(lam, "c", fire=False)
    chi_fire = buckling_reduction_factor(lam, "c", fire=True)
    assert chi_amb != chi_fire  # the regime selection is observable
    expected = chi_amb * m.A * m.yield_stress - abs(m.axial_force)
    assert _math.isclose(_engine()._buckling_margin(m), expected, rel_tol=1e-12)


def test_buckling_margin_requires_yield_stress_under_chi():
    assert np.isnan(_engine()._buckling_margin(_member(fy=None)))


def test_pf_empirical_counts_failures_and_carries_exact_ci():
    from truss_analysis.reliability import _statistics

    margins = np.array([1.0, -2.0, 3.0, -0.5, np.nan, 2.0, -1.0, 4.0, 5.0, 6.0])
    stat = _statistics(LimitState.YIELD, "m", margins)
    # 3 failures out of 9 valid samples (NaN excluded).
    assert stat.valid_samples == 9
    assert stat.pf_empirical == pytest.approx(3.0 / 9.0)
    low, high = stat.pf_empirical_ci
    assert 0.0 < low < stat.pf_empirical < high < 1.0

    # Zero observed failures: point estimate 0, upper bound ~3/n (the
    # resolution limit of the study), never a bare 0-width interval.
    safe = _statistics(LimitState.YIELD, "m", np.ones(1000))
    assert safe.pf_empirical == 0.0
    lo, hi = safe.pf_empirical_ci
    assert lo == 0.0
    assert 0.0 < hi < 5.0 / 1000

    # All-failure series closes at 1.
    bad = _statistics(LimitState.YIELD, "m", -np.ones(10))
    assert bad.pf_empirical == 1.0
    assert bad.pf_empirical_ci[1] == 1.0

    # No valid samples: NaN margin stats, no interval claim.
    empty = _statistics(LimitState.YIELD, "m", np.full(4, np.nan))
    assert np.isnan(empty.pf_empirical)
    assert empty.pf_empirical_ci is None
