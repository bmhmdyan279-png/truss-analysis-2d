"""Phase 2 tests for the Monte Carlo reliability engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from unittest.mock import MagicMock

import numpy as np
import pytest
from truss_analysis.reliability import (
    AnalysisSample,
    Direction,
    LimitState,
    MemberResponse,
    ReliabilityEngine,
    ServiceLimit,
    _beta_hat,
    _pf_from_beta,
    sample_named_variables,
)
from truss_analysis.solver import solve
from truss_analysis.uncertainty import LognormalRV

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
    assert np.isfinite(stat.beta_hat)

    assert abs(stat.mean - _MARGIN_MEAN) <= 0.10 * abs(_MARGIN_MEAN)
    assert abs(stat.std - _LOAD_STD) <= 0.10 * _LOAD_STD
    assert abs(stat.beta_hat - _BETA_THEORY) <= 0.20 * abs(_BETA_THEORY)
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
    assert np.isnan(stat.beta_hat)
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
    assert abs(largest.beta_hat - _BETA_THEORY) <= 0.30 * abs(_BETA_THEORY)


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
    assert np.isfinite(stat.beta_hat)


def test_service_limit_rejects_negative_limit() -> None:
    with pytest.raises(ValueError):
        ServiceLimit(node_id=2, direction=Direction.X, limit=-1.0)


def test_engine_rejects_nonpositive_sample_size() -> None:
    engine = ReliabilityEngine(variables={}, analyze_fn=_empty_analysis)
    with pytest.raises(ValueError):
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


def test_beta_hat_and_pf_edge_cases() -> None:
    assert _beta_hat(10.0, 0.0) == float("inf")
    assert _beta_hat(-10.0, 0.0) == float("-inf")
    assert np.isnan(_beta_hat(0.0, 0.0))
    assert np.isnan(_beta_hat(float("nan"), 1.0))

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
    assert np.isnan(stat.beta_hat)


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
