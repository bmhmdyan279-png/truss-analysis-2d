"""C16: parallel Monte Carlo must change the wall clock and nothing else.

The reliability engine draws its whole sample stream up front and then walks
the samples in index order, so ``n_jobs`` cannot affect a result. These tests
pin that property rather than trusting the docstring: a parallel path that
silently reordered margins would produce plausible-looking statistics for the
wrong samples, and nothing downstream would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from truss_analysis.reliability import (
    PARALLEL_CHUNK_SAMPLES,
    AnalysisSample,
    LimitState,
    MemberResponse,
    ReliabilityEngine,
    _evaluate_chunk,
    _resolve_n_jobs,
)
from truss_analysis.uncertainty import LognormalRV

FY_MEAN = 355e6
E_MEAN = 210e9
SEED = 20260916


def fresh_variables() -> dict[str, LognormalRV]:
    """New random variables per run -- their private generators hold state.

    Reusing one ``LognormalRV`` across two engine runs advances its RNG, so
    the second run would draw different numbers and a determinism test would
    fail for a reason that has nothing to do with parallelism.
    """
    return {
        "fy": LognormalRV(mean=FY_MEAN, cov=0.07, seed=SEED),
        "E": LognormalRV(mean=E_MEAN, cov=0.05, seed=SEED),
    }


def analyze(sample: dict[str, float]) -> AnalysisSample:
    """A callback with enough linear algebra in it to release the GIL."""
    fy = sample["fy"]
    E = sample["E"]
    x = np.linspace(0.0, fy / 1e5, 64)
    mat = np.eye(32) * 3.0 + np.outer(x[:32], x[:32])
    np.linalg.solve(mat, x[:32])
    return AnalysisSample(
        nodal_displacements=np.zeros(4),
        member_responses={
            "m1": MemberResponse(
                axial_force=-1.2e5 * (fy / FY_MEAN),
                E=E,
                A=4e-3,
                I_sec=1.2e-5,
                length=3.0,
                effective_length_factor=1.0,
                yield_stress=fy,
                temperature=20.0,
            ),
            # tension member: its buckling margin is NaN by construction, so
            # the NaN-aware comparison below is exercised on every run
            "m2": MemberResponse(
                axial_force=0.6e5,
                E=E,
                A=2e-3,
                I_sec=4e-9,
                length=2.0,
                effective_length_factor=1.0,
                yield_stress=fy,
                temperature=20.0,
            ),
        },
    )


N_SAMPLES = 400


def _run(n_jobs: int, backend: str = "threads", sizes=(N_SAMPLES,)):
    engine = ReliabilityEngine(
        fresh_variables(), analyze, n_jobs=n_jobs, parallel_backend=backend
    )
    return engine.run_convergence(sizes)


def _same_float(x: float, y: float) -> bool:
    """Exact equality that treats ``nan`` as equal to ``nan``."""
    if np.isnan(x) and np.isnan(y):
        return True
    return x == y


def assert_reports_identical(a, b) -> None:
    """Bit-for-bit equality of two convergence reports."""
    assert set(a) == set(b)
    for size in a:
        ra, rb = a[size], b[size]
        assert ra.sample_size == rb.sample_size
        assert len(ra.statistics) == len(rb.statistics)
        for sa, sb in zip(ra.statistics, rb.statistics, strict=True):
            assert sa.limit_state == sb.limit_state
            assert str(sa.target_id) == str(sb.target_id)
            for field in ("mean", "std", "beta_mom", "pf_approx", "pf_empirical"):
                assert _same_float(getattr(sa, field), getattr(sb, field)), (
                    f"{field} differs for {sa.limit_state}/{sa.target_id}: "
                    f"{getattr(sa, field)!r} vs {getattr(sb, field)!r}"
                )
            assert (sa.pf_empirical_ci is None) == (sb.pf_empirical_ci is None)
            if sa.pf_empirical_ci is not None:
                assert all(
                    _same_float(x, y)
                    for x, y in zip(sa.pf_empirical_ci, sb.pf_empirical_ci, strict=True)
                )
            assert np.array_equal(sa.margins, sb.margins, equal_nan=True), (
                f"margin series differ for {sa.limit_state}/{sa.target_id}"
            )


# --------------------------------------------------------------------------
# n_jobs normalisation
# --------------------------------------------------------------------------


def test_resolve_n_jobs_follows_the_sklearn_convention() -> None:
    import os

    cpus = os.cpu_count() or 1
    assert _resolve_n_jobs(None) == 1
    assert _resolve_n_jobs(1) == 1
    assert _resolve_n_jobs(4) == 4
    assert _resolve_n_jobs(-1) == cpus
    assert _resolve_n_jobs(-2) == max(1, cpus - 1)
    # more negative than there are CPUs must clamp, not go to zero
    assert _resolve_n_jobs(-(cpus + 10)) == 1


def test_resolve_n_jobs_rejects_zero() -> None:
    with pytest.raises(ValueError, match="non-zero integer"):
        _resolve_n_jobs(0)


def test_unknown_parallel_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="parallel_backend must be"):
        ReliabilityEngine(
            fresh_variables(),
            analyze,
            parallel_backend="ray",  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------
# determinism: the property that actually matters
# --------------------------------------------------------------------------


def test_threads_are_bit_identical_to_serial() -> None:
    assert_reports_identical(_run(1), _run(4, "threads"))


def test_negative_n_jobs_is_bit_identical_to_serial() -> None:
    assert_reports_identical(_run(1), _run(-1, "threads"))


def test_processes_are_bit_identical_to_serial() -> None:
    """The process backend pickles the callback; the numbers must not move."""
    assert_reports_identical(_run(1), _run(2, "processes"))


def test_per_call_override_matches_the_constructor_default() -> None:
    serial = _run(1)
    engine = ReliabilityEngine(fresh_variables(), analyze, n_jobs=1)
    overridden = engine.run_convergence((N_SAMPLES,), n_jobs=4)
    assert_reports_identical(serial, overridden)


@pytest.mark.parametrize("jobs", [1, 2, 3, 5, 8])
def test_convergence_reports_across_sizes_are_identical_for_any_jobs(
    jobs: int,
) -> None:
    """Prefix reuse must survive parallelism too.

    ``run_convergence`` simulates the largest size and reports the smaller
    ones from prefixes of the same stream, so ``margins[:200]`` of the 400
    report has to equal the 200 report exactly -- at every worker count.
    """
    sizes = (100, 200, N_SAMPLES)
    report = _run(jobs, "threads", sizes)
    assert set(report) == set(sizes)
    full = {stat.limit_state: stat for stat in report[N_SAMPLES].statistics}
    for size in (100, 200):
        for stat in report[size].statistics:
            reference = full[stat.limit_state]
            if str(stat.target_id) != str(reference.target_id):
                continue
            assert np.array_equal(
                stat.margins, reference.margins[:size], equal_nan=True
            )


def test_chunking_boundaries_do_not_drop_or_duplicate_samples() -> None:
    """Every worker-count must see all ``N_SAMPLES``, none twice.

    The chunk/block arithmetic is the one place a parallel rewrite can go
    quietly wrong: an off-by-one there loses a tail sample or repeats one,
    and the statistics still look reasonable.
    """
    for jobs in (1, 2, 3, 4, 7):
        report = _run(jobs, "threads")
        for stat in report[N_SAMPLES].statistics:
            assert stat.sample_size == N_SAMPLES
            assert stat.margins.shape == (N_SAMPLES,)


def test_small_sample_streams_stay_serial() -> None:
    """``max_n <= 1`` must not spin up a pool for a single sample."""
    engine = ReliabilityEngine(fresh_variables(), analyze, n_jobs=4)
    report = engine.run_convergence((1,))
    assert report[1].sample_size == 1
    for stat in report[1].statistics:
        assert stat.valid_samples <= 1


def test_chunk_size_is_a_declared_constant() -> None:
    """The chunking unit is public so a study can reason about memory."""
    assert PARALLEL_CHUNK_SAMPLES >= 1
    assert isinstance(PARALLEL_CHUNK_SAMPLES, int)


def test_evaluate_chunk_preserves_input_order() -> None:
    """The worker itself must be order-preserving, independent of the pool."""
    samples = {"v": np.arange(10, dtype=float)}
    seen: list[float] = []

    def record(scalar: dict[str, float]) -> AnalysisSample:
        seen.append(scalar["v"])
        return AnalysisSample(
            nodal_displacements=np.zeros(2),
            member_responses={
                "m": MemberResponse(
                    axial_force=-1.0,
                    E=E_MEAN,
                    A=1e-3,
                    I_sec=1e-9,
                    length=1.0,
                    effective_length_factor=1.0,
                    yield_stress=FY_MEAN,
                )
            },
        )

    out = _evaluate_chunk((record, samples, range(3, 8)))
    assert len(out) == 5
    assert seen == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert all(isinstance(r, AnalysisSample) for r in out)


def test_limit_states_covered_by_the_parallel_path() -> None:
    """The fixture exercises both margin types, so determinism is not vacuous."""
    report = _run(4, "threads")
    kinds = {(s.limit_state, str(s.target_id)) for s in report[N_SAMPLES].statistics}
    assert (LimitState.YIELD, "m1") in kinds
    assert (LimitState.BUCKLING, "m1") in kinds
    # m2 is in tension: its buckling margin series must be all-NaN, which is
    # exactly the case a naive equality check would get wrong
    m2_buckling = next(
        s
        for s in report[N_SAMPLES].statistics
        if s.limit_state == LimitState.BUCKLING and str(s.target_id) == "m2"
    )
    assert np.all(np.isnan(m2_buckling.margins))
