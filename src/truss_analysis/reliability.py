"""Monte Carlo reliability engine for sampled safety margins.

This module provides a small, dependency-light engine that propagates
random variables through a user-supplied analysis callback and summarises
the resulting safety margins (yield, buckling and serviceability) with
first-order reliability statistics.

The theory behind the margin definitions and the estimator conventions
is documented in ``docs/theory.md``.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from .exceptions import BucklingCheckWarning
from .limitstates import DEFAULT_BUCKLING_CURVE, GAMMA_M_FIRE, BucklingModel
from .sections import (
    buckling_reduction_factor,
    euler_buckling_load,
    non_dimensional_slenderness,
)
from .uncertainty import RandomVariable

_DEPRECATION_BETA_HAT = (
    "MarginStatistics.beta_hat is deprecated and renamed to beta_mom: the "
    "quantity is a method-of-moments mean/std of the sampled margin, not a "
    "Hasofer-Lind / FORM reliability index (no design point is searched for)."
)

TargetId: TypeAlias = int | str
ScalarSample: TypeAlias = Mapping[str, float]


class LimitState(str, Enum):
    """Limit state whose safety margin is tracked by the engine."""

    YIELD = "yield"
    BUCKLING = "buckling"
    SERVICEABILITY = "serviceability"


class Direction(str, Enum):
    """Displacement direction used by serviceability limits."""

    X = "x"
    Y = "y"
    MAGNITUDE = "magnitude"


@dataclass(frozen=True)
class MemberResponse:
    """Axial response and section properties of one truss member.

    Attributes
    ----------
    axial_force : float
        Member axial force; tension is positive.
    E : float
        Young's modulus of the member material.
    A : float
        Cross-sectional area.
    I_sec : float
        Second moment of area (named ``I_sec`` to avoid the ambiguous
        single-letter ``I`` and to match the element model).
    length : float
        Member length.
    effective_length_factor : float
        Buckling effective-length factor ``k``.
    yield_stress : float or None, optional
        Yield stress; when ``None`` the yield margin is not evaluated.
    temperature : float, default 20.0
        Steel temperature [degC] of the member state. Selects the fire
        buckling curve (EN 1993-1-2 imperfection factor ``0.65 alpha``)
        when above ambient and the ambient curve (EN 1993-1-1) at 20 degC;
        see :class:`ReliabilityEngine`. ``E`` and ``yield_stress`` are taken
        **as given** (the analysis callback decides whether they are already
        temperature-reduced), so ``temperature`` never rescales them -- it
        only selects the code regime for ``chi``.
    """

    axial_force: float
    E: float
    A: float
    I_sec: float
    length: float
    effective_length_factor: float
    yield_stress: float | None = None
    temperature: float = 20.0


@dataclass(frozen=True)
class AnalysisSample:
    """Result of analysing one sampled variable vector.

    Attributes
    ----------
    member_responses : Mapping[TargetId, MemberResponse]
        Per-member responses keyed by member identifier (``int`` or ``str``).
    nodal_displacements : Mapping[TargetId, tuple[float, float]]
        Per-node displacements ``(ux, uy)`` keyed by node identifier.
    """

    member_responses: Mapping[TargetId, MemberResponse]
    nodal_displacements: Mapping[TargetId, tuple[float, float]]


@dataclass(frozen=True)
class ServiceLimit:
    """Allowable displacement at one node and direction.

    Attributes
    ----------
    node_id : TargetId
        Identifier of the monitored node.
    direction : Direction
        Monitored displacement direction.
    limit : float
        Non-negative allowable displacement.
    name : str or None, optional
        Human-readable key; defaults to a name derived from node and direction.
    """

    node_id: TargetId
    direction: Direction
    limit: float
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate that the allowable displacement is non-negative."""
        if self.limit < 0:
            raise ValueError("Serviceability limit must be non-negative.")

    @property
    def key(self) -> str:
        """Return the stable identifier of this limit.

        Returns
        -------
        str
            ``name`` when provided, otherwise ``node_<id>_<direction>``.
        """
        return self.name or f"node_{self.node_id}_{self.direction.value}"


@dataclass
class MarginStatistics:
    """Summary statistics of one sampled margin series.

    Attributes
    ----------
    limit_state : LimitState
        Limit state that produced the margin.
    target_id : TargetId
        Member or serviceability-limit identifier.
    sample_size : int
        Number of samples in the series (including invalid ones).
    valid_samples : int
        Number of finite samples actually summarised.
    mean : float
        Arithmetic mean of the finite margins.
    std : float
        Sample standard deviation (``ddof=1``) of the finite margins.
    beta_mom : float
        **Method-of-moments** index ``mean / std`` of the sampled margin.

        Renamed from ``beta_hat`` in round 6 because the old name read as a
        reliability index in the Hasofer-Lind / FORM sense, which this is
        not.  A FORM index is the distance from the origin to the design
        point, minimised over the actual limit-state surface in standard
        normal space; ``beta_mom`` is the first two moments of a sampled
        margin and nothing else.  The two coincide only when the margin is
        exactly Gaussian *and* the limit state is linear in the standard
        normal variables.  No FORM/SORM search is performed here -- see the
        module docstring's scope statement.
    pf_approx : float
        First-order failure probability ``Phi(-beta_mom)``. This is a
        **normality assumption**, not a measurement: ``beta_mom`` is a
        method-of-moments index, so ``pf_approx`` is accurate only when the
        margin is close to Gaussian. Margins built from skewed inputs (a
        lognormal ``f_y``, a Gumbel live load) are themselves skewed, and in
        the far tail -- exactly where ``pf`` lives -- the normal
        approximation can be off by orders of magnitude. Read
        :attr:`pf_empirical` alongside it.
    pf_empirical : float
        Observed failure rate ``mean(margin < 0)`` over the *valid* (finite)
        samples. Assumption-free, but with ``n`` samples it cannot resolve
        probabilities much below ``1/n``: a rare-event study with zero
        observed failures reports ``0.0``, which means "not observed", never
        "impossible". Quantify that with :attr:`pf_empirical_ci`.
    pf_empirical_ci : tuple[float, float] or None
        Clopper-Pearson (exact binomial) 95 % confidence interval on
        :attr:`pf_empirical`, or ``None`` when there are no valid samples.
        The interval, not the point estimate, is the honest output of a
        crude-Monte-Carlo failure probability: its upper bound at zero
        observed failures is ``~3/n``, the resolution limit of the study.
    margins : numpy.ndarray
        Raw margin series; ``NaN`` marks samples where the margin is undefined.
    """

    limit_state: LimitState
    target_id: TargetId
    sample_size: int
    valid_samples: int
    mean: float
    std: float
    beta_mom: float
    pf_approx: float
    pf_empirical: float
    pf_empirical_ci: tuple[float, float] | None
    margins: npt.NDArray[np.float64] = field(repr=False)

    @property
    def beta_hat(self) -> float:
        """Deprecated alias of :attr:`beta_mom` (round-6 rename, C8).

        The old name implied a Hasofer-Lind reliability index; the quantity
        is a method-of-moments ``mean / std``.  Emits
        :class:`DeprecationWarning`.
        """
        warnings.warn(_DEPRECATION_BETA_HAT, DeprecationWarning, stacklevel=2)
        return self.beta_mom


@dataclass
class ReliabilityReport:
    """Reliability statistics for one fixed sample size.

    Attributes
    ----------
    sample_size : int
        Number of Monte Carlo samples behind this report.
    statistics : tuple[MarginStatistics, ...]
        Per-target statistics, sorted by limit state then target identifier.
    """

    sample_size: int
    statistics: tuple[MarginStatistics, ...]

    def get(
        self, limit_state: LimitState, target_id: TargetId
    ) -> MarginStatistics | None:
        """Return statistics for one limit state and target.

        Parameters
        ----------
        limit_state : LimitState
            Limit state to look up.
        target_id : TargetId
            Member or serviceability-limit identifier.

        Returns
        -------
        MarginStatistics or None
            Matching statistics, or ``None`` when the target is absent.
        """
        for stat in self.statistics:
            if stat.limit_state == limit_state and stat.target_id == target_id:
                return stat
        return None


AnalyzeSample: TypeAlias = Callable[[ScalarSample], AnalysisSample]


def sample_named_variables(
    variables: Mapping[str, RandomVariable],
    n_samples: int,
) -> dict[str, npt.NDArray[np.float64]]:
    """Draw ``n_samples`` realisations from each named random variable.

    Parameters
    ----------
    variables : Mapping[str, RandomVariable]
        Random variables keyed by name.
    n_samples : int
        Number of samples to draw from every variable; must be positive.

    Returns
    -------
    dict[str, numpy.ndarray]
        Sample arrays keyed by variable name, each of shape ``(n_samples,)``.

    Raises
    ------
    ValueError
        If ``n_samples`` is not positive or a variable returns a sample
        array whose shape is not ``(n_samples,)``.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    samples: dict[str, npt.NDArray[np.float64]] = {}
    for name, rv in variables.items():
        arr = np.asarray(rv.sample(n_samples), dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != n_samples:
            raise ValueError(
                f"Random variable '{name}' returned an invalid sample shape."
            )
        samples[name] = arr
    return samples


#: Samples evaluated per worker task when ``n_jobs != 1``.  The chunk is the
#: unit of both load balancing and (for the process backend) pickling, so it
#: has to be large enough to amortise shipping the sample arrays and small
#: enough that a straggler does not idle the pool at the end.
PARALLEL_CHUNK_SAMPLES = 64

ParallelBackend = Literal["threads", "processes"]
"""Where :class:`ReliabilityEngine` runs its samples when ``n_jobs != 1``.

``"threads"`` (the default) needs nothing of the caller: the analysis
callback is never pickled, and because the expensive work is numpy/scipy
linear algebra -- which releases the GIL -- threads scale.  ``"processes"``
sidesteps the GIL entirely but requires ``analyze_fn`` and everything it
closes over to be picklable, which rules out the closures and lambdas most
call sites use.
"""


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Normalise a scikit-learn-style ``n_jobs`` to a concrete worker count.

    ``None`` and ``1`` are serial; ``-1`` is every CPU; ``-k`` is every CPU
    but ``k - 1``.  ``0`` is rejected rather than quietly meaning one of the
    two things a caller might have intended.
    """
    if n_jobs is None:
        return 1
    jobs = int(n_jobs)
    if jobs == 0:
        msg = "n_jobs must be a non-zero integer (-1 = every CPU)"
        raise ValueError(msg)
    if jobs < 0:
        cpus = os.cpu_count() or 1
        return max(1, cpus + 1 + jobs)
    return jobs


def _evaluate_chunk(
    payload: tuple[AnalyzeSample, dict[str, npt.NDArray[np.float64]], range],
) -> list[AnalysisSample]:
    """Evaluate one contiguous block of sample indices, in order.

    Module level (not a closure) so the *process* backend can pickle it.  The
    sample arrays travel once per chunk rather than once per sample, which is
    what makes that backend viable at all.
    """
    fn, samples, indices = payload
    return [
        fn({name: float(values[i]) for name, values in samples.items()})
        for i in indices
    ]


class ReliabilityEngine:
    """Monte Carlo engine that summarises sampled safety margins.

    Parameters
    ----------
    variables : Mapping[str, RandomVariable]
        Random variables propagated through the analysis callback.
    analyze_fn : AnalyzeSample
        Callback mapping one sampled variable vector to an
        :class:`AnalysisSample`.
    service_limits : Sequence[ServiceLimit], optional
        Serviceability limits evaluated on nodal displacements.
    buckling_model : BucklingModel, default EUROCODE_CHI
        Compression capacity model behind the buckling margin; the default
        matches the limit-state layer so reliability indices and DCRs speak
        the same language (round-5 audit: the pre-2.8 engine used bare
        Euler while the DCR chain used ``chi``, making every reported
        buckling ``beta`` systematically optimistic).
    buckling_curve : str, default "c"
        Flexural buckling curve, only used by ``EUROCODE_CHI``.
    n_jobs : int, default 1
        Workers used to evaluate ``analyze_fn`` across the sample stream
        (C16).  ``1`` is serial; ``-1`` is every CPU; ``-k`` is every CPU but
        ``k - 1``.  Results are *bit-identical* to the serial run for any
        value: the random draws are taken up front from one stream and the
        per-sample margins are accumulated strictly in index order, so
        parallelism changes the wall clock and nothing else.
    parallel_backend : {"threads", "processes"}, default "threads"
        See :data:`ParallelBackend`.  ``"processes"`` requires
        ``analyze_fn`` to be picklable.
    """

    def __init__(
        self,
        variables: Mapping[str, RandomVariable],
        analyze_fn: AnalyzeSample,
        service_limits: Sequence[ServiceLimit] = (),
        buckling_model: BucklingModel = BucklingModel.EUROCODE_CHI,
        buckling_curve: str = DEFAULT_BUCKLING_CURVE,
        n_jobs: int = 1,
        parallel_backend: ParallelBackend = "threads",
    ) -> None:
        if parallel_backend not in ("threads", "processes"):
            msg = (
                "parallel_backend must be 'threads' or 'processes'; got "
                f"{parallel_backend!r}"
            )
            raise ValueError(msg)
        self._variables = dict(variables)
        self._analyze_fn = analyze_fn
        self._service_limits = tuple(service_limits)
        self._buckling_model = buckling_model
        self._buckling_curve = buckling_curve
        self._n_jobs = _resolve_n_jobs(n_jobs)
        self._parallel_backend = parallel_backend

    def run(self, n_samples: int) -> ReliabilityReport:
        """Run the engine once at a fixed sample size.

        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo samples; must be positive.

        Returns
        -------
        ReliabilityReport
            Statistics for the requested sample size.
        """
        reports = self.run_convergence((n_samples,))
        return reports[n_samples]

    def run_convergence(
        self,
        sample_sizes: Sequence[int],
        n_jobs: int | None = None,
    ) -> dict[int, ReliabilityReport]:
        """Run the engine once and report statistics at several sample sizes.

        The largest requested size is simulated; smaller reports reuse the
        corresponding prefix of the same sample stream, which keeps the
        comparison across sizes free of resampling noise.

        Parameters
        ----------
        sample_sizes : Sequence[int]
            Positive sample sizes; duplicates are ignored.
        n_jobs : int or None, optional
            Per-call override of the constructor's ``n_jobs``; ``None`` uses
            the engine default.  Convenience for running a large study in
            parallel after a small serial sanity check on the same engine.

        Returns
        -------
        dict[int, ReliabilityReport]
            Report for each requested sample size, keyed by that size.

        Raises
        ------
        ValueError
            If ``sample_sizes`` is empty or contains non-positive values.
        """
        if not sample_sizes:
            raise ValueError("sample_sizes must not be empty.")

        sizes = sorted({int(s) for s in sample_sizes})
        if any(size <= 0 for size in sizes):
            raise ValueError("All sample sizes must be positive.")

        max_n = sizes[-1]
        samples = sample_named_variables(self._variables, max_n)

        margin_lists: dict[tuple[LimitState, TargetId], list[float]] = {}

        jobs = self._n_jobs if n_jobs is None else _resolve_n_jobs(n_jobs)
        for i, response in self._iter_responses(samples, max_n, jobs):
            seen: set[tuple[LimitState, TargetId]] = set()

            for member_id, member in response.member_responses.items():
                mid: TargetId = member_id

                if member.yield_stress is not None:
                    key: tuple[LimitState, TargetId] = (LimitState.YIELD, mid)
                    seen.add(key)
                    if key not in margin_lists:
                        margin_lists[key] = [float("nan")] * i
                    margin_lists[key].append(
                        member.yield_stress * member.A - abs(member.axial_force)
                    )

                key_buck: tuple[LimitState, TargetId] = (LimitState.BUCKLING, mid)
                seen.add(key_buck)
                if key_buck not in margin_lists:
                    margin_lists[key_buck] = [float("nan")] * i
                margin_lists[key_buck].append(self._buckling_margin(member))

            for service_limit in self._service_limits:
                key_serv: tuple[LimitState, TargetId] = (
                    LimitState.SERVICEABILITY,
                    service_limit.key,
                )
                seen.add(key_serv)
                if key_serv not in margin_lists:
                    margin_lists[key_serv] = [float("nan")] * i
                margin_lists[key_serv].append(
                    self._service_margin(response, service_limit)
                )

            for key, values in margin_lists.items():
                if key not in seen:
                    values.append(float("nan"))

        reports: dict[int, ReliabilityReport] = {}
        for size in sizes:
            statistics: list[MarginStatistics] = []
            for key, values in margin_lists.items():
                arr: npt.NDArray[np.float64] = np.asarray(
                    values[:size], dtype=np.float64
                )
                statistics.append(_statistics(key[0], key[1], arr))

            statistics.sort(
                key=lambda stat: (stat.limit_state.value, str(stat.target_id))
            )
            reports[size] = ReliabilityReport(
                sample_size=size,
                statistics=tuple(statistics),
            )

        return reports

    def _iter_responses(
        self,
        samples: dict[str, npt.NDArray[np.float64]],
        max_n: int,
        jobs: int,
    ) -> Iterator[tuple[int, AnalysisSample]]:
        """Yield ``(index, response)`` pairs in strict index order.

        With ``jobs == 1`` this is the original serial loop and no executor is
        created.  Otherwise the stream is cut into chunks of
        :data:`PARALLEL_CHUNK_SAMPLES` ``* jobs`` indices, one chunk is in
        flight at a time, and the chunk's results are consumed in order
        before the next is submitted.  That bounds resident responses to one
        chunk -- the serial path holds none -- instead of materialising all
        ``max_n`` of them, which an eager ``executor.map`` over the whole
        range would do.

        Determinism does not depend on the pool: the sample stream is drawn
        once up front and the accumulation below walks indices ascending, so
        a given ``max_n`` produces a given report at any ``jobs``.
        """
        if jobs <= 1 or max_n <= 1:
            for i in range(max_n):
                scalar = {name: float(values[i]) for name, values in samples.items()}
                yield i, self._analyze_fn(scalar)
            return

        chunk = max(jobs, min(max_n, PARALLEL_CHUNK_SAMPLES * jobs))
        executor_cls = (
            ThreadPoolExecutor
            if self._parallel_backend == "threads"
            else ProcessPoolExecutor
        )
        with executor_cls(max_workers=jobs) as pool:
            for start in range(0, max_n, chunk):
                stop = min(start + chunk, max_n)
                # Sub-blocks of `jobs` indices, so one chunk keeps every
                # worker busy while `pool.map` still returns them in order.
                blocks = [
                    range(a, min(a + jobs, stop)) for a in range(start, stop, jobs)
                ]
                payloads = [(self._analyze_fn, samples, block) for block in blocks]
                # Each index comes from its own block, never from a running
                # counter, so a short block cannot silently mislabel a margin.
                for block, responses in zip(
                    blocks, pool.map(_evaluate_chunk, payloads), strict=True
                ):
                    for i, response in zip(block, responses, strict=True):
                        yield i, response

    def _buckling_margin(self, member: MemberResponse) -> float:
        """Return the buckling safety margin of a compressed member.

        Under the default :attr:`BucklingModel.EUROCODE_CHI` the margin is
        ``chi * A * f_y / gamma_M - |N|`` with ``chi`` from the *same*
        buckling-curve machinery the limit-state layer uses
        (:func:`truss_analysis.sections.non_dimensional_slenderness` +
        :func:`truss_analysis.sections.buckling_reduction_factor`), so this
        engine and the DCR chain can no longer disagree about what a
        member's compression capacity is. The fire imperfection factor
        (``0.65 alpha``, EN 1993-1-2 4.2.3.1(3)) applies when the member
        temperature is above ambient, the ambient curve (EN 1993-1-1) at
        20 degC.

        The historical :attr:`BucklingModel.EULER_ONLY` returns the bare
        ``P_cr - |N|``; it is kept for reproducing pre-2.8 numbers and is
        documented in :class:`BucklingModel` as optimistic at intermediate
        slenderness (up to ~15 % at lambda_bar ~ 2, more near lambda_bar 1).

        Parameters
        ----------
        member : MemberResponse
            Member response; tension (non-negative axial force) or any
            non-positive stiffness/geometry yields ``NaN``. Under
            ``EUROCODE_CHI`` a missing ``yield_stress`` also yields ``NaN``:
            without ``f_y`` the non-dimensional slenderness -- and hence
            ``chi`` -- is undefined, and silently falling back to Euler
            would re-create the two-capacity-models split this fix removes.

        Returns
        -------
        float
            Capacity minus ``|N|`` [N], or ``NaN`` when undefined.
        """
        if member.axial_force >= 0.0:
            return float("nan")

        if (
            member.E <= 0.0
            or member.I_sec <= 0.0
            or member.length <= 0.0
            or member.effective_length_factor <= 0.0
        ):
            return float("nan")

        p_cr = euler_buckling_load(
            member.I_sec,
            member.length,
            member.E,
            member.effective_length_factor,
        )
        if self._buckling_model is BucklingModel.EULER_ONLY:
            warnings.warn(
                "BucklingModel.EULER_ONLY ignores residual stresses and initial "
                "out-of-straightness, overestimating capacity of intermediate-"
                "slenderness members. The reported reliability indices are optimistic. "
                "Use EUROCODE_CHI for code-compliant assessment.",
                BucklingCheckWarning,
                stacklevel=2,
            )
            return float(p_cr - abs(member.axial_force))

        if member.yield_stress is None or member.yield_stress <= 0.0:
            return float("nan")
        if member.A <= 0.0:
            return float("nan")

        lambda_bar = non_dimensional_slenderness(member.A, member.yield_stress, p_cr)
        chi = buckling_reduction_factor(
            lambda_bar,
            self._buckling_curve,
            fire=member.temperature > 20.0,
        )
        capacity = chi * member.A * member.yield_stress / GAMMA_M_FIRE
        return float(capacity - abs(member.axial_force))

    @staticmethod
    def _service_margin(
        response: AnalysisSample,
        limit: ServiceLimit,
    ) -> float:
        """Return the displacement margin ``limit - |u|`` at one node.

        Parameters
        ----------
        response : AnalysisSample
            Sampled analysis result providing nodal displacements.
        limit : ServiceLimit
            Serviceability limit to evaluate.

        Returns
        -------
        float
            Margin for the monitored direction, or ``NaN`` when the node is
            missing from the response.
        """
        displacement = response.nodal_displacements.get(limit.node_id)
        if displacement is None:
            return float("nan")

        ux, uy = displacement

        if limit.direction is Direction.X:
            value = abs(ux)
        elif limit.direction is Direction.Y:
            value = abs(uy)
        else:
            value = float(np.hypot(ux, uy))

        return float(limit.limit - value)


def _clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial (Clopper-Pearson) interval for ``k`` failures in ``n``.

    Chosen over a Wald/normal interval deliberately: at the small failure
    counts a crude Monte Carlo reliability study produces (often ``k = 0``)
    the Wald interval is degenerate or even extends below zero, while
    Clopper-Pearson is exact by construction and gives the honest
    ``~3/n`` upper bound at ``k = 0``.
    """
    alpha = 1.0 - confidence
    low = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low, high


def _statistics(
    limit_state: LimitState,
    target_id: TargetId,
    margins: npt.NDArray[np.float64],
) -> MarginStatistics:
    """Summarise one margin series into :class:`MarginStatistics`."""
    finite = margins[np.isfinite(margins)]
    valid_samples = int(finite.size)

    if valid_samples == 0:
        return MarginStatistics(
            limit_state=limit_state,
            target_id=target_id,
            sample_size=int(margins.size),
            valid_samples=0,
            mean=float("nan"),
            std=float("nan"),
            beta_mom=float("nan"),
            pf_approx=float("nan"),
            pf_empirical=float("nan"),
            pf_empirical_ci=None,
            margins=margins,
        )

    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if valid_samples > 1 else 0.0
    beta = _beta_mom(mean, std)
    pf = _pf_from_beta(beta)
    failures = int(np.count_nonzero(finite < 0.0))
    pf_emp = failures / valid_samples
    pf_ci = _clopper_pearson(failures, valid_samples)

    return MarginStatistics(
        limit_state=limit_state,
        target_id=target_id,
        sample_size=int(margins.size),
        valid_samples=valid_samples,
        mean=mean,
        std=std,
        beta_mom=beta,
        pf_approx=pf,
        pf_empirical=pf_emp,
        pf_empirical_ci=pf_ci,
        margins=margins,
    )


def _beta_mom(mean: float, std: float) -> float:
    """Return the method-of-moments index ``mean / std`` with degenerate guards.

    Not a Hasofer-Lind reliability index -- no design point is searched for.
    The guards matter: a zero-dispersion margin has an infinite index whose
    *sign* is the whole answer, and ``0/0`` must stay ``nan`` rather than
    becoming a confident ``0.0``.
    """
    if not np.isfinite(mean) or not np.isfinite(std):
        return float("nan")

    scale = max(1.0, abs(mean))
    if std <= np.finfo(float).eps * scale:
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return float("nan")

    return float(mean / std)


def _pf_from_beta(beta: float) -> float:
    """Return the approximate failure probability ``Phi(-beta)``."""
    if np.isnan(beta):
        return float("nan")
    if np.isposinf(beta):
        return 0.0
    if np.isneginf(beta):
        return 1.0
    return float(norm.cdf(-beta))
