"""Monte Carlo reliability engine for sampled safety margins.

This module provides a small, dependency-light engine that propagates
random variables through a user-supplied analysis callback and summarises
the resulting safety margins (yield, buckling and serviceability) with
first-order reliability statistics.

The theory behind the margin definitions and the estimator conventions
is documented in ``docs/theory.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from .uncertainty import RandomVariable

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
    """

    axial_force: float
    E: float
    A: float
    I_sec: float
    length: float
    effective_length_factor: float
    yield_stress: float | None = None


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
    beta_hat : float
        Reliability index estimate ``mean / std``.
    pf_approx : float
        Approximate failure probability ``Phi(-beta_hat)``.
    margins : numpy.ndarray
        Raw margin series; ``NaN`` marks samples where the margin is undefined.
    """

    limit_state: LimitState
    target_id: TargetId
    sample_size: int
    valid_samples: int
    mean: float
    std: float
    beta_hat: float
    pf_approx: float
    margins: npt.NDArray[np.float64] = field(repr=False)


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
    """

    def __init__(
        self,
        variables: Mapping[str, RandomVariable],
        analyze_fn: AnalyzeSample,
        service_limits: Sequence[ServiceLimit] = (),
    ) -> None:
        self._variables = dict(variables)
        self._analyze_fn = analyze_fn
        self._service_limits = tuple(service_limits)

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
        self, sample_sizes: Sequence[int]
    ) -> dict[int, ReliabilityReport]:
        """Run the engine once and report statistics at several sample sizes.

        The largest requested size is simulated; smaller reports reuse the
        corresponding prefix of the same sample stream, which keeps the
        comparison across sizes free of resampling noise.

        Parameters
        ----------
        sample_sizes : Sequence[int]
            Positive sample sizes; duplicates are ignored.

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

        for i in range(max_n):
            scalar_sample = {name: float(values[i]) for name, values in samples.items()}
            response = self._analyze_fn(scalar_sample)
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

    @staticmethod
    def _buckling_margin(member: MemberResponse) -> float:
        """Return the Euler buckling margin of a compressed member.

        Parameters
        ----------
        member : MemberResponse
            Member response; tension (non-negative axial force) or any
            non-positive stiffness/geometry yields ``NaN``.

        Returns
        -------
        float
            ``P_cr - |N|`` with ``P_cr`` the Euler critical load, or ``NaN``
            when the margin is undefined.
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

        p_cr = (np.pi**2 * member.E * member.I_sec) / (
            (member.effective_length_factor * member.length) ** 2
        )
        return float(p_cr - abs(member.axial_force))

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
            beta_hat=float("nan"),
            pf_approx=float("nan"),
            margins=margins,
        )

    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if valid_samples > 1 else 0.0
    beta = _beta_hat(mean, std)
    pf = _pf_from_beta(beta)

    return MarginStatistics(
        limit_state=limit_state,
        target_id=target_id,
        sample_size=int(margins.size),
        valid_samples=valid_samples,
        mean=mean,
        std=std,
        beta_hat=beta,
        pf_approx=pf,
        margins=margins,
    )


def _beta_hat(mean: float, std: float) -> float:
    """Return the reliability index ``mean / std`` with degenerate guards."""
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
