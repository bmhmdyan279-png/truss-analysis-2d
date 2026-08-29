"""Phase 2: Monte Carlo engine for sampled safety margins."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias, Union

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from .uncertainty import RandomVariable

TargetId: TypeAlias = Union[int, str]
ScalarSample: TypeAlias = Mapping[str, float]


class LimitState(str, Enum):
    YIELD = "yield"
    BUCKLING = "buckling"
    SERVICEABILITY = "serviceability"


class Direction(str, Enum):
    X = "x"
    Y = "y"
    MAGNITUDE = "magnitude"


@dataclass(frozen=True)
class MemberResponse:
    axial_force: float
    E: float
    A: float
    I_sec: float  # Renamed from I to avoid E741 and match Element model
    length: float
    effective_length_factor: float
    yield_stress: float | None = None


@dataclass(frozen=True)
class AnalysisSample:
    # تغییر int به TargetId برای پشتیبانی از شناسه‌های str در مدل واقعی
    member_responses: Mapping[TargetId, MemberResponse]
    nodal_displacements: Mapping[TargetId, tuple[float, float]]


@dataclass(frozen=True)
class ServiceLimit:
    node_id: TargetId  # تغییر از int به TargetId برای پشتیبانی از str
    direction: Direction
    limit: float
    name: str | None = None

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("Serviceability limit must be non-negative.")

    @property
    def key(self) -> str:
        return self.name or f"node_{self.node_id}_{self.direction.value}"


@dataclass
class MarginStatistics:
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
    sample_size: int
    statistics: tuple[MarginStatistics, ...]

    def get(
        self, limit_state: LimitState, target_id: TargetId
    ) -> MarginStatistics | None:
        for stat in self.statistics:
            if stat.limit_state == limit_state and stat.target_id == target_id:
                return stat
        return None


AnalyzeSample: TypeAlias = Callable[[ScalarSample], AnalysisSample]


def sample_named_variables(
    variables: Mapping[str, RandomVariable],
    n_samples: int,
) -> dict[str, npt.NDArray[np.float64]]:
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
        reports = self.run_convergence((n_samples,))
        return reports[n_samples]

    def run_convergence(
        self, sample_sizes: Sequence[int]
    ) -> dict[int, ReliabilityReport]:
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
    if np.isnan(beta):
        return float("nan")
    if np.isposinf(beta):
        return 0.0
    if np.isneginf(beta):
        return 1.0
    return float(norm.cdf(-beta))
