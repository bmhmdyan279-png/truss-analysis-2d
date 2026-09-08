"""Probabilistic ranking of members from per-sample criticality fields.

Definition implemented here (four-step procedure):

1. For each Monte Carlo sample of the random inputs (load, resistance,
   temperature), the responses ``u_max_base`` and ``u_max_(i, alpha)`` are
   computed by the caller via the criticality engine.
2. ``CI_i`` is computed for that sample.
3. The probabilistic rank of member ``i`` is determined from the
   MEAN of ``CI_i`` over all samples.
4. The deterministic reference rank is determined from ``CI_i`` at the MEAN
   VALUES of the random variables (no sampling).

:func:`probabilistic_ranking` implements steps (3)-(4) given per-sample CI
fields; steps (1)-(2) are produced by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..criticality.ranking import rank_members

__all__ = ["ProbabilisticRanking", "probabilistic_ranking"]


@dataclass(frozen=True)
class ProbabilisticRanking:
    """Mean-CI field with probabilistic and deterministic rankings.

    Attributes
    ----------
    mean_ci : dict[str, float]
        Member id -> mean of ``CI_i`` over the Monte Carlo samples.
    rank_probabilistic : list[str]
        Member ids ranked by ``mean_ci`` descending.
    rank_deterministic : list[str]
        Member ids ranked by the CI field at mean inputs.
    top5_probabilistic : list[str]
        First five entries of ``rank_probabilistic``.
    ci_std : dict[str, float]
        Member id -> sample standard deviation of ``CI_i`` (ddof=1).
    """

    mean_ci: dict[str, float] = field(default_factory=dict)
    rank_probabilistic: list[str] = field(default_factory=list)
    rank_deterministic: list[str] = field(default_factory=list)
    top5_probabilistic: list[str] = field(default_factory=list)
    ci_std: dict[str, float] = field(default_factory=dict)


def probabilistic_ranking(
    ci_per_sample: Mapping[str, np.ndarray],
    ci_at_mean_inputs: Mapping[str, float],
) -> ProbabilisticRanking:
    """Rank members from per-sample CI fields (steps 3-4 of the definition).

    The probabilistic rank of each member is derived from the MEAN of its
    CI over all Monte Carlo samples; the deterministic reference rank comes
    from the CI field computed at the mean values of the random variables
    (no sampling). Sampling itself (steps 1-2) is performed by the caller
    through the criticality engine.

    Parameters
    ----------
    ci_per_sample : Mapping[str, np.ndarray]
        Member id -> array of CI values, one entry per Monte Carlo sample.
    ci_at_mean_inputs : Mapping[str, float]
        CI field computed at the mean values of the random variables
        (deterministic reference).

    Returns
    -------
    ProbabilisticRanking
        Mean/std CI fields plus both rankings and the probabilistic top-5.
    """
    mean_ci = {mid: float(np.mean(vals)) for mid, vals in ci_per_sample.items()}
    std_ci = {mid: float(np.std(vals, ddof=1)) for mid, vals in ci_per_sample.items()}
    rank_prob = rank_members(mean_ci)
    rank_det = rank_members(dict(ci_at_mean_inputs))
    return ProbabilisticRanking(
        mean_ci=mean_ci,
        rank_probabilistic=rank_prob,
        rank_deterministic=rank_det,
        top5_probabilistic=rank_prob[:5],
        ci_std=std_ci,
    )
