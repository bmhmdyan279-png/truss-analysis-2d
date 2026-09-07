"""Probabilistic ranking — the operational definition of proposal §5.4.

The proposal's four-step definition, quoted verbatim (translation of §5.4):

    "Probabilistic ranking for each member i is computed as: (1) for each
    Monte Carlo sample (load, resistance, temperature), the responses
    u_max^base and u_max^(i,alpha) are computed; (2) CI_i is computed for that
    sample; (3) the probabilistic rank of member i is determined from the
    MEAN of CI_i over all samples; (4) the deterministic rank is determined
    from CI_i at the MEAN VALUES of the variables (no sampling)."

:func:`probabilistic_ranking` implements steps (3)-(4) given per-sample CI
fields (steps (1)-(2) are produced by the caller via the engine).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping

import numpy as np
from truss_analysis.criticality.ranking import rank_members

__all__ = ["ProbabilisticRanking", "probabilistic_ranking"]


@dataclass(frozen=True)
class ProbabilisticRanking:
    """Mean-CI field with probabilistic and deterministic rankings."""

    mean_ci: Dict[str, float] = field(default_factory=dict)
    rank_probabilistic: List[str] = field(default_factory=list)
    rank_deterministic: List[str] = field(default_factory=list)
    top5_probabilistic: List[str] = field(default_factory=list)
    ci_std: Dict[str, float] = field(default_factory=dict)


def probabilistic_ranking(
    ci_per_sample: Mapping[str, np.ndarray],
    ci_at_mean_inputs: Mapping[str, float],
) -> ProbabilisticRanking:
    """Steps (3)-(4) of the proposal §5.4 operational definition.

    Operational definition quoted (proposal §5.4): probabilistic ranking for
    each member i: (1) for each Monte Carlo sample (load, resistance,
    temperature) the responses u_max^base and u_max^(i,alpha) are computed;
    (2) CI_i is computed for that sample; (3) the probabilistic rank of
    member i is determined from the MEAN of CI_i over all samples; (4) the
    deterministic rank is determined from CI_i at the MEAN VALUES of the
    variables (no sampling).

    Parameters
    ----------
    ci_per_sample:
        member id -> array of CI values, one entry per Monte Carlo sample
        (steps 1-2 performed by the caller).
    ci_at_mean_inputs:
        CI field computed at the mean values of the random variables
        (step 4, deterministic reference).
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
