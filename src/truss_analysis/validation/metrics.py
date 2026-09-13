"""Agreement metrics and the pre-coded three-branch verdict.

The reference-validation protocol fixes its decision tree *before* any
correlation is measured, so a low value has a documented path instead of an
ad-hoc reaction:

=============  ============================================================
rho >= 0.90    FULL_VALIDATION — agreement is sufficient; record and go on.
0.70 .. 0.90   CAUSE_ANALYSIS  — do not proceed blindly: inspect the
               per-member/per-node deviations (ties, unit consistency,
               boundary-condition mapping, degradation mapping), document
               the cause, and only then decide.
rho < 0.70     MODEL_REVIEW    — stop: the comparison itself is suspect
               (geometry/connectivity mapping, loads, supports, E-field
               transfer); review the model and re-run.
NaN            treated as MODEL_REVIEW (a degenerate comparison carries no
               evidence — never silently promoted to a pass).
=============  ============================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import spearmanr

from ..criticality.ranking import natural_sort_key

__all__ = [
    "RHO_FLOOR",
    "RHO_FULL",
    "RhoBranch",
    "RhoVerdict",
    "classify_rho",
    "rank_correlation",
]

RHO_FULL: float = 0.90
RHO_FLOOR: float = 0.70


def rank_correlation(
    a: Mapping[str, float],
    b: Mapping[str, float],
    quantize: float | None = None,
) -> float:
    """Spearman rank correlation between two fields over shared keys.

    Keys are aligned by :func:`natural_sort_key` (deterministic).  Ties are
    handled by scipy's average-rank convention.  If either field is constant
    over the shared keys the correlation is undefined and ``float('nan')`` is
    returned — callers must route NaN through :func:`classify_rho`, never
    treat it as a pass.

    ``quantize`` (optional, e.g. ``1e-10``): values are snapped to a grid of
    that resolution before ranking — the same convention as
    :func:`truss_analysis.criticality.ranking.tau_b` (sub-tolerance
    floating-point noise between near-tied members is not a rank signal).
    """
    keys = sorted(set(a) & set(b), key=natural_sort_key)
    if len(keys) < 2:
        return float("nan")
    va = np.asarray([a[k] for k in keys], dtype=float)
    vb = np.asarray([b[k] for k in keys], dtype=float)
    if quantize:
        va = np.floor(va / quantize + 0.5)
        vb = np.floor(vb / quantize + 0.5)
    if float(np.std(va)) == 0.0 or float(np.std(vb)) == 0.0:
        return float("nan")
    return float(spearmanr(va, vb).statistic)


class RhoVerdict(str, Enum):
    """Outcome branch of a reference-correlation check."""

    FULL_VALIDATION = "full_validation"
    CAUSE_ANALYSIS = "cause_analysis"
    MODEL_REVIEW = "model_review"


@dataclass(frozen=True)
class RhoBranch:
    """Verdict plus the prescribed action for the branch that was taken."""

    rho: float
    verdict: RhoVerdict
    prescribed_action: str


def classify_rho(rho: float) -> RhoBranch:
    """Map a measured rank correlation onto the pre-coded decision tree."""
    if np.isnan(rho):
        return RhoBranch(
            rho=float(rho),
            verdict=RhoVerdict.MODEL_REVIEW,
            prescribed_action=(
                "Correlation is undefined (constant or <2 shared keys): the "
                "comparison carries no evidence. Review the case setup."
            ),
        )
    if rho >= RHO_FULL:
        return RhoBranch(
            rho=float(rho),
            verdict=RhoVerdict.FULL_VALIDATION,
            prescribed_action=(
                "Agreement sufficient: record rho, per-quantity deviations "
                "and the solver versions; continue."
            ),
        )
    if rho >= RHO_FLOOR:
        return RhoBranch(
            rho=float(rho),
            verdict=RhoVerdict.CAUSE_ANALYSIS,
            prescribed_action=(
                "Inspect per-member deviations before proceeding: tie "
                "handling and quantisation, unit consistency, support and "
                "load mapping, degraded-E field transfer. Document the "
                "cause of the gap."
            ),
        )
    return RhoBranch(
        rho=float(rho),
        verdict=RhoVerdict.MODEL_REVIEW,
        prescribed_action=(
            "Stop: review geometry/connectivity mapping, boundary "
            "conditions, loads and the E-field passed to the reference "
            "solver; re-run the comparison."
        ),
    )
