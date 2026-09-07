"""Validation package: external-reference checks and transfer metrics.

Public API (prompt-08, multi-level validation suite):

* :mod:`.metrics` — rank correlation and the three-branch verdict for
  reference-solver agreement (full validation / cause analysis / model
  review), pre-coded so a low correlation has a documented path instead of
  an ad-hoc decision.
* :mod:`.opensees_reference` — bridge to OpenSeesPy (optional dependency):
  builds the same pin-jointed model, sets ``E_i(T) = k_E(T_i) E_i`` per
  member explicitly (OpenSees ``Truss`` elements carry no temperature
  dependence of their own), and reproduces the criticality sweep as n+1
  independent solves — the strongest possible witness for the rank-1
  (Sherman–Morrison) engine.
* :mod:`.surrogate` — the cross-family transfer check: a member-level
  ridge surrogate on statics+geometry features, trained on one truss
  family and rank-correlated against the exact CI field of held-out
  families.

No function in this package ever fabricates a reference result: when
OpenSeesPy is unavailable the bridge raises, and callers (tests) skip
explicitly.
"""

from __future__ import annotations

from .metrics import (
    RHO_FLOOR,
    RHO_FULL,
    RhoBranch,
    RhoVerdict,
    classify_rho,
    rank_correlation,
)
from .opensees_reference import (
    RHO_QUANTIZE,
    OpenseesSolveError,
    OpenseesUnavailableError,
    StateComparison,
    ci_sweep_in_opensees,
    compare_ci_ranking,
    compare_state,
    opensees_available,
    opensees_version,
    solve_truss_in_opensees,
)
from .surrogate import (
    CV_RHO_GATE,
    FEATURES,
    CrossValidationResult,
    RidgeSurrogate,
    StateRow,
    cross_family_cv,
    feature_matrix,
)

__all__ = [
    "CV_RHO_GATE",
    "FEATURES",
    "RHO_FLOOR",
    "RHO_FULL",
    "RHO_QUANTIZE",
    "CrossValidationResult",
    "OpenseesSolveError",
    "OpenseesUnavailableError",
    "RhoBranch",
    "RhoVerdict",
    "RidgeSurrogate",
    "StateComparison",
    "StateRow",
    "ci_sweep_in_opensees",
    "classify_rho",
    "compare_ci_ranking",
    "compare_state",
    "cross_family_cv",
    "feature_matrix",
    "opensees_available",
    "opensees_version",
    "rank_correlation",
    "solve_truss_in_opensees",
]
