"""Normalized Criticality Index - one function, one documented policy.

Policy: when the CI field carries no ordering information
(``max(CI) ~= min(CI)`` within ``1e-12``) the degenerate case returns
``values=None`` with ``is_degenerate=True``; callers must surface the flag
instead of inventing an informative-looking number.  This module defines
the package's only ``compute_n*`` function.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["NciResult", "compute_nci"]

_DEGENERACY_ATOL = 1e-12


@dataclass(frozen=True)
class NciResult:
    """NCI field plus its degeneracy flag and the observed CI range."""

    values: dict[str, float] | None
    is_degenerate: bool
    min_ci: float
    max_ci: float


def compute_nci(ci_dict: Mapping[str, float]) -> NciResult:
    """NCI_i = (CI_i - min_j CI_j) / (max_j CI_j - min_j CI_j), bounded [0, 1].

    Degenerate input (empty range within ``1e-12``, e.g. a uniform CI field):
    ``values=None``, ``is_degenerate=True``.  An empty input maps to an empty
    non-degenerate result.
    """
    if not ci_dict:
        return NciResult({}, False, 0.0, 0.0)
    values = list(ci_dict.values())
    min_ci = min(values)
    max_ci = max(values)
    if max_ci - min_ci <= _DEGENERACY_ATOL:
        return NciResult(None, True, min_ci, max_ci)
    span = max_ci - min_ci
    return NciResult(
        {eid: (ci - min_ci) / span for eid, ci in ci_dict.items()},
        False,
        min_ci,
        max_ci,
    )
