"""Normalized Criticality Index — one function, one documented policy.

Pre-prompt-4 the package carried two NCI implementations with conflicting
degenerate-case policies (``compute_normalized_ci`` returned 1.0,
``compute_nci`` returned 0.5; CONTEXT_LOCK §4.5 B2).  Both fabricated
informative-looking numbers for a field that carries no information when
``max(CI) ≈ min(CI)``.

Policy now (DL-022): the degenerate case returns ``values=None`` with
``is_degenerate=True``; callers must surface the flag instead of inventing a
number.  This module defines the package's only ``compute_n*`` function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

__all__ = ["NciResult", "compute_nci"]

_DEGENERACY_ATOL = 1e-12


@dataclass(frozen=True)
class NciResult:
    """NCI field plus its degeneracy flag and the observed CI range."""

    values: Optional[Dict[str, float]]
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
