"""Deterministic ranking and Kendall tau-b computed from CI **values**.

Pre-prompt-4 defects fixed here (CONTEXT_LOCK §4.5):
* B1 — ``rank_members`` sorted ties by the raw identifier string, so
  ``'10' < '2'``; ties now break on :func:`natural_sort_key`, which is
  deterministic, documented and independent of the identifier format.
* B4 — ``compute_kendall_tau`` consumed pre-sorted identifier *lists*, which
  destroyed ties (tau-b silently degenerated to tau-a) and returned a silent
  1.0 in the all-ties case.  :func:`tau_b` consumes the CI value mappings
  directly, reports tie counts, and returns ``tau=None`` with
  ``is_degenerate=True`` when the comparison carries no information.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Mapping, Optional

from scipy.stats import kendalltau

__all__ = ["TauResult", "natural_sort_key", "rank_members", "tau_b"]


@dataclass(frozen=True)
class TauResult:
    """Structured Kendall tau-b outcome (never a silent number)."""

    tau: Optional[float]
    n_pairs: int
    n_ties_a: int
    n_ties_b: int
    p_value: Optional[float]
    is_degenerate: bool


def natural_sort_key(key: str) -> List[object]:
    """Sort key comparing numeric chunks numerically: ``'2' < '10'``."""
    parts = re.split(r"(\d+)", key)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def rank_members(ci_dict: Mapping[str, float]) -> List[str]:
    """Rank member ids by CI descending.

    Tie-break: :func:`natural_sort_key` of the identifier — deterministic,
    format-independent and documented (prompt-04 §C9).
    """
    return sorted(ci_dict, key=lambda eid: (-ci_dict[eid], natural_sort_key(eid)))


def _tie_pairs(values: List[float]) -> int:
    counts = Counter(values)
    return sum(c * (c - 1) // 2 for c in counts.values())


def tau_b(
    ci_a: Mapping[str, float],
    ci_b: Mapping[str, float],
    quantize: Optional[float] = None,
) -> TauResult:
    """Kendall tau-b between two CI fields, computed from the **values**.

    Members are aligned on the intersection of the identifier sets (ordered
    by :func:`natural_sort_key` for determinism).  If fewer than two pairs
    exist, or either field is constant (all pairs tied), the comparison is
    degenerate: ``tau=None`` and ``is_degenerate=True`` — never a silent
    1.0 or 0.0 (prompt-04 §C10, CONTEXT_LOCK §4.5 B4).  Fields containing
    non-finite values (mechanism members carry ``CI = +inf``) are likewise
    degenerate: the quantisation grid is undefined for infinities and the
    comparison carries no rank information (prompt-08, DR-025).

    ``quantize`` (optional, e.g. ``1e-10``): values are snapped to a grid of
    that resolution before comparing.  Kendall's tau is a discrete rank
    statistic: sub-tolerance floating-point noise between near-equal members
    (mirror-symmetric pairs of a symmetric truss) would otherwise flip pairs
    and report spurious discordance.  Quantising at the stated tolerance makes
    "ranks unchanged up to the tolerance" the measured quantity (prompt-04 §B7
    lemma convention); noise below the tolerance is not a rank signal.
    """
    keys = sorted(set(ci_a) & set(ci_b), key=natural_sort_key)
    n = len(keys)
    n_pairs = n * (n - 1) // 2
    a_vals = [ci_a[k] for k in keys]
    b_vals = [ci_b[k] for k in keys]
    # non-finite values (members flagged as mechanisms, CI = +inf) cannot be
    # quantised and carry no rank information in this statistic: degenerate,
    # never a silent number and never a math.floor(inf) crash (prompt-08,
    # DR-025).  Tie counts reported here are the raw (unquantised) ones.
    if not all(math.isfinite(v) for v in a_vals) or not all(
        math.isfinite(v) for v in b_vals
    ):
        return TauResult(
            None, n_pairs, _tie_pairs(a_vals), _tie_pairs(b_vals), None, True
        )
    if quantize:
        a_vals = [math.floor(v / quantize + 0.5) for v in a_vals]
        b_vals = [math.floor(v / quantize + 0.5) for v in b_vals]
    ties_a = _tie_pairs(a_vals)
    ties_b = _tie_pairs(b_vals)
    if n < 2 or ties_a == n_pairs or ties_b == n_pairs:
        return TauResult(None, n_pairs, ties_a, ties_b, None, True)
    res = kendalltau(a_vals, b_vals)
    tau = float(res.statistic)
    if math.isnan(tau):  # defensive: scipy degeneracy not caught above
        return TauResult(None, n_pairs, ties_a, ties_b, None, True)
    raw_p = res.pvalue
    p_value = None if raw_p is None or math.isnan(float(raw_p)) else float(raw_p)
    return TauResult(tau, n_pairs, ties_a, ties_b, p_value, False)
