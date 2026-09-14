"""Central numerical-tolerance policy for the whole library.

Every threshold that decides "is this zero / singular / ill-conditioned /
converged" lives here. Before this module existed the same concept was
hard-coded independently in several places and the values silently
disagreed — most visibly the relative rank cutoff, which was ``1e-13`` in
:mod:`truss_analysis.solver` but ``1e-9`` in
:mod:`truss_analysis.graph_validation`. A model could therefore be reported
as a mechanism by the validator and as perfectly solvable by the solver,
with no way for a user to tell which verdict to trust.

The two cutoffs are *deliberately* different and now documented as such:

``rank_rel_cutoff`` (tight, 1e-13)
    A **hard gate** used by the solver. Rejecting a genuinely solvable
    structure is a worse failure than solving a marginal one, so only a
    truly singular ``K_ff`` is refused.

``near_singular_rel_cutoff`` (loose, 1e-9)
    A **diagnostic** used by model validation. Flagging a near-mechanism as
    suspect is cheap and informative, so the validator warns earlier than
    the solver refuses.

Both are exposed through one frozen dataclass so callers can retune the
whole library consistently for a different unit system or model size
(a millimetre-scale model and a kilometre-scale bridge should not share an
absolute joule or newton threshold), and so the relationship between the
gates is explicit rather than accidental.

All tolerances in this module are **relative and dimensionless** wherever
physics allows. The few absolute ones (length) are documented with their
unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = [
    "DEFAULT_TOLERANCES",
    "NumericalStatus",
    "NumericalTolerances",
    "singular_value_screen",
]


class NumericalStatus(str, Enum):
    """Verdict on the numerical quality of a stiffness matrix.

    ``ill_conditioned`` is explicitly *not* ``invalid``: the solve still
    proceeds and the result is returned, but the caller can see that the
    answer carries amplified round-off. Keeping the three verdicts separate
    matters for criticality studies, where the members that matter most are
    precisely the ones that drive the structure towards a mechanism — there,
    conditioning is part of the phenomenon rather than a defect to hide.
    """

    STABLE = "stable"
    ILL_CONDITIONED = "ill_conditioned"
    SINGULAR = "singular"


@dataclass(frozen=True)
class NumericalTolerances:
    """Relative thresholds governing every numerical decision in the library.

    Parameters
    ----------
    length_tol : float
        Absolute geometric zero-length threshold **in metres**. Below this a
        member has no meaningful axial stiffness (``k = E A / L`` diverges)
        and is rejected or skipped. This is the one genuinely absolute
        tolerance in the policy: it guards a division, not a comparison.
    rank_rel_cutoff : float
        Relative singular-value cutoff for the **hard** singularity gate.
        Singular values below ``s_max * rank_rel_cutoff`` count as zero when
        estimating the numerical rank of ``K_ff``.
    near_singular_rel_cutoff : float
        Relative cutoff for the **diagnostic** near-mechanism test used by
        model validation. Must be looser than ``rank_rel_cutoff`` so the
        validator always warns at least as early as the solver refuses.
    cond_warning : float
        Condition number above which
        :class:`~truss_analysis.exceptions.IllConditionedWarning` is emitted.
        Consistent with ``near_singular_rel_cutoff`` when
        ``cond_warning == 1 / near_singular_rel_cutoff``.
    energy_rel_tol : float
        Maximum tolerated *relative* imbalance in the Clapeyron energy check.
        A linear-elastic solve closes it to machine precision.
    energy_roundoff_rel : float
        Round-off floor for the energy check, expressed relative to the
        characteristic imposed-strain energy
        :func:`~truss_analysis.postprocess.imposed_strain_energy`. Imposed-strain
        problems can make every balance term vanish analytically (free thermal
        expansion); without this floor a purely relative test divides
        round-off by round-off and reports a 100% error on an exact result.
    force_zero_rel : float
        Axial-force classification band as a fraction of the member's own
        ``E A``. Since ``N / (E A)`` is the axial strain, the default means
        "strain below 1e-12 counts as zero", which is invariant to unit
        system and model size.
    displacement_zero_rel : float
        Displacement classification band as a fraction of the largest nodal
        displacement magnitude in the model.

    Notes
    -----
    The instance is frozen so a tolerance set cannot be mutated halfway
    through an analysis. Build a replacement with
    :func:`dataclasses.replace` instead.
    """

    length_tol: float = 1e-12
    rank_rel_cutoff: float = 1e-13
    near_singular_rel_cutoff: float = 1e-9
    cond_warning: float = 1e12
    energy_rel_tol: float = 1e-8
    energy_roundoff_rel: float = 1e-9
    force_zero_rel: float = 1e-12
    displacement_zero_rel: float = 1e-12

    def __post_init__(self) -> None:
        """Reject internally inconsistent threshold sets at construction time.

        An ordering violation here would let the validator and the solver
        disagree about the same matrix, which is exactly the silent
        inconsistency this module exists to prevent.
        """
        if self.length_tol <= 0.0:
            raise ValueError("length_tol must be positive")
        if not 0.0 < self.rank_rel_cutoff < self.near_singular_rel_cutoff < 1.0:
            raise ValueError(
                "require 0 < rank_rel_cutoff < near_singular_rel_cutoff < 1, "
                f"got rank_rel_cutoff={self.rank_rel_cutoff}, "
                f"near_singular_rel_cutoff={self.near_singular_rel_cutoff}"
            )
        if self.cond_warning <= 1.0:
            raise ValueError("cond_warning must exceed 1")
        if not 0.0 < self.energy_rel_tol < 1.0:
            raise ValueError("energy_rel_tol must lie in (0, 1)")
        if self.energy_roundoff_rel <= 0.0:
            raise ValueError("energy_roundoff_rel must be positive")
        if self.energy_roundoff_rel >= self.energy_rel_tol:
            raise ValueError(
                "energy_roundoff_rel must be tighter than energy_rel_tol, "
                "otherwise the round-off floor could mask a real imbalance"
            )
        if not 0.0 < self.force_zero_rel < 1.0:
            raise ValueError("force_zero_rel must lie in (0, 1)")
        if not 0.0 < self.displacement_zero_rel < 1.0:
            raise ValueError("displacement_zero_rel must lie in (0, 1)")


#: Library-wide default policy. Import this rather than re-declaring numbers.
DEFAULT_TOLERANCES = NumericalTolerances()


def singular_value_screen(
    K: np.ndarray,
    tolerances: NumericalTolerances = DEFAULT_TOLERANCES,
) -> tuple[int, float, float, NumericalStatus]:
    """Estimate numerical rank, condition number and status from one SVD.

    Performs a single singular-value decomposition and derives every verdict
    from it, so the rank gate, the conditioning warning and the reported
    status can never disagree with each other.

    Parameters
    ----------
    K : np.ndarray
        Square symmetric matrix to screen, typically the free-free stiffness
        sub-matrix ``K_ff``.
    tolerances : NumericalTolerances, optional
        Policy supplying the cutoffs.

    Returns
    -------
    tuple[int, float, float, NumericalStatus]
        ``(rank, s_max, cond, status)`` where ``rank`` counts singular values
        above ``s_max * rank_rel_cutoff``, ``s_max`` is the largest singular
        value, ``cond`` is ``s_max / s_min`` (``inf`` when ``s_min`` is zero
        or the matrix is empty) and ``status`` classifies the result. An
        empty matrix yields ``(0, 0.0, 1.0, NumericalStatus.STABLE)`` because
        a fully constrained model has no free DOFs to be singular in.
    """
    n = K.shape[0]
    if n == 0:
        return 0, 0.0, 1.0, NumericalStatus.STABLE

    sv = np.linalg.svd(K, compute_uv=False)
    s_max = float(sv[0]) if sv.size else 0.0
    if s_max <= 0.0:
        return 0, 0.0, float("inf"), NumericalStatus.SINGULAR

    rank = int(np.count_nonzero(sv > s_max * tolerances.rank_rel_cutoff))
    s_min = float(sv[-1])
    cond = s_max / s_min if s_min > 0.0 else float("inf")

    if rank < n:
        status = NumericalStatus.SINGULAR
    elif cond > tolerances.cond_warning:
        status = NumericalStatus.ILL_CONDITIONED
    else:
        status = NumericalStatus.STABLE
    return rank, s_max, cond, status
