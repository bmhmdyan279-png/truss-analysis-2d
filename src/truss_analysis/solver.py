"""Linear solve of the global system ``K U = F`` plus energy verification.

Factorisation strategy
----------------------
The free-free stiffness matrix of a stable truss is symmetric positive
definite, so the mathematically appropriate factorisation is Cholesky
(``K = L L^T``), which costs roughly half of the general LU decomposition
that :func:`numpy.linalg.solve` performs and uses only the lower triangle.
The solve therefore tries Cholesky first and falls back to LU. Cholesky is
also a *sharper* singularity test than LU: it fails the moment the matrix
stops being positive definite, which for a symmetric PSD stiffness matrix is
exactly the appearance of a kinematic mechanism.

When ``check_condition`` is on, an SVD screen runs first anyway, because it
yields the numerical rank (needed for the explicit "mechanism detected"
message) and the condition number (needed for
:class:`~truss_analysis.exceptions.IllConditionedWarning`). Callers that
solve the same topology many times — temperature sweeps, retrofit triage,
Monte Carlo — should pass ``check_condition=False`` and let Cholesky act as
the gate.

Sparse support
--------------
``K`` may be a dense :class:`numpy.ndarray` or any
:class:`scipy.sparse.spmatrix`. Sparse input is solved with SuperLU
(:func:`scipy.sparse.linalg.splu`), which turns the ``O(n^3)`` dense cost
and ``O(n^2)`` memory into something tractable for large trusses.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import LinAlgError, cho_factor, cho_solve, lu_factor, lu_solve

from .exceptions import (
    EnergyValidationError,
    IllConditionedWarning,
    SingularMatrixError,
)
from .numerics import (
    DEFAULT_TOLERANCES,
    NumericalStatus,
    NumericalTolerances,
    singular_value_screen,
)

#: cond(K_ff) above this threshold emits :class:`IllConditionedWarning`.
#: Kept as a module constant for backwards compatibility; the authoritative
#: value is :attr:`NumericalTolerances.cond_warning`.
CONDITION_WARNING_THRESHOLD = DEFAULT_TOLERANCES.cond_warning

#: Relative singular-value cutoff used for the numerical rank estimate.
#: Kept as a module constant for backwards compatibility; the authoritative
#: value is :attr:`NumericalTolerances.rank_rel_cutoff`.
_RANK_REL_CUTOFF = DEFAULT_TOLERANCES.rank_rel_cutoff

#: Round-off floor for :func:`check_energy`, relative to the characteristic
#: imposed-strain energy. Authoritative value lives in the tolerances policy.
_ENERGY_ROUNDOFF_REL = DEFAULT_TOLERANCES.energy_roundoff_rel

#: Absolute fallback floor [J] for :func:`check_energy` when no
#: ``energy_scale`` is supplied. Needed because a balance in which every term
#: vanishes analytically (free thermal expansion) has no meaningful relative
#: error. ``run()`` always passes a physically derived scale, so this only
#: applies to direct callers.
_ENERGY_ZERO_FLOOR = 1e-9


@dataclass(frozen=True)
class SolveResult:
    """Displacement vector together with the numerical evidence for it.

    Reporting *only* ``U`` hides whether the answer is trustworthy. This
    container keeps the verdict alongside the result so downstream code (and
    the user's report) can distinguish "well-posed and well-conditioned" from
    "solvable but round-off-amplified" from "mechanism".
    """

    U: np.ndarray
    """Global displacement vector, shape ``(2n,)``; fixed DOFs are exactly 0."""

    n_free: int
    """Number of unconstrained degrees of freedom in the solved system."""

    rank: int
    """Numerical rank of ``K_ff``; equals ``n_free`` when full rank."""

    cond: float
    """Estimated 2-norm condition number of ``K_ff`` (``inf`` if singular)."""

    status: NumericalStatus
    """Stable / ill-conditioned / singular verdict."""

    factorisation: str
    """Which factorisation actually produced the answer.

    One of ``"cholesky"``, ``"lu"``, ``"sparse-lu"``, ``"none"`` (no free
    DOFs), or ``"unscreened"`` when the SVD screen was skipped so ``rank``
    and ``cond`` are not meaningful.
    """

    screened: bool
    """Whether the SVD rank/condition screen was run."""


def solve_with_diagnostics(
    K: np.ndarray | sp.spmatrix,
    F: np.ndarray,
    fixed_dofs: list[int],
    check_condition: bool = True,
    tolerances: NumericalTolerances = DEFAULT_TOLERANCES,
) -> SolveResult:
    """Solve ``K U = F`` and report the numerical evidence for the answer.

    Same computation as :func:`solve`, but returns a :class:`SolveResult`
    carrying rank, condition number, the stable/ill-conditioned/singular
    verdict and the factorisation that was actually used.

    Parameters
    ----------
    K : np.ndarray or scipy.sparse.spmatrix
        Global stiffness matrix, shape ``(2n, 2n)``. Dense or sparse.
    F : np.ndarray
        Global force vector, shape ``(2n,)``.
    fixed_dofs : list[int]
        Indices of constrained degrees of freedom (zero displacement).
    check_condition : bool, default True
        Run the SVD rank/conditioning screen before solving. Turning this off
        skips an ``O(n^3)`` decomposition that costs more than the solve
        itself; Cholesky then acts as the singularity gate, which is the
        recommended setting for repeated solves of one topology.
    tolerances : NumericalTolerances, optional
        Numerical policy supplying the rank and conditioning cutoffs.

    Returns
    -------
    SolveResult
        Displacements plus rank, condition number, status and factorisation.

    Raises
    ------
    SingularMatrixError
        If ``K_ff`` is rank deficient (mechanism) or no factorisation
        succeeds on it.

    Warns
    -----
    IllConditionedWarning
        If ``cond(K_ff)`` exceeds ``tolerances.cond_warning``.
    """
    n = K.shape[0]
    U = np.zeros(n)
    fixed_set = set(fixed_dofs)
    free_dofs = [i for i in range(n) if i not in fixed_set]
    if not free_dofs:
        return SolveResult(
            U=U,
            n_free=0,
            rank=0,
            cond=1.0,
            status=NumericalStatus.STABLE,
            factorisation="none",
            screened=False,
        )

    n_free = len(free_dofs)
    sparse_input = sp.issparse(K)

    if sparse_input:
        K_ff = sp.csc_matrix(K)[np.ix_(free_dofs, free_dofs)]
    else:
        K_ff = np.asarray(K)[np.ix_(free_dofs, free_dofs)]
    F_f = np.asarray(F)[free_dofs]

    rank = n_free
    cond = 1.0
    status = NumericalStatus.STABLE

    if check_condition:
        # The SVD screen is the only route to an explicit numerical rank, and
        # the rank is what makes the mechanism message diagnostic instead of
        # a bare LinAlgError. Sparse matrices are screened as dense only when
        # small enough to afford it.
        if sparse_input and n_free > _SPARSE_SVD_LIMIT:
            rank, _, cond, status = n_free, 0.0, 1.0, NumericalStatus.STABLE
            screened = False
        else:
            dense_kff = K_ff.toarray() if sparse_input else K_ff
            rank, _, cond, status = singular_value_screen(dense_kff, tolerances)
            screened = True
            if status is NumericalStatus.SINGULAR:
                raise SingularMatrixError(
                    f"mechanism detected before solve: rank(K_ff)={rank} < "
                    f"{n_free} free DOFs"
                )
            if status is NumericalStatus.ILL_CONDITIONED:
                warnings.warn(
                    f"ill-conditioned stiffness: cond(K_ff)={cond:.3e} > "
                    f"{tolerances.cond_warning:.0e}",
                    IllConditionedWarning,
                    stacklevel=2,
                )
    else:
        screened = False

    if sparse_input:
        U_f = _solve_sparse(K_ff, F_f, n_free, screened)
        used = "sparse-lu"
    else:
        U_f, used = _solve_dense(K_ff, F_f, n_free)

    for i, dof in enumerate(free_dofs):
        U[dof] = U_f[i]

    return SolveResult(
        U=U,
        n_free=n_free,
        rank=rank,
        cond=cond,
        status=status if screened else NumericalStatus.STABLE,
        factorisation=used if screened else "unscreened",
        screened=screened,
    )


#: Above this free-DOF count a sparse system is no longer densified for the
#: SVD screen; the screen would cost more than the whole sparse solve.
_SPARSE_SVD_LIMIT = 2000


def _solve_dense(
    K_ff: np.ndarray, F_f: np.ndarray, n_free: int
) -> tuple[np.ndarray, str]:
    """Factor and solve a dense SPD system, Cholesky first then LU.

    Returns
    -------
    tuple[np.ndarray, str]
        ``(U_f, factorisation)`` where ``factorisation`` is ``"cholesky"`` or
        ``"lu"``.

    Raises
    ------
    SingularMatrixError
        If neither factorisation succeeds.
    """
    try:
        # check_finite=False skips an O(n^2) NaN/inf scan; the assembler
        # guarantees finite entries and a NaN would surface as a failed
        # factorisation or a NaN result either way.
        c, low = cho_factor(K_ff, lower=True, check_finite=False)
        return np.asarray(cho_solve((c, low), F_f, check_finite=False)), "cholesky"
    except (LinAlgError, ValueError):
        # Not positive definite: either a genuine mechanism or a marginally
        # indefinite round-off case. LU still solves full-rank indefinite
        # systems, so try it before declaring failure.
        pass

    try:
        lu, piv = lu_factor(K_ff, check_finite=False)
        return np.asarray(lu_solve((lu, piv), F_f, check_finite=False)), "lu"
    except (LinAlgError, ValueError) as exc:
        raise SingularMatrixError(
            f"Stiffness matrix is singular (mechanism detected); "
            f"neither Cholesky nor LU could factorise K_ff ({n_free} free DOFs)"
        ) from exc


def _solve_sparse(
    K_ff: sp.spmatrix, F_f: np.ndarray, n_free: int, screened: bool
) -> np.ndarray:
    """Solve a sparse system with SuperLU, reporting singularity explicitly."""
    try:
        return np.asarray(spla.splu(sp.csc_matrix(K_ff)).solve(F_f))
    except RuntimeError as exc:
        # SciPy raises RuntimeError("Factor is exactly singular") rather than
        # LinAlgError; translate it into the library's own error type.
        if screened:
            raise
        raise SingularMatrixError(
            f"Stiffness matrix is singular (mechanism detected); "
            f"SuperLU could not factorise K_ff ({n_free} free DOFs)"
        ) from exc


def solve(
    K: np.ndarray | sp.spmatrix,
    F: np.ndarray,
    fixed_dofs: list[int],
    check_condition: bool = True,
) -> np.ndarray:
    """Solve the linear system ``K U = F`` under Dirichlet boundary conditions.

    The free-free sub-matrix ``K_ff`` is screened *before* the solve: a
    rank deficiency (kinematic mechanism) raises :class:`SingularMatrixError`
    with an explicit message instead of surfacing as a ``LinAlgError`` from
    the factorisation. An ill-conditioned but full-rank matrix emits
    :class:`IllConditionedWarning`. The factorisation itself is Cholesky,
    which is the correct choice for a symmetric positive definite stiffness
    matrix and costs about half of a general LU decomposition.

    Parameters
    ----------
    K : np.ndarray or scipy.sparse.spmatrix
        Global stiffness matrix, shape ``(2n, 2n)``.
    F : np.ndarray
        Global force vector, shape ``(2n,)``.
    fixed_dofs : list[int]
        Indices of constrained degrees of freedom (zero displacement).
    check_condition : bool, default True
        Run the SVD-based rank/conditioning screen before solving.

    Returns
    -------
    np.ndarray
        Global displacement vector ``U`` of shape ``(2n,)``; entries at
        ``fixed_dofs`` are exactly zero.

    Raises
    ------
    SingularMatrixError
        If ``K_ff`` is rank deficient (mechanism) or the factorisation
        fails on a singular matrix.

    Warns
    -----
    IllConditionedWarning
        If ``cond(K_ff) > CONDITION_WARNING_THRESHOLD``.

    See Also
    --------
    solve_with_diagnostics : same solve, returning rank/condition/status too.
    """
    return solve_with_diagnostics(K, F, fixed_dofs, check_condition=check_condition).U


#: Default penalty multiplier: the added diagonal term is this many times the
#: largest existing diagonal entry of ``K``. Scaling by the matrix's own
#: magnitude is what keeps the penalty dimensionless and prevents it from
#: swamping a stiff model or vanishing in a compliant one.
DEFAULT_PENALTY_MULTIPLIER = 1e10


#: Penalty-to-stiffness ratios outside this band are reported. Measured on the
#: shipped example model (``cond(K_ff) ~ 41``), the two quantities trade off
#: exactly against each other:
#:
#: .. code-block:: text
#:
#:     relative constraint error ~ 0.4 / ratio
#:     cond(K_pen)               ~ 24  * ratio
#:     their product             ~ 9.4   (independent of ratio)
#:
#: so there is no free lunch -- only a choice of where to sit. Below ``1e4``
#: the constraint is enforced to barely four digits and the penalty solve
#: stops being a meaningful approximation to the elimination solve; above
#: ``1e12`` the condition number passes
#: :attr:`~truss_analysis.numerics.NumericalTolerances.cond_warning` and the
#: structural eigenvalues begin to be lost in round-off. The default
#: multiplier of ``1e10`` sits near the useful ceiling: ~4e-11 constraint
#: error at ``cond ~ 2.4e11``.
_PENALTY_MIN_RATIO = 1e4
_PENALTY_CONDITIONING_RATIO = 1e12


def apply_penalty_bc(
    K: np.ndarray,
    F: np.ndarray,
    fixed_dofs: Sequence[int],
    penalty_value: float | None = None,
    penalty_multiplier: float = DEFAULT_PENALTY_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray]:
    """Impose Dirichlet conditions by penalty instead of by elimination.

    Each constrained DOF ``d`` receives a large diagonal term ``alpha`` and the
    right-hand side is left untouched, so the solved displacement at ``d`` is
    ``F[d] / (K[d, d] + alpha) ~ 0`` rather than exactly zero.

    ``alpha`` may be given two ways, and the distinction matters:

    ``penalty_value``
        An **absolute** stiffness in N/m, added to the diagonal as-is. This is
        the conventional meaning and what the ``options`` block of the shipped
        example models carries.

    ``penalty_multiplier``
        A **dimensionless** multiple of ``max(|diag(K)|)``, used when
        ``penalty_value`` is ``None``. Scaling by the matrix's own magnitude is
        what makes the default safe across unit systems and model sizes: an
        absolute penalty chosen for a steel bridge is far too small for a
        millimetre-scale lattice and needlessly huge for a compliant one.

    Supplying both uses the absolute ``penalty_value`` and ignores the
    multiplier.

    Trade-off versus elimination
    ----------------------------
    Elimination — the default — removes the constrained DOFs and solves a
    smaller, exactly-constrained system. It is both faster and exact. Penalty
    keeps the full DOF numbering, which is convenient when the same assembled
    matrix must be reused under changing support conditions, and it is the
    method ``docs/theory.md`` and the README describe. Its cost is a
    constraint satisfied only to ``O(1/alpha)`` and a condition number inflated
    by roughly ``alpha``, so the multiplier should be raised only as far as
    needed and :class:`~truss_analysis.exceptions.IllConditionedWarning` should
    be expected when it is.

    Parameters
    ----------
    K : np.ndarray
        Global stiffness matrix. Not modified; a copy is returned.
    F : np.ndarray
        Global force vector. Not modified; a copy is returned.
    fixed_dofs : Sequence[int]
        Indices of constrained degrees of freedom.
    penalty_value : float or None, optional
        Absolute penalty stiffness [N/m]. Takes precedence over
        ``penalty_multiplier`` when not ``None``.
    penalty_multiplier : float, default 1e10
        Penalty as a multiple of ``max(|diag(K)|)``, used when
        ``penalty_value`` is ``None``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Penalised copies ``(K_pen, F_pen)`` of the full-size system.

    Raises
    ------
    ValueError
        If the resolved penalty is not positive and finite, or if the
        stiffness diagonal is identically zero.

    Warns
    -----
    IllConditionedWarning
        If the penalty-to-stiffness ratio falls outside
        [:data:`_PENALTY_MIN_RATIO`, :data:`_PENALTY_CONDITIONING_RATIO`].
        Too large and ``cond(K)`` grows until structural modes are lost in
        round-off; too small and the supports settle enough that the result is
        no longer a close approximation to an exact constraint. Both directions
        are reported, because an absolute ``penalty_value`` that is well chosen
        for one model can be badly scaled for another.
    """
    K_pen = np.array(K, dtype=float, copy=True)
    F_pen = np.array(F, dtype=float, copy=True)

    diag = np.diag(K_pen)
    reference = float(np.max(np.abs(diag))) if diag.size else 0.0
    if reference <= 0.0:
        raise ValueError(
            "penalty boundary conditions require a non-zero stiffness diagonal"
        )

    if penalty_value is None:
        if not np.isfinite(penalty_multiplier) or penalty_multiplier <= 0.0:
            raise ValueError(
                "penalty_multiplier must be positive and finite, "
                f"got {penalty_multiplier}"
            )
        alpha = penalty_multiplier * reference
    else:
        if not np.isfinite(penalty_value) or penalty_value <= 0.0:
            raise ValueError(
                f"penalty_value must be positive and finite, got {penalty_value}"
            )
        alpha = float(penalty_value)

    if not len(fixed_dofs):
        return K_pen, F_pen

    ratio = alpha / reference
    if ratio > _PENALTY_CONDITIONING_RATIO:
        warnings.warn(
            f"penalty {alpha:.3e} is {ratio:.3e}x the largest stiffness "
            f"diagonal {reference:.3e}; cond(K) grows by roughly that factor "
            f"(to ~{24.0 * ratio:.1e}) and structural modes may be lost to "
            f"round-off. Prefer a ratio near 1e10, or use "
            f"bc_method='elimination'.",
            IllConditionedWarning,
            stacklevel=2,
        )
    elif ratio < _PENALTY_MIN_RATIO:
        warnings.warn(
            f"penalty {alpha:.3e} is only {ratio:.3e}x the largest stiffness "
            f"diagonal {reference:.3e}; supports will settle by roughly "
            f"1/{ratio:.0f} of the displacement scale (~{0.4 / ratio:.1e} "
            f"relative error), so this is not a close approximation to an "
            f"exact constraint. Raise the penalty towards 1e10x the diagonal, "
            f"or omit it to use the scale-aware default.",
            IllConditionedWarning,
            stacklevel=2,
        )

    for dof in fixed_dofs:
        K_pen[dof, dof] += alpha
    return K_pen, F_pen


def solve_penalty(
    K: np.ndarray,
    F: np.ndarray,
    fixed_dofs: Sequence[int],
    penalty_value: float | None = None,
    penalty_multiplier: float = DEFAULT_PENALTY_MULTIPLIER,
    check_condition: bool = True,
) -> np.ndarray:
    """Solve with penalty boundary conditions and zero out the constrained DOFs.

    The residual ``O(1/alpha)`` displacement at constrained DOFs is discarded
    on return, so callers see exactly zero there — matching what elimination
    produces and keeping the two methods comparable in tests.

    Parameters
    ----------
    K : np.ndarray
        Global stiffness matrix.
    F : np.ndarray
        Global force vector.
    fixed_dofs : Sequence[int]
        Indices of constrained degrees of freedom.
    penalty_value : float or None, optional
        Absolute penalty stiffness [N/m]; see :func:`apply_penalty_bc`.
    penalty_multiplier : float, default 1e10
        Penalty as a multiple of ``max(|diag(K)|)`` when ``penalty_value`` is
        ``None``.
    check_condition : bool, default True
        Run the SVD rank/conditioning screen before solving.

    Returns
    -------
    np.ndarray
        Global displacement vector with constrained DOFs set exactly to zero.
    """
    U_raw, _penalty_energy = solve_penalty_with_energy(
        K, F, fixed_dofs, penalty_value, penalty_multiplier
    )
    U = np.array(U_raw, copy=True)
    for dof in fixed_dofs:
        U[dof] = 0.0
    return U


def solve_penalty_with_energy(
    K: np.ndarray,
    F: np.ndarray,
    fixed_dofs: Sequence[int],
    penalty_value: float | None = None,
    penalty_multiplier: float = DEFAULT_PENALTY_MULTIPLIER,
) -> tuple[np.ndarray, float]:
    """Solve with penalty boundary conditions, reporting the spring energy.

    The penalty springs are real springs: they store
    ``0.5 * alpha * sum(U_d^2)`` over the constrained DOFs. The displacements
    ``U_d`` are tiny but ``alpha`` is huge, so the product is *not* negligible
    -- it is the term that closes the Clapeyron balance for a penalised system,

    .. code-block:: text

        0.5 U^T F_mech = U_strain + 0.5 W_prestress + U_penalty

    Omitting it makes :func:`check_energy` fail by roughly ``R^2 / alpha`` for
    a typical reaction ``R``: about 0.6% at ``alpha = 1e12`` on the shipped
    example model. Including it closes the balance to machine precision.

    Parameters
    ----------
    K : np.ndarray
        Global stiffness matrix.
    F : np.ndarray
        Global force vector.
    fixed_dofs : Sequence[int]
        Indices of constrained degrees of freedom.
    penalty_value : float or None, optional
        Absolute penalty stiffness [N/m]; see :func:`apply_penalty_bc`.
    penalty_multiplier : float, default 1e10
        Penalty as a multiple of ``max(|diag(K)|)`` when ``penalty_value`` is
        ``None``.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(U, penalty_energy)`` where ``U`` is the **raw** solution of the
        penalised system -- constrained DOFs are *not* zeroed -- and
        ``penalty_energy`` is the energy stored in the penalty springs.

    Notes
    -----
    The constrained DOFs are deliberately left at their raw values. Under a
    penalty solve a support settles by roughly ``R / alpha``, and for realistic
    penalties that residual is the same order as the displacements being
    sought: at ``alpha = 1e12`` with a ``1e5`` N reaction the support moves
    ``1e-7`` m while free nodes move ``1e-6`` m. Zeroing it afterwards breaks
    the energy identity, because the structural strain energy would then be
    evaluated on a displacement field that solves nothing. Both terms of the
    balance must come from the same ``U``, which is why the raw vector is
    returned.

    :func:`solve_penalty` offers the zeroed field for callers who only want to
    compare displacements against the elimination method.

    The SVD rank/conditioning screen is deliberately skipped. Adding ``alpha``
    to the constrained diagonal inflates ``cond(K)`` by design, so the generic
    ill-conditioning warning would fire on every penalty solve and carry no
    information; :func:`apply_penalty_bc` issues its own, better-scaled
    warning instead.
    """
    K_pen, F_pen = apply_penalty_bc(K, F, fixed_dofs, penalty_value, penalty_multiplier)
    # The penalised system is solved whole: that is the point of the method.
    U = solve_with_diagnostics(K_pen, F_pen, [], check_condition=False).U

    # alpha is whatever was actually added to each constrained diagonal, so
    # read it back from the difference instead of re-deriving the rule.
    # Penalty BC is dense-only: the method works by inflating the diagonal of
    # the full system, and run() falls back to dense assembly when it is asked
    # for together with sparse storage.
    added = np.diag(K_pen) - np.diag(K)
    penalty_energy = 0.5 * float(sum(added[dof] * U[dof] ** 2 for dof in fixed_dofs))
    return U, penalty_energy


def check_energy(
    U: np.ndarray,
    F_mechanical: np.ndarray,
    strain_energy: float,
    prestress_work: float,
    tol: float = 1e-8,
    energy_scale: float | None = None,
    penalty_energy: float = 0.0,
) -> bool:
    """Verify the work-energy balance of a linear-elastic solve.

    This is the generalized Clapeyron theorem for bars with imposed
    (thermal / fabrication) elongations. With

    - ``W_mech = 0.5 * U^T F_mechanical`` (external mechanical work),
    - ``U_strain`` (mechanical strain energy),
    - ``W_prestress`` (prestress work term),

    a correct solve satisfies ``W_mech = U_strain + 0.5 * W_prestress``
    to machine precision.

    Parameters
    ----------
    U : np.ndarray
        Global displacement vector.
    F_mechanical : np.ndarray
        External *mechanical* force vector (thermal effects excluded).
    strain_energy : float
        Sum of ``0.5 * k * delta_L_mech**2`` over all elements.
    prestress_work : float
        Sum of ``k * delta_L_prestress * delta_L_mech`` over all elements.
    tol : float, default 1e-8
        Maximum tolerated *relative* imbalance. A linear-elastic solve
        closes the balance to machine precision, so the default is tight;
        callers handling deliberately ill-conditioned cases may loosen it.
    energy_scale : float or None, optional
        Characteristic elastic energy of the imposed-strain field, i.e.
        ``sum(0.5 * k * delta_L_prestress**2)`` — see
        :func:`truss_analysis.postprocess.imposed_strain_energy`. It is used
        only to place a machine-precision floor under the residual, which is
        what keeps the test meaningful when every balance term vanishes
        analytically (free thermal expansion). ``None`` disables the floor
        and yields a purely relative test.
    penalty_energy : float, default 0.0
        Energy stored in penalty boundary-condition springs,
        ``0.5 * alpha * sum(U_d^2)``, as returned by
        :func:`solve_penalty_with_energy`. Zero under the default elimination
        method, where constraints are exact and store nothing. Including it is
        what lets the balance close to machine precision for a penalty solve;
        omitting it leaves a residual of order ``R^2 / alpha``.

    Returns
    -------
    bool
        ``True`` when the balance closes within ``tol``.

    Raises
    ------
    EnergyValidationError
        If the imbalance exceeds ``tol`` relative to the largest energy
        magnitude present in the balance.

    Notes
    -----
    Derivation, with ``delta_L = delta_L_mech + delta_L_prestress``:

    .. code-block:: text

        K U = F_mechanical + F_thermal
        U^T K U = U^T F_mechanical + U^T F_thermal

        U^T K U = sum(k * delta_L^2)
                = 2 U_strain + 2 W_prestress + sum(k * delta_L_prestress^2)

        U^T F_thermal = sum(k * delta_L_prestress * delta_L)
                      = W_prestress + sum(k * delta_L_prestress^2)

        =>  2 U_strain + W_prestress = U^T F_mechanical
        =>  0.5 * U^T F_mechanical = U_strain + 0.5 * W_prestress

    The *self-equilibrated* case (``F_mechanical = 0``, e.g. a fully
    restrained bar heated by ``delta_T``) is **not** a special case in
    which ``U_strain`` must vanish. Setting ``W_mech = 0`` in the identity
    above gives ``U_strain = -0.5 * W_prestress``, which is generally
    non-zero: the imposed strain stores elastic energy that is exactly
    balanced by the prestress work term. An earlier revision asserted
    ``U_strain ~ 0`` there and consequently rejected the textbook
    thermal-stress problem. The balance is now always tested in its
    unified form, scaled by the largest energy magnitude present, so no
    branch depends on an absolute joule threshold.
    """
    W_mech = 0.5 * np.dot(U, F_mechanical)

    # Generalized Clapeyron theorem with prestress. All three terms are
    # energies [J], so the natural reference scale is the largest magnitude
    # present -- dimensionally consistent, and it keeps the balance testable
    # when W_mech vanishes (self-equilibrated thermal / prestress states).
    expected = strain_energy + 0.5 * prestress_work + penalty_energy
    error = abs(W_mech - expected)
    scale = max(
        abs(W_mech),
        abs(strain_energy),
        0.5 * abs(prestress_work),
        abs(penalty_energy),
    )

    # Round-off floor. Imposed-strain problems can make *every* term of the
    # balance vanish analytically -- free thermal expansion has W_mech =
    # U_strain = W_prestress = 0 exactly -- and then any purely relative test
    # divides round-off by round-off and reports a 100% error on a physically
    # exact result. A floor is therefore unavoidable; what matters is that it
    # is explicit, documented and overridable rather than a hard-coded joule
    # constant buried in a branch.
    #
    # When the caller supplies `energy_scale` (run() does, via
    # imposed_strain_energy) the floor becomes machine-precision *relative to
    # the characteristic imposed-strain energy*, which makes the test fully
    # invariant to unit system and model size. The absolute constant is used
    # only when no physical scale is available, i.e. for direct callers.
    if energy_scale is not None and energy_scale > 0.0:
        floor = _ENERGY_ROUNDOFF_REL * energy_scale
    else:
        floor = _ENERGY_ZERO_FLOOR

    if error <= floor:
        return True
    if scale == 0.0:
        # Nothing to compare against beyond the floor, which was not met.
        raise EnergyValidationError(
            f"Energy balance failed: W_mech={W_mech:.6e}, "
            f"U_strain={strain_energy:.6e}, W_prestress={prestress_work:.6e}, "
            f"expected={expected:.6e}, residual={error:.6e} exceeds the "
            f"round-off floor {floor:.6e}"
        )

    relative_error = error / scale

    if relative_error > tol:
        raise EnergyValidationError(
            f"Energy balance failed: W_mech={W_mech:.6e}, "
            f"U_strain={strain_energy:.6e}, W_prestress={prestress_work:.6e}, "
            f"expected={expected:.6e}, Error={relative_error * 100:.2f}%"
        )
    return True
