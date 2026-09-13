"""Linear solve of the global system ``K U = F`` plus energy verification."""

from __future__ import annotations

import warnings

import numpy as np

from .exceptions import (
    EnergyValidationError,
    IllConditionedWarning,
    SingularMatrixError,
)

#: cond(K_ff) above this threshold emits :class:`IllConditionedWarning`.
CONDITION_WARNING_THRESHOLD = 1e12

#: Relative singular-value cutoff used for the numerical rank estimate.
_RANK_REL_CUTOFF = 1e-13


def solve(
    K: np.ndarray,
    F: np.ndarray,
    fixed_dofs: list[int],
    check_condition: bool = True,
) -> np.ndarray:
    """Solve the linear system ``K U = F`` under Dirichlet boundary conditions.

    The free-free sub-matrix ``K_ff`` is screened *before* the solve: a
    rank deficiency (kinematic mechanism) raises :class:`SingularMatrixError`
    with an explicit message instead of surfacing as a ``LinAlgError`` from
    the factorisation. An ill-conditioned but full-rank matrix emits
    :class:`IllConditionedWarning`.

    Parameters
    ----------
    K : np.ndarray
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
    """
    n = len(K)
    U = np.zeros(n)
    fixed_set = set(fixed_dofs)
    free_dofs = [i for i in range(n) if i not in fixed_set]
    if not free_dofs:
        return U

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]

    if check_condition:
        sv = np.linalg.svd(K_ff, compute_uv=False)
        s_max = float(sv[0]) if sv.size else 0.0
        rank = (
            int(np.count_nonzero(sv > s_max * _RANK_REL_CUTOFF)) if s_max > 0.0 else 0
        )
        if rank < len(free_dofs):
            raise SingularMatrixError(
                f"mechanism detected before solve: rank(K_ff)={rank} < "
                f"{len(free_dofs)} free DOFs"
            )
        if s_max > 0.0 and sv[-1] > 0.0:
            cond = s_max / float(sv[-1])
            if cond > CONDITION_WARNING_THRESHOLD:
                warnings.warn(
                    f"ill-conditioned stiffness: cond(K_ff)={cond:.3e} > "
                    f"{CONDITION_WARNING_THRESHOLD:.0e}",
                    IllConditionedWarning,
                    stacklevel=2,
                )

    try:
        U_f = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError as exc:
        raise SingularMatrixError(
            "Stiffness matrix is singular (mechanism detected)"
        ) from exc

    for i, dof in enumerate(free_dofs):
        U[dof] = U_f[i]
    return U


def check_energy(
    U: np.ndarray,
    F_mechanical: np.ndarray,
    strain_energy: float,
    prestress_work: float,
    tol: float = 1e-8,
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

    Returns
    -------
    bool
        ``True`` when the balance closes within ``tol``.

    Raises
    ------
    EnergyValidationError
        If the relative imbalance exceeds ``tol``, or if a
        self-equilibrated problem (no net mechanical work) carries
        non-negligible strain energy.

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
    """
    W_mech = 0.5 * np.dot(U, F_mechanical)

    # For self-equilibrated problems (no external mechanical loads)
    if abs(W_mech) < 1e-12:
        if abs(strain_energy) > tol:
            raise EnergyValidationError(
                f"Self-equilibrated problem: U_strain = {strain_energy:.6e} "
                f"(expected approx 0)"
            )
        return True

    # Generalized Clapeyron theorem with prestress
    expected = strain_energy + 0.5 * prestress_work
    error = abs(W_mech - expected)
    relative_error = error / abs(W_mech)

    if relative_error > tol:
        raise EnergyValidationError(
            f"Energy balance failed: W_mech={W_mech:.6e}, "
            f"U_strain={strain_energy:.6e}, W_prestress={prestress_work:.6e}, "
            f"expected={expected:.6e}, Error={relative_error * 100:.2f}%"
        )
    return True
