"""Solver: KU=F and energy validation."""

from __future__ import annotations

import warnings

import numpy as np

from .exceptions import (
    EnergyValidationError,
    IllConditionedWarning,
    SingularMatrixError,
)

CONDITION_WARNING_THRESHOLD = 1e12


def solve(
    K: np.ndarray,
    F: np.ndarray,
    fixed_dofs: list[int],
    check_condition: bool = True,
) -> np.ndarray:
    """Solve the linear system KU=F with boundary conditions.

    Prompt-7 hardening (D-012 verification): the legacy code only caught
    ``np.linalg.LinAlgError`` AFTER attempting the solve; the rank (SVD) check
    now runs BEFORE the solve and raises ``SingularMatrixError`` with an
    explicit mechanism message.  ``cond(K_ff) > 1e12`` emits an
    ``IllConditionedWarning`` instead of failing silently.  ``fixed_dofs``
    membership uses a set (the legacy ``i not in list`` scan was O(n*m)).
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
        rank = int(np.count_nonzero(sv > s_max * 1e-13)) if s_max > 0.0 else 0
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
    except np.linalg.LinAlgError:
        raise SingularMatrixError("Stiffness matrix is singular (mechanism detected)")

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
    """Check thermodynamic energy balance (generalized Clapeyron theorem).

    The correct formula with thermal/fabrication effects is:
    W_mech = U_strain + 0.5 * W_prestress

    where:
    W_mech = 0.5 * U^T F_mechanical (external mechanical work)
    U_strain = sum of 0.5 * k * (delta_L_mech)^2 (mechanical strain energy)
    W_prestress = sum of k * delta_L_prestress * delta_L_mech

    Derivation:
    K U = F_mechanical + F_thermal
    U^T K U = U^T F_mechanical + U^T F_thermal

    U^T K U = sum(k * delta_L^2) = sum(k * (delta_L_mech + delta_L_prestress)^2)
            = 2 U_strain + 2 W_prestress + sum(k * delta_L_prestress^2)

    U^T F_thermal = sum(k * delta_L_prestress * delta_L)
                  = W_prestress + sum(k * delta_L_prestress^2)

    Therefore: 2 U_strain + W_prestress = U^T F_mechanical
    Or: 0.5 * U^T F_mechanical = U_strain + 0.5 * W_prestress
    """
    # Prompt-7 (DL-030): the default tolerance was 1% with no documented
    # reason; a linear-elastic solve closes the Clapeyron balance to machine
    # precision, so the default is now 1e-8 (relative).  Callers handling
    # deliberately ill-conditioned cases may pass a looser tol.
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
