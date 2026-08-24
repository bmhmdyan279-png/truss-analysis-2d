"""Solver: KU=F and energy validation."""

from __future__ import annotations

import numpy as np

from .exceptions import EnergyValidationError, SingularMatrixError


def solve(
    K: np.ndarray,
    F: np.ndarray,
    fixed_dofs: list[int],
) -> np.ndarray:
    """Solve the linear system KU=F with boundary conditions."""
    n = len(K)
    U = np.zeros(n)
    free_dofs = [i for i in range(n) if i not in fixed_dofs]
    if not free_dofs:
        return U

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]

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
    tol: float = 0.01,
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
