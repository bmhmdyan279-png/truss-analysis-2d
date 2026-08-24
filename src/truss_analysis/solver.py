"""Solver: KU=F and energy validation."""
from __future__ import annotations

import numpy as np

from .exceptions import SingularMatrixError, EnergyValidationError


def solve(
    K: np.ndarray,
    F: np.ndarray,
    fixed_dofs: list[int],
) -> np.ndarray:
    """Solve the linear system KU=F with boundary conditions.

    Args:
        K: Global stiffness matrix
        F: Global force vector
        fixed_dofs: List of fixed DOF indices

    Returns:
        U: Displacement vector
    """
    n = len(K)
    U = np.zeros(n)

    # Free DOFs
    free_dofs = [i for i in range(n) if i not in fixed_dofs]

    if not free_dofs:
        return U

    # Extract submatrices
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]

    # Check for singular matrix
    try:
        U_f = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError:
        raise SingularMatrixError("Stiffness matrix is singular (mechanism detected)")

    # Assemble full displacement vector
    for i, dof in enumerate(free_dofs):
        U[dof] = U_f[i]

    return U


def check_energy(
    U: np.ndarray,
    F_mechanical: np.ndarray,
    strain_energy: float,
    prestress_work: float,
    tol: float = 0.01,
) -> None:
    """Check thermodynamic energy balance (generalized Clapeyron's theorem).

    The correct formula with thermal/fabrication effects is:
        W_mech = U_strain + W_prestress

    where:
        W_mech = ½ U^T F_mechanical (external mechanical work)
        U_strain = Σ ½ k ΔL_mech² (mechanical strain energy)
        W_prestress = Σ k ΔL_prestress ΔL_mech (prestress work)
    """
    W_mech = 0.5 * np.dot(U, F_mechanical)

    # Check for self-equilibrated problems (no external work)
    if abs(W_mech) < 1e-12:
        # For self-equilibrated problems, check U_strain + W_prestress ≈ 0
        total_energy = strain_energy + prestress_work
        if abs(total_energy) > tol:
            raise EnergyValidationError(
                f"Self-equilibrated problem: U_strain + W_prestress = {total_energy:.6e} "
                f"(expected ≈ 0)"
            )
        return

    # Generalized Clapeyron's theorem
    error = abs(W_mech - (strain_energy + prestress_work))
    relative_error = error / abs(W_mech)

    if relative_error > tol:
        raise EnergyValidationError(
            f"Energy balance failed: W_mech={W_mech:.6e}, "
            f"U_strain={strain_energy:.6e}, W_prestress={prestress_work:.6e}, "
            f"Error={relative_error*100:.2f}%"
        )
