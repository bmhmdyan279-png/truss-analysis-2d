from __future__ import annotations

import numpy as np

from .exceptions import EnergyValidationError, SingularMatrixError


def solve(K, F, fixed_dofs):
    all_dofs = np.arange(K.shape[0])
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    if len(free_dofs) == 0:
        return np.zeros_like(F)

    K_free = K[np.ix_(free_dofs, free_dofs)]
    F_free = F[free_dofs]

    if np.linalg.matrix_rank(K_free) < K_free.shape[0]:
        raise SingularMatrixError("Global stiffness matrix is singular.")

    U_free = np.linalg.solve(K_free, F_free)

    U = np.zeros_like(F)
    U[free_dofs] = U_free
    return U


def check_energy(U, F_mechanical, strain_energy, prestress_work, tol=0.01):
    W_mech = 0.5 * np.dot(U, F_mechanical)
    expected = strain_energy + prestress_work

    if abs(W_mech) < 1e-12 and abs(expected) < 1e-12:
        return True

    denominator = max(abs(expected), 1e-12)
    err = abs(W_mech - expected) / denominator

    if err > tol:
        msg = (
            f"Energy validation failed: W_mech={W_mech:.4e}, "
            f"U_strain={strain_energy:.4e}, W_prestress={prestress_work:.4e}, "
            f"Error {err:.4%} exceeds tolerance {tol:.2%}"
        )
        raise EnergyValidationError(msg)

    return True


solve_truss = solve
