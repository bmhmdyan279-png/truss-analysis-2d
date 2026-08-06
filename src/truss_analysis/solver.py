import numpy as np


class SingularMatrixError(Exception):
    pass


def solve(K, F, fixed_dofs, use_sparse=False):
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


def check_energy(U, F, strain_energy, tol=0.01):
    W = 0.5 * np.dot(U, F)
    if abs(W) < 1e-12 and abs(strain_energy) < 1e-12:
        return True
    if abs(W) < 1e-12:
        msg = (
            "Energy validation failed: External work is zero but strain energy is "
            f"{strain_energy}"
        )
        raise ValueError(msg)

    err = abs(strain_energy - W) / abs(W)
    if err > tol:
        msg2 = (
            "Energy validation failed: Error {err_val:.4%} exceeds tolerance "
            "{tol_val:.2%}"
        )
        raise ValueError(msg2.format(err_val=err, tol_val=tol))
    return True


solve_truss = solve
