from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

from .exceptions import EnergyValidationError, SingularMatrixError


def solve(K, F, fixed_dofs, use_sparse=False):
    all_dofs = np.arange(K.shape[0])
    free = np.setdiff1d(all_dofs, fixed_dofs)
    if len(free) == 0:
        return np.zeros(K.shape[0])

    K_ff = K[np.ix_(free, free)]
    F_f = F[free]

    K_dense = K_ff.toarray() if sparse.issparse(K_ff) else K_ff

    # TRUSS-3001: Fail-Fast Rank Check (No silent zero-row deletion)
    if np.linalg.matrix_rank(K_dense) < K_dense.shape[0]:
        raise SingularMatrixError()

    try:
        if use_sparse and sparse.issparse(K_ff):
            U_f = spsolve(K_ff, F_f)
        else:
            U_f = np.linalg.solve(K_dense, F_f)
    except np.linalg.LinAlgError as e:
        raise SingularMatrixError(str(e)) from e

    U = np.zeros(K.shape[0])
    U[free] = U_f
    return U


def check_energy(U, F, strain_energy, tol=0.01):
    W = 0.5 * np.dot(U, F)
    if W == 0 and strain_energy == 0:
        return True
    if W == 0:
        raise EnergyValidationError("کار خارجی صفر است اما انرژی کرنشی وجود دارد.")
    err = abs(strain_energy - W) / abs(W)
    if err > tol:
        raise EnergyValidationError(f"عدم توازن انرژی: {err*100:.2f}% > {tol*100}%")
    return True
