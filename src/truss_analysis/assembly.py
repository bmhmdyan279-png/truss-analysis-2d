from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

from .model import Element, Node


def assemble_global_matrices(
    nodes: dict[str, Node], elements: dict[str, Element]
) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray, list[int]]:
    n_dofs = len(nodes) * 2
    K = np.zeros((n_dofs, n_dofs))
    F_thermal = np.zeros(n_dofs)

    dof_map = {}
    for i, nid in enumerate(nodes.keys()):
        dof_map[nid] = (i * 2, i * 2 + 1)

    for elem in elements.values():
        i_u, i_v = dof_map[elem.node_i]
        j_u, j_v = dof_map[elem.node_j]

        ni = nodes[elem.node_i]
        nj = nodes[elem.node_j]

        dx = nj.x - ni.x
        dy = nj.y - ni.y
        L = np.hypot(dx, dy)
        if L <= 0:
            continue

        c, s = dx / L, dy / L
        k_local = (elem.E * elem.A / L) * np.array(
            [
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ]
        )

        dofs = [i_u, i_v, j_u, j_v]
        K[np.ix_(dofs, dofs)] += k_local

        # CRITICAL-001 Fix: Correct Thermal Force Sign Convention
        if getattr(elem, "delta_L_free", 0) != 0:
            f_mag = (elem.E * elem.A / L) * elem.delta_L_free
            f_local = f_mag * np.array([-c, -s, c, s])
            F_thermal[dofs] += f_local

    fixed_dofs = []
    for i, node in enumerate(nodes.values()):
        if node.is_support:
            if node.support_dx:
                fixed_dofs.append(i * 2)
            if node.support_dy:
                fixed_dofs.append(i * 2 + 1)

    if n_dofs > 100:
        K = sparse.csr_matrix(K)

    return K, F_thermal, fixed_dofs
