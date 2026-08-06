from __future__ import annotations

import numpy as np

from .exceptions import AssemblyError
from .model import Element, Node


def assemble_global_matrices(nodes: list[Node], elements: list[Element]):
    node_map = {node.id: i for i, node in enumerate(nodes)}
    num_dofs = len(nodes) * 2
    K = np.zeros((num_dofs, num_dofs))
    F_ext = np.zeros(num_dofs)
    fixed_dofs = []

    for node in nodes:
        idx = node_map[node.id]
        if node.is_support:
            if node.support_dx:
                fixed_dofs.append(idx * 2)
            if node.support_dy:
                fixed_dofs.append(idx * 2 + 1)

    for elem in elements:
        if elem.node_i not in node_map or elem.node_j not in node_map:
            raise AssemblyError(f"Element {elem.id} references non-existent node.")

        n1_idx = node_map[elem.node_i]
        n2_idx = node_map[elem.node_j]
        n1 = nodes[n1_idx]
        n2 = nodes[n2_idx]

        dx = n2.x - n1.x
        dy = n2.y - n1.y
        L = np.hypot(dx, dy)

        if L < 1e-12:
            raise AssemblyError(f"Element {elem.id} has zero or negative length.")

        c = dx / L
        s = dy / L
        k_axial = elem.E * elem.A / L

        k_local = (
            np.array(
                [
                    [c * c, c * s, -c * c, -c * s],
                    [c * s, s * s, -c * s, -s * s],
                    [-c * c, -c * s, c * c, c * s],
                    [-c * s, -s * s, c * s, s * s],
                ]
            )
            * k_axial
        )

        dofs = [n1_idx * 2, n1_idx * 2 + 1, n2_idx * 2, n2_idx * 2 + 1]
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += k_local[i, j]

        delta_L_thermal = elem.alpha * elem.delta_T * L + elem.delta_L_free
        f_thermal = k_axial * delta_L_thermal

        F_ext[n1_idx * 2] -= f_thermal * c
        F_ext[n1_idx * 2 + 1] -= f_thermal * s
        F_ext[n2_idx * 2] += f_thermal * c
        F_ext[n2_idx * 2 + 1] += f_thermal * s

    return K, F_ext, sorted(list(set(fixed_dofs)))
