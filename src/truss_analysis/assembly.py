import numpy as np

from .model import Element, Node


class AssemblyError(Exception):
    pass


def assemble_global_matrices(nodes: list[Node], elements: list[Element]):
    n_dofs = len(nodes) * 2
    K_dense = np.zeros((n_dofs, n_dofs))
    F_ext = np.zeros(n_dofs)

    node_map = {node.id: i for i, node in enumerate(nodes)}
    fixed_dofs = []

    for i, node in enumerate(nodes):
        if node.is_support:
            if node.support_dx:
                fixed_dofs.append(i * 2)
            if node.support_dy:
                fixed_dofs.append(i * 2 + 1)

    for elem in elements:
        n1_idx = node_map.get(elem.node_i)
        n2_idx = node_map.get(elem.node_j)
        if n1_idx is None or n2_idx is None:
            raise AssemblyError(f"Element {elem.id} references unknown nodes.")

        n1 = nodes[n1_idx]
        n2 = nodes[n2_idx]

        dx = n2.x - n1.x
        dy = n2.y - n1.y
        L = np.hypot(dx, dy)

        if L <= 1e-12:
            raise AssemblyError(
                f"Element {elem.id} has zero or negative length (L={L})."
            )

        c = dx / L
        s = dy / L
        delta_L_thermal = elem.alpha * elem.delta_T * L
        k_coeff = elem.E * elem.A / L

        # Pre-formatted array to strictly avoid E501
        k_local = k_coeff * np.array(
            [
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ]
        )

        dofs = [n1_idx * 2, n1_idx * 2 + 1, n2_idx * 2, n2_idx * 2 + 1]
        for i_local, i_global in enumerate(dofs):
            for j_local, j_global in enumerate(dofs):
                K_dense[i_global, j_global] += k_local[i_local, j_local]

        f_thermal = (elem.E * elem.A / L) * delta_L_thermal
        F_ext[n1_idx * 2] -= f_thermal * c
        F_ext[n1_idx * 2 + 1] -= f_thermal * s
        F_ext[n2_idx * 2] += f_thermal * c
        F_ext[n2_idx * 2 + 1] += f_thermal * s

    return K_dense, F_ext, fixed_dofs
