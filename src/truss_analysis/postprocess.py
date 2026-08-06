import numpy as np

from .model import Element, Node


def calculate_element_forces(nodes: list[Node], elements: list[Element], U):
    node_map = {node.id: i for i, node in enumerate(nodes)}
    results = []
    total_strain_energy = 0.0

    for elem in elements:
        n1_idx = node_map[elem.node_i]
        n2_idx = node_map[elem.node_j]
        n1 = nodes[n1_idx]
        n2 = nodes[n2_idx]

        dx = n2.x - n1.x
        dy = n2.y - n1.y
        L = np.hypot(dx, dy)
        c = dx / L
        s = dy / L

        u1x = U[n1_idx * 2]
        u1y = U[n1_idx * 2 + 1]
        u2x = U[n2_idx * 2]
        u2y = U[n2_idx * 2 + 1]

        delta_l_mech = (u2x - u1x) * c + (u2y - u1y) * s
        delta_l_thermal = elem.alpha * elem.delta_T * L
        delta_l_total = delta_l_mech + delta_l_thermal

        k_axial = elem.E * elem.A / L
        n_force = k_axial * delta_l_total
        u_elem = 0.5 * k_axial * (delta_l_total**2)
        total_strain_energy += u_elem

        results.append(
            {
                "element": elem.id,
                "force": n_force,
                "stress": n_force / elem.A,
                "strain": delta_l_total / L,
                "energy": u_elem,
            }
        )

    return results, total_strain_energy
