from __future__ import annotations

import numpy as np

from .model import Element, Node


def calculate_element_forces(nodes: list[Node], elements: list[Element], U):
    node_map = {node.id: i for i, node in enumerate(nodes)}
    results = []
    total_strain_energy = 0.0
    total_prestress_work = 0.0

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

        delta_l_u = (u2x - u1x) * c + (u2y - u1y) * s
        delta_l_thermal = elem.alpha * elem.delta_T * L + elem.delta_L_free
        delta_l_mech = delta_l_u - delta_l_thermal

        k_axial = elem.E * elem.A / L
        n_force = k_axial * delta_l_mech
        u_elem = 0.5 * k_axial * (delta_l_mech**2)
        total_strain_energy += u_elem

        w_prestress = k_axial * delta_l_thermal * delta_l_mech
        total_prestress_work += w_prestress

        results.append(
            {
                "element": elem.id,
                "force": n_force,
                "stress": n_force / elem.A,
                "strain": delta_l_mech / L,
                "energy": u_elem,
            }
        )

    return results, total_strain_energy, total_prestress_work


def calculate_percentages(results):
    total_e = sum(r.get("energy", 0.0) for r in results)
    for r in results:
        r["pct_U"] = (
            (r.get("energy", 0.0) / total_e * 100.0) if total_e > 1e-12 else 0.0
        )
    return results


def calculate_displacement_scale_factor(nodes, U):
    max_u = np.max(np.abs(U)) if len(U) > 0 else 0.0
    max_dim = max(max(abs(n.x), abs(n.y)) for n in nodes) if nodes else 0.0
    if max_u < 1e-12 or max_dim < 1e-12:
        return 1000.0
    scale = max_dim / max_u
    return min(scale, 1000.0)
