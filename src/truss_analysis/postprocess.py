from __future__ import annotations

import numpy as np

from .model import Element, Node

MIN_SCALE = 1.0
MAX_SCALE = 1000.0
ZERO_ENERGY_TOL = 1e-12
ZERO_DISP_TOL = 1e-12


def calculate_element_forces(
    nodes: dict[str, Node],
    elements: dict[str, Element],
    u_vec: np.ndarray,
    dof_map: dict,
) -> list[dict]:
    results = []
    for elem in elements.values():
        i_u, i_v = dof_map[elem.node_i]
        j_u, j_v = dof_map[elem.node_j]

        ni = nodes[elem.node_i]
        nj = nodes[elem.node_j]

        dx = nj.x - ni.x
        dy = nj.y - ni.y
        length = np.hypot(dx, dy)
        if length <= 0:
            continue

        c, s = dx / length, dy / length
        u_local = np.array([-c, -s, c, s]) @ u_vec[[i_u, i_v, j_u, j_v]]

        delta_l_mech = u_local - getattr(elem, "delta_L_free", 0.0)
        n_force = (elem.E * elem.A / length) * delta_l_mech
        u_elem = 0.5 * n_force * delta_l_mech

        results.append(
            {
                "element_id": elem.id,
                "force": n_force,
                "stress": n_force / elem.A,
                "strain": delta_l_mech / length,
                "energy": max(0.0, u_elem),
                "length": length,
            }
        )
    return results


def calculate_percentages(results: list[dict]) -> list[dict]:
    total_energy = sum(r["energy"] for r in results if r["energy"] > 0)

    if total_energy <= ZERO_ENERGY_TOL:
        for r in results:
            r["pct_U"] = 0.0
    else:
        for r in results:
            energy = r.get("energy", 0.0)
            if energy < 0 or not np.isfinite(energy):
                r["pct_U"] = 0.0
            else:
                r["pct_U"] = 100.0 * energy / total_energy
    return results


def calculate_displacement_scale_factor(
    nodes: dict[str, Node], u_vec: np.ndarray
) -> float:
    if len(u_vec) == 0:
        return 1.0

    max_disp = np.max(np.abs(u_vec))
    if max_disp < ZERO_DISP_TOL:
        return 1.0

    xs = [n.x for n in nodes.values()]
    ys = [n.y for n in nodes.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if span <= 0:
        span = 1.0

    target_ratio = 0.1
    scale_factor = target_ratio * span / max_disp

    return float(np.clip(scale_factor, MIN_SCALE, MAX_SCALE))
