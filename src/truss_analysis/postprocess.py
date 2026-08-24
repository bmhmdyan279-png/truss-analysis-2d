"""Post-processing: element forces, reactions, equilibrium, buckling."""

from __future__ import annotations

import numpy as np


def calculate_element_forces(nodes, elements, U):
    """Calculate axial forces, strain energy, and prestress work.

    Physical model:
    - delta_L_total: total elongation from nodal displacements
    - delta_L_thermal: thermal elongation = alpha * delta_T * L
    - delta_L_prestress: total prestress elongation (thermal + fabrication)
    - delta_L_mech: mechanical elongation = delta_L_total - delta_L_prestress
    - N: axial force from mechanical elongation only
    """
    results = []
    strain_energy = 0.0
    prestress_work = 0.0
    node_map = {node.id: i for i, node in enumerate(nodes)}

    for elem in elements:
        i = node_map[elem.node_i]
        j = node_map[elem.node_j]
        dx = nodes[j].x - nodes[i].x
        dy = nodes[j].y - nodes[i].y
        L = np.sqrt(dx**2 + dy**2)

        if L < 1e-12:
            results.append({"id": elem.id, "N": 0.0, "status": "ZERO_LENGTH"})
            continue

        c = dx / L
        s = dy / L

        # Nodal displacements
        ui, vi = U[2 * i], U[2 * i + 1]
        uj, vj = U[2 * j], U[2 * j + 1]

        # Total elongation from nodal displacements
        delta_L_total = (uj - ui) * c + (vj - vi) * s

        # Thermal/fabrication elongation
        delta_L_thermal = elem.alpha * elem.delta_T * L
        delta_L_prestress = delta_L_thermal + elem.delta_L_free

        # Mechanical elongation (what causes stress)
        delta_L_mech = delta_L_total - delta_L_prestress

        # Axial stiffness
        k = elem.E * elem.A / L

        # Axial force (positive = tension)
        N = k * delta_L_mech

        # Strain energy (mechanical only)
        strain_energy += 0.5 * k * delta_L_mech**2

        # Prestress work
        prestress_work += k * delta_L_prestress * delta_L_mech

        status = "Tension" if N > 1e-9 else ("Compression" if N < -1e-9 else "Zero")
        results.append(
            {
                "id": elem.id,
                "N": float(N),
                "delta_L_mech": float(delta_L_mech),
                "delta_L_prestress": float(delta_L_prestress),
                "status": status,
            }
        )

    return results, float(strain_energy), float(prestress_work)


def calculate_reactions(nodes, K, U, F_ext, fixed_dofs):
    """Support reactions: R = K*U - F_ext at constrained DOFs."""
    R = K @ U - F_ext
    fixed = set(fixed_dofs)
    reactions = {}
    for i, node in enumerate(nodes):
        dx_fixed = 2 * i in fixed
        dy_fixed = 2 * i + 1 in fixed
        if dx_fixed or dy_fixed:
            reactions[node.id] = {
                "Fx": float(R[2 * i]) if dx_fixed else 0.0,
                "Fy": float(R[2 * i + 1]) if dy_fixed else 0.0,
            }
    return reactions


def check_equilibrium(nodes, reactions, applied_loads, tol=1e-6):
    """Global static equilibrium: sum(Fx)=0, sum(Fy)=0, sum(M)=0."""
    coords = {node.id: (node.x, node.y) for node in nodes}
    sum_fx = sum_fy = sum_m = 0.0

    # Sum reactions
    for nid, rec in reactions.items():
        x, y = coords[nid]
        sum_fx += rec["Fx"]
        sum_fy += rec["Fy"]
        sum_m += x * rec["Fy"] - y * rec["Fx"]

    # Sum applied loads
    for lf in applied_loads:
        x, y = coords[str(lf["node_id"])]
        sum_fx += lf["Fx"]
        sum_fy += lf["Fy"]
        sum_m += x * lf["Fy"] - y * lf["Fx"]

    # Tolerance scaling
    ref = 0.0
    for lf in applied_loads:
        ref += abs(lf["Fx"]) + abs(lf["Fy"])
    for rec in reactions.values():
        ref += abs(rec["Fx"]) + abs(rec["Fy"])
    limit = tol * max(1.0, ref)

    return {
        "sum_fx": float(sum_fx),
        "sum_fy": float(sum_fy),
        "sum_m": float(sum_m),
        "is_valid": bool(
            abs(sum_fx) <= limit
            and abs(sum_fy) <= limit
            and abs(sum_m) <= limit * max(1.0, ref)
        ),
    }


def calculate_buckling(nodes, elements, results, tol=1e-12):
    """Euler buckling: P_cr = pi^2*E*I/L^2 for compressed members."""
    coords = {node.id: node for node in nodes}
    forces = {str(r.get("id")): float(r.get("N", 0.0)) for r in results}
    report = []

    for e in elements:
        ni, nj = coords[e.node_i], coords[e.node_j]
        L = float(np.hypot(nj.x - ni.x, nj.y - ni.y))
        N = forces.get(str(e.id), 0.0)
        entry = {
            "id": e.id,
            "N": N,
            "length": L,
            "P_cr": None,
            "ratio": 0.0,
            "slenderness": None,
            "safe": True,
        }

        if N < -tol and L > tol and e.I_sec > tol:
            p_cr = float(np.pi**2 * e.E * e.I_sec / L**2)
            r_gyr = float(np.sqrt(e.I_sec / e.A)) if e.A > tol else 0.0
            entry["P_cr"] = p_cr
            entry["ratio"] = -N / p_cr
            entry["slenderness"] = L / r_gyr if r_gyr > tol else None
            entry["safe"] = bool(entry["ratio"] < 1.0)

        report.append(entry)

    return report
