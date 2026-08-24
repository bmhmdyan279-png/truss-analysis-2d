"""Post-processing: forces, energy, reactions, equilibrium."""

from __future__ import annotations

import numpy as np

from .model import Element, Node


def calculate_element_forces(
    nodes: list[Node],
    elements: list[Element],
    U: np.ndarray,
) -> tuple[list[dict], float, float]:
    """Calculate element forces, strain energy, and prestress work.

    Returns:
        results: List of dicts with element forces
        strain_energy: Total strain energy (mechanical)
        prestress_work: Work done by prestress (thermal/fabrication)
    """
    results = []
    strain_energy = 0.0
    prestress_work = 0.0

    for elem in elements:
        i = next(j for j, n in enumerate(nodes) if n.id == elem.node_i)
        j = next(j for j, n in enumerate(nodes) if n.id == elem.node_j)

        # Element geometry
        dx = nodes[j].x - nodes[i].x
        dy = nodes[j].y - nodes[i].y
        L = np.sqrt(dx**2 + dy**2)

        if L < 1e-12:
            raise ValueError(f"Element {elem.id} has zero length")

        c = dx / L
        s = dy / L

        # Displacements
        u_i = U[2 * i : 2 * i + 2]
        u_j = U[2 * j : 2 * j + 2]

        # Axial deformation
        delta_L = (u_j[0] - u_i[0]) * c + (u_j[1] - u_i[1]) * s

        # Thermal/fabrication effects
        delta_L_thermal = elem.alpha * elem.delta_T * L
        delta_L_free = elem.delta_L_free
        delta_L_prestress = delta_L_thermal + delta_L_free

        # Mechanical deformation
        delta_L_mech = delta_L - delta_L_prestress

        # Axial stiffness
        k = elem.E * elem.A / L

        # Axial force (positive = tension)
        force = k * delta_L_mech

        # Strain energy (mechanical only)
        U_elem = 0.5 * k * delta_L_mech**2
        strain_energy += U_elem

        # Prestress work
        W_prestress_elem = k * delta_L_prestress * delta_L_mech
        prestress_work += W_prestress_elem

        # Buckling check (Euler)
        slenderness_ratio = None
        buckling_warning = None
        if elem.I_sec > 0:
            # Critical buckling load (Euler)
            P_cr = (
                np.pi**2 * elem.E * elem.I_sec / (elem.effective_length_factor * L) ** 2
            )
            slenderness_ratio = L / np.sqrt(elem.I_sec / elem.A)

            if abs(force) > 0 and force < 0:  # Compression
                safety_factor = abs(P_cr / force)
                if safety_factor < 1.0:
                    buckling_warning = f"BUCKLING! SF={safety_factor:.2f} < 1"
                elif safety_factor < 2.0:
                    buckling_warning = f"Warning: SF={safety_factor:.2f} < 2"

        results.append(
            {
                "element": elem.id,
                "force": force,
                "stress": force / elem.A,
                "strain": delta_L_mech / L,
                "length": L,
                "slenderness_ratio": slenderness_ratio,
                "buckling_warning": buckling_warning,
            }
        )

    return results, strain_energy, prestress_work


def calculate_reactions(
    nodes: list[Node],
    elements: list[Element],
    U: np.ndarray,
    F_ext: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Calculate support reactions using R = KU - F_ext.

    Returns:
        Dictionary mapping node IDs to reaction forces {Fx, Fy}
    """
    from .assembly import assemble_global_matrices

    K, _, _, _ = assemble_global_matrices(nodes, elements)

    # Full force vector: R = KU - F_ext
    F_total = K @ U
    R = F_total - F_ext

    reactions = {}
    for i, node in enumerate(nodes):
        if node.is_support:
            reactions[node.id] = {
                "Rx": R[2 * i],
                "Ry": R[2 * i + 1],
            }

    return reactions


def check_equilibrium(
    nodes: list[Node],
    reactions: dict[str, dict[str, float]],
    F_ext: np.ndarray,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Check static equilibrium: ΣFx=0, ΣFy=0, ΣM=0.

    Returns:
        Dictionary with equilibrium errors
    """
    # Sum of external forces
    Fx_ext = sum(F_ext[2 * i] for i in range(len(nodes)))
    Fy_ext = sum(F_ext[2 * i + 1] for i in range(len(nodes)))

    # Sum of reactions
    Rx_sum = sum(r["Rx"] for r in reactions.values())
    Ry_sum = sum(r["Ry"] for r in reactions.values())

    # Equilibrium check
    delta_Fx = Fx_ext + Rx_sum
    delta_Fy = Fy_ext + Ry_sum

    # Moment about first support node
    support_nodes = [(n, reactions[n.id]) for n in nodes if n.id in reactions]
    if support_nodes:
        ref_node = support_nodes[0][0]
        M_ext = 0.0
        for i, node in enumerate(nodes):
            dx = node.x - ref_node.x
            dy = node.y - ref_node.y
            M_ext += dx * F_ext[2 * i + 1] - dy * F_ext[2 * i]

        M_react = 0.0
        for node, react in support_nodes:
            dx = node.x - ref_node.x
            dy = node.y - ref_node.y
            M_react += dx * react["Ry"] - dy * react["Rx"]

        delta_M = M_ext + M_react
    else:
        delta_M = 0.0

    errors = {
        "delta_Fx": delta_Fx,
        "delta_Fy": delta_Fy,
        "delta_M": delta_M,
    }

    # Check if within tolerance
    max_error = max(abs(delta_Fx), abs(delta_Fy), abs(delta_M))
    if max_error > tol * max(1.0, abs(Fx_ext), abs(Fy_ext)):
        print("⚠️  Equilibrium check failed:")
        print(f"   ΣFx error: {delta_Fx:.6e}")
        print(f"   ΣFy error: {delta_Fy:.6e}")
        print(f"   ΣM error: {delta_M:.6e}")

    return errors


def calculate_displacement_scale_factor(
    nodes: list[Node],
    U: np.ndarray,
    max_scale: float = 1000.0,
) -> float:
    """Calculate optimal scale factor for deformation visualization.

    The scale factor is chosen so that the maximum displacement
    is visible but not exaggerated beyond max_scale.

    Args:
        nodes: List of nodes
        U: Displacement vector
        max_scale: Maximum allowed scale factor

    Returns:
        Optimal scale factor
    """
    max_disp = 0.0
    for i in range(len(nodes)):
        ux = U[2 * i]
        uy = U[2 * i + 1]
        disp = np.sqrt(ux**2 + uy**2)
        max_disp = max(max_disp, disp)

    if max_disp < 1e-12:
        return 1.0

    # Scale so max displacement is about 5% of structure size
    max_x = max(n.x for n in nodes) - min(n.x for n in nodes)
    max_y = max(n.y for n in nodes) - min(n.y for n in nodes)
    structure_size = max(max_x, max_y, 1.0)

    target_disp = 0.05 * structure_size
    scale = target_disp / max_disp

    return min(scale, max_scale)


def calculate_percentages(
    results: list[dict],
    total_energy: float | None = None,
) -> list[dict]:
    """Calculate percentage contribution of each element's energy.

    Args:
        results: List of element result dicts (must have 'energy' key)
        total_energy: Total strain energy (computed if None)

    Returns:
        Results list with added 'pct_U' field
    """
    if total_energy is None:
        total_energy = sum(r.get("energy", 0.0) for r in results)

    for r in results:
        energy = r.get("energy", 0.0)
        if total_energy > 0:
            r["pct_U"] = (energy / total_energy) * 100.0
        else:
            r["pct_U"] = 0.0

    return results
