"""Assembly: global stiffness matrix and force vectors."""

from __future__ import annotations

import numpy as np

from .exceptions import AssemblyError
from .model import Element, Node


def assemble_global_matrices(
    nodes: list[Node],
    elements: list[Element],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Assemble global stiffness matrix and force vectors.

    Returns:
        K: Global stiffness matrix
        F_ext: External force vector (mechanical + thermal)
        F_mechanical: Mechanical force vector only
        fixed_dofs: List of fixed DOF indices
    """
    n = len(nodes)
    K = np.zeros((2 * n, 2 * n))
    F_ext = np.zeros(2 * n)
    F_mechanical = np.zeros(2 * n)
    fixed_dofs = []

    # Build node index map
    node_map = {node.id: i for i, node in enumerate(nodes)}

    # Assemble element contributions
    for elem in elements:
        if elem.node_i not in node_map or elem.node_j not in node_map:
            raise AssemblyError(f"Element {elem.id} references non-existent nodes")

        i = node_map[elem.node_i]
        j = node_map[elem.node_j]

        # Element geometry
        dx = nodes[j].x - nodes[i].x
        dy = nodes[j].y - nodes[i].y
        L = np.sqrt(dx**2 + dy**2)

        if L < 1e-12:
            raise AssemblyError(f"Element {elem.id} has zero or negative length")

        c = dx / L
        s = dy / L

        # Element stiffness matrix (local to global transformation)
        k = elem.E * elem.A / L
        ke = k * np.array(
            [
                [c**2, c * s, -(c**2), -c * s],
                [c * s, s**2, -c * s, -(s**2)],
                [-(c**2), -c * s, c**2, c * s],
                [-c * s, -(s**2), c * s, s**2],
            ]
        )

        # DOF indices
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]

        # Assemble into global matrix
        for ii in range(4):
            for jj in range(4):
                K[dofs[ii], dofs[jj]] += ke[ii, jj]

        # Thermal/fabrication forces
        if elem.alpha != 0 or elem.delta_L_free != 0:
            delta_L_thermal = elem.alpha * elem.delta_T * L
            delta_L_prestress = delta_L_thermal + elem.delta_L_free

            # Equivalent nodal forces (in global coordinates)
            F_thermal = k * delta_L_prestress
            F_ext[2 * i] -= F_thermal * c
            F_ext[2 * i + 1] -= F_thermal * s
            F_ext[2 * j] += F_thermal * c
            F_ext[2 * j + 1] += F_thermal * s

            # F_mechanical does NOT include thermal forces
            # (they are internal, not external)

    # Apply boundary conditions
    for i, node in enumerate(nodes):
        if node.is_support:
            if node.support_dx:
                fixed_dofs.append(2 * i)
            if node.support_dy:
                fixed_dofs.append(2 * i + 1)

    return K, F_ext, F_mechanical, fixed_dofs
