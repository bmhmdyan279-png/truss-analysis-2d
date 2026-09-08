"""Independent member-sensitivity cross-checks: adjoint DDM and strain energy.

This module provides verification tools that are *independent* of the main
assembly/solve path:

* the direct differentiation method (DDM) in adjoint form, giving the
  sensitivity of the maximum nodal displacement magnitude with respect to
  each member's cross-sectional area, and
* per-member strain energy computed directly from the element stiffness
  in global coordinates.

Both quantities are standard, self-contained checks of a solved model and
are used to corroborate ranking-based tools elsewhere in the library.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .assembly import assemble_global_matrices
from .model import Element, Node
from .solver import solve


@dataclass(frozen=True)
class SensitivityResult:
    """Result of the independent sensitivity/energy check for one member.

    Attributes
    ----------
    member_id : str
        Identifier of the element.
    ddm_sensitivity : float
        Derivative of the maximum nodal displacement magnitude with
        respect to the member area, ``d(|u|_max)/dA_i`` [m per m^2].
    strain_energy : float
        Member strain energy ``0.5 * u_e^T k_e u_e`` [J], clipped at zero
        against negative floating-point round-off.
    """

    member_id: str
    ddm_sensitivity: float
    strain_energy: float


class IndependentValidator:
    """Cross-validate member importance rankings with DDM and strain energy.

    Parameters
    ----------
    nodes : list[Node]
        Model nodes.
    elements : list[Element]
        Model elements.
    loads : list[Any]
        Nodal loads; each item must expose ``node_id`` and either
        ``fx``/``fy`` or ``Fx``/``Fy`` attributes.
    """

    def __init__(
        self,
        nodes: list[Node],
        elements: list[Element],
        loads: list[Any],
    ) -> None:
        self.nodes = nodes
        self.elements = elements
        self.loads = loads
        self.node_map = {node.id: i for i, node in enumerate(nodes)}

    def compute_baseline(self) -> tuple[np.ndarray, list[int]]:
        """Solve the baseline system ``K U = F`` including the nodal loads.

        Returns
        -------
        tuple[np.ndarray, list[int]]
            ``(U, fixed_dofs)``: the global displacement vector and the
            list of constrained DOF indices.
        """
        K, F_ext, _, fixed_dofs = assemble_global_matrices(self.nodes, self.elements)

        # Apply external loads
        for load in self.loads:
            idx = self.node_map[load.node_id]
            fx = getattr(load, "fx", getattr(load, "Fx", 0.0))
            fy = getattr(load, "fy", getattr(load, "Fy", 0.0))
            F_ext[2 * idx] += fx
            F_ext[2 * idx + 1] += fy

        U = solve(K, F_ext, fixed_dofs)
        return U, fixed_dofs

    def compute_all(self) -> list[SensitivityResult]:
        """Compute DDM sensitivity and strain energy for every element.

        The DDM uses a rank-1 formulation: one LU factorisation of ``K_ff``
        plus one triangular solve for the whole compatibility matrix ``B``
        replaces any explicit inverse. With ``dU/dk_i = -Z_i (b_i^T U_f)``
        and ``dk_i/dA = E/L``:

        ``dU/dA_i = -(E_i / L_i) * Z_i * (b_i^T U_f)``

        Returns
        -------
        list[SensitivityResult]
            One result per element, in element order.
        """
        U, fixed_dofs = self.compute_baseline()
        n = len(self.nodes)
        free_dofs = [i for i in range(2 * n) if i not in fixed_dofs]

        K, _, _, _ = assemble_global_matrices(self.nodes, self.elements)
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        U_f = U[free_dofs]

        # Rank-1 DDM machinery: one LU factorisation plus one solve for the
        # whole compatibility matrix B; no explicit inverse, and singularity
        # is caught up-front by the rank check inside solve().
        node_idx = {nd.id: i for i, nd in enumerate(self.nodes)}
        b_free = np.zeros((len(self.elements), len(free_dofs)))
        for e_i, elem in enumerate(self.elements):
            i_idx = node_idx[elem.node_i]
            j_idx = node_idx[elem.node_j]
            dx = self.nodes[j_idx].x - self.nodes[i_idx].x
            dy = self.nodes[j_idx].y - self.nodes[i_idx].y
            length = math.hypot(dx, dy)
            c, s = dx / length, dy / length
            dofs = [2 * i_idx, 2 * i_idx + 1, 2 * j_idx, 2 * j_idx + 1]
            for k, dof in enumerate(dofs):
                with contextlib.suppress(ValueError):
                    b_free[e_i, free_dofs.index(dof)] = (-c, -s, c, s)[k]
        z_mat = lu_solve(lu_factor(K_ff), b_free.T)

        # Find critical node for max displacement
        disp_magnitudes = np.hypot(U[0::2], U[1::2])
        crit_node_idx = int(np.argmax(disp_magnitudes))
        d_max = disp_magnitudes[crit_node_idx]

        if d_max < 1e-15:
            d_max = 1.0

        crit_dof_x = 2 * crit_node_idx
        crit_dof_y = 2 * crit_node_idx + 1

        results: list[SensitivityResult] = []

        for e_i, elem in enumerate(self.elements):
            i_idx = self.node_map[elem.node_i]
            j_idx = self.node_map[elem.node_j]

            dx = self.nodes[j_idx].x - self.nodes[i_idx].x
            dy = self.nodes[j_idx].y - self.nodes[i_idx].y
            L = math.hypot(dx, dy)
            c = dx / L
            s = dy / L

            # 1. Strain energy in global coordinates: 0.5 * u_e^T k_e u_e
            #    with k_e = (E A / L) * [direction dyadic pattern].
            k_e = (elem.E * elem.A / L) * np.array(
                [
                    [c**2, c * s, -(c**2), -c * s],
                    [c * s, s**2, -c * s, -(s**2)],
                    [-(c**2), -c * s, c**2, c * s],
                    [-c * s, -(s**2), c * s, s**2],
                ]
            )

            u_e = np.array(
                [
                    U[2 * i_idx],
                    U[2 * i_idx + 1],
                    U[2 * j_idx],
                    U[2 * j_idx + 1],
                ]
            )

            # Strain energy is theoretically non-negative for stable
            # structures; max(0.0, ...) guards against floating-point
            # negative zeros from round-off.
            raw_energy = 0.5 * float(u_e.T @ k_e @ u_e)
            strain_energy = max(0.0, raw_energy)

            # 2. DDM (adjoint formulation on free DOFs), with dK/dA = K_i / A
            z_i = z_mat[:, e_i]
            b_i_u = float(b_free[e_i] @ U_f)
            dU_f_dA = -(elem.E / L) * z_i * b_i_u

            try:
                idx_x = free_dofs.index(crit_dof_x)
                du_x_dA = dU_f_dA[idx_x]
            except ValueError:
                du_x_dA = 0.0

            try:
                idx_y = free_dofs.index(crit_dof_y)
                du_y_dA = dU_f_dA[idx_y]
            except ValueError:
                du_y_dA = 0.0

            u_x = U[crit_dof_x]
            u_y = U[crit_dof_y]

            dmax_sensitivity = float((u_x * du_x_dA + u_y * du_y_dA) / d_max)

            results.append(
                SensitivityResult(
                    member_id=str(elem.id),
                    ddm_sensitivity=dmax_sensitivity,
                    strain_energy=strain_energy,
                )
            )

        return results
