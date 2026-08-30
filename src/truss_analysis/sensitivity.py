"""Phase 5: Independent SCF Validation (Adjoint/DDM and Strain Energy)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from .assembly import assemble_global_matrices
from .model import Element, Node
from .solver import solve


@dataclass(frozen=True)
class SensitivityResult:
    """Result of independent sensitivity and energy validation for a single member."""

    member_id: str
    ddm_sensitivity: float
    strain_energy: float


class IndependentValidator:
    """Validates Phase 4 SCF rankings using DDM and Strain Energy methods."""

    def __init__(
        self,
        nodes: List[Node],
        elements: List[Element],
        loads: List[Any],
    ) -> None:
        self.nodes = nodes
        self.elements = elements
        self.loads = loads
        self.node_map = {node.id: i for i, node in enumerate(nodes)}

    def compute_baseline(self) -> tuple[np.ndarray, list[int]]:
        """Solves the baseline system KU=F."""
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
        """Computes DDM sensitivity and strain energy for all elements."""
        U, fixed_dofs = self.compute_baseline()
        n = len(self.nodes)
        free_dofs = [i for i in range(2 * n) if i not in fixed_dofs]

        K, _, _, _ = assemble_global_matrices(self.nodes, self.elements)
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        U_f = U[free_dofs]

        # Pre-invert K_ff (safe since solve() would have caught singularity)
        K_ff_inv = np.linalg.inv(K_ff)

        # Find critical node for max displacement
        disp_magnitudes = np.hypot(U[0::2], U[1::2])
        crit_node_idx = int(np.argmax(disp_magnitudes))
        d_max = disp_magnitudes[crit_node_idx]

        if d_max < 1e-15:
            d_max = 1.0

        crit_dof_x = 2 * crit_node_idx
        crit_dof_y = 2 * crit_node_idx + 1

        results: list[SensitivityResult] = []

        for elem in self.elements:
            i_idx = self.node_map[elem.node_i]
            j_idx = self.node_map[elem.node_j]

            dx = self.nodes[j_idx].x - self.nodes[i_idx].x
            dy = self.nodes[j_idx].y - self.nodes[i_idx].y
            L = math.hypot(dx, dy)
            c = dx / L
            s = dy / L

            # 1. Strain Energy (Global Coordinates)
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

            # FIX: Use max(0.0, ...) to prevent floating-point negative zeros
            # Strain energy is theoretically non-negative for stable structures.
            raw_energy = 0.5 * float(u_e.T @ k_e @ u_e)
            strain_energy = max(0.0, raw_energy)

            # 2. DDM (Adjoint Formulation on Free DOFs)
            # dK/dA = K_i / A
            k_i = (elem.E / L) * np.array(
                [
                    [c**2, c * s, -(c**2), -c * s],
                    [c * s, s**2, -c * s, -(s**2)],
                    [-(c**2), -c * s, c**2, c * s],
                    [-c * s, -(s**2), c * s, s**2],
                ]
            )

            K_i_global = np.zeros((2 * n, 2 * n))
            dofs = [2 * i_idx, 2 * i_idx + 1, 2 * j_idx, 2 * j_idx + 1]
            for ii in range(4):
                for jj in range(4):
                    K_i_global[dofs[ii], dofs[jj]] = k_i[ii, jj]

            K_i_ff = K_i_global[np.ix_(free_dofs, free_dofs)]

            dU_f_dA = -K_ff_inv @ (K_i_ff / elem.A) @ U_f

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
