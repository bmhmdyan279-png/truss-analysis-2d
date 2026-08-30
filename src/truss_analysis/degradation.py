"""Phase 4: Alpha-degradation operator and Geometric Scaling Rule."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .assembly import assemble_global_matrices
from .exceptions import SingularMatrixError
from .model import Element, Node
from .reliability_adapter import NodalLoad
from .solver import solve


@dataclass(frozen=True)
class DegradationPoint:
    """Result of a single alpha-degradation solve."""

    alpha: float
    max_disp: float
    axial_forces: dict[str, float]
    is_singular: bool
    error: str | None = None


@dataclass
class MemberSensitivityProfile:
    """Sensitivity profile for a single member across multiple alpha values."""

    member_id: str
    baseline_max_disp: float
    baseline_axial_forces: dict[str, float]
    points: list[DegradationPoint]
    sensitivity_slope: float
    scf_alpha_min: float
    is_key_element: bool
    mechanism_detected_at: float | None = None


class DamageOperator:
    """Applies alpha-degradation to truss members to assess system sensitivity."""

    def __init__(
        self,
        nodes: list[Node],
        elements: list[Element],
        loads: list[NodalLoad],
    ) -> None:
        self.nodes = nodes
        self.elements = elements
        self.loads = loads
        self.node_map = {node.id: i for i, node in enumerate(nodes)}

    def _solve(
        self,
        current_nodes: list[Node],
        current_elements: list[Element],
    ) -> tuple[np.ndarray, dict[str, float]]:
        K, F_ext, _, fixed_dofs = assemble_global_matrices(
            current_nodes, current_elements
        )
        for load in self.loads:
            idx = self.node_map[load.node_id]
            F_ext[2 * idx] += load.fx
            F_ext[2 * idx + 1] += load.fy

        U = solve(K, F_ext, fixed_dofs)

        axial_forces: dict[str, float] = {}
        for elem in current_elements:
            i = self.node_map[elem.node_i]
            j = self.node_map[elem.node_j]
            ui, vi = U[2 * i], U[2 * i + 1]
            uj, vj = U[2 * j], U[2 * j + 1]
            dx = current_nodes[j].x - current_nodes[i].x
            dy = current_nodes[j].y - current_nodes[i].y
            length = math.hypot(dx, dy)
            if length < 1e-12:
                axial_forces[elem.id] = 0.0
                continue
            c, s = dx / length, dy / length
            delta_L_mech = c * (uj - ui) + s * (vj - vi)
            delta_L_prestress = (elem.alpha * elem.delta_T * length) + elem.delta_L_free
            k = elem.E * elem.A / length
            axial_forces[elem.id] = k * (delta_L_mech - delta_L_prestress)

        return U, axial_forces

    def _check_mechanism(
        self,
        current_nodes: list[Node],
        current_elements: list[Element],
    ) -> bool:
        K, _, _, fixed_dofs = assemble_global_matrices(current_nodes, current_elements)
        n = len(current_nodes)
        free_dofs = [i for i in range(2 * n) if i not in fixed_dofs]
        if not free_dofs:
            return False
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        rank = np.linalg.matrix_rank(K_ff, tol=1e-9)
        return rank < len(free_dofs)

    def _apply_geometric_scaling(
        self,
        elements: list[Element],
        target_id: str,
        alpha: float,
    ) -> list[Element]:
        new_elements = []
        for elem in elements:
            if elem.id == target_id:
                new_elem = deepcopy(elem)
                new_elem.A = elem.A * alpha
                new_elem.I_sec = elem.I_sec * (alpha**2)
                new_elements.append(new_elem)
            else:
                new_elements.append(elem)
        return new_elements

    def analyze_member(
        self,
        target_id: str,
        alphas: Sequence[float],
        probe_near_zero: bool = True,
    ) -> MemberSensitivityProfile:
        baseline_U, baseline_N = self._solve(self.nodes, self.elements)
        baseline_max_disp = float(np.max(np.hypot(baseline_U[0::2], baseline_U[1::2])))

        points: list[DegradationPoint] = []
        mechanism_detected_at: float | None = None

        for alpha in alphas:
            degraded_elements = self._apply_geometric_scaling(
                self.elements, target_id, alpha
            )
            try:
                U, N = self._solve(self.nodes, degraded_elements)
                max_disp = float(np.max(np.hypot(U[0::2], U[1::2])))
                points.append(
                    DegradationPoint(
                        alpha=alpha,
                        max_disp=max_disp,
                        axial_forces=N,
                        is_singular=False,
                    )
                )
            except SingularMatrixError as e:
                points.append(
                    DegradationPoint(
                        alpha=alpha,
                        max_disp=float("inf"),
                        axial_forces={},
                        is_singular=True,
                        error=str(e),
                    )
                )
                if mechanism_detected_at is None:
                    mechanism_detected_at = alpha

        is_key = False
        if probe_near_zero:
            probe_alpha = 1e-6
            degraded_elements = self._apply_geometric_scaling(
                self.elements, target_id, probe_alpha
            )
            if self._check_mechanism(self.nodes, degraded_elements):
                is_key = True
                if mechanism_detected_at is None:
                    mechanism_detected_at = probe_alpha

        valid_points = [p for p in points if not p.is_singular]
        if not valid_points or baseline_max_disp < 1e-15:
            slope = 0.0
            scf_min = float("inf")
        else:
            scfs = [p.max_disp / baseline_max_disp for p in valid_points]
            alphas_valid = [p.alpha for p in valid_points]
            x = np.array(alphas_valid)
            y = np.array(scfs)
            if len(x) > 1:
                A_mat = np.vstack([x, np.ones(len(x))]).T
                slope = float(np.linalg.lstsq(A_mat, y, rcond=None)[0][0])
            else:
                slope = 0.0
            scf_min = scfs[-1] if scfs else float("inf")

        return MemberSensitivityProfile(
            member_id=target_id,
            baseline_max_disp=baseline_max_disp,
            baseline_axial_forces=baseline_N,
            points=points,
            sensitivity_slope=float(slope),
            scf_alpha_min=scf_min,
            is_key_element=is_key,
            mechanism_detected_at=mechanism_detected_at,
        )

    def analyze_all(
        self,
        alphas: Sequence[float] = (1.0, 0.9, 0.8, 0.7),
    ) -> list[MemberSensitivityProfile]:
        return [self.analyze_member(elem.id, alphas) for elem in self.elements]
