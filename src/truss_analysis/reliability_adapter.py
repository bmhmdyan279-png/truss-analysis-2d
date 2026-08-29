"""Adapter for connecting the Reliability Engine to the Truss FEM Model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .assembly import assemble_global_matrices
from .model import Element, Node
from .reliability import AnalysisSample, MemberResponse, TargetId
from .solver import solve


@dataclass
class NodalLoad:
    """Simple DTO for nodal loads."""

    node_id: str
    fx: float
    fy: float


class TrussReliabilityModel:
    """Wraps the deterministic truss model for Monte Carlo sampling."""

    def __init__(
        self,
        nodes: list[Node],
        elements: list[Element],
        loads: list[NodalLoad],
        yield_stress_map: Mapping[str, float] | None = None,
    ) -> None:
        self.nodes = nodes
        self.elements = elements
        self.loads = loads
        self.yield_stress_map = yield_stress_map or {}
        self.node_map: dict[str, int] = {node.id: i for i, node in enumerate(nodes)}

    def create_analyze_fn(self) -> Callable[[Mapping[str, float]], AnalysisSample]:
        """Returns a callback for the ReliabilityEngine."""

        def analyze(sample: Mapping[str, float]) -> AnalysisSample:
            # 1. Deepcopy to prevent mutation across Monte Carlo iterations
            current_nodes = deepcopy(self.nodes)
            current_elements = deepcopy(self.elements)
            current_loads = deepcopy(self.loads)

            # 2. Apply sampled values to the model
            for var_name, value in sample.items():
                self._apply_sample(
                    var_name, float(value), current_elements, current_loads
                )

            # 3. Assemble global matrices (includes thermal/prestress forces)
            K, F_ext, _, fixed_dofs = assemble_global_matrices(
                current_nodes, current_elements
            )

            # 4. Add external mechanical loads to F_ext
            for load in current_loads:
                idx = self.node_map[load.node_id]
                F_ext[2 * idx] += load.fx
                F_ext[2 * idx + 1] += load.fy

            # 5. Solve deterministic system
            U = solve(K, F_ext, fixed_dofs)

            # 6. Post-process: Calculate member forces and nodal displacements
            member_responses: dict[TargetId, MemberResponse] = {}
            for elem in current_elements:
                i = self.node_map[elem.node_i]
                j = self.node_map[elem.node_j]

                ui, vi = U[2 * i], U[2 * i + 1]
                uj, vj = U[2 * j], U[2 * j + 1]

                dx = current_nodes[j].x - current_nodes[i].x
                dy = current_nodes[j].y - current_nodes[i].y
                L = np.sqrt(dx**2 + dy**2)
                c = dx / L
                s = dy / L

                # Mechanical elongation
                delta_L_mech = c * (uj - ui) + s * (vj - vi)
                # Thermal/fabrication elongation
                delta_L_prestress = (elem.alpha * elem.delta_T * L) + elem.delta_L_free

                k = elem.E * elem.A / L
                axial_force = k * (delta_L_mech - delta_L_prestress)

                member_responses[elem.id] = MemberResponse(
                    axial_force=axial_force,
                    E=elem.E,
                    A=elem.A,
                    I_sec=elem.I_sec,
                    length=L,
                    effective_length_factor=elem.effective_length_factor,
                    yield_stress=self.yield_stress_map.get(elem.id),
                )

            nodal_displacements: dict[TargetId, tuple[float, float]] = {}
            for node_id, idx in self.node_map.items():
                nodal_displacements[node_id] = (
                    float(U[2 * idx]),
                    float(U[2 * idx + 1]),
                )

            return AnalysisSample(
                member_responses=member_responses,
                nodal_displacements=nodal_displacements,
            )

        return analyze

    @staticmethod
    def _apply_sample(
        var_name: str,
        value: float,
        elements: list[Element],
        loads: list[NodalLoad],
    ) -> None:
        """Parses variable names like 'E_1', 'A_2', 'Fy_0' and updates the model."""
        parts = var_name.split("_", 1)
        if len(parts) != 2:
            return

        param, target_id = parts

        # Update element properties
        if param in (
            "E",
            "A",
            "alpha",
            "delta_T",
            "delta_L_free",
            "I_sec",
            "effective_length_factor",
        ):
            for elem in elements:
                if elem.id == target_id:
                    setattr(elem, param, value)
                    break

        # Update nodal loads (target_id is the index in the loads list)
        elif param in ("Fx", "Fy"):
            idx = int(target_id)
            if 0 <= idx < len(loads):
                if param == "Fx":
                    loads[idx].fx = value
                else:
                    loads[idx].fy = value
