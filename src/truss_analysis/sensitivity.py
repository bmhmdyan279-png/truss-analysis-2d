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

        Two documented conventions (round-5 audit, C8-7):

        1. **Subgradient at ties/switches.** ``|u|_max`` is differentiated
           at the *base-state* argmax node. The map ``A -> max_k |u_k|`` is
           piecewise smooth: when perturbing ``A_i`` moves the identity of
           the critical node (or when several nodes tie), the reported
           value is one valid **subgradient** of the max function, not a
           Frechet derivative. As a screening/ranking quantity this is the
           intended use; a smooth alternative is the p-norm aggregate.
        2. **Degenerate-displacement floor.** When every nodal displacement
           is below ``1e-15`` m (an unloaded or fully restrained model), the
           normalising magnitude ``d_max`` is set to ``1.0`` to avoid a
           ``0/0``. In that case the value is the *un-normalised* numerator
           ``u . du/dA`` (numerically ~0) and carries no physical unit; do
           not interpret it as [m/m^2].
    strain_energy : float
        **Mechanical** member strain energy ``0.5 * k * delta_L_mech^2`` [J],
        where ``delta_L_mech = delta_L_total - delta_L_prestress``. This is the
        energy actually stored as stress. Clipped at zero against negative
        floating-point round-off.
    strain_energy_total : float
        ``0.5 * u_e^T k_e u_e = 0.5 * k * delta_L_total^2`` [J], i.e. the
        quadratic form over the *total* elongation including imposed
        (thermal / fabrication) strain. Retained for cross-checking: the two
        coincide exactly when the member carries no imposed strain, and they
        differ precisely by the prestress contribution otherwise. A member
        that expands freely has ``strain_energy = 0`` but a non-zero
        ``delta_L_total``; a fully restrained heated member has
        ``delta_L_total = 0`` but a large ``strain_energy``.
    """

    member_id: str
    ddm_sensitivity: float
    strain_energy: float
    strain_energy_total: float = 0.0


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

    def _assemble_and_solve(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """One assembly + one solve; returns ``(K, U, fixed_dofs)``.

        Shared by :meth:`compute_baseline` and :meth:`compute_all` so the
        DDM pass no longer re-assembles the global matrix it just solved
        (round-5 audit, C8 perf): assembly is O(m) with a large constant,
        and the duplicated call was pure waste on every invocation.
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
        return K, U, fixed_dofs

    def compute_baseline(self) -> tuple[np.ndarray, list[int]]:
        """Solve the baseline system ``K U = F`` including the nodal loads.

        Returns
        -------
        tuple[np.ndarray, list[int]]
            ``(U, fixed_dofs)``: the global displacement vector and the
            list of constrained DOF indices.
        """
        _K, U, fixed_dofs = self._assemble_and_solve()
        return U, fixed_dofs

    def compute_all(self) -> list[SensitivityResult]:
        """Compute DDM sensitivity and strain energy for every element.

        The DDM uses a rank-1 formulation: one LU factorisation of ``K_ff``
        plus one triangular solve for the whole compatibility matrix ``B``
        replaces any explicit inverse. The area ``A_i`` enters the solved
        system through **two** channels: the stiffness ``k_i = E_i A_i / L_i``
        *and* the imposed-force term ``F_pre,i = k_i dL_pre,i b_i`` that
        ``assemble_global_matrices`` puts on the right-hand side whenever the
        member carries thermal/fabrication strain. Differentiating
        ``K U = F_mech + F_pre`` gives

        .. code-block:: text

            K dU/dA_i = (E_i/L_i) dL_pre,i b_i - (E_i/L_i) b_i (b_i^T U_f)

        so with ``Z_i = K_ff^-1 b_i``:

        ``dU/dA_i = -(E_i / L_i) * Z_i * (b_i^T U_f - dL_pre,i)``

        The numerator is the member's **mechanical** elongation
        (``N_i / k_i``), not its total elongation — the same convention as
        the rank-1 numerator in :func:`truss_analysis.criticality.engine.ci_sweep`.
        Using the total elongation is wrong whenever ``dL_pre,i != 0``: for a
        nearly fully restrained heated member the total elongation is almost
        pure imposed strain, and the sensitivity came out with the wrong
        sign and two orders of magnitude too large (pinned against central
        finite differences in ``tests/test_sensitivity.py``). For members
        without imposed strain the formula reduces to the previous one.

        Returns
        -------
        list[SensitivityResult]
            One result per element, in element order.
        """
        K, U, fixed_dofs = self._assemble_and_solve()
        n = len(self.nodes)
        free_dofs = [i for i in range(2 * n) if i not in fixed_dofs]

        K_ff = K[np.ix_(free_dofs, free_dofs)]
        U_f = U[free_dofs]

        # Rank-1 DDM machinery: one LU factorisation plus one solve for the
        # whole compatibility matrix B; no explicit inverse, and singularity
        # is caught up-front by the rank check inside solve().
        node_idx = {nd.id: i for i, nd in enumerate(self.nodes)}
        # DOF -> column position in the free-DOF ordering. list.index() here
        # would be an O(n_free) linear search inside a 4M-iteration loop,
        # i.e. O(M * n_free) wasted comparisons on top of the O(n^3)
        # factorisation; the dict makes every lookup O(1).
        free_pos = {dof: pos for pos, dof in enumerate(free_dofs)}
        b_free = np.zeros((len(self.elements), len(free_dofs)))
        elem_rows: list[tuple[float, float, float]] = []  # (c, s, L) per element
        for e_i, elem in enumerate(self.elements):
            i_idx = node_idx[elem.node_i]
            j_idx = node_idx[elem.node_j]
            dx = self.nodes[j_idx].x - self.nodes[i_idx].x
            dy = self.nodes[j_idx].y - self.nodes[i_idx].y
            length = math.hypot(dx, dy)
            c, s = dx / length, dy / length
            elem_rows.append((c, s, length))
            dofs = [2 * i_idx, 2 * i_idx + 1, 2 * j_idx, 2 * j_idx + 1]
            for k, dof in enumerate(dofs):
                pos = free_pos.get(dof)
                if pos is not None:
                    b_free[e_i, pos] = (-c, -s, c, s)[k]
        z_mat = lu_solve(lu_factor(K_ff), b_free.T)

        # Find critical node for max displacement
        disp_magnitudes = np.hypot(U[0::2], U[1::2])
        crit_node_idx = int(np.argmax(disp_magnitudes))
        d_max = disp_magnitudes[crit_node_idx]

        if d_max < 1e-15:
            # Degenerate model (no measurable displacement): avoid 0/0. The
            # reported value is then the un-normalised numerator, NOT a
            # [m/m^2] sensitivity -- see SensitivityResult.ddm_sensitivity.
            d_max = 1.0

        crit_dof_x = 2 * crit_node_idx
        crit_dof_y = 2 * crit_node_idx + 1
        # Positions of the critical node's DOFs in the free-DOF ordering,
        # resolved once (they do not depend on the member being differentiated).
        crit_pos_x = free_pos.get(crit_dof_x)
        crit_pos_y = free_pos.get(crit_dof_y)

        results: list[SensitivityResult] = []

        for e_i, elem in enumerate(self.elements):
            i_idx = self.node_map[elem.node_i]
            j_idx = self.node_map[elem.node_j]

            c, s, L = elem_rows[e_i]

            # 1. Strain energy. Two routes are computed and cross-checked:
            #
            #    (a) the full 4x4 quadratic form u_e^T k_e u_e in global
            #        coordinates, which equals k * delta_L_total^2 because
            #        k_e is the rank-1 dyad k * b b^T;
            #    (b) the compatibility vector b dotted with u_e, giving
            #        delta_L_total directly.
            #
            #    Agreement between them verifies the element matrix and the
            #    DOF mapping independently of the assembler.
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

            k_axial = elem.E * elem.A / L
            quad_total = float(u_e @ k_e @ u_e)  # == k * delta_L_total^2
            delta_L_total = float(np.array([-c, -s, c, s]) @ u_e)

            # The energy *stored* in the member comes only from the mechanical
            # part of the elongation. The total elongation includes the imposed
            # (thermal + fabrication) strain, which produces displacement
            # without stress when the member is free to expand. Reporting
            # 0.5 * u_e^T k_e u_e as the strain energy therefore overstated it
            # whenever delta_T or delta_L_free was non-zero -- and understated
            # it to exactly zero for a fully restrained heated bar, where the
            # nodes do not move but the member is fully stressed.
            delta_L_prestress = elem.alpha * elem.delta_T * L + elem.delta_L_free
            delta_L_mech = delta_L_total - delta_L_prestress
            raw_energy = 0.5 * k_axial * delta_L_mech**2
            # Strain energy is theoretically non-negative for stable
            # structures; max(0.0, ...) guards against floating-point
            # negative zeros from round-off.
            strain_energy = max(0.0, raw_energy)
            strain_energy_total = max(0.0, 0.5 * quad_total)

            # 2. DDM (adjoint formulation on free DOFs), with dK/dA = K_i / A
            # and dF_pre/dA = (E_i/L_i) dL_pre,i b_i: the numerator is the
            # MECHANICAL elongation (total minus imposed), matching the
            # derivation in compute_all's docstring and the rank-1 numerator
            # convention of the criticality engine.
            z_i = z_mat[:, e_i]
            b_i_u = float(b_free[e_i] @ U_f) - delta_L_prestress
            dU_f_dA = -(elem.E / L) * z_i * b_i_u

            du_x_dA = float(dU_f_dA[crit_pos_x]) if crit_pos_x is not None else 0.0
            du_y_dA = float(dU_f_dA[crit_pos_y]) if crit_pos_y is not None else 0.0

            u_x = U[crit_dof_x]
            u_y = U[crit_dof_y]

            dmax_sensitivity = float((u_x * du_x_dA + u_y * du_y_dA) / d_max)

            results.append(
                SensitivityResult(
                    member_id=str(elem.id),
                    ddm_sensitivity=dmax_sensitivity,
                    strain_energy=strain_energy,
                    strain_energy_total=strain_energy_total,
                )
            )

        return results
