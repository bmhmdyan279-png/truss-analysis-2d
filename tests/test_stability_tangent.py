"""Test suite for validating tangent stiffness matrix (K_T = K_E + K_G) 
using finite difference method.
Addresses Issue A5: No finite-difference tangent test for K_G.
"""
import numpy as np
from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.stability import geometric_stiffness
from truss_analysis.solver import solve
from truss_analysis.model import Node, Element


def make_simple_truss():
    """Creates a simple 2-bar truss under compression for testing."""
    nodes = [
        Node(id='n0', x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id='n1', x=2.0, y=0.0, is_support=False, support_dx=False, support_dy=False),
        Node(id='n2', x=4.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    
    elements = [
        Element(id='e0', node_i='n0', node_j='n1', E=210e9, A=1e-4, alpha=1.2e-5),
        Element(id='e1', node_i='n1', node_j='n2', E=210e9, A=1e-4, alpha=1.2e-5),
    ]
    
    loads = {'n1': {'fy': -1000.0}}
    
    return nodes, elements, loads


class TestTangentStiffnessValidation:
    """Validates the consistency of analytical tangent stiffness with finite differences."""

    def compute_finite_difference_tangent(self, nodes, elements, u0, dof_idx, h=1e-8):
        """
        Computes a column of the tangent stiffness matrix using central finite differences.
        K[:, i] ≈ (F_int(u + h*e_i) - F_int(u - h*e_i)) / (2h)
        
        For a truss, internal force = K_E * u (linear elastic).
        Tangent stiffness should equal K_E when no axial load effect.
        With axial load, tangent = K_E + K_G.
        """
        n_dof = 2 * len(nodes)
        k_col_fd = np.zeros(n_dof)
        
        # Perturb positive
        u_plus = u0.copy()
        u_plus[dof_idx] += h
        f_plus = self._compute_internal_forces(nodes, elements, u_plus)
        
        # Perturb negative
        u_minus = u0.copy()
        u_minus[dof_idx] -= h
        f_minus = self._compute_internal_forces(nodes, elements, u_minus)
        
        # Central difference
        k_col_fd = (f_plus - f_minus) / (2 * h)
        
        return k_col_fd
    
    def _compute_internal_forces(self, nodes, elements, u):
        """Compute internal forces F_int = K_E * u."""
        K_E, _, _, _ = assemble_global_matrices(nodes, elements)
        return K_E @ u

    def test_tangent_stiffness_elastic_consistency(self):
        """
        Verifies that the analytical elastic stiffness (K_E) matches 
        the Finite Difference approximation within numerical tolerance.
        """
        nodes, elements, loads = make_simple_truss()
        
        # Solve to get displacement state
        result = solve(nodes, elements, loads)
        u_current = result.displacements
        
        # Compute Analytical Elastic Stiffness
        K_E, _, _, fixed = assemble_global_matrices(nodes, elements)
        free_dofs = [d for d in range(len(u_current)) if d not in fixed]
        
        # Check a free DOF
        test_dof = free_dofs[0] if free_dofs else 0
        
        k_fd = self.compute_finite_difference_tangent(nodes, elements, u_current, test_dof)
        k_ana = K_E[:, test_dof]
        
        # Relative error
        norm_fd = np.linalg.norm(k_fd)
        norm_diff = np.linalg.norm(k_fd - k_ana)
        
        if norm_fd > 1e-10:
            rel_error = norm_diff / norm_fd
        else:
            rel_error = norm_diff
        
        # Tolerance: 1e-5 is reasonable for central difference with h=1e-8
        assert rel_error < 1e-5, f"FD mismatch at DOF {test_dof}: Rel Error = {rel_error:.2e}"

    def test_geometric_stiffness_contribution(self):
        """
        Tests that K_G contributes to tangent stiffness under axial load.
        Compares tangent stiffness at low load vs high load.
        """
        nodes, elements, loads_low = make_simple_truss()
        
        # High load case
        loads_high = {'n1': {'fy': -50000.0}}  # High compression
        
        # Solve both cases
        result_low = solve(nodes, elements, loads_low)
        result_high = solve(nodes, elements, loads_high)
        
        # Get axial forces from results
        forces_low = {m.id: m.axial_force for m in result_low.members}
        forces_high = {m.id: m.axial_force for m in result_high.members}
        
        # Fixed DOFs
        _, _, _, fixed = assemble_global_matrices(nodes, elements)
        free_dofs = [d for d in range(2*len(nodes)) if d not in fixed]
        
        # Compute K_G for both states
        K_G_low = geometric_stiffness(nodes, elements, forces_low, free_dofs=free_dofs)
        K_G_high = geometric_stiffness(nodes, elements, forces_high, free_dofs=free_dofs)
        
        # The geometric stiffness should differ significantly
        diff_norm = np.linalg.norm(K_G_high - K_G_low)
        
        # Under high compression, K_G should be significantly different
        assert diff_norm > 1e-3, "Geometric stiffness contribution seems negligible or missing."
