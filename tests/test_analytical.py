"""Analytical tests for scientific correctness."""

from truss_analysis import Element, Node, solve
from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.postprocess import (
    calculate_element_forces,
    calculate_reactions,
    check_equilibrium,
)


def test_pure_mechanical_truss():
    """Test simple truss with pure mechanical loading."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=1.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
    ]
    elements = [
        Element(id="1", node_i="1", node_j="2", E=200e9, A=0.001),
    ]

    K, F_ext, F_mech, fixed_dofs = assemble_global_matrices(nodes, elements)

    # Apply 10 kN in X direction
    F_ext[2] = 10000.0
    F_mech[2] = 10000.0

    U = solve(K, F_ext, fixed_dofs)

    # Expected displacement: u = FL/(EA) = 10000 * 1.0 / (200e9 * 0.001) = 5e-5 m
    assert abs(U[2] - 5e-5) < 1e-8

    results, U_strain, W_prestress = calculate_element_forces(nodes, elements, U)

    # Expected force: 10000 N (tension)
    assert abs(results[0]["N"] - 10000.0) < 1e-6

    # Expected strain energy: ½ * F * u = 0.5 * 10000 * 5e-5 = 0.25 J
    assert abs(U_strain - 0.25) < 1e-8

    # No prestress
    assert abs(W_prestress) < 1e-12


def test_free_thermal_expansion():
    """Test bar with free thermal expansion (no stress)."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=1.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
    ]
    elements = [
        Element(
            id="1",
            node_i="1",
            node_j="2",
            E=200e9,
            A=0.001,
            alpha=1.2e-5,
            delta_T=100.0,
        ),
    ]

    K, F_ext, F_mech, fixed_dofs = assemble_global_matrices(nodes, elements)
    U = solve(K, F_ext, fixed_dofs)

    # Free expansion: u = α·ΔT·L = 1.2e-5 * 100 * 1.0 = 1.2e-3 m
    assert abs(U[2] - 1.2e-3) < 1e-6

    results, U_strain, W_prestress = calculate_element_forces(nodes, elements, U)

    # No mechanical deformation, so no force
    assert abs(results[0]["N"]) < 1e-6

    # No strain energy
    assert abs(U_strain) < 1e-12

    # Prestress work should be zero (no mechanical deformation)
    assert abs(W_prestress) < 1e-12


def test_constrained_thermal_expansion():
    """Test bar with constrained thermal expansion (thermal stress)."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [
        Element(
            id="1",
            node_i="1",
            node_j="2",
            E=200e9,
            A=0.001,
            alpha=1.2e-5,
            delta_T=100.0,
        ),
    ]

    K, F_ext, F_mech, fixed_dofs = assemble_global_matrices(nodes, elements)
    U = solve(K, F_ext, fixed_dofs)

    # Fully constrained: u = 0
    assert abs(U[2]) < 1e-12

    results, U_strain, W_prestress = calculate_element_forces(nodes, elements, U)

    # Thermal stress: σ = -E·α·ΔT = -200e9 * 1.2e-5 * 100 = -240 MPa
    # Force: F = σ·A = -240e6 * 0.001 = -240000 N (compression)
    expected_force = -200e9 * 1.2e-5 * 100 * 0.001
    assert abs(results[0]["N"] - expected_force) < 1.0

    # Strain energy: ½·k·(ΔL_mech)² where ΔL_mech = -α·ΔT·L
    k = 200e9 * 0.001 / 1.0
    delta_L_mech = -1.2e-5 * 100 * 1.0
    expected_U_strain = 0.5 * k * delta_L_mech**2
    assert abs(U_strain - expected_U_strain) < 1e-6


def test_reactions_and_equilibrium():
    """Test calculation of reactions and equilibrium check."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=2.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="3", x=1.0, y=1.5, is_support=False),
    ]
    elements = [
        Element(id="1", node_i="1", node_j="3", E=210e9, A=0.01),
        Element(id="2", node_i="2", node_j="3", E=210e9, A=0.01),
        Element(id="3", node_i="1", node_j="2", E=210e9, A=0.01),
    ]

    K, F_ext, F_mech, fixed_dofs = assemble_global_matrices(nodes, elements)

    # Apply 10 kN downward at node 3
    F_ext[5] = -10000.0
    F_mech[5] = -10000.0

    U = solve(K, F_ext, fixed_dofs)

    reactions = calculate_reactions(nodes, K, U, F_ext, fixed_dofs)

    # Sum of vertical reactions should equal applied load
    Ry_sum = sum(r["Fy"] for r in reactions.values())
    assert abs(Ry_sum - 10000.0) < 1.0

    # Check equilibrium
    # Convert F_ext array to applied_loads list format
    applied_loads = []
    for i, node in enumerate(nodes):
        fx, fy = F_ext[2 * i], F_ext[2 * i + 1]
        if abs(fx) > 1e-9 or abs(fy) > 1e-9:
            applied_loads.append({"node_id": node.id, "Fx": fx, "Fy": fy})

    errors = check_equilibrium(nodes, reactions, applied_loads)
    assert abs(errors["sum_fx"]) < 1e-6
    assert abs(errors["sum_fy"]) < 1e-6
    assert abs(errors["sum_m"]) < 1e-6
