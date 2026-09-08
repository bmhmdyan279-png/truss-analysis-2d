import numpy as np

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.model import Element, Node
from truss_analysis.postprocess import calculate_element_forces
from truss_analysis.solver import check_energy, solve


def test_golden_simple_truss():
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=3.0, y=0.0, is_support=False),
        # Node 3 must be a full pin: with support_dx=False the vertical
        # member 1-3 provides no X stiffness and the structure is a mechanism.
        Node(id="3", x=0.0, y=4.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [
        Element(id="1", node_i="1", node_j="2", E=200e9, A=0.001),
        Element(id="2", node_i="2", node_j="3", E=200e9, A=0.002),
        Element(id="3", node_i="1", node_j="3", E=200e9, A=0.0015),
    ]
    K, F_ext, F_mech, fixed_dofs = assemble_global_matrices(nodes, elements)

    F_ext[2] += 10000.0
    F_mech[2] += 10000.0

    U = solve(K, F_ext, fixed_dofs)

    _results, strain_energy, prestress_work = calculate_element_forces(
        nodes, elements, U
    )

    check_energy(U, F_mech, strain_energy, prestress_work)

    # Node 2 must displace under the applied load
    assert abs(U[2]) > 1e-10, "Node 2 should displace under load."


def test_golden_thermal_loading():
    """Free thermal expansion: the strain energy must be zero."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        # Node 2: roller along X (free in X so the bar can expand,
        # constrained in Y to prevent a rotational mechanism)
        Node(id="2", x=3.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
    ]
    elements = [
        Element(
            id="1",
            node_i="1",
            node_j="2",
            E=200e9,
            A=0.001,
            alpha=1.2e-5,
            delta_T=50,
        ),
    ]

    K, F_ext, F_mech, fixed_dofs = assemble_global_matrices(nodes, elements)
    U = solve(K, F_ext, fixed_dofs)

    _results, strain_energy, prestress_work = calculate_element_forces(
        nodes, elements, U
    )

    # Node 2 is free along X, so the bar expands freely: delta_l_mech = 0
    assert strain_energy < 1e-6, (
        f"Free thermal expansion should produce zero strain energy, got {strain_energy}"
    )

    check_energy(U, F_mech, strain_energy, prestress_work)


def test_golden_thermal_constrained():
    """Constrained thermal expansion: the bar sits between rigid supports."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=3.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [
        Element(
            id="1",
            node_i="1",
            node_j="2",
            E=200e9,
            A=0.001,
            alpha=1.2e-5,
            delta_T=50,
        ),
    ]

    K, F_ext, _, fixed_dofs = assemble_global_matrices(nodes, elements)
    U = solve(K, F_ext, fixed_dofs)

    _results, strain_energy, _prestress_work = calculate_element_forces(
        nodes, elements, U
    )

    # Both nodes are constrained, so U must be zero
    assert np.allclose(U, 0)

    # delta_l_mech = 0 - delta_l_thermal = -delta_l_thermal
    # hence the strain energy must be positive and substantial
    L = 3.0
    delta_l_thermal = 1.2e-5 * 50 * L
    k_axial = 200e9 * 0.001 / L
    expected_energy = 0.5 * k_axial * (delta_l_thermal**2)

    assert abs(strain_energy - expected_energy) < 1e-3, (
        f"Constrained thermal: expected energy {expected_energy}, got {strain_energy}"
    )
