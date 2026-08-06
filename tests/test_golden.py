import numpy as np

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.model import Element, Node
from truss_analysis.postprocess import calculate_element_forces
from truss_analysis.solver import check_energy, solve


def test_golden_simple_truss():
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=3.0, y=0.0, is_support=False),
        # گره ۳: تکیه‌گاه غلتکی (آزاد در X، مقید در Y)
        Node(id="3", x=0.0, y=4.0, is_support=True, support_dx=False, support_dy=True),
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

    results, strain_energy, prestress_work = calculate_element_forces(
        nodes, elements, U
    )

    check_energy(U, F_mech, strain_energy, prestress_work)

    # DOF 4 = U_x گره ۳ (که غلتکی است، پس باید جابجا شود)
    assert abs(U[4]) > 1e-10, "CRITICAL: Roller support is artificially locked!"


def test_golden_thermal_loading():
    """تست انبساط حرارتی آزاد (انرژی کرنشی باید صفر باشد)"""
    nodes = [
        # گره ۱: مقید کامل
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        # گره ۲: کاملاً آزاد (تا میله بتواند آزادانه منبسط شود)
        Node(id="2", x=3.0, y=0.0, is_support=False),
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

    results, strain_energy, prestress_work = calculate_element_forces(
        nodes, elements, U
    )

    # چون گره ۲ آزاد است، میله منبسط می‌شود و delta_l_mech = 0 خواهد بود
    # در نتیجه انرژی کرنشی باید نزدیک به صفر باشد
    assert strain_energy < 1e-6, (
        f"Free thermal expansion should produce zero strain energy, got {strain_energy}"
    )

    # F_mechanical صفر است (بار مکانیکی نداریم)
    # بنابراین W_mech = 0 و W_prestress هم باید با strain_energy برابر باشد
    check_energy(U, F_mech, strain_energy, prestress_work)


def test_golden_thermal_constrained():
    """تست انبساط حرارتی مقید (میله بین دو تکیه‌گاه صلب)"""
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

    K, F_ext, F_mech, fixed_dofs = assemble_global_matrices(nodes, elements)
    U = solve(K, F_ext, fixed_dofs)

    results, strain_energy, prestress_work = calculate_element_forces(
        nodes, elements, U
    )

    # چون هر دو گره مقید هستند، U باید صفر باشد
    assert np.allclose(U, 0)

    # delta_l_mech = 0 - delta_l_thermal = -delta_l_thermal
    # بنابراین انرژی کرنشی باید مثبت و قابل توجه باشد
    L = 3.0
    delta_l_thermal = 1.2e-5 * 50 * L
    k_axial = 200e9 * 0.001 / L
    expected_energy = 0.5 * k_axial * (delta_l_thermal**2)

    assert abs(strain_energy - expected_energy) < 1e-3, (
        f"Constrained thermal: expected energy {expected_energy}, got {strain_energy}"
    )
