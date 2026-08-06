import numpy as np

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.model import Element, Node
from truss_analysis.postprocess import calculate_element_forces
from truss_analysis.solver import check_energy, solve


def test_golden_simple_truss():
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=3.0, y=0.0, is_support=False),
        # گره ۳ باید مفصلی کامل باشد. اگر support_dx=False باشد، چون المان 1-3 عمودی است
        # هیچ سختی در راستای X ندارد و سازه دچار مکانیزم (Singular Matrix) می‌شود.
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

    results, strain_energy, prestress_work = calculate_element_forces(
        nodes, elements, U
    )

    check_energy(U, F_mech, strain_energy, prestress_work)

    # بررسی اینکه گره ۲ زیر بار جابجا شده است
    assert abs(U[2]) > 1e-10, "Node 2 should displace under load."


def test_golden_thermal_loading():
    """تست انبساط حرارتی آزاد (انرژی کرنشی باید صفر باشد)"""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        # گره ۲: غلتکی در راستای X (آزاد در X برای انبساط،
        # مقید در Y برای جلوگیری از مکانیزم چرخشی)
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

    results, strain_energy, prestress_work = calculate_element_forces(
        nodes, elements, U
    )

    # چون گره ۲ در راستای X آزاد است، میله منبسط می‌شود و delta_l_mech = 0 خواهد بود
    assert strain_energy < 1e-6, (
        f"Free thermal expansion should produce zero strain energy, got {strain_energy}"
    )

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
