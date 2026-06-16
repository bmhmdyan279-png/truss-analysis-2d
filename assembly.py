"""
مونتاژ ماتریس‌های سراسری بهینه‌شده
"""

import numpy as np
from typing import Tuple
from scipy import sparse
from model import TrussModel, Element
from constants import FORCE_CONVENTION


def calculate_element_stiffness(element: Element) -> np.ndarray:
    """
    محاسبه ماتریس سختی عنصر در مختصات سراسری
    """
    AE_L = element.A * element.E / element.L
    c, s = element.c, element.s

    # ماتریس سختی 4x4
    k_e = AE_L * np.array([
        [c * c, c * s, -c * c, -c * s],
        [c * s, s * s, -c * s, -s * s],
        [-c * c, -c * s, c * c, c * s],
        [-c * s, -s * s, c * s, s * s]
    ])

    return k_e


def calculate_element_thermal_force(element: Element) -> np.ndarray:
    """
    محاسبه بردار نیروی معادل اثرات حرارتی و خطای ساخت
    فرمول اصلاح‌شده: f_e = AE/L * delta_L_free * [-c, -s, c, s]

    منطق: وقتی عضو گرم می‌شود (delta_L_free > 0) → می‌خواهد منبسط شود
          اما در سیستم محدود → نیروی فشاری ایجاد می‌شود
          نیروی فشاری = نیرو به سمت داخل عضو
    """
    if element.delta_L_free is None:
        element.calculate_thermal_effects()

    AE_L = element.A * element.E / element.L
    c, s = element.c, element.s

    # فرمول نهایی: f_e = AE/L * delta_L_free * [-c, -s, c, s]
    f_e = AE_L * element.delta_L_free * np.array([-c, -s, c, s])

    return f_e


def assemble_global_matrices(truss: TrussModel, K_global, F_global):
    """
    کاهش سیستم با حذف DOFهای قفل شده
    """
    free_dofs = np.array(truss.free_dofs, dtype=int)
    fixed_dofs = np.array(truss.fixed_dofs, dtype=int)

    # بخش‌بندی ماتریس‌ها
    if isinstance(K_global, np.ndarray):
        # ماتریس متراکم
        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    else:
        # ماتریس تنک
        K_ff = K_global[free_dofs, :][:, free_dofs]

    F_f = F_global[free_dofs]

    return K_ff, F_f, free_dofs, fixed_dofs


def build_global_matrices(truss: TrussModel):
    """
    مونتاژ ماتریس سختی سراسری و بردار نیروی سراسری
    """
    from scipy import sparse

    n_dof = truss.n_dof
    use_sparse = truss.options.get('use_sparse', True)

    if use_sparse:
        K_global = sparse.lil_matrix((n_dof, n_dof))
    else:
        K_global = np.zeros((n_dof, n_dof))

    F_global = np.zeros(n_dof)

    # مونتاژ ماتریس سختی و بردار نیرو از عناصر
    for element in truss.elements.values():
        k_e = calculate_element_stiffness(element)
        f_e = calculate_element_thermal_force(element)

        # DOFهای عنصر
        node_i_dofs = truss.nodes[element.node_i.id].dofs
        node_j_dofs = truss.nodes[element.node_j.id].dofs
        dofs = [node_i_dofs[0], node_i_dofs[1], node_j_dofs[0], node_j_dofs[1]]

        # افزودن به ماتریس سراسری
        for i in range(4):
            for j in range(4):
                K_global[dofs[i], dofs[j]] += k_e[i, j]
            F_global[dofs[i]] += f_e[i]

    # افزودن بارهای گره‌ای
    for load in truss.loads:
        node_id = load['node_id']
        node_dofs = truss.nodes[node_id].dofs
        F_global[node_dofs[0]] += load['Fx']
        F_global[node_dofs[1]] += load['Fy']

    if use_sparse:
        K_global = K_global.tocsr()

    return K_global, F_global
# ایجاد نام مستعار برای سازگاری با solver.py
get_reduced_system = assemble_global_matrices