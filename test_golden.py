"""
تست مرجع ساده‌شده
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import TrussModel
from assembly import build_global_matrices
from solver import solve_displacements, calculate_element_results

def test_golden_reference():
    """تست ساده‌شده با یک عضو افقی"""
    print("🧪 اجرای تست مرجع ساده...")

    input_data = {
        'units': 'SI',
        'nodes': [
            {'id': 1, 'x': 0, 'y': 0, 'is_support': True},
            {'id': 2, 'x': 2, 'y': 0, 'is_support': True}  # فقط دو گره
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 2,
                'A': 0.01,
                'E': 210e9,
                'alpha': 1.2e-5,
                'delta_T': 100.0  # تغییر دمای 100 درجه
            }
        ],
        'loads': {'node_forces': []}
    }

    # تحلیل
    truss = TrussModel(input_data)
    K, F = build_global_matrices(truss)
    displacements = solve_displacements(truss, K, F)
    results = calculate_element_results(truss, displacements)

    # نتایج
    element = results[0]

    print(f"📊 نتایج محاسبه شده:")
    print(f"  N = {element['N']:.1f} N")
    print(f"  U = {element['U']:.1f} J")
    print(f"  δL_eff = {element['delta_L_eff']:.6f} m")

    # محاسبات تحلیلی
    L = 2.0
    A = 0.01
    E = 210e9
    alpha = 1.2e-5
    delta_T = 100.0

    delta_L_free = alpha * delta_T * L  # 0.0024 m
    AE_L = A * E / L  # 1.05e9 N/m
    N_expected = -AE_L * delta_L_free  # -2.52e6 N
    U_expected = 0.5 * abs(N_expected) * abs(delta_L_free)  # 3024 J

    print(f"\n🎯 نتایج مورد انتظار:")
    print(f"  N = {N_expected:.1f} N")
    print(f"  U = {U_expected:.1f} J")
    print(f"  δL_free = {delta_L_free:.6f} m")

    # تحمل خطا
    tolerance = 0.01  # 1%

    if abs(element['N'] - N_expected) / abs(N_expected) < tolerance:
        print(f"\n✅ تست با موفقیت گذشت!")
    else:
        print(f"\n❌ تست شکست خورد!")
        print(f"  خطا: {(abs(element['N'] - N_expected) / abs(N_expected)) * 100:.2f}%")
        return False

if __name__ == "__main__":
    success = test_golden_reference()
    sys.exit(0 if success else 1)