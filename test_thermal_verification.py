"""
تست تأیید علامت حرارتی با کد اصلی
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from model import TrussModel
from assembly import build_global_matrices
from solver import solve_displacements, calculate_element_results

print("=" * 60)
print("🧪 تست تأیید علامت حرارتی با کد اصلی")
print("=" * 60)

# داده‌های تست
input_data = {
    'units': 'SI',
    'nodes': [
        {'id': 1, 'x': 0, 'y': 0, 'is_support': True},
        {'id': 2, 'x': 2, 'y': 0, 'is_support': True}
    ],
    'elements': [
        {
            'id': 1,
            'node_i': 1,
            'node_j': 2,
            'A': 0.01,
            'E': 210e9,
            'alpha': 1.2e-5,
            'delta_T': 100.0
        }
    ],
    'loads': {'node_forces': []}
}

# تحلیل
truss = TrussModel(input_data)
K, F = build_global_matrices(truss)
displacements = solve_displacements(truss, K, F)
results = calculate_element_results(truss, displacements)

# نمایش نتایج
element = results[0]
print(f"\n📊 نتایج تحلیل:")
print(f"  نیروی محوری N = {element['N']:.1f} N")
print(f"  وضعیت = {element['status']}")
print(f"  تغییر طول مؤثر = {element['delta_L_eff']:.6f} m")

# محاسبات تحلیلی
delta_L_free = 1.2e-5 * 100 * 2  # α·ΔT·L = 0.0024 m
AE_L = 0.01 * 210e9 / 2  # 1.05e9 N/m
N_expected = -AE_L * delta_L_free  # -2,520,000 N

print(f"\n🎯 نتایج مورد انتظار:")
print(f"  N_expected = {N_expected:.1f} N")

# بررسی
if abs(element['N'] - N_expected) / abs(N_expected) < 0.01:  # 1% خطا
    print("\n✅ تست با موفقیت گذشت!")
    if element['status'] == 'Compression' and element['N'] < 0:
        print("✅ وضعیت فشاری تأیید شد!")
else:
    print(f"\n❌ تست شکست خورد!")
    print(f"  خطا: {abs(element['N'] - N_expected)/abs(N_expected)*100:.2f}%")

print("\n" + "=" * 60)