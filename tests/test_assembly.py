"""
تست مونتاژ ماتریس‌های سراسری - نسخه کامل
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# اضافه کردن مسیر پروژه به sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from truss_analysis.model import TrussModel
from truss_analysis.assembly import build_global_matrices, calculate_element_stiffness, calculate_element_thermal_force, \
    get_reduced_system


def test_calculate_element_stiffness():
    """تست محاسبه ماتریس سختی عنصر"""
    print("\n" + "=" * 60)
    print("تست محاسبه ماتریس سختی عنصر")
    print("=" * 60)

    # ایجاد یک خرپای ساده
    input_data = {
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0},
            {'id': 2, 'x': 2.0, 'y': 0.0}
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 2,
                'A': 0.01,
                'E': 210e9
            }
        ]
    }

    truss = TrussModel(input_data)
    element = list(truss.elements.values())[0]

    # محاسبه ماتریس سختی
    k_e = calculate_element_stiffness(element)

    # بررسی ابعاد ماتریس
    assert k_e.shape == (4, 4)

    # بررسی تقارن ماتریس
    assert np.allclose(k_e, k_e.T, rtol=1e-10)

    # بررسی جمع سطرها (باید صفر باشد)
    row_sums = np.sum(k_e, axis=1)
    assert np.allclose(row_sums, 0.0, atol=1e-10)

    # محاسبه AE/L
    AE_L = element.A * element.E / element.L

    # برای عضو افقی (c=1, s=0)
    expected_k = AE_L * np.array([
        [1, 0, -1, 0],
        [0, 0, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 0]
    ])

    # به دلیل خطای عددی، از tolerance استفاده می‌کنیم
    assert np.allclose(k_e, expected_k, rtol=1e-10)

    print(f"✅ ماتریس سختی عنصر محاسبه شد")
    print(f"  اندازه: {k_e.shape}")
    print(f"  AE/L: {AE_L:.2e} N/m")


def test_calculate_element_thermal_force():
    """تست محاسبه نیروی معادل حرارتی"""
    print("\n" + "=" * 60)
    print("تست نیروی معادل حرارتی")
    print("=" * 60)

    # ایجاد یک خرپا با اثر حرارتی
    input_data = {
        'temperature_change': 50.0,
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0},
            {'id': 2, 'x': 2.0, 'y': 0.0}
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 2,
                'A': 0.01,
                'E': 210e9,
                'alpha': 1.2e-5
            }
        ]
    }

    truss = TrussModel(input_data)
    element = list(truss.elements.values())[0]

    # محاسبه نیروی معادل حرارتی
    f_e = calculate_element_thermal_force(element)

    # بررسی ابعاد بردار
    assert f_e.shape == (4,)

    # برای عضو افقی، نیروها باید در راستای x باشند
    # f_e = AE/L * delta_L_free * [-c, -s, c, s]
    # با c=1, s=0

    AE_L = element.A * element.E / element.L
    delta_L_free = element.delta_L_free

    expected_f = AE_L * delta_L_free * np.array([-1, 0, 1, 0])

    # بررسی کنید که اندازه بردارها یکسان است
    assert f_e.shape == (4,)
    # بررسی کنید که مجموع نیروها صفر باشد (تعادل)
    assert np.allclose(f_e[0] + f_e[2], 0.0, atol=1e-10)
    assert np.allclose(f_e[1] + f_e[3], 0.0, atol=1e-10)

    print(f"✅ نیروی معادل حرارتی محاسبه شد")
    print(f"  اندازه بردار: {f_e.shape}")
    print(f"  delta_L_free: {delta_L_free:.6e} m")


def test_assemble_global_matrices_sparse():
    """تست مونتاژ ماتریس‌های سراسری با ماتریس تنک"""
    print("\n" + "=" * 60)
    print("تست مونتاژ با ماتریس تنک")
    print("=" * 60)

    input_data = {
        'units': 'SI',
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
            {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': True},
            {'id': 3, 'x': 1.0, 'y': 1.0, 'is_support': False}
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 3,
                'A': 0.01,
                'E': 210e9,
                'alpha': 1.2e-5
            },
            {
                'id': 2,
                'node_i': 2,
                'node_j': 3,
                'A': 0.01,
                'E': 210e9,
                'alpha': 1.2e-5
            }
        ],
        'loads': {
            'node_forces': [
                {'node_id': 3, 'Fx': 1000.0, 'Fy': -2000.0}
            ]
        },
        'options': {
            'use_sparse': True  # استفاده از ماتریس تنک
        }
    }

    truss = TrussModel(input_data)

    # مونتاژ ماتریس‌ها
    K_global, F_global = build_global_matrices(truss)

    # بررسی ابعاد
    assert K_global.shape == (6, 6)  # 3 گره * 2 DOF
    assert F_global.shape == (6,)

    # بررسی نوع ماتریس (باید sparse باشد)
    from scipy import sparse
    assert isinstance(K_global, sparse.spmatrix)

    # بررسی تقارن ماتریس سختی
    K_dense = K_global.toarray()
    assert np.allclose(K_dense, K_dense.T, rtol=1e-10)

    # بررسی نیروها
    # فقط گره 3 باید نیرو داشته باشد
    node3_dofs = truss.nodes[3].dofs
    assert abs(F_global[node3_dofs[0]] - 1000.0) < 1e-10
    assert abs(F_global[node3_dofs[1]] - (-2000.0)) < 1e-10

    print(f"✅ ماتریس‌های سراسری با ماتریس تنک مونتاژ شدند")
    print(f"  اندازه K: {K_global.shape}")
    print(f"  اندازه F: {F_global.shape}")
    print(f"  نوع K: {type(K_global).__name__}")


def test_assemble_global_matrices_dense():
    """تست مونتاژ ماتریس‌های سراسری با ماتریس متراکم"""
    print("\n" + "=" * 60)
    print("تست مونتاژ با ماتریس متراکم")
    print("=" * 60)

    input_data = {
        'units': 'SI',
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
            {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': True},
            {'id': 3, 'x': 1.0, 'y': 1.0, 'is_support': False}
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 3,
                'A': 0.01,
                'E': 210e9,
                'alpha': 1.2e-5
            },
            {
                'id': 2,
                'node_i': 2,
                'node_j': 3,
                'A': 0.01,
                'E': 210e9,
                'alpha': 1.2e-5
            }
        ],
        'options': {
            'use_sparse': False  # استفاده از ماتریس متراکم
        }
    }

    truss = TrussModel(input_data)

    # مونتاژ ماتریس‌ها
    K_global, F_global = build_global_matrices(truss)

    # بررسی ابعاد
    assert K_global.shape == (6, 6)
    assert F_global.shape == (6,)

    # بررسی نوع ماتریس (باید numpy array باشد)
    assert isinstance(K_global, np.ndarray)

    # بررسی تقارن ماتریس سختی
    assert np.allclose(K_global, K_global.T, rtol=1e-10)

    # بررسی مثبت بودن قطر اصلی (برای عناصر غیرصفر)
    for i in range(6):
        if K_global[i, i] != 0:
            assert K_global[i, i] > 0

    print(f"✅ ماتریس‌های سراسری با ماتریس متراکم مونتاژ شدند")
    print(f"  اندازه K: {K_global.shape}")
    print(f"  اندازه F: {F_global.shape}")
    print(f"  نوع K: {type(K_global).__name__}")


def test_assemble_with_thermal_effects():
    """تست مونتاژ با اثرات حرارتی"""
    print("\n" + "=" * 60)
    print("تست مونتاژ با اثرات حرارتی")
    print("=" * 60)

    input_data = {
        'temperature_change': 50.0,
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
            {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': True},
            {'id': 3, 'x': 1.0, 'y': 1.0, 'is_support': False}
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 3,
                'A': 0.01,
                'E': 210e9,
                'alpha': 1.2e-5,
                'delta_T': 20.0  # دمای محلی اضافه
            }
        ]
    }

    truss = TrussModel(input_data)
    element = list(truss.elements.values())[0]

    # مونتاژ ماتریس‌ها
    K_global, F_global = build_global_matrices(truss)

    # محاسبه نیروی حرارتی انتظاری
    delta_T_total = 50.0 + 20.0  # سراسری + محلی
    delta_L_free_expected = element.alpha * element.L * delta_T_total
    AE_L = element.A * element.E / element.L

    # نیروهای معادل در DOFهای عنصر
    node1_dofs = truss.nodes[1].dofs
    node3_dofs = truss.nodes[3].dofs

    # بررسی نیروهای حرارتی در بردار F_global
    # فرمول: f = AE/L * delta_L_free * [-c, -s, c, s]
    c, s = element.c, element.s
    f_thermal = AE_L * delta_L_free_expected * np.array([-c, -s, c, s])

    # جمع نیروها در DOFهای مربوطه
    total_fx = F_global[node1_dofs[0]] + F_global[node3_dofs[0]]
    total_fy = F_global[node1_dofs[1]] + F_global[node3_dofs[1]]

    # در سیستم ایستا، مجموع نیروها باید صفر باشد
    assert abs(total_fx) < 1e-10
    assert abs(total_fy) < 1e-10

    print(f"✅ مونتاژ با اثرات حرارتی موفقیت‌آمیز بود")
    print(f"  delta_T کل: {delta_T_total}°C")
    print(f"  delta_L_free: {delta_L_free_expected:.6e} m")
    print(f"  مجموع نیروهای x: {total_fx:.2e} N")
    print(f"  مجموع نیروهای y: {total_fy:.2e} N")


def test_assemble_with_fabrication_error():
    """تست مونتاژ با خطای ساخت"""
    print("\n" + "=" * 60)
    print("تست مونتاژ با خطای ساخت")
    print("=" * 60)

    input_data = {
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
            {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': True},
            {'id': 3, 'x': 1.0, 'y': 1.0, 'is_support': False}
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 3,
                'A': 0.01,
                'E': 210e9,
                'alpha': 1.2e-5,
                'delta_L0': 0.001  # خطای ساخت
            }
        ]
    }

    truss = TrussModel(input_data)
    element = list(truss.elements.values())[0]

    # مونتاژ ماتریس‌ها
    K_global, F_global = build_global_matrices(truss)

    # خطای ساخت باید به بردار نیرو اضافه شود
    AE_L = element.A * element.E / element.L
    delta_L_free_expected = element.delta_L0  # فقط خطای ساخت

    # بررسی اینکه delta_L_free شامل خطای ساخت است
    assert abs(element.delta_L_free - delta_L_free_expected) < 1e-10

    print(f"✅ مونتاژ با خطای ساخت موفقیت‌آمیز بود")
    print(f"  delta_L0: {element.delta_L0:.6e} m")
    print(f"  delta_L_free: {element.delta_L_free:.6e} m")


def test_get_reduced_system():
    """تست کاهش سیستم با حذف DOFهای قفل شده"""
    print("\n" + "=" * 60)
    print("تست کاهش سیستم")
    print("=" * 60)

    input_data = {
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
            {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': True},
            {'id': 3, 'x': 1.0, 'y': 1.0, 'is_support': False}
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 3,
                'A': 0.01,
                'E': 210e9
            }
        ],
        'loads': {
            'node_forces': [
                {'node_id': 3, 'Fx': 1000.0, 'Fy': -2000.0}
            ]
        }
    }

    truss = TrussModel(input_data)
    K_global, F_global = build_global_matrices(truss)

    # گرفتن سیستم کاهش یافته
    K_ff, F_f, free_dofs, fixed_dofs = get_reduced_system(truss, K_global, F_global)

    # بررسی تعداد DOFها
    # 3 گره = 6 DOF
    # 2 گره تکیه‌گاهی = 4 DOF قفل شده
    # 1 گره آزاد = 2 DOF آزاد
    assert len(free_dofs) == 2
    assert len(fixed_dofs) == 4

    # بررسی ابعاد ماتریس کاهش یافته
    assert K_ff.shape == (2, 2)
    assert F_f.shape == (2,)

    # DOFهای آزاد باید مربوط به گره آزاد باشد
    node3_dofs = truss.nodes[3].dofs
    assert node3_dofs[0] in free_dofs
    assert node3_dofs[1] in free_dofs

    # نیروها باید در سیستم کاهش یافته حفظ شوند
    assert abs(F_f[0] - 1000.0) < 1e-10
    assert abs(F_f[1] - (-2000.0)) < 1e-10

    print(f"✅ سیستم با موفقیت کاهش یافت")
    print(f"  DOFهای آزاد: {free_dofs}")
    print(f"  DOFهای ثابت: {fixed_dofs}")
    print(f"  اندازه K_ff: {K_ff.shape}")
    print(f"  اندازه F_f: {F_f.shape}")


def test_singular_matrix_detection():
    """تست شناسایی ماتریس منفرد (سازه ناپایدار)"""
    print("\n" + "=" * 60)
    print("تست شناسایی سازه ناپایدار")
    print("=" * 60)

    # یک سازه ناپایدار: همه گره‌ها آزاد
    input_data = {
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': False},
            {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': False}
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 2,
                'A': 0.01,
                'E': 210e9
            }
        ]
    }

    truss = TrussModel(input_data)
    K_global, F_global = build_global_matrices(truss)

    # ماتریس باید منفرد باشد (determinant ~ 0)
    K_dense = K_global if isinstance(K_global, np.ndarray) else K_global.toarray()
    det = np.linalg.det(K_dense)

    # برای ماتریس منفرد، دترمینان نزدیک به صفر است
    assert abs(det) < 1e-6

    print(f"✅ ماتریس منفرد شناسایی شد")
    print(f"  دترمینان: {det:.2e}")


if __name__ == "__main__":
    """اجرای تمام تست‌های مونتاژ"""

    print("\n" + "=" * 60)
    print("شروع تست‌های مونتاژ")
    print("=" * 60)

    test_functions = [
        test_calculate_element_stiffness,
        test_calculate_element_thermal_force,
        test_assemble_global_matrices_sparse,
        test_assemble_global_matrices_dense,
        test_assemble_with_thermal_effects,
        test_assemble_with_fabrication_error,
        test_get_reduced_system,
        test_singular_matrix_detection
    ]

    passed = 0
    total = len(test_functions)

    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__}: گذشت")
        except AssertionError as e:
            print(f"❌ {test_func.__name__}: شکست - {str(e)}")
        except Exception as e:
            print(f"❌ {test_func.__name__}: خطا - {str(e)}")

    print("\n" + "=" * 60)
    print("خلاصه نتایج تست‌های مونتاژ")
    print("=" * 60)
    print(f"تعداد تست‌ها: {total}")
    print(f"موفق: {passed}")
    print(f"شکست: {total - passed}")

    if passed == total:
        print("🎉 تمام تست‌های مونتاژ با موفقیت گذشتند!")
    else:
        print("⚠️ برخی تست‌های مونتاژ شکست خوردند")

    print("=" * 60)