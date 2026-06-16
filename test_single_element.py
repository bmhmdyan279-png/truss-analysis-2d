"""
تست تحلیل یک عضو منفرد - برای بررسی صحت محاسبات پایه
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from truss_analysis.model import TrussModel
from truss_analysis.assembly import build_global_matrices
from truss_analysis.solver import solve_displacements, calculate_element_results


def test_single_element_tension():
    """تست یک عضو تحت کشش"""
    print("\n" + "=" * 60)
    print("تست یک عضو تحت کشش")
    print("=" * 60)

    input_data = {
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
            {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': False}
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 2,
                'A': 0.01,  # 0.01 m²
                'E': 210e9,  # 210 GPa
                'alpha': 1.2e-5
            }
        ],
        'loads': {
            'node_forces': [
                {'node_id': 2, 'Fx': 10000.0, 'Fy': 0.0}
            ]
        }
    }

    truss = TrussModel(input_data)
    print(f"تعداد گره‌ها: {len(truss.nodes)}")
    print(f"تعداد اعضا: {len(truss.elements)}")
    print(f"DOFهای آزاد: {truss.free_dofs}")
    print(f"DOFهای ثابت: {truss.fixed_dofs}")

    K, F = build_global_matrices(truss)
    print(f"اندازه ماتریس سختی: {K.shape}")

    displacements = solve_displacements(truss, K, F)
    print(f"جابجایی‌ها:\n{displacements}")

    results = calculate_element_results(truss, displacements)

    # بررسی نتایج
    element_result = results[0]
    print(f"\nنتایج عضو:")
    print(f"  طول عضو: {element_result['L']:.4f} m")
    print(f"  تغییر طول مؤثر: {element_result['delta_L_eff']:.6f} m")
    print(f"  نیروی محوری: {element_result['N']:.2f} N")
    print(f"  وضعیت: {element_result['status']}")
    print(f"  انرژی کرنشی: {element_result['U']:.6f} J")

    # محاسبات تحلیلی
    L = 2.0  # طول عضو
    A = 0.01  # مساحت مقطع
    E = 210e9  # مدول الاستیسیته
    F_applied = 10000.0  # نیروی اعمال شده

    # تغییر طول تحلیلی: δ = FL/(EA)
    delta_analytical = F_applied * L / (A * E)
    print(f"\nتحلیلی:")
    print(f"  تغییر طول انتظاری: {delta_analytical:.6e} m")

    # نیروی تحلیلی: باید برابر با نیروی اعمال شده باشد
    N_analytical = F_applied

    # انرژی کرنشی تحلیلی: U = 0.5 * F * δ
    U_analytical = 0.5 * F_applied * delta_analytical

    # استفاده از pytest.approx برای مقایسه اعداد اعشاری
    assert element_result['status'] == 'Tension'
    assert element_result['N'] > 0
    assert element_result['delta_L_eff'] == pytest.approx(delta_analytical, rel=1e-6)
    assert element_result['N'] == pytest.approx(N_analytical, rel=1e-6)
    assert element_result['U'] == pytest.approx(U_analytical, rel=1e-6)

    print("✅ تست یک عضو تحت کشش با موفقیت گذشت")


def test_single_element_compression():
    """تست یک عضو تحت فشار"""
    print("\n" + "=" * 60)
    print("تست یک عضو تحت فشار")
    print("=" * 60)

    input_data = {
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
            {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': False}
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
        ],
        'loads': {
            'node_forces': [
                {'node_id': 2, 'Fx': -10000.0, 'Fy': 0.0}
            ]
        }
    }

    truss = TrussModel(input_data)
    K, F = build_global_matrices(truss)
    displacements = solve_displacements(truss, K, F)
    results = calculate_element_results(truss, displacements)

    element_result = results[0]

    print(f"نتایج عضو:")
    print(f"  نیروی محوری: {element_result['N']:.2f} N")
    print(f"  وضعیت: {element_result['status']}")

    # اعتبارسنجی
    assert element_result['status'] == 'Compression'
    assert element_result['N'] < 0
    assert element_result['N'] == pytest.approx(-10000.0, rel=1e-6)

    print("✅ تست یک عضو تحت فشار با موفقیت گذشت")


def test_single_element_thermal_expansion():
    """تست یک عضو با انبساط حرارتی"""
    print("\n" + "=" * 60)
    print("تست یک عضو با انبساط حرارتی")
    print("=" * 60)

    input_data = {
        'temperature_change': 50.0,
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
            {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': False}
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
        # بدون بار خارجی
    }

    truss = TrussModel(input_data)
    print(f"تغییر دمای اعمال شده: {truss.global_delta_T}°C")

    K, F = build_global_matrices(truss)
    displacements = solve_displacements(truss, K, F)
    results = calculate_element_results(truss, displacements)

    element_result = results[0]

    print(f"\nنتایج عضو:")
    print(f"  تغییر طول آزاد: {element_result['delta_L_free']:.6f} m")
    print(f"  نیروی محوری: {element_result['N']:.2f} N")
    print(f"  وضعیت: {element_result['status']}")

    # محاسبات تحلیلی
    L = 2.0
    alpha = 1.2e-5
    delta_T = 50.0
    A = 0.01
    E = 210e9

    # تغییر طول آزاد ناشی از حرارت
    delta_L_free_analytical = alpha * delta_T * L

    print(f"\nتحلیلی:")
    print(f"  تغییر طول آزاد انتظاری: {delta_L_free_analytical:.6f} m")

    # استفاده از pytest.approx برای مقایسه
    assert element_result['delta_L_free'] == pytest.approx(delta_L_free_analytical, rel=1e-6)

    # وقتی یک انتها آزاد است، عضو می‌تواند آزادانه منبسط شود
    # نیرو باید نزدیک به صفر باشد
    assert abs(element_result['N']) < 1e-8  # تغییر از < 0 به < 1e-8
    # وضعیت می‌تواند 'Tension' یا 'Compression' باشد بسته به علامت عددی
    # فقط بررسی کنیم که نیرو نزدیک به صفر است

    print("✅ تست یک عضو با انبساط حرارتی با موفقیت گذشت")


def test_single_element_fabrication_error():
    """تست یک عضو با خطای ساخت"""
    print("\n" + "=" * 60)
    print("تست یک عضو با خطای ساخت")
    print("=" * 60)

    input_data = {
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
            {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': True}  # هر دو انتها تکیه‌گاه
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 2,
                'A': 0.01,
                'E': 210e9,
                'alpha': 1.2e-5,
                'delta_L0': 0.001  # عضو 1mm بلندتر ساخته شده
            }
        ]
        # بدون بار خارجی
    }

    truss = TrussModel(input_data)
    K, F = build_global_matrices(truss)

    displacements = solve_displacements(truss, K, F)
    results = calculate_element_results(truss, displacements)

    element_result = results[0]

    print(f"\nنتایج عضو:")
    print(f"  تغییر طول آزاد: {element_result['delta_L_free']:.6f} m")
    print(f"  نیروی محوری: {element_result['N']:.2f} N")
    print(f"  وضعیت: {element_result['status']}")

    # در این حالت، چون هر دو انتها تکیه‌گاه هستند
    # و عضو بلندتر ساخته شده، باید نیروی فشاری ایجاد شود
    assert element_result['status'] == 'Compression'
    assert element_result['N'] < 0

    # محاسبه نیروی انتظاری
    L = 2.0
    A = 0.01
    E = 210e9
    delta_L0 = 0.001
    N_expected = -A * E * delta_L0 / L

    assert element_result['N'] == pytest.approx(N_expected, rel=1e-6)

    print("✅ تست یک عضو با خطای ساخت با موفقیت گذشت")


def test_single_element_combined():
    """تست یک عضو با ترکیب اثرات"""
    print("\n" + "=" * 60)
    print("تست یک عضو با ترکیب اثرات")
    print("=" * 60)

    input_data = {
        'temperature_change': 30.0,
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
            {'id': 2, 'x': 3.0, 'y': 0.0, 'is_support': False}
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 2,
                'A': 0.015,
                'E': 200e9,
                'alpha': 1.1e-5,
                'delta_T': 20.0,  # دمای محلی اضافه
                'delta_L0': 0.0005  # خطای ساخت
            }
        ],
        'loads': {
            'node_forces': [
                {'node_id': 2, 'Fx': 8000.0, 'Fy': 0.0}
            ]
        }
    }

    truss = TrussModel(input_data)

    print(f"پارامترهای عضو:")
    element = list(truss.elements.values())[0]
    print(f"  طول: {element.L:.3f} m")
    print(f"  دمای کل: {element.delta_T:.1f}°C")
    print(f"  خطای ساخت: {element.delta_L0:.6f} m")

    K, F = build_global_matrices(truss)
    displacements = solve_displacements(truss, K, F)
    results = calculate_element_results(truss, displacements)

    element_result = results[0]

    print(f"\nنتایج عضو:")
    print(f"  تغییر طول آزاد: {element_result['delta_L_free']:.6f} m")
    print(f"  تغییر طول مؤثر: {element_result['delta_L_eff']:.6f} m")
    print(f"  نیروی محوری: {element_result['N']:.2f} N")
    print(f"  وضعیت: {element_result['status']}")
    print(f"  انرژی کرنشی: {element_result['U']:.6f} J")

    # محاسبات تحلیلی
    L = 3.0
    A = 0.015
    E = 200e9
    alpha = 1.1e-5
    delta_T_total = 30.0 + 20.0  # دمای سراسری + محلی
    delta_L0 = 0.0005
    F_applied = 8000.0

    # تغییر طول آزاد ناشی از حرارت و خطای ساخت
    delta_L_free_analytical = alpha * delta_T_total * L + delta_L0

    # تغییر طول مؤثر: δ_eff = u - δ_free
    # برای محاسبه تحلیلی: از رابطه F = (EA/L) * δ_eff
    # بنابراین: δ_eff = F * L / (EA)
    delta_L_eff_analytical = F_applied * L / (A * E)

    print(f"\nتحلیلی:")
    print(f"  تغییر طول آزاد انتظاری: {delta_L_free_analytical:.6f} m")
    print(f"  تغییر طول مؤثر انتظاری: {delta_L_eff_analytical:.6f} m")

    # استفاده از pytest.approx برای مقایسه
    assert element_result['delta_L_free'] == pytest.approx(delta_L_free_analytical, rel=1e-6)
    assert element_result['delta_L_eff'] == pytest.approx(delta_L_eff_analytical, rel=1e-6)

    print("✅ تست یک عضو با ترکیب اثرات با موفقیت گذشت")


if __name__ == "__main__":
    """اجرای تمام تست‌های عضو منفرد"""

    print("\n" + "=" * 60)
    print("شروع تست‌های عضو منفرد")
    print("=" * 60)

    test_functions = [
        test_single_element_tension,
        test_single_element_compression,
        test_single_element_thermal_expansion,
        test_single_element_fabrication_error,
        test_single_element_combined
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
    print("خلاصه نتایج تست‌های عضو منفرد")
    print("=" * 60)
    print(f"تعداد تست‌ها: {total}")
    print(f"موفق: {passed}")
    print(f"شکست: {total - passed}")

    if passed == total:
        print("🎉 تمام تست‌های عضو منفرد با موفقیت گذشتند!")
    else:
        print("⚠️ برخی تست‌های عضو منفرد شکست خوردند")

    print("=" * 60)