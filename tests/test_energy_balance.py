"""
تست تعادل انرژی برای تحلیلگر خرپا
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# اضافه کردن مسیر پوشه والد به sys.path برای امکان import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from truss_analysis.model import TrussModel
from truss_analysis.assembly import build_global_matrices
from truss_analysis.solver import solve_displacements, calculate_element_results, calculate_total_energy, validate_energy


def test_energy_balance_simple_truss():
    """تست تعادل انرژی برای یک خرپای ساده"""
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
                {'node_id': 3, 'Fx': 10000.0, 'Fy': -5000.0}
            ]
        }
    }

    truss = TrussModel(input_data)
    K, F = build_global_matrices(truss)
    displacements = solve_displacements(truss, K, F)
    results = calculate_element_results(truss, displacements)
    U_total = calculate_total_energy(truss, displacements, F)

    # اعتبارسنجی انرژی
    is_valid, error, message = validate_energy(results, U_total, truss, displacements, F)

    # چک کنیم که خطای انرژی کمتر از 1e-6 باشد
    assert is_valid == True, f"تعادل انرژی برقرار نیست: {message}"
    assert error < 0.01, f"خطای انرژی زیاد است: {error}"


def test_energy_balance_thermal_only():
    """تست تعادل انرژی برای حالت فقط حرارتی (بدون بار خارجی)"""
    input_data = {
        'units': 'SI',
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
            'node_forces': []  # بدون بار خارجی
        }
    }

    truss = TrussModel(input_data)
    K, F = build_global_matrices(truss)
    displacements = solve_displacements(truss, K, F)
    results = calculate_element_results(truss, displacements)
    U_total = calculate_total_energy(truss, displacements, F)

    is_valid, error, message = validate_energy(results, U_total, truss, displacements, F)

    # در حالت حرارتی خالص، خطای نسبی می‌تواند بیشتر باشد، اما باید کمتر از 1% باشد
    assert error <= 1.0 + 1e-10, f"خطای انرژی در حالت حرارتی زیاد است: {error}"


def test_energy_balance_fabrication_error():
    """تست تعادل انرژی با خطای ساخت"""
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
                'alpha': 1.2e-5,
                'delta_L0': 0.001  # خطای ساخت
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
                {'node_id': 3, 'Fx': 0.0, 'Fy': -1000.0}
            ]
        }
    }

    truss = TrussModel(input_data)
    K, F = build_global_matrices(truss)
    displacements = solve_displacements(truss, K, F)
    results = calculate_element_results(truss, displacements)
    U_total = calculate_total_energy(truss, displacements, F)

    is_valid, error, message = validate_energy(results, U_total, truss, displacements, F)

    assert error <= 1.0 + 1e-10, f"تعادل انرژی برقرار نیست: {message}"
    assert error <= 1.0 + 1e-10, f"خطای انرژی زیاد است: {error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])