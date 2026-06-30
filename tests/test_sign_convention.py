"""
تست قرارداد علامت برای تحلیلگر خرپا
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# اضافه کردن مسیر پوشه والد به sys.path برای امکان import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from truss_analysis.model import TrussModel


def test_sign_convention_no_force_on_support():
    """تست اینکه هیچ نیرویی به تکیه‌گاه اعمال نشود"""
    input_data = {
        'units': 'SI',
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
                'E': 210e9
            }
        ],
        'loads': {
            'node_forces': [
                {'node_id': 1, 'Fx': 1000.0, 'Fy': 0.0}  # نیرو به تکیه‌گاه!
            ]
        }
    }

    truss = TrussModel(input_data)
    # این باید False برگرداند زیرا نیرو به تکیه‌گاه اعمال شده
    assert truss.validate_sign_convention() == False


def test_sign_convention_valid_force():
    """تست قرارداد علامت معتبر"""
    input_data = {
        'units': 'SI',
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
                'E': 210e9
            }
        ],
        'loads': {
            'node_forces': [
                {'node_id': 2, 'Fx': 1000.0, 'Fy': 0.0}  # نیرو به گره آزاد
            ]
        }
    }

    truss = TrussModel(input_data)
    assert truss.validate_sign_convention() == True


def test_sign_convention_no_loads():
    """تست خرپا بدون بار"""
    input_data = {
        'units': 'SI',
        'nodes': [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
            {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': True}
        ],
        'elements': [
            {
                'id': 1,
                'node_i': 1,
                'node_j': 2,
                'A': 0.01,
                'E': 210e9
            }
        ],
        'loads': {
            'node_forces': []  # بدون بار
        }
    }

    truss = TrussModel(input_data)
    assert truss.validate_sign_convention() == True


def test_sign_convention_multiple_supports():
    """تست خرپا با چند تکیه‌گاه"""
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
                'E': 210e9
            },
            {
                'id': 2,
                'node_i': 2,
                'node_j': 3,
                'A': 0.01,
                'E': 210e9
            }
        ],
        'loads': {
            'node_forces': [
                {'node_id': 3, 'Fx': 0.0, 'Fy': -1000.0}
            ]
        }
    }

    truss = TrussModel(input_data)
    assert truss.validate_sign_convention() == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])