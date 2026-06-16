"""
تست‌های کامل تحلیلگر خرپا - نسخه نهایی بدون shortcuts
"""

import pytest
import numpy as np
import json
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# اضافه کردن مسیر پوشه والد برای import کردن ماژول‌ها
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from truss_analysis.model import TrussModel, Node, Element  # اضافه کردن Node و Element
from truss_analysis.assembly import assemble_global_matrices, calculate_element_stiffness, get_reduced_system
from truss_analysis.solver import solve_displacements, calculate_element_results, calculate_total_energy, validate_energy
from truss_analysis.postprocess import sort_elements, calculate_percentages, generate_report
from truss_analysis.fileio import parse_input, validate_units, write_output

class TestNode:
    """تست کامل کلاس Node"""

    def test_node_creation_basic(self):
        """تست ایجاد گره با پارامترهای اولیه"""
        node = Node(id=1, x=1.5, y=2.5, is_support=False)
        assert node.id == 1
        assert node.x == 1.5
        assert node.y == 2.5
        assert node.is_support == False
        assert node.dofs is None

    def test_node_creation_support(self):
        """تست ایجاد گره تکیه‌گاهی"""
        node = Node(id=2, x=0.0, y=0.0, is_support=True)
        assert node.is_support == True

    def test_node_set_dofs(self):
        """تست تنظیم DOFهای گره"""
        node = Node(id=3, x=3.0, y=4.0)
        dof_map = {3: (6, 7)}
        node.set_dofs(dof_map)
        assert node.dofs == (6, 7)

    def test_node_displacement(self):
        """تست ذخیره جابجایی گره"""
        node = Node(id=4, x=0.0, y=0.0)
        node.displacement = np.array([0.001, -0.002])
        assert np.allclose(node.displacement, [0.001, -0.002])

    def test_node_repr(self):
        """تست نمایش رشته‌ای گره"""
        node = Node(id=5, x=1.0, y=2.0, is_support=True)
        repr_str = repr(node)
        assert "Node(5" in repr_str
        assert "support=True" in repr_str

class TestElement:
    """تست کامل کلاس Element"""

    def setup_method(self):
        """آماده‌سازی قبل از هر تست"""
        self.node1 = Node(id=1, x=0.0, y=0.0)
        self.node2 = Node(id=2, x=3.0, y=4.0)

    def test_element_creation(self):
        """تست ایجاد عضو"""
        element = Element(
            id=1,
            node_i=self.node1,
            node_j=self.node2,
            A=0.01,
            E=210e9,
            alpha=1.2e-5
        )
        assert element.id == 1
        assert element.A == 0.01
        assert element.E == 210e9
        assert element.alpha == 1.2e-5
        assert np.isclose(element.L, 5.0)
        assert np.isclose(element.c, 0.6)
        assert np.isclose(element.s, 0.8)

    def test_element_length_zero(self):
        """تست خطا برای طول صفر"""
        node_same = Node(id=3, x=1.0, y=1.0)
        with pytest.raises(ValueError):
            Element(
                id=2,
                node_i=node_same,
                node_j=node_same,  # همان گره
                A=0.01,
                E=210e9
            )

    def test_element_thermal_effects(self):
        """تست اثرات حرارتی"""
        element = Element(
            id=3,
            node_i=self.node1,
            node_j=Node(id=3, x=2.0, y=0.0),  # طول = 2
            A=0.01,
            E=210e9,
            alpha=1.2e-5,
            delta_T=50.0,
            delta_L0=0.001
        )
        delta_free = element.calculate_thermal_effects()
        expected = 1.2e-5 * 50.0 * 2.0 + 0.001
        assert np.isclose(delta_free, expected)
        assert np.isclose(element.delta_L_free, expected)

    def test_element_buckling_load(self):
        """تست محاسبه بار کمانش"""
        element = Element(
            id=4,
            node_i=self.node1,
            node_j=Node(id=4, x=2.0, y=0.0),
            A=0.01,
            E=210e9,
            I=7.85e-9,
            effective_length_factor=1.0
        )
        P_cr = element.calculate_buckling_load()
        expected = (np.pi**2 * 210e9 * 7.85e-9) / (2.0**2)
        assert np.isclose(P_cr, expected)

    def test_element_no_buckling(self):
        """تست عضو بدون ممان اینرسی"""
        element = Element(
            id=5,
            node_i=self.node1,
            node_j=self.node2,
            A=0.01,
            E=210e9,
            I=None  # بدون ممان اینرسی
        )
        P_cr = element.calculate_buckling_load()
        assert P_cr is None

    def test_element_repr(self):
        """تست نمایش رشته‌ای عضو"""
        element = Element(
            id=6,
            node_i=self.node1,
            node_j=self.node2,
            A=0.01,
            E=210e9
        )
        repr_str = repr(element)
        assert "Element(6" in repr_str
        assert "L=5.000" in repr_str

class TestTrussModel:
    """تست کامل کلاس TrussModel"""

    def test_truss_creation_simple(self):
        """تست ایجاد خرپای ساده"""
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
                },
                {
                    'id': 2,
                    'node_i': 2,
                    'node_j': 3,
                    'A': 0.01,
                    'E': 210e9
                }
            ]
        }

        truss = TrussModel(input_data)
        assert len(truss.nodes) == 3
        assert len(truss.elements) == 2
        assert truss.n_dof == 6
        assert len(truss.free_nodes) == 1
        assert len(truss.supported_nodes) == 2

    def test_truss_with_loads(self):
        """تست خرپا با بار"""
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
                    'E': 210e9
                }
            ],
            'loads': {
                'node_forces': [
                    {'node_id': 2, 'Fx': 10000.0, 'Fy': -5000.0}
                ]
            }
        }

        truss = TrussModel(input_data)
        assert len(truss.loads) == 1
        load = truss.loads[0]
        assert load['node_id'] == 2
        assert load['Fx'] == 10000.0
        assert load['Fy'] == -5000.0

    def test_truss_units_si_mm(self):
        """تست خرپا با واحدهای SI-mm"""
        input_data = {
            'units': 'SI-mm',
            'nodes': [
                {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
                {'id': 2, 'x': 2000.0, 'y': 0.0, 'is_support': False}  # 2000 mm = 2 m
            ],
            'elements': [
                {
                    'id': 1,
                    'node_i': 1,
                    'node_j': 2,
                    'A': 100,  # 100 mm² = 0.0001 m²
                    'E': 210e9
                }
            ]
        }

        truss = TrussModel(input_data)
        # بعد از تبدیل باید طول 2 متر باشد
        assert np.isclose(truss.nodes[2].x, 2.0)

    def test_truss_thermal_global(self):
        """تست اثر دمای سراسری"""
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
                    'alpha': 1.2e-5
                }
            ]
        }

        truss = TrussModel(input_data)
        assert truss.global_delta_T == 50.0
        element = truss.elements[1]
        assert element.delta_T == 50.0

    def test_truss_thermal_local(self):
        """تست اثر دمای محلی"""
        input_data = {
            'temperature_change': 20.0,
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
                    'E': 210e9,
                    'alpha': 1.2e-5,
                    'delta_T': 30.0  # دمای محلی
                }
            ]
        }

        truss = TrussModel(input_data)
        element = truss.elements[1]
        # دمای کل = سراسری + محلی
        assert element.delta_T == 50.0

    def test_truss_fabrication_error(self):
        """تست خطای ساخت"""
        input_data = {
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
                    'E': 210e9,
                    'delta_L0': 0.001
                }
            ]
        }

        truss = TrussModel(input_data)
        element = truss.elements[1]
        assert element.delta_L0 == 0.001

    def test_truss_invalid_element(self):
        """تست خطا برای عضو نامعتبر"""
        input_data = {
            'nodes': [{'id': 1, 'x': 0.0, 'y': 0.0}],
            'elements': [
                {
                    'id': 1,
                    'node_i': 1,
                    'node_j': 99,  # گره وجود ندارد
                    'A': 0.01,
                    'E': 210e9
                }
            ]
        }

        with pytest.raises(KeyError):  # تغییر از ValueError به KeyError
            TrussModel(input_data)

    def test_truss_invalid_units(self):
        """تست خطا برای واحد نامعتبر"""
        input_data = {
            'units': 'INVALID',
            'nodes': [{'id': 1, 'x': 0.0, 'y': 0.0}],
            'elements': []
        }

        with pytest.raises(ValueError):
            TrussModel(input_data)

    def test_truss_options(self):
        """تست گزینه‌های خرپا"""
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
                    'E': 210e9
                }
            ],
            'options': {
                'use_sparse': False,
                'bc_method': 'penalty',
                'penalty_value': 1e10,
                'plot_results': True,
                'displacement_scale': 50.0
            }
        }

        truss = TrussModel(input_data)
        assert truss.options['use_sparse'] == False
        assert truss.options['bc_method'] == 'penalty'
        assert truss.options['penalty_value'] == 1e10
        assert truss.options['plot_results'] == True
        assert truss.options['displacement_scale'] == 50.0

    def test_truss_non_sequential_node_ids(self):
        """تست شناسه‌های گره غیرمتوالی"""
        input_data = {
            'nodes': [
                {'id': 10, 'x': 0.0, 'y': 0.0, 'is_support': True},
                {'id': 20, 'x': 2.0, 'y': 0.0, 'is_support': True},
                {'id': 30, 'x': 1.0, 'y': 1.0, 'is_support': False}
            ],
            'elements': [
                {
                    'id': 1,
                    'node_i': 10,
                    'node_j': 30,
                    'A': 0.01,
                    'E': 210e9
                },
                {
                    'id': 2,
                    'node_i': 20,
                    'node_j': 30,
                    'A': 0.01,
                    'E': 210e9
                }
            ]
        }

        truss = TrussModel(input_data)
        assert 10 in truss.nodes
        assert 20 in truss.nodes
        assert 30 in truss.nodes
        assert truss.n_dof == 6  # 3 گره * 2 DOF

    def test_truss_dof_mapping(self):
        """تست نگاشت DOFها"""
        input_data = {
            'nodes': [
                {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
                {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': False}
            ],
            'elements': []
        }

        truss = TrussModel(input_data)
        dof_map = truss.get_dof_indices()
        assert dof_map[1] == (0, 1)  # DOFهای گره 1
        assert dof_map[2] == (2, 3)  # DOFهای گره 2

    def test_truss_validate_sign_convention_empty(self):
        """تست اعتبارسنجی قرارداد علامت برای خرپای خالی"""
        input_data = {
            'nodes': [
                {'id': 1, 'x': 0.0, 'y': 0.0, 'is_support': True},
                {'id': 2, 'x': 2.0, 'y': 0.0, 'is_support': True}
            ],
            'elements': []
        }

        truss = TrussModel(input_data)
        # بدون اعضا باید True برگرداند
        assert truss.validate_sign_convention() == True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])