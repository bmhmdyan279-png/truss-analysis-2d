"""
مدل‌سازی خرپا - نسخه نهایی کامل
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# ایمپورت از فایل جدید utils
from utils import validate_units, convert_to_si

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, id, x, y, is_support=False):
        self.id = id
        self.x = x
        self.y = y
        self.is_support = is_support
        self.dofs = None
        self.displacement = np.array([0.0, 0.0])

    def set_dofs(self, dof_map: Dict[int, Tuple[int, int]]):
        """تعیین شاخص DOFهای گره"""
        if self.id in dof_map:
            self.dofs = dof_map[self.id]

    def __repr__(self):
        return f"Node({self.id}, ({self.x:.3f}, {self.y:.3f}), support={self.is_support})"


class Element:
    """کلاس عضو خرپا"""

    def __init__(self, element_id: int = None, node_i: 'Node' = None, node_j: 'Node' = None,
                 A: float = None, E: float = None, alpha: float = 1.2e-5,
                 delta_T: float = 0.0, delta_L0: float = 0.0,
                 I: Optional[float] = None,
                 effective_length_factor: float = 1.0,
                 section_type: str = 'general', **kwargs):
        # backward compatibility: accept 'id' kwarg
        if element_id is None and 'id' in kwargs:
            element_id = kwargs.get('id')
        self.id = element_id
        self.node_i = node_i
        self.node_j = node_j
        self.A = A
        self.E = E
        self.alpha = alpha
        self.I = I
        self.K_eff = effective_length_factor
        self.section_type = section_type

        # محاسبه طول و کسینوس‌ها
        dx = node_j.x - node_i.x
        dy = node_j.y - node_i.y
        self.L = np.sqrt(dx**2 + dy**2)

        if self.L == 0:
            raise ValueError(f"طول عضو {element_id} صفر است.")

        self.c = dx / self.L
        self.s = dy / self.L

        # اثرات حرارتی و خطای ساخت
        self.delta_T = delta_T
        self.delta_L0 = delta_L0

        # محاسبه تغییر طول آزاد (حرارتی + خطای ساخت)
        self.delta_L_free = self.calculate_thermal_effects()

        # نتایج
        self.delta_L_eff = None
        self.N = None
        self.U = None
        self.status = None

    def calculate_thermal_effects(self) -> float:
        """محاسبه تغییر طول ناشی از حرارت و خطای ساخت"""
        delta_thermal = self.alpha * self.delta_T * self.L
        return delta_thermal + self.delta_L0

    def calculate_buckling_load(self) -> Optional[float]:
        """محاسبه بار کمانش برای عضو فشاری"""
        if self.I is None or self.I <= 0:
            return None

        # فرمول اویلر: P_cr = π²EI / (K_eff * L)²
        if self.L > 0 and self.E > 0:
            return (np.pi**2 * self.E * self.I) / (self.K_eff * self.L)**2
        return None

    def __repr__(self):
        return f"Element({self.id}: {self.node_i.id}→{self.node_j.id}, L={self.L:.3f}, A={self.A:.6f}, E={self.E:.3e})"


class TrussModel:
    """مدل کامل خرپا"""

    def __init__(self, input_data: Dict):
        # تنظیم واحدها
        self.units = input_data.get('units', 'SI')
        # اعتبارسنجی واحدها
        self._units_conv = validate_units(self.units)
        self._convert_to_si = convert_to_si
        self.global_delta_T = input_data.get('temperature_change', 0.0)

        # گزینه‌ها
        self.options = {
            'use_sparse': True,
            'bc_method': 'elimination',
            'penalty_value': 1e12,
            'plot_results': True,
            'displacement_scale': 'auto'
        }
        if 'options' in input_data:
            self.options.update(input_data['options'])

        # ذخیره داده ورودی
        self.input_data = input_data

        # ایجاد گره‌ها
        self.nodes = self._create_nodes(input_data['nodes'])

        # ایجاد اعضا
        self.elements = self._create_elements(input_data['elements'])

        # بارها
        self.loads = self._create_loads(input_data.get('loads', {}))

        # محاسبه DOFها
        self._setup_dofs()

    def _create_nodes(self, nodes_data: List[Dict]) -> Dict[int, Node]:
        """ایجاد دیکشنری گره‌ها"""
        nodes = {}
        for node_data in nodes_data:
            node = Node(
                id=node_data['id'],  # تغییر node_id به id
                x=convert_to_si(float(node_data['x']), self.units, 'length'),
                y=convert_to_si(float(node_data['y']), self.units, 'length'),
                is_support=bool(node_data.get('is_support', False))
            )
            nodes[node.id] = node
        return nodes

    def _create_elements(self, elements_data: List[Dict]) -> Dict[int, Element]:
        """ایجاد دیکشنری اعضا"""
        elements = {}
        for element_data in elements_data:
            element_id = element_data['id']

            # پیدا کردن گره‌ها
            node_i = self.nodes[element_data['node_i']]
            node_j = self.nodes[element_data['node_j']]

            # دمای کل (سراسری + محلی)
            delta_T_total = self.global_delta_T + element_data.get('delta_T', 0.0)

            # تبدیل مساحت مقطع به SI
            if element_data.get('A') is not None:
                A_value = convert_to_si(float(element_data['A']), self.units, 'length')
                # برای مساحت: اگر واحد طول mm باشد، مساحت mm² است که باید به m² تبدیل شود
                # اما convert_to_si فقط تبدیل طول انجام می‌دهد، پس باید مربع آن را بگیریم
                if self.units == 'SI-mm':
                    A_si = A_value**2  # تبدیل mm² به m²
                elif self.units == 'SI-cm':
                    A_si = A_value**2  # تبدیل cm² به m²
                else:
                    A_si = element_data['A']  # در واحد SI مساحت مستقیماً m² است
            else:
                A_si = None

            element = Element(
                element_id=element_id,
                node_i=node_i,
                node_j=node_j,
                A=A_si,
                E=float(element_data['E']),
                alpha=float(element_data.get('alpha', 1.2e-5)),
                delta_T=delta_T_total,
                delta_L0=float(element_data.get('delta_L0', 0.0)),
                I=element_data.get('I'),
                effective_length_factor=float(element_data.get('effective_length_factor', 1.0)),
                section_type=element_data.get('section_type', 'general')
            )
            elements[element.id] = element
        return elements

    def _create_loads(self, loads_data: Dict) -> List[Dict]:
        """ایجاد لیست بارها"""
        loads = []
        if 'node_forces' in loads_data:
            for i, load_data in enumerate(loads_data['node_forces']):
                loads.append({
                    'id': load_data.get('id', load_data.get('node_id', i + 1)),  # انعطاف‌پذیر
                    'node_id': load_data.get('node_id'),  # اضافه کردن node_id
                    'Fx': float(load_data.get('Fx', 0.0)),
                    'Fy': float(load_data.get('Fy', 0.0))
                })
        return loads

    def _setup_dofs(self):
        """تنظیم DOFهای سیستم"""
        self.n_dof = 2 * len(self.nodes)
        self.free_dofs = []
        self.fixed_dofs = []
        self.free_nodes = []
        self.supported_nodes = []

        # ایجاد نگاشت DOFها
        self.dof_map = {}
        dof_index = 0
        for id, node in sorted(self.nodes.items()):
            self.dof_map[id] = (dof_index, dof_index + 1)
            dof_index += 2

        # تعیین DOFهای آزاد و قفل شده
        for id, node in self.nodes.items():
            node.set_dofs(self.dof_map)
            dof_x, dof_y = self.dof_map[id]

            if node.is_support:
                self.fixed_dofs.extend([dof_x, dof_y])
                self.supported_nodes.append(node)
            else:
                self.free_dofs.extend([dof_x, dof_y])
                self.free_nodes.append(node)

        # مرتب‌سازی لیست DOFها
        self.free_dofs.sort()
        self.fixed_dofs.sort()

    def get_dof_indices(self) -> Dict[int, Tuple[int, int]]:
        """دریافت نگاشت DOFها"""
        return self.dof_map

    def validate_sign_convention(self) -> bool:
        """
        اعتبارسنجی قرارداد علامت برای نیروها و جابجایی‌ها
        بازگشت: True اگر قرارداد علامت رعایت شده باشد
        """
        try:
            # بررسی جهت نیروهای خارجی
            for load in self.loads:
                # هیچ نیرویی نباید به تکیه‌گاه اعمال شود
                node = self.nodes[load['id']]
                if node.is_support:
                    logger.warning(f"⚠️ نیرو به تکیه‌گاه (گره {node.id}) اعمال شده است.")
                    return False

            # بررسی جهت‌های عضوها
            for element in self.elements.values():
                # محاسبه بردار عضو
                dx = element.node_j.x - element.node_i.x
                dy = element.node_j.y - element.node_i.y

                # زاویه عضو
                angle = np.arctan2(dy, dx)

                # بررسی منطقی بودن زاویه (اختیاری)
                if not np.isfinite(angle):
                    logger.warning(f"⚠️ زاویه نامعتبر برای عضو {element.id}")
                    return False

            return True

        except Exception as e:
            logger.error(f"❌ خطا در اعتبارسنجی قرارداد علامت: {e}")
            return False