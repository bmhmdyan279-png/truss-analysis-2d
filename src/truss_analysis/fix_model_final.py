# fix_model_final.py
"""بازیابی کامل model.py با تمام propertyها و validation"""

model_content = '''"""
مدل‌سازی خرپای دوبعدی با validation کامل
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class Node:
    """کلاس گره"""
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
        return (
            f"Node({self.id}, ({self.x:.3f}, {self.y:.3f}), support={self.is_support})"
        )


class Element:
    """کلاس عضو"""
    def __init__(
        self,
        id,
        node_i: Node,
        node_j: Node,
        E: float,
        A: float,
        alpha: float = 0.0,
        delta_T: float = 0.0,
        delta_L0: float = 0.0,
        I: Optional[float] = None,
        effective_length_factor: float = 1.0,
    ):
        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.E = E
        self.A = A
        self.alpha = alpha
        self.delta_T = delta_T
        self.delta_L0 = delta_L0
        self.I = I
        self.effective_length_factor = effective_length_factor

    @property
    def L(self) -> float:
        """طول عضو"""
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        return np.sqrt(dx**2 + dy**2)

    @property
    def direction_cosines(self) -> Tuple[float, float]:
        """کسینوس‌های جهت"""
        L = self.L
        if L == 0:
            raise ValueError(f"Element {self.id} has zero length")
        cx = (self.node_j.x - self.node_i.x) / L
        cy = (self.node_j.y - self.node_i.y) / L
        return cx, cy

    def calculate_thermal_effects(self) -> float:
        """محاسبه تغییر طول آزاد ناشی از دما"""
        return self.alpha * self.delta_T * self.L

    def calculate_buckling_load(self) -> float:
        """محاسبه بار کمانش اویلر"""
        if self.I is None or self.I == 0:
            return np.inf
        L_eff = self.effective_length_factor * self.L
        return np.pi**2 * self.E * self.I / (L_eff**2)

    def __repr__(self):
        return (
            f"Element({self.id}, {self.node_i.id}->{self.node_j.id}, "
            f"E={self.E:.2e}, A={self.A:.4f})"
        )


class TrussModel:
    """مدل خرپا"""
    def __init__(self, input_data: Dict[str, Any], options: Optional[Dict] = None):
        """
        مقداردهی اولیه

        Args:
            input_data: داده‌های ورودی
            options: گزینه‌های مدل
        """
        # ✅ Validation ورودی
        if not isinstance(input_data, dict):
            raise TypeError(
                f"input_data must be a dict, got {type(input_data).__name__}"
            )

        self.units = input_data.get("units", "SI")
        self.global_delta_T = input_data.get("temperature_change", 0.0)
        self.options = options or {"use_sparse": True}

        # ایجاد اجزا
        self.nodes: Dict[int, Node] = self._create_nodes(input_data.get("nodes", []))
        self.elements: Dict[int, Element] = self._create_elements(
            input_data.get("elements", [])
        )
        self.loads: Dict[int, Dict[str, float]] = self._create_loads(
            input_data.get("loads", {})
        )

        # تنظیم DOFها
        self._setup_dofs()

        # تبدیل واحد
        self._convert_units()

    def _create_nodes(self, nodes_data: Any) -> Dict[int, Node]:
        """ایجاد گره‌ها"""
        # ✅ Validation
        if not isinstance(nodes_data, list):
            raise TypeError(
                f"'nodes' must be a list, got {type(nodes_data).__name__}"
            )

        nodes = {}
        for i, node_data in enumerate(nodes_data):
            if not isinstance(node_data, dict):
                raise TypeError(f"nodes[{i}] must be a dict")

            required_fields = ["id", "x", "y"]
            for field in required_fields:
                if field not in node_data:
                    raise ValueError(f"nodes[{i}] missing '{field}'")

            node_id = node_data["id"]
            if node_id in nodes:
                raise ValueError(f"Duplicate node ID: {node_id}")

            node = Node(
                id=node_id,
                x=float(node_data["x"]),
                y=float(node_data["y"]),
                is_support=bool(node_data.get("is_support", False))
            )
            nodes[node_id] = node

        logger.info(f"Created {len(nodes)} nodes")
        return nodes

    def _create_elements(self, elements_data: Any) -> Dict[int, Element]:
        """ایجاد اعضا"""
        if not isinstance(elements_data, list):
            raise TypeError(f"'elements' must be a list")

        elements = {}
        for i, elem_data in enumerate(elements_data):
            if not isinstance(elem_data, dict):
                raise TypeError(f"elements[{i}] must be a dict")

            required_fields = ["id", "node_i", "node_j", "E", "A"]
            for field in required_fields:
                if field not in elem_data:
                    raise ValueError(f"elements[{i}] missing '{field}'")

            elem_id = elem_data["id"]
            if elem_id in elements:
                raise ValueError(f"Duplicate element ID: {elem_id}")

            node_i_id = elem_data["node_i"]
            node_j_id = elem_data["node_j"]

            if node_i_id not in self.nodes:
                raise ValueError(f"node_i={node_i_id} does not exist")
            if node_j_id not in self.nodes:
                raise ValueError(f"node_j={node_j_id} does not exist")

            element = Element(
                id=elem_id,
                node_i=self.nodes[node_i_id],
                node_j=self.nodes[node_j_id],
                E=float(elem_data["E"]),
                A=float(elem_data["A"]),
                alpha=float(elem_data.get("alpha", 0.0)),
                delta_T=float(elem_data.get("delta_T", 0.0)),
                delta_L0=float(elem_data.get("delta_L0", 0.0)),
                I=float(elem_data["I"]) if "I" in elem_data else None,
                effective_length_factor=float(elem_data.get("effective_length_factor", 1.0))
            )
            elements[elem_id] = element

        logger.info(f"Created {len(elements)} elements")
        return elements

    def _create_loads(self, loads_data: Any) -> Dict[int, Dict[str, float]]:
        """ایجاد بارها"""
        if not isinstance(loads_data, dict):
            raise TypeError(f"'loads' must be a dict")

        loads = {}
        node_forces = loads_data.get("node_forces", [])

        for i, force_data in enumerate(node_forces):
            if not isinstance(force_data, dict):
                raise TypeError(f"loads.node_forces[{i}] must be a dict")

            if "node_id" not in force_data:
                raise ValueError(f"loads.node_forces[{i}] missing 'node_id'")

            node_id = force_data["node_id"]
            if node_id not in self.nodes:
                raise ValueError(f"node_id={node_id} does not exist")

            if node_id in loads:
                raise ValueError(f"Duplicate load on node {node_id}")

            loads[node_id] = {
                "fx": float(force_data.get("Fx", 0.0)),
                "fy": float(force_data.get("Fy", 0.0))
            }

        logger.info(f"Created {len(loads)} loads")
        return loads

    def _setup_dofs(self):
        """تنظیم DOFها"""
        dof_map = {}
        dof_idx = 0
        for node_id in sorted(self.nodes.keys()):
            dof_map[node_id] = (dof_idx, dof_idx + 1)
            dof_idx += 2

        for node in self.nodes.values():
            node.set_dofs(dof_map)

        self.n_dof = dof_idx

    def _convert_units(self):
        """تبدیل واحد به SI"""
        if self.units == "SI_mm":
            for node in self.nodes.values():
                node.x /= 1000.0
                node.y /= 1000.0

    @property
    def supported_nodes(self) -> List[Node]:
        """لیست گره‌های تکیه‌گاهی"""
        return [n for n in self.nodes.values() if n.is_support]

    @property
    def fixed_dofs(self) -> List[int]:
        """لیست DOFهای قفل‌شده"""
        dofs = []
        for node in self.supported_nodes:
            dofs.extend([node.dofs[0], node.dofs[1]])
        return sorted(dofs)

    def validate_sign_convention(self) -> bool:
        """اعتبارسنجی قرارداد علامت"""
        for node_id, load in self.loads.items():
            if self.nodes[node_id].is_support:
                if load["fx"] != 0 or load["fy"] != 0:
                    return False
        return True
'''

with open("src/truss_analysis/model.py", "w", encoding="utf-8") as f:
    f.write(model_content)

print("✅ model.py restored successfully")