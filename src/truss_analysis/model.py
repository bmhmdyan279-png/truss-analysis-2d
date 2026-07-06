"""
مدل‌سازی خرپای دوبعدی با validation کامل
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """کلاس گره"""
    id: int
    x: float
    y: float
    is_support: bool = False
    
    def __post_init__(self):
        if not isinstance(self.id, int):
            raise TypeError(f"Node ID must be int, got {type(self.id).__name__}")
        if self.id <= 0:
            raise ValueError(f"Node ID must be positive, got {self.id}")


@dataclass
class Element:
    """کلاس عضو"""
    id: int
    node_i: Node
    node_j: Node
    E: float
    A: float
    alpha: float = 0.0
    delta_T: float = 0.0
    delta_L0: float = 0.0
    I: Optional[float] = None
    effective_length_factor: float = 1.0
    
    def __post_init__(self):
        if not isinstance(self.id, int):
            raise TypeError(f"Element ID must be int, got {type(self.id).__name__}")
        if self.id <= 0:
            raise ValueError(f"Element ID must be positive, got {self.id}")
        if self.E <= 0:
            raise ValueError(f"Element {self.id}: E must be positive, got {self.E}")
        if self.A <= 0:
            raise ValueError(f"Element {self.id}: A must be positive, got {self.A}")
        if self.node_i.id == self.node_j.id:
            raise ValueError(f"Element {self.id}: node_i and node_j cannot be the same")


class TrussModel:
    """مدل خرپا با validation کامل"""
    
    def __init__(self, input_data: Dict[str, Any]):
        """
        مقداردهی اولیه با validation کامل
        
        Args:
            input_data: دیکشنری داده‌های ورودی
            
        Raises:
            TypeError: اگر ساختار داده اشتباه باشد
            ValueError: اگر مقادیر نامعتبر باشند
        """
        if not isinstance(input_data, dict):
            raise TypeError(
                f"input_data must be a dict, got {type(input_data).__name__}"
            )
        
        self.units = input_data.get("units", "SI")
        self.temperature_change = input_data.get("temperature_change", 0.0)
        
        # validation و ایجاد اجزا
        self.nodes: Dict[int, Node] = self._create_nodes(input_data.get("nodes", []))
        self.elements: Dict[int, Element] = self._create_elements(
            input_data.get("elements", [])
        )
        self.loads: Dict[int, Dict[str, float]] = self._create_loads(
            input_data.get("loads", {})
        )
        
        # بررسی نهایی
        self._validate_model()
    
    def _create_nodes(self, nodes_data: Any) -> Dict[int, Node]:
        """ایجاد گره‌ها با validation کامل"""
        # ✅ validation نوع
        if not isinstance(nodes_data, list):
            raise TypeError(
                f"'nodes' must be a list, got {type(nodes_data).__name__}.\n"
                f"Example: [{{'id': 1, 'x': 0, 'y': 0, 'is_support': true}}]"
            )
        
        nodes = {}
        for i, node_data in enumerate(nodes_data):
            # validation هر گره
            if not isinstance(node_data, dict):
                raise TypeError(
                    f"nodes[{i}] must be a dict, got {type(node_data).__name__}"
                )
            
            # بررسی فیلدهای ضروری
            required_fields = ["id", "x", "y"]
            for field in required_fields:
                if field not in node_data:
                    raise ValueError(
                        f"nodes[{i}] missing required field: '{field}'"
                    )
            
            # بررسی نوع فیلدها
            if not isinstance(node_data["id"], int):
                raise TypeError(
                    f"nodes[{i}]['id'] must be int, got {type(node_data['id']).__name__}"
                )
            
            # بررسی تکراری نبودن ID
            node_id = node_data["id"]
            if node_id in nodes:
                raise ValueError(
                    f"Duplicate node ID: {node_id}. "
                    f"Each node must have a unique ID."
                )
            
            # ایجاد گره
            try:
                node = Node(
                    id=node_id,
                    x=float(node_data["x"]),
                    y=float(node_data["y"]),
                    is_support=bool(node_data.get("is_support", False))
                )
                nodes[node_id] = node
            except (ValueError, TypeError) as e:
                raise ValueError(f"nodes[{i}] (id={node_id}): {e}") from e
        
        logger.info(f"Created {len(nodes)} nodes")
        return nodes
    
    def _create_elements(self, elements_data: Any) -> Dict[int, Element]:
        """ایجاد اعضا با validation کامل"""
        if not isinstance(elements_data, list):
            raise TypeError(
                f"'elements' must be a list, got {type(elements_data).__name__}.\n"
                f"Example: [{{'id': 1, 'node_i': 1, 'node_j': 2, 'E': 200e9, 'A': 0.01}}]"
            )
        
        elements = {}
        for i, elem_data in enumerate(elements_data):
            if not isinstance(elem_data, dict):
                raise TypeError(
                    f"elements[{i}] must be a dict, got {type(elem_data).__name__}"
                )
            
            # فیلدهای ضروری
            required_fields = ["id", "node_i", "node_j", "E", "A"]
            for field in required_fields:
                if field not in elem_data:
                    raise ValueError(
                        f"elements[{i}] missing required field: '{field}'"
                    )
            
            elem_id = elem_data["id"]
            
            # بررسی تکراری نبودن ID
            if elem_id in elements:
                raise ValueError(f"Duplicate element ID: {elem_id}")
            
            # بررسی وجود گره‌ها
            node_i_id = elem_data["node_i"]
            node_j_id = elem_data["node_j"]
            
            if node_i_id not in self.nodes:
                raise ValueError(
                    f"elements[{i}] (id={elem_id}): node_i={node_i_id} does not exist"
                )
            if node_j_id not in self.nodes:
                raise ValueError(
                    f"elements[{i}] (id={elem_id}): node_j={node_j_id} does not exist"
                )
            
            # بررسی مقادیر فیزیکی
            E = float(elem_data["E"])
            A = float(elem_data["A"])
            
            if E <= 0:
                raise ValueError(
                    f"elements[{i}] (id={elem_id}): E must be positive, got {E}"
                )
            if A <= 0:
                raise ValueError(
                    f"elements[{i}] (id={elem_id}): A must be positive, got {A}"
                )
            
            # ایجاد عضو
            try:
                element = Element(
                    id=elem_id,
                    node_i=self.nodes[node_i_id],
                    node_j=self.nodes[node_j_id],
                    E=E,
                    A=A,
                    alpha=float(elem_data.get("alpha", 0.0)),
                    delta_T=float(elem_data.get("delta_T", 0.0)),
                    delta_L0=float(elem_data.get("delta_L0", 0.0)),
                    I=float(elem_data["I"]) if "I" in elem_data else None,
                    effective_length_factor=float(
                        elem_data.get("effective_length_factor", 1.0)
                    )
                )
                elements[elem_id] = element
            except (ValueError, TypeError) as e:
                raise ValueError(f"elements[{i}] (id={elem_id}): {e}") from e
        
        logger.info(f"Created {len(elements)} elements")
        return elements
    
    def _create_loads(self, loads_data: Any) -> Dict[int, Dict[str, float]]:
        """ایجاد بارها با validation کامل"""
        if not isinstance(loads_data, dict):
            raise TypeError(
                f"'loads' must be a dict, got {type(loads_data).__name__}.\n"
                f"Example: {{'node_forces': [{{'node_id': 1, 'Fx': 1000, 'Fy': 0}}]}}"
            )
        
        loads = {}
        
        # بارهای نقطه‌ای
        node_forces = loads_data.get("node_forces", [])
        if not isinstance(node_forces, list):
            raise TypeError(
                f"'loads.node_forces' must be a list, got {type(node_forces).__name__}"
            )
        
        for i, force_data in enumerate(node_forces):
            if not isinstance(force_data, dict):
                raise TypeError(
                    f"loads.node_forces[{i}] must be a dict, got {type(force_data).__name__}"
                )
            
            # فیلدهای ضروری
            if "node_id" not in force_data:
                raise ValueError(f"loads.node_forces[{i}] missing 'node_id'")
            
            node_id = force_data["node_id"]
            
            # بررسی وجود گره
            if node_id not in self.nodes:
                raise ValueError(
                    f"loads.node_forces[{i}]: node_id={node_id} does not exist"
                )
            
            # بررسی تکراری نبودن
            if node_id in loads:
                raise ValueError(
                    f"Duplicate load on node {node_id}. "
                    f"Combine forces into a single entry."
                )
            
            # استخراج مقادیر
            fx = float(force_data.get("Fx", 0.0))
            fy = float(force_data.get("Fy", 0.0))
            
            loads[node_id] = {"fx": fx, "fy": fy}
        
        logger.info(f"Created {len(loads)} loads")
        return loads
    
    def _validate_model(self):
        """بررسی نهایی مدل"""
        # بررسی وجود حداقل یک گره
        if len(self.nodes) == 0:
            raise ValueError("Model must have at least one node")
        
        # بررسی وجود حداقل یک عضو
        if len(self.elements) == 0:
            raise ValueError("Model must have at least one element")
        
        # بررسی وجود حداقل یک تکیهگاه
        supported_nodes = [n for n in self.nodes.values() if n.is_support]
        if len(supported_nodes) == 0:
            raise ValueError(
                "Model must have at least one support. "
                "Set 'is_support': true for at least one node."
            )
        
        # بررسی وجود حداقل یک بار (اگر اثر حرارتی/خطای ساخت نداریم)
        has_thermal = any(
            e.alpha != 0 and (e.delta_T != 0 or self.temperature_change != 0)
            for e in self.elements.values()
        )
        has_fabrication = any(e.delta_L0 != 0 for e in self.elements.values())
        
        if len(self.loads) == 0 and not has_thermal and not has_fabrication:
            logger.warning(
                "Model has no loads, thermal effects, or fabrication errors. "
                "All displacements and forces will be zero."
            )
        
        logger.info("Model validation passed")
    
    @property
    def supported_nodes(self) -> List[Node]:
        """لیست گره‌های تکیه‌گاهی"""
        return [n for n in self.nodes.values() if n.is_support]
    
    @property
    def fixed_dofs(self) -> List[int]:
        """لیست DOFهای قفل‌شده"""
        dofs = []
        for node in self.supported_nodes:
            dofs.extend([2 * (node.id - 1), 2 * (node.id - 1) + 1])
        return sorted(dofs)
