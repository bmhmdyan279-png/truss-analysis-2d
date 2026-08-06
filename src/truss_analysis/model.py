from __future__ import annotations

import math
from dataclasses import dataclass

from .exceptions import InputValidationError


@dataclass
class Node:
    id: str
    x: float
    y: float
    is_support: bool = False
    support_dx: bool = False
    support_dy: bool = False


@dataclass
class Element:
    id: str
    node_i: str
    node_j: str
    E: float
    A: float
    I_sec: float = 0.0
    alpha: float = 0.0
    delta_T: float = 0.0
    delta_L_free: float = 0.0


def validate_inputs(nodes, elements):
    for n in nodes if isinstance(nodes, list) else nodes.values():
        if math.isnan(n.x) or math.isnan(n.y):
            raise InputValidationError(f"Node {n.id} has NaN coordinates.")
    for e in elements if isinstance(elements, list) else elements.values():
        if e.A <= 0:
            raise InputValidationError(f"Element {e.id} has non-positive area.")
