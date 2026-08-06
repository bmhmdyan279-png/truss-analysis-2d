from dataclasses import dataclass


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
    I: float = 0.0  # noqa: E741
    alpha: float = 0.0
    delta_T: float = 0.0
