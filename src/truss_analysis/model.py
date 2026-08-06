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
    support_dx: bool = False  # True = Fixed, False = Roller/Free
    support_dy: bool = False


@dataclass
class Element:
    id: str
    node_i: str
    node_j: str
    E: float
    A: float
    I: float | None = None  # noqa: E741
    delta_L_free: float = 0.0  # noqa: N815


def validate_inputs(nodes: dict, elements: dict):
    for nid, n in nodes.items():
        if not all(math.isfinite(v) for v in [n.x, n.y]):
            raise InputValidationError(f"گره {nid}: مختصات نامعتبر (NaN/Inf).")
    for eid, e in elements.items():
        if not math.isfinite(e.E) or e.E <= 0:
            raise InputValidationError(
                f"عضو {eid}: مدول الاستیسیته (E) باید مثبت و محدود باشد."
            )
        if not math.isfinite(e.A) or e.A <= 0:
            raise InputValidationError(f"عضو {eid}: مساحت (A) باید مثبت و محدود باشد.")
        if e.I is not None and (not math.isfinite(e.I) or e.I < 0):
            raise InputValidationError(f"عضو {eid}: ممان اینرسی (I) نامعتبر.")
