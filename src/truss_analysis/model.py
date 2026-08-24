"""Pure DTOs for Truss Analysis."""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import InputValidationError


@dataclass
class Node:
    """A node in the truss structure."""

    id: str
    x: float
    y: float
    is_support: bool = False
    support_dx: bool = False
    support_dy: bool = False

    def __post_init__(self):
        if not isinstance(self.id, str):
            raise InputValidationError(f"Node ID must be string, got {type(self.id)}")
        if not all(isinstance(v, (int, float)) for v in [self.x, self.y]):
            raise InputValidationError("Node coordinates must be numeric")


@dataclass
class Element:
    """A truss element (bar) connecting two nodes."""

    id: str
    node_i: str
    node_j: str
    E: float  # Young's modulus
    A: float  # Cross-sectional area
    I_sec: float = 0.0  # Second moment of area
    alpha: float = 0.0  # Thermal expansion coefficient
    delta_T: float = 0.0  # Temperature change
    delta_L_free: float = 0.0  # Free length change (fabrication error)
    density: float = 0.0  # Material density (for self-weight)
    effective_length_factor: float = 1.0  # For buckling calculation

    def __post_init__(self):
        if not isinstance(self.id, str):
            raise InputValidationError(
                f"Element ID must be string, got {type(self.id)}"
            )
        if self.E <= 0:
            raise InputValidationError(
                f"Element {self.id}: E must be positive, got {self.E}"
            )
        if self.A <= 0:
            raise InputValidationError(
                f"Element {self.id}: A must be positive, got {self.A}"
            )
        if self.node_i == self.node_j:
            raise InputValidationError(
                f"Element {self.id}: node_i and node_j cannot be the same"
            )


def validate_inputs(nodes: list[Node], elements: list[Element]) -> None:
    """Validate input data for consistency and correctness."""

    # Check unique node IDs
    node_ids = {n.id for n in nodes}
    if len(node_ids) != len(nodes):
        raise InputValidationError("Duplicate node IDs found")

    # Check unique element IDs
    elem_ids = {e.id for e in elements}
    if len(elem_ids) != len(elements):
        raise InputValidationError("Duplicate element IDs found")

    # Check that all element node references exist
    for elem in elements:
        if elem.node_i not in node_ids:
            raise InputValidationError(
                f"Element {elem.id} references non-existent node {elem.node_i}"
            )
        if elem.node_j not in node_ids:
            raise InputValidationError(
                f"Element {elem.id} references non-existent node {elem.node_j}"
            )

    # Check kinematic stability (at least 3 constraints total)
    total_constraints = sum(
        (n.support_dx + n.support_dy) for n in nodes if n.is_support
    )
    if total_constraints < 3:
        raise InputValidationError(
            f"Insufficient constraints for stability: {total_constraints} < 3"
        )
