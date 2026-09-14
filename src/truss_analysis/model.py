"""Pure data-transfer objects for the truss model (nodes and elements).

Support semantics
-----------------
A degree of freedom is constrained if and only if the node is flagged
``is_support`` **and** the matching ``support_dx`` / ``support_dy`` restraint
is set. The flag alone is not sufficient, and a restraint without the flag is
not honoured. Because that combination is easy to write by accident and
silently produces a *different structure* from the one described, the pairing
is enforced as an invariant in :func:`validate_inputs`.

The mapping from nodes to constrained DOF indices lives in exactly one place
(:func:`fixed_dof_indices`) and is consumed by both the assembler and the
criticality engine. They previously each carried a private copy, so a change
in support semantics could put the two solvers in disagreement.
"""

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

    def __post_init__(self) -> None:
        """Validate node identity and coordinates after initialisation."""
        if not isinstance(self.id, str):  # pragma: no branch
            raise InputValidationError(f"Node ID must be string, got {type(self.id)}")
        if not all(  # pragma: no branch
            isinstance(v, (int, float)) for v in [self.x, self.y]
        ):
            raise InputValidationError("Node coordinates must be numeric")

    @property
    def has_restraint(self) -> bool:
        """Whether at least one translational restraint is declared."""
        return bool(self.support_dx or self.support_dy)

    @property
    def is_consistent_support(self) -> bool:
        """Whether ``is_support`` agrees with the declared restraints.

        ``False`` means the node carries a contradictory description: either
        flagged as a support with no restraint at all, or carrying restraints
        without the flag. :func:`validate_inputs` rejects both.
        """
        return bool(self.is_support) == self.has_restraint


def fixed_dof_indices(nodes: list[Node]) -> list[int]:
    """Return the sorted global DOF indices that are constrained.

    DOF numbering is ``2*i`` for the x translation and ``2*i + 1`` for the y
    translation of node ``i``. A DOF is fixed when its node is flagged
    ``is_support`` and the matching restraint is set.

    Parameters
    ----------
    nodes : list[Node]
        Model nodes in the order that defines the DOF map.

    Returns
    -------
    list[int]
        Ascending indices of constrained degrees of freedom.
    """
    fixed: list[int] = []
    for i, node in enumerate(nodes):
        if not node.is_support:
            continue
        if node.support_dx:
            fixed.append(2 * i)
        if node.support_dy:
            fixed.append(2 * i + 1)
    return fixed


@dataclass
class Element:
    """A truss element (bar) connecting two nodes.

    Only ``E`` and ``A`` enter the stiffness matrix: a pin-jointed truss
    member carries axial force alone, so the solver uses ``k = E A / L``.
    ``I_sec`` and ``effective_length_factor`` are *stability* properties
    consumed by the buckling checks
    (:func:`truss_analysis.postprocess.calculate_buckling`,
    :mod:`truss_analysis.limitstates`) and never affect the displacement
    solution. ``density`` and ``alpha``/``delta_T``/``delta_L_free`` likewise
    feed loading rather than stiffness.
    """

    id: str
    node_i: str
    node_j: str
    E: float  # Young's modulus
    A: float  # Cross-sectional area
    I_sec: float = 0.0  # Second moment of area (stability checks only)
    alpha: float = 0.0  # Thermal expansion coefficient
    delta_T: float = 0.0  # Temperature change
    delta_L_free: float = 0.0  # Free length change (fabrication error)
    density: float = 0.0  # Material density (for self-weight)
    effective_length_factor: float = 1.0  # Buckling effective length factor K

    def __post_init__(self) -> None:
        """Validate element identity, material and geometry after initialisation."""
        if not isinstance(self.id, str):  # pragma: no branch
            raise InputValidationError(
                f"Element ID must be string, got {type(self.id)}"
            )
        if self.E <= 0:  # pragma: no branch
            raise InputValidationError(
                f"Element {self.id}: E must be positive, got {self.E}"
            )
        if self.A <= 0:  # pragma: no branch
            raise InputValidationError(
                f"Element {self.id}: A must be positive, got {self.A}"
            )
        if self.node_i == self.node_j:  # pragma: no branch
            raise InputValidationError(
                f"Element {self.id}: node_i and node_j cannot be the same"
            )
        if self.I_sec < 0:  # pragma: no branch
            raise InputValidationError(
                f"Element {self.id}: I_sec must be non-negative, got {self.I_sec}"
            )
        if self.density < 0:  # pragma: no branch
            raise InputValidationError(
                f"Element {self.id}: density must be non-negative, got {self.density}"
            )
        if self.effective_length_factor <= 0:  # pragma: no branch
            raise InputValidationError(
                f"Element {self.id}: effective_length_factor must be positive, "
                f"got {self.effective_length_factor}"
            )


def validate_inputs(nodes: list[Node], elements: list[Element]) -> None:
    """Validate input data for consistency and correctness.

    This is an *input consistency* check, not a proof of stability. Kinematic
    stability is decided numerically from the rank of ``K_ff`` in
    :func:`truss_analysis.solver.solve` and reported by
    :func:`truss_analysis.graph_validation.validate_topology`; static
    determinacy (``m + r - 2j``) is an algebraic classification only. The
    three notions are deliberately kept apart, since counting scalar
    constraints can neither prove nor disprove stability: three parallel
    rollers in a line satisfy the count and are still a mechanism.

    Raises
    ------
    InputValidationError
        On duplicate IDs, dangling node references, contradictory support
        declarations, or fewer than three restraints in total.
    """
    # Check unique node IDs
    node_ids = {n.id for n in nodes}
    if len(node_ids) != len(nodes):  # pragma: no branch
        raise InputValidationError("Duplicate node IDs found")

    # Check unique element IDs
    elem_ids = {e.id for e in elements}
    if len(elem_ids) != len(elements):  # pragma: no branch
        raise InputValidationError("Duplicate element IDs found")

    # Check that all element node references exist
    for elem in elements:
        if elem.node_i not in node_ids:  # pragma: no branch
            raise InputValidationError(
                f"Element {elem.id} references non-existent node {elem.node_i}"
            )
        if elem.node_j not in node_ids:  # pragma: no branch
            raise InputValidationError(
                f"Element {elem.id} references non-existent node {elem.node_j}"
            )

    # Support declarations must be self-consistent. A restraint without the
    # is_support flag is silently ignored by the DOF map, which would yield a
    # different structure from the one the user described; a flag without any
    # restraint describes a support that restrains nothing.
    for node in nodes:
        if node.is_consistent_support:
            continue
        if node.is_support:  # pragma: no branch
            raise InputValidationError(
                f"Node {node.id}: is_support=True but neither support_dx nor "
                f"support_dy is set; the node restrains nothing"
            )
        raise InputValidationError(  # pragma: no cover
            f"Node {node.id}: support_dx/support_dy set without is_support=True; "
            f"the restraint would be silently ignored"
        )

    # Minimum-restraint sanity check for 2D statics. Necessary, not
    # sufficient: see the docstring.
    total_constraints = sum(
        (n.support_dx + n.support_dy) for n in nodes if n.is_support
    )
    if total_constraints < 3:  # pragma: no branch
        raise InputValidationError(
            f"Insufficient constraints for stability: {total_constraints} < 3"
        )
