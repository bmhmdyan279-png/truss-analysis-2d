from __future__ import annotations

import math

import pytest

from truss_analysis.exceptions import InputValidationError
from truss_analysis.model import Element, Node, validate_inputs


def test_node_creation():
    n = Node("1", 0.0, 0.0)
    assert n.id == "1"
    assert not n.is_support


def test_element_creation():
    e = Element("e1", "1", "2", E=200e9, A=0.01)
    assert e.E == 200e9


def test_validate_inputs_rejects_negative_area():
    # Element __post_init__ raises the error, so Element() must be inside the with block
    with pytest.raises(InputValidationError, match="A must be positive"):
        Element("e1", "1", "2", E=200e9, A=-0.01)


def test_validate_inputs_rejects_nan_coords():
    nodes = [Node("1", math.nan, 0), Node("2", 1, 0)]
    elements = [Element("e1", "1", "2", E=200e9, A=0.01)]
    with pytest.raises(InputValidationError):
        validate_inputs(nodes, elements)


def test_node_id_must_be_string():
    """Test that non-string node ID raises InputValidationError."""
    with pytest.raises(InputValidationError, match="Node ID must be string"):
        Node(123, 0.0, 0.0)  # type: ignore


def test_node_coordinates_must_be_numeric():
    """Test that non-numeric node coordinates raise InputValidationError."""
    with pytest.raises(InputValidationError, match="Node coordinates must be numeric"):
        Node("1", "not_a_number", 0.0)  # type: ignore


def test_element_id_must_be_string():
    """Test that non-string element ID raises InputValidationError."""
    with pytest.raises(InputValidationError, match="Element ID must be string"):
        Element(123, "1", "2", E=200e9, A=0.01)  # type: ignore


def test_element_e_must_be_positive():
    """Test that non-positive Young's modulus raises InputValidationError."""
    with pytest.raises(InputValidationError, match="E must be positive"):
        Element("e1", "1", "2", E=0, A=0.01)
    
    with pytest.raises(InputValidationError, match="E must be positive"):
        Element("e2", "1", "2", E=-100, A=0.01)


def test_element_nodes_cannot_be_same():
    """Test that element with same start and end node raises InputValidationError."""
    with pytest.raises(InputValidationError, match="node_i and node_j cannot be the same"):
        Element("e1", "1", "1", E=200e9, A=0.01)


def test_validate_duplicate_node_ids():
    """Test that duplicate node IDs are rejected."""
    nodes = [Node("1", 0.0, 0.0), Node("1", 1.0, 0.0)]
    elements = [Element("e1", "1", "2", E=200e9, A=0.01)]
    with pytest.raises(InputValidationError, match="Duplicate node IDs"):
        validate_inputs(nodes, elements)


def test_validate_duplicate_element_ids():
    """Test that duplicate element IDs are rejected."""
    nodes = [Node("1", 0.0, 0.0), Node("2", 1.0, 0.0)]
    elements = [
        Element("e1", "1", "2", E=200e9, A=0.01),
        Element("e1", "2", "1", E=200e9, A=0.01)
    ]
    with pytest.raises(InputValidationError, match="Duplicate element IDs"):
        validate_inputs(nodes, elements)


def test_validate_element_references_nonexistent_node_i():
    """Test that element referencing non-existent node_i is rejected."""
    nodes = [Node("1", 0.0, 0.0)]
    elements = [Element("e1", "1", "nonexistent", E=200e9, A=0.01)]
    with pytest.raises(InputValidationError, match="references non-existent node"):
        validate_inputs(nodes, elements)


def test_validate_element_references_nonexistent_node_j():
    """Test that element referencing non-existent node_j is rejected."""
    nodes = [Node("1", 0.0, 0.0)]
    elements = [Element("e1", "nonexistent", "1", E=200e9, A=0.01)]
    with pytest.raises(InputValidationError, match="references non-existent node"):
        validate_inputs(nodes, elements)


def test_validate_insufficient_constraints():
    """Test that insufficient constraints for stability are rejected."""
    # Only 2 constraints (less than required 3)
    nodes = [Node("1", 0.0, 0.0, is_support=True, support_dx=True, support_dy=False)]
    elements = [Element("e1", "1", "2", E=200e9, A=0.01), Element("e2", "2", "3", E=200e9, A=0.01)]
    nodes.append(Node("2", 1.0, 0.0))
    nodes.append(Node("3", 2.0, 0.0))
    with pytest.raises(InputValidationError, match="Insufficient constraints"):
        validate_inputs(nodes, elements)

