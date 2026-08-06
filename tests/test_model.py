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
    nodes = {"1": Node("1", 0, 0), "2": Node("2", 1, 0)}
    elements = {"e1": Element("e1", "1", "2", E=200e9, A=-0.01)}
    with pytest.raises(InputValidationError):
        validate_inputs(nodes, elements)


def test_validate_inputs_rejects_nan_coords():
    nodes = {"1": Node("1", math.nan, 0), "2": Node("2", 1, 0)}
    elements = {"e1": Element("e1", "1", "2", E=200e9, A=0.01)}
    with pytest.raises(InputValidationError):
        validate_inputs(nodes, elements)
