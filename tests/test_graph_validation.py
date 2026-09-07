"""Tests for graph_validation module."""

from __future__ import annotations

import pytest
from truss_analysis.graph_validation import TopologyValidationError, validate_topology


def _minimal_valid_model() -> dict:
    return {
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "is_support": True},
            {"id": 2, "x": 1.0, "y": 0.0, "is_support": False},
        ],
        "elements": [
            {"id": 1, "node_i": 1, "node_j": 2, "A": 0.01, "E": 210e9},
        ],
        "loads": [],
    }


def test_valid_model_passes() -> None:
    validate_topology(_minimal_valid_model())


def test_orphan_node_detected() -> None:
    model = _minimal_valid_model()
    model["nodes"].append({"id": 3, "x": 2.0, "y": 0.0, "is_support": False})
    with pytest.raises(TopologyValidationError, match="Orphan"):
        validate_topology(model)


def test_zero_length_element_detected() -> None:
    model = _minimal_valid_model()
    model["nodes"][1]["x"] = 0.0  # collapse onto node 1
    with pytest.raises(TopologyValidationError, match="Zero-length"):
        validate_topology(model)


def test_disconnected_graph_detected() -> None:
    model = _minimal_valid_model()
    # Add an isolated pair of nodes + element (disconnected component)
    model["nodes"].append({"id": 3, "x": 5.0, "y": 0.0, "is_support": False})
    model["nodes"].append({"id": 4, "x": 6.0, "y": 0.0, "is_support": False})
    model["elements"].append({"id": 2, "node_i": 3, "node_j": 4, "A": 0.01, "E": 210e9})
    with pytest.raises(TopologyValidationError, match="Disconnected"):
        validate_topology(model)
