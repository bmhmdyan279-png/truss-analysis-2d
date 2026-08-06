from __future__ import annotations

import numpy as np

from truss_analysis.model import Element, Node
from truss_analysis.postprocess import (
    calculate_displacement_scale_factor,
    calculate_element_forces,
    calculate_percentages,
)


def test_calculate_element_forces():
    nodes = [Node("1", 0.0, 0.0), Node("2", 1.0, 0.0)]
    elements = [Element("e1", "1", "2", E=200e9, A=0.01)]
    u = np.array([0.0, 0.0, 0.001, 0.0])
    res, energy = calculate_element_forces(nodes, elements, u)
    assert len(res) == 1
    assert res[0]["element"] == "e1"
    assert abs(res[0]["force"] - 2e6) < 1.0


def test_calculate_percentages_zero_energy():
    res = [{"energy": 0.0}, {"energy": 0.0}]
    res = calculate_percentages(res)
    assert res[0]["pct_U"] == 0.0


def test_calculate_percentages_normal():
    res = [{"energy": 25.0}, {"energy": 75.0}]
    res = calculate_percentages(res)
    assert abs(res[0]["pct_U"] - 25.0) < 1e-6
    assert abs(res[1]["pct_U"] - 75.0) < 1e-6


def test_scale_factor_limits():
    nodes = [Node("1", 0.0, 0.0), Node("2", 10.0, 0.0)]
    u = np.array([0.0, 0.0, 1e-9, 0.0])
    scale = calculate_displacement_scale_factor(nodes, u)
    assert scale == 1000.0
