from __future__ import annotations

import numpy as np

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.model import Element, Node


def test_assemble_simple_truss():
    nodes = [
        Node("1", 0.0, 0.0, is_support=True, support_dx=True, support_dy=True),
        Node("2", 1.0, 0.0, is_support=True, support_dx=False, support_dy=True),
        Node("3", 0.5, 1.0, is_support=False),
    ]
    elements = [
        Element("e1", "1", "3", E=200e9, A=0.01),
        Element("e2", "2", "3", E=200e9, A=0.01),
    ]
    k, f_ext, f_mech, fixed = assemble_global_matrices(nodes, elements)
    assert k.shape == (6, 6)
    assert sorted(fixed) == [0, 1, 3]
    assert np.allclose(k, k.T)
    assert np.allclose(f_mech, 0)


def test_thermal_force_sign_convention():
    nodes = [
        Node("1", 0.0, 0.0, is_support=True, support_dx=True, support_dy=True),
        Node("2", 1.0, 0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [Element("e1", "1", "2", E=200e9, A=0.01, delta_L_free=0.001)]
    k, f_ext, f_mech, fixed = assemble_global_matrices(nodes, elements)
    assert np.any(f_ext != 0)
    assert np.allclose(f_mech, 0)
