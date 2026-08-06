from __future__ import annotations

import numpy as np

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.model import Element, Node
from truss_analysis.solver import solve


def test_golden_3bar_truss_roller_and_thermal():
    nodes = {
        "1": Node("1", 0.0, 0.0, is_support=True, support_dx=True, support_dy=True),
        "2": Node("2", 4.0, 0.0, is_support=True, support_dx=False, support_dy=True),
        "3": Node("3", 2.0, 3.0, is_support=False),
    }
    elements = {
        "e1": Element("e1", "1", "3", E=200e9, A=0.001),
        "e2": Element("e2", "2", "3", E=200e9, A=0.001),
        "e3": Element("e3", "1", "2", E=200e9, A=0.001, delta_L_free=0.005),
    }

    K, F_th, fixed = assemble_global_matrices(nodes, elements)
    F_ext = np.zeros(len(nodes) * 2)
    F_ext[5] = -10000.0

    F_total = F_ext + F_th
    U = solve(K, F_total, fixed)

    u2_x = U[2]
    assert abs(u2_x) > 1e-10, "CRITICAL: Roller support is artificially locked!"
    assert np.any(U != 0), "Solver failed to compute displacements"
