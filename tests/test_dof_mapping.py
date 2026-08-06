from __future__ import annotations

import numpy as np
from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.model import Element, Node


def test_dof_mapping_mixed_boundary_conditions():
    nodes = {
        "1": Node("1", 0.0, 0.0, is_support=True, support_dx=True, support_dy=True),
        "2": Node("2", 1.0, 0.0, is_support=False),
        "3": Node("3", 2.0, 0.0, is_support=True, support_dx=True, support_dy=True),
        "4": Node("4", 1.0, 1.0, is_support=False),
    }
    elements = {
        "e1": Element("e1", "1", "2", E=200e9, A=0.01),
        "e2": Element("e2", "2", "3", E=200e9, A=0.01),
        "e3": Element("e3", "2", "4", E=200e9, A=0.01),
    }

    k_glob, f_th, fixed_dofs = assemble_global_matrices(nodes, elements)

    assert sorted(fixed_dofs) == [0, 1, 4, 5]

    all_dofs = set(range(8))
    free_dofs = sorted(list(all_dofs - set(fixed_dofs)))
    assert free_dofs == [2, 3, 6, 7]

    assert k_glob.shape == (8, 8)
    assert np.allclose(k_glob, k_glob.T)
