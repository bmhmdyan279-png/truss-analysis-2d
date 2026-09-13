import numpy as np
import pytest

from truss_analysis.assembly import AssemblyError, assemble_global_matrices
from truss_analysis.exceptions import SingularMatrixError
from truss_analysis.model import Element, Node
from truss_analysis.solver import solve


def test_zero_length_element():
    nodes = [Node(id="1", x=0.0, y=0.0), Node(id="2", x=0.0, y=0.0)]
    elements = [Element(id="1", node_i="1", node_j="2", E=200e9, A=0.001)]
    with pytest.raises(AssemblyError, match="zero or negative length"):
        assemble_global_matrices(nodes, elements)


def test_singular_matrix_error():
    K = np.array([[1.0, -1.0], [-1.0, 1.0]])
    F = np.array([1.0, -1.0])
    with pytest.raises(SingularMatrixError):
        solve(K, F, fixed_dofs=[])
