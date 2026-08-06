from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.model import Element, Node
from truss_analysis.solver import solve


def test_golden_simple_truss():
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=3.0, y=0.0, is_support=False),
        Node(id="3", x=0.0, y=4.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [
        Element(id="1", node_i="1", node_j="2", E=200e9, A=0.001),
        Element(id="2", node_i="2", node_j="3", E=200e9, A=0.002),
        Element(id="3", node_i="1", node_j="3", E=200e9, A=0.0015),
    ]
    K, F_ext, fixed_dofs = assemble_global_matrices(nodes, elements)
    F_ext[2] += 10000.0
    U = solve(K, F_ext, fixed_dofs)
    assert U is not None
