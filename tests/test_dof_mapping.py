from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.model import Node


def test_dof_mapping():
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=1.0, y=0.0, is_support=False),
    ]
    K, F, fixed = assemble_global_matrices(nodes, [])
    assert 0 in fixed and 1 in fixed
    assert 2 not in fixed and 3 not in fixed
