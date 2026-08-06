from __future__ import annotations

import sys

from .assembly import assemble_global_matrices
from .fileio import load_json
from .model import Element, Node, validate_inputs
from .postprocess import calculate_element_forces
from .solver import check_energy, solve
from .units import to_si


def run(filepath, unit_sys="SI"):
    data = load_json(filepath)
    nodes = [
        Node(
            id=str(n["id"]),
            x=to_si(n["x"], unit_sys, "L"),
            y=to_si(n["y"], unit_sys, "L"),
            is_support=n.get("is_support", False),
            support_dx=n.get("support_dx", False),
            support_dy=n.get("support_dy", False),
        )
        for n in data["nodes"]
    ]
    elements = [
        Element(
            id=str(e["id"]),
            node_i=str(e["node_i"]),
            node_j=str(e["node_j"]),
            E=to_si(e["E"], unit_sys, "E"),
            A=to_si(e["A"], unit_sys, "A"),
            I_sec=to_si(e.get("I_sec", e.get("I", 0.0)), unit_sys, "I_sec"),
            alpha=to_si(e.get("alpha", 0.0), unit_sys, "alpha"),
            delta_T=to_si(e.get("delta_T", 0.0), unit_sys, "delta_T"),
            delta_L_free=to_si(e.get("delta_L_free", 0.0), unit_sys, "L"),
        )
        for e in data["elements"]
    ]
    validate_inputs(nodes, elements)

    K, F_ext, F_mechanical, fixed_dofs = assemble_global_matrices(nodes, elements)

    loads = data.get("loads", [])
    node_map = {node.id: i for i, node in enumerate(nodes)}
    for lf in loads:
        nid = str(lf.get("node_id", lf.get("id")))
        if nid in node_map:
            idx = node_map[nid]
            Fx = to_si(lf.get("Fx", 0.0), unit_sys, "F")
            Fy = to_si(lf.get("Fy", 0.0), unit_sys, "F")
            F_ext[idx * 2] += Fx
            F_ext[idx * 2 + 1] += Fy
            F_mechanical[idx * 2] += Fx
            F_mechanical[idx * 2 + 1] += Fy

    U = solve(K, F_ext, fixed_dofs)

    results, strain_energy, prestress_work = calculate_element_forces(
        nodes, elements, U
    )

    check_energy(U, F_mechanical, strain_energy, prestress_work)

    print("Analysis successful. Energy balanced (with thermal effects).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "SI")
