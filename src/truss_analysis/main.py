from __future__ import annotations

import logging
import sys

import numpy as np

from .assembly import assemble_global_matrices
from .fileio import load_json
from .model import Element, Node, validate_inputs
from .postprocess import (
    calculate_displacement_scale_factor,
    calculate_element_forces,
    calculate_percentages,
)
from .solver import solve
from .units import UnitSystem, to_si

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_analysis(filepath: str) -> None:
    logger.info(f"Loading model from {filepath}...")
    data = load_json(filepath)

    unit_sys = UnitSystem(data.get("units", "SI"))

    nodes = {}
    for nid, ndata in data.get("nodes", {}).items():
        nodes[nid] = Node(
            id=nid,
            x=to_si(ndata["x"], unit_sys, "L"),
            y=to_si(ndata["y"], unit_sys, "L"),
            is_support=ndata.get("is_support", False),
            support_dx=ndata.get("support_dx", False),
            support_dy=ndata.get("support_dy", False),
        )

    elements = {}
    for eid, edata in data.get("elements", {}).items():
        elements[eid] = Element(
            id=eid,
            node_i=edata["node_i"],
            node_j=edata["node_j"],
            E=to_si(edata["E"], unit_sys, "E"),
            A=to_si(edata["A"], unit_sys, "L2"),
            I=to_si(edata.get("I"), unit_sys, "L4"),  # noqa: E741
            delta_L_free=to_si(edata.get("delta_L_free"), unit_sys, "L"),
        )

    validate_inputs(nodes, elements)

    k_global, f_th, fixed_dofs = assemble_global_matrices(nodes, elements)

    f_ext = np.zeros(len(nodes) * 2)
    node_list = list(nodes.keys())
    for load in data.get("loads", []):
        nid = load["node_id"]
        idx = node_list.index(nid)
        if "Fx" in load:
            f_ext[idx * 2] += to_si(load["Fx"], unit_sys, "F")
        if "Fy" in load:
            f_ext[idx * 2 + 1] += to_si(load["Fy"], unit_sys, "F")

    f_total = f_ext + f_th

    logger.info("Solving displacements...")
    u_vec = solve(k_global, f_total, fixed_dofs)

    dof_map = {nid: (i * 2, i * 2 + 1) for i, nid in enumerate(node_list)}
    results = calculate_element_forces(nodes, elements, u_vec, dof_map)
    results = calculate_percentages(results)
    scale = calculate_displacement_scale_factor(nodes, u_vec)

    logger.info(f"Analysis complete. Max displacement scale factor: {scale:.1f}")
    for r in results:
        logger.info(
            f"Element {r['element_id']}: "
            f"Force={r['force']:.2f} N, Energy={r['pct_U']:.1f}%"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m truss_analysis.main <input.json>")
        sys.exit(1)
    run_analysis(sys.argv[1])
