from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from .assembly import assemble_global_matrices
from .fileio import load_json
from .model import Element, Node, validate_inputs
from .postprocess import calculate_element_forces
from .solver import check_energy, solve
from .units import to_si


@dataclass
class AnalysisResult:
    displacements: dict
    forces: list
    strain_energy: float
    prestress_work: float


def run(filepath: str, unit_sys: str = "SI") -> AnalysisResult:
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
            I_sec=to_si(
                e.get("I_sec", e.get("I", 0.0)), unit_sys, "I_sec"
            ),
            alpha=to_si(e.get("alpha", 0.0), unit_sys, "alpha"),
            delta_T=to_si(e.get("delta_T", 0.0), unit_sys, "delta_T"),
            delta_L_free=to_si(
                e.get("delta_L_free", 0.0), unit_sys, "L"
            ),
        )
        for e in data["elements"]
    ]
    validate_inputs(nodes, elements)

    K, F_ext, F_mechanical, fixed_dofs = assemble_global_matrices(
        nodes, elements
    )

    loads = data.get("loads", [])
    if isinstance(loads, dict):
        loads = loads.get("node_forces", [])

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

    displacements = {
        node.id: (U[i * 2], U[i * 2 + 1]) for i, node in enumerate(nodes)
    }

    return AnalysisResult(
        displacements=displacements,
        forces=results,
        strain_energy=strain_energy,
        prestress_work=prestress_work,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="2D Truss Analysis Tool")
    parser.add_argument("filepath", help="Path to the JSON input file")
    parser.add_argument(
        "unit_sys",
        nargs="?",
        default="SI",
        help="Unit system (SI or Imperial)",
    )

    args = parser.parse_args()

    try:
        result = run(args.filepath, args.unit_sys)
        print("✅ تحلیل با موفقیت انجام شد.")
        print(f"انرژی کرنشی: {result.strain_energy:.4f} J")
        print(f"کار پیش‌تنیدگی: {result.prestress_work:.4f} J")
        print("نیروهای اعضا:")
        for r in result.forces:
            print(f"  - المان {r['element']}: {r['force']:.2f} N")
        return 0
    except Exception as e:
        print(f"❌ خطا در تحلیل: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
