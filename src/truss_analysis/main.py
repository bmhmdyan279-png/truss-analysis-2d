"""CLI entry point and high-level orchestration."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .assembly import assemble_global_matrices
from .fileio import load_json
from .model import Element, Node, validate_inputs
from .postprocess import (
    calculate_buckling,
    calculate_element_forces,
    calculate_reactions,
    check_equilibrium,
)
from .solver import check_energy, solve
from .units import to_si

G = 9.80665


def _pure(obj):
    """Convert numpy scalars to plain Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _pure(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pure(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


@dataclass
class AnalysisResult:
    """Structured container for all analysis outputs."""

    status: str
    displacements: dict
    element_forces: list
    reactions: dict
    equilibrium: dict
    buckling: list = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=" * 62, " TRUSS ANALYSIS RESULT", "=" * 62]
        for r in self.element_forces:
            nid = r.get("id")
            force = r.get("N", 0.0)
            status = r.get("status", "?")
            lines.append(f"Element {nid}: N = {force:14.3f}  [{status}]")
        for nid, rec in self.reactions.items():
            fx = rec["Fx"]
            fy = rec["Fy"]
            lines.append(f"Reaction @ {nid}: Fx = {fx:12.3f}  Fy = {fy:12.3f}")
        eq = self.equilibrium
        ok = "OK" if eq["is_valid"] else "FAIL"
        lines.append(
            f"Equilibrium: dFx={eq['sum_fx']:.2e} "
            f"dFy={eq['sum_fy']:.2e} dM={eq['sum_m']:.2e} -> {ok}"
        )
        for b in self.buckling:
            if b["N"] < 0 and b["P_cr"]:
                flag = "OK" if b["safe"] else "BUCKLING RISK"
                lines.append(f"Buckling {b['id']}: ratio={b['ratio']:.3f} -> {flag}")
        lines.append(f"Status: {self.status}")
        return "\n".join(lines)


def _write_markdown(result: AnalysisResult, path: str) -> None:
    """Write Markdown report."""
    lines = [
        "# Truss Analysis Report",
        "",
        "## Axial Forces",
        "",
        "| Element | N (force) | Status |",
        "|---|---|---|",
    ]
    for r in result.element_forces:
        eid = r.get("id")
        force = r.get("N", 0.0)
        status = r.get("status", "-")
        lines.append(f"| {eid} | {force:.3f} | {status} |")
    lines += [
        "",
        "## Reactions",
        "",
        "| Node | Fx | Fy |",
        "|---|---|---|",
    ]
    for nid, rec in result.reactions.items():
        lines.append(f"| {nid} | {rec['Fx']:.3f} | {rec['Fy']:.3f} |")
    eq = result.equilibrium
    lines += [
        "",
        "## Static Equilibrium",
        "",
        f"- sum(Fx) = {eq['sum_fx']:.3e}",
        f"- sum(Fy) = {eq['sum_fy']:.3e}",
        f"- sum(M) = {eq['sum_m']:.3e}",
        f"- Valid: {eq['is_valid']}",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run(
    filepath,
    unit_sys="SI",
    plot=False,
    check_buckling=False,
    output=None,
    csv_path=None,
    report_path=None,
    plot_path=None,
):
    """Run the full analysis pipeline and return an AnalysisResult."""
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
    node_map = {node.id: i for i, node in enumerate(nodes)}
    applied_loads = []

    for lf in data.get("loads", []):
        nid = str(lf.get("node_id", lf.get("id")))
        if nid not in node_map:
            continue
        idx = node_map[nid]
        fx = to_si(lf.get("Fx", 0.0), unit_sys, "F")
        fy = to_si(lf.get("Fy", 0.0), unit_sys, "F")
        F_ext[2 * idx] += fx
        F_ext[2 * idx + 1] += fy
        F_mechanical[2 * idx] += fx
        F_mechanical[2 * idx + 1] += fy
        applied_loads.append({"node_id": nid, "Fx": fx, "Fy": fy})

    # Optional self-weight via per-element density rho [kg/m^3]
    raw_elems = {str(e["id"]): e for e in data["elements"]}
    weight_per_node = {}
    for elem in elements:
        rho = float(raw_elems.get(elem.id, {}).get("rho", 0.0) or 0.0)
        if rho <= 0.0:
            continue
        i, j = node_map[elem.node_i], node_map[elem.node_j]
        L = float(np.hypot(nodes[j].x - nodes[i].x, nodes[j].y - nodes[i].y))
        W = rho * elem.A * L * G
        weight_per_node[i] = weight_per_node.get(i, 0.0) + W / 2.0
        weight_per_node[j] = weight_per_node.get(j, 0.0) + W / 2.0
    for idx, W in weight_per_node.items():
        F_ext[2 * idx + 1] -= W
        F_mechanical[2 * idx + 1] -= W
        applied_loads.append({"node_id": nodes[idx].id, "Fx": 0.0, "Fy": -W})

    U = solve(K, F_ext, fixed_dofs)
    element_forces, strain_energy, prestress_work = calculate_element_forces(
        nodes, elements, U
    )
    check_energy(U, F_mechanical, strain_energy, prestress_work)
    reactions = calculate_reactions(nodes, K, U, F_ext, fixed_dofs)
    equilibrium = check_equilibrium(nodes, reactions, applied_loads)
    buckling = (
        calculate_buckling(nodes, elements, element_forces) if check_buckling else []
    )
    displacements = {
        node.id: {"ux": float(U[2 * i]), "uy": float(U[2 * i + 1])}
        for i, node in enumerate(nodes)
    }

    result = AnalysisResult(
        status="SUCCESS",
        displacements=_pure(displacements),
        element_forces=_pure(element_forces),
        reactions=_pure(reactions),
        equilibrium=_pure(equilibrium),
        buckling=_pure(buckling),
    )

    if output:
        Path(output).write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["element_id", "axial_force", "status"])
            for r in result.element_forces:
                w.writerow([r.get("id"), r.get("N"), r.get("status")])
    if report_path:
        _write_markdown(result, report_path)
    if plot or plot_path:
        from .visualization import plot_truss

        plot_truss(nodes, elements, U=U, results=element_forces, save_path=plot_path)
    print(result.summary())
    return result


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="truss-analysis", description="2D Truss Analysis Tool"
    )
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("--units", default="SI", choices=["SI", "Imperial"])
    parser.add_argument("-o", "--output", help="Write JSON result")
    parser.add_argument("--csv", dest="csv_path", help="Write CSV")
    parser.add_argument("--report", dest="report_path", help="Write MD")
    parser.add_argument("--plot", action="store_true", help="Show plot")
    parser.add_argument("--plot-path", help="Save plot PNG")
    parser.add_argument("--check-buckling", action="store_true")
    args = parser.parse_args(argv)
    run(
        args.input,
        unit_sys=args.units,
        plot=args.plot or bool(args.plot_path),
        check_buckling=args.check_buckling,
        output=args.output,
        csv_path=args.csv_path,
        report_path=args.report_path,
        plot_path=args.plot_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
