"""Main entry point for truss analysis with advanced CLI."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

from .assembly import assemble_global_matrices
from .fileio import load_json
from .model import Element, Node, validate_inputs
from .plotter import plot_axial_force, plot_truss, setup_persian_font
from .postprocess import (
    calculate_element_forces,
    calculate_reactions,
    check_equilibrium,
)
from .report import to_csv, to_json, to_markdown
from .solver import check_energy, solve
from .units import to_si


@dataclass
class AnalysisResult:
    """Complete analysis results."""

    displacements: dict[str, tuple[float, float]]
    forces: list[dict]
    reactions: dict[str, dict[str, float]]
    strain_energy: float
    prestress_work: float
    equilibrium_errors: dict[str, float]
    nodes: list[Node] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)


def run(filepath: str, unit_sys: str = "SI") -> AnalysisResult:
    """Run complete truss analysis."""
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
            density=to_si(e.get("density", 0.0), unit_sys, "density"),
            effective_length_factor=e.get("effective_length_factor", 1.0),
        )
        for e in data["elements"]
    ]

    validate_inputs(nodes, elements)
    K, F_ext, F_mechanical, fixed_dofs = assemble_global_matrices(nodes, elements)

    # Add self-weight if density is provided
    g = 9.81  # m/s²
    for elem in elements:
        if elem.density > 0:
            i = next(j for j, n in enumerate(nodes) if n.id == elem.node_i)
            j = next(j for j, n in enumerate(nodes) if n.id == elem.node_j)
            dx = nodes[j].x - nodes[i].x
            dy = nodes[j].y - nodes[i].y
            L = (dx**2 + dy**2) ** 0.5
            weight = elem.density * elem.A * L * g
            F_ext[2 * i + 1] -= weight / 2
            F_ext[2 * j + 1] -= weight / 2
            F_mechanical[2 * i + 1] -= weight / 2
            F_mechanical[2 * j + 1] -= weight / 2

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

    reactions = calculate_reactions(nodes, elements, U, F_ext)
    equilibrium_errors = check_equilibrium(nodes, reactions, F_ext)
    displacements = {node.id: (U[i * 2], U[i * 2 + 1]) for i, node in enumerate(nodes)}

    return AnalysisResult(
        displacements=displacements,
        forces=results,
        reactions=reactions,
        strain_energy=strain_energy,
        prestress_work=prestress_work,
        equilibrium_errors=equilibrium_errors,
        nodes=nodes,
        elements=elements,
    )


def main() -> int:
    # Set UTF-8 encoding for stdout/stderr (Windows compatibility)
    import io

    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except (AttributeError, Exception):
        pass

    parser = argparse.ArgumentParser(
        description="2D Truss Analysis Tool with Advanced Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py example.json
  python main.py example.json --plot
  python main.py example.json --output results --format all
  python main.py example.json --save-plots --scale 50
  python main.py example.json --check-buckling
        """,
    )
    parser.add_argument("filepath", help="Path to the JSON input file")
    parser.add_argument(
        "unit_sys",
        nargs="?",
        default="SI",
        help="Unit system (SI or Imperial)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show interactive plots of the truss",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots as PNG files",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file prefix (default: results)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "markdown", "all"],
        default="all",
        help="Output format(s) to generate",
    )
    parser.add_argument(
        "--check-buckling",
        action="store_true",
        help="Check Euler buckling for compression members",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="Deformation scale factor for plots (default: 10)",
    )

    args = parser.parse_args()

    try:
        result = run(args.filepath, args.unit_sys)

        # Console output
        print("✅ تحلیل با موفقیت انجام شد.")
        print(f"انرژی کرنشی: {result.strain_energy:.4f} J")
        print(f"کار پیش‌تنیدگی: {result.prestress_work:.4f} J")
        print("\n📊 نیروهای اعضا:")
        for r in result.forces:
            status = "📈 کشش" if r["force"] > 0 else "📉 فشار"
            print(f"  المان {r['element']}: {r['force']:.2f} N ({status})")
            if r.get("buckling_warning"):
                print(f"    ⚠️  {r['buckling_warning']}")

        print("\n🔧 عکس‌العمل‌های تکیه‌گاهی:")
        for node_id, react in result.reactions.items():
            print(f"  گره {node_id}: Rx={react['Rx']:.2f} N, Ry={react['Ry']:.2f} N")

        print("\n✅ بررسی تعادل استاتیکی:")
        print(f"  خطای ΣFx: {result.equilibrium_errors['delta_Fx']:.6e}")
        print(f"  خطای ΣFy: {result.equilibrium_errors['delta_Fy']:.6e}")
        print(f"  خطای ΣM: {result.equilibrium_errors['delta_M']:.6e}")

        # Generate plots if requested
        if args.plot or args.save_plots:
            font_prop = setup_persian_font()
            prefix = args.output or "results"

            if args.save_plots:
                plot_truss(
                    result.nodes,
                    result.elements,
                    result.displacements,
                    show=args.plot,
                    filename=f"{prefix}_truss.png",
                    scale=args.scale,
                    font_prop=font_prop,
                )
                plot_axial_force(
                    result.nodes,
                    result.elements,
                    result.forces,
                    filename=f"{prefix}_forces.png",
                    font_prop=font_prop,
                )
            elif args.plot:
                plot_truss(
                    result.nodes,
                    result.elements,
                    result.displacements,
                    show=True,
                    scale=args.scale,
                    font_prop=font_prop,
                )
                plot_axial_force(
                    result.nodes,
                    result.elements,
                    result.forces,
                    font_prop=font_prop,
                )

        # Generate structured outputs
        if args.output:
            prefix = args.output
            if args.format in ["json", "all"]:
                to_json(result, f"{prefix}.json")
            if args.format in ["csv", "all"]:
                to_csv(result, prefix)
            if args.format in ["markdown", "all"]:
                to_markdown(result, f"{prefix}.md")

        return 0

    except Exception as e:
        print(f"❌ خطا در تحلیل: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
