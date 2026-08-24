"""Structured output generation module."""

import csv
import json
from pathlib import Path


def to_json(result, filename: str = "results.json"):
    """Export results to JSON format."""
    data = {
        "status": "success",
        "strain_energy": result.strain_energy,
        "prestress_work": result.prestress_work,
        "equilibrium_errors": result.equilibrium_errors,
        "nodal_displacements": {
            nid: {"ux": u[0], "uy": u[1]} for nid, u in result.displacements.items()
        },
        "member_forces": result.forces,
        "reactions": result.reactions,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"📄 JSON report saved to {filename}")


def to_csv(result, filename_prefix: str = "results"):
    """Export results to CSV files (nodes and members separately)."""
    # Nodes CSV
    nodes_file = f"{filename_prefix}_nodes.csv"
    with open(nodes_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Node_ID", "Ux", "Uy"])
        for nid, (ux, uy) in result.displacements.items():
            writer.writerow([nid, f"{ux:.6f}", f"{uy:.6f}"])

    # Members CSV
    members_file = f"{filename_prefix}_members.csv"
    with open(members_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Member_ID", "Axial_Force", "Stress", "Strain", "Buckling_Warning"]
        )
        for force_data in result.forces:
            writer.writerow(
                [
                    force_data.get("element", ""),
                    f"{force_data.get('force', 0):.4f}",
                    f"{force_data.get('stress', 0):.4f}",
                    f"{force_data.get('strain', 0):.6f}",
                    force_data.get("buckling_warning", ""),
                ]
            )

    # Reactions CSV
    reactions_file = f"{filename_prefix}_reactions.csv"
    with open(reactions_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Node_ID", "Rx", "Ry"])
        for nid, react in result.reactions.items():
            writer.writerow([nid, f"{react['Rx']:.4f}", f"{react['Ry']:.4f}"])

    print(f"📄 CSV reports saved to {nodes_file}, {members_file}, {reactions_file}")


def to_markdown(result, filename: str = "results.md"):
    """Export results to Markdown format."""
    lines = [
        "# گزارش تحلیل خرپای دوبعدی\n",
        "## خلاصه\n",
        f"- انرژی کرنشی: {result.strain_energy:.6f} J",
        f"- کار پیش‌تنیدگی: {result.prestress_work:.6f} J\n",
        "## جابه‌جایی گره‌ها\n",
        "| گره | Ux (m) | Uy (m) |",
        "|-----|--------|--------|",
    ]

    for nid, (ux, uy) in result.displacements.items():
        lines.append(f"| {nid} | {ux:.6f} | {uy:.6f} |")

    lines.extend(
        [
            "\n## نیروهای اعضا\n",
            "| عضو | نیرو (N) | تنش (Pa) | کرنش | هشدار کمانش |",
            "|-----|----------|----------|------|-------------|",
        ]
    )

    for force_data in result.forces:
        status = "کشش 📈" if force_data.get("force", 0) > 0 else "فشار 📉"
        lines.append(
            f"| {force_data.get('element', '')} | "
            f"{force_data.get('force', 0):.4f} ({status}) | "
            f"{force_data.get('stress', 0):.4f} | "
            f"{force_data.get('strain', 0):.6f} | "
            f"{force_data.get('buckling_warning', '-')} |"
        )

    lines.extend(
        [
            "\n## عکس‌العمل‌های تکیه‌گاهی\n",
            "| گره | Rx (N) | Ry (N) |",
            "|-----|--------|--------|",
        ]
    )

    for nid, react in result.reactions.items():
        lines.append(f"| {nid} | {react['Rx']:.4f} | {react['Ry']:.4f} |")

    lines.extend(
        [
            "\n## بررسی تعادل استاتیکی\n",
            f"- خطای ΣFx: {result.equilibrium_errors['delta_Fx']:.6e}",
            f"- خطای ΣFy: {result.equilibrium_errors['delta_Fy']:.6e}",
            f"- خطای ΣM: {result.equilibrium_errors['delta_M']:.6e}",
        ]
    )

    Path(filename).write_text("\n".join(lines), encoding="utf-8")
    print(f"📝 Markdown report saved to {filename}")
