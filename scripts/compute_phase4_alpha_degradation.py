"""Compute Phase 4 alpha-degradation metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
for _candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    _candidate_str = str(_candidate)
    if _candidate_str not in sys.path:
        sys.path.insert(0, _candidate_str)

from truss_analysis.degradation import (  # noqa: E402
    DamageOperator,
    MemberSensitivityProfile,
)
from truss_analysis.model import Element, Node, validate_inputs  # noqa: E402
from truss_analysis.reliability_adapter import NodalLoad  # noqa: E402

_MISSING: Final[object] = object()
_SCRIPT_NAME: Final = "scripts/compute_phase4_alpha_degradation.py"


@dataclass(frozen=True)
class ProblemData:
    nodes: list[Node]
    elements: list[Element]
    loads: list[NodalLoad]
    source_path: Path


def _field(
    raw: dict[str, Any],
    key: str,
    default: object = _MISSING,
) -> Any:
    candidates = (key, key.strip(), f"{key} ", f" {key}")
    for candidate in candidates:
        if candidate in raw:
            return raw[candidate]
    if default is not _MISSING:
        return default
    raise KeyError(f"Missing field: {key}")


def _find_problem_path(raw_path: str) -> Path:
    path = Path(raw_path)
    candidates = [
        path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "examples" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Problem file not found: {raw_path}")


def load_problem(path: Path) -> ProblemData:
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    raw_nodes = _field(data, "nodes", [])
    nodes: list[Node] = []
    for raw_node in raw_nodes:
        nodes.append(
            Node(
                id=str(_field(raw_node, "id")),
                x=float(_field(raw_node, "x")),
                y=float(_field(raw_node, "y")),
                is_support=bool(_field(raw_node, "is_support", False)),
                support_dx=bool(_field(raw_node, "support_dx", False)),
                support_dy=bool(_field(raw_node, "support_dy", False)),
            )
        )

    raw_elements = _field(data, "elements", [])
    elements: list[Element] = []
    for raw_elem in raw_elements:
        elements.append(
            Element(
                id=str(_field(raw_elem, "id")),
                node_i=str(_field(raw_elem, "node_i")),
                node_j=str(_field(raw_elem, "node_j")),
                E=float(_field(raw_elem, "E")),
                A=float(_field(raw_elem, "A")),
                I_sec=float(_field(raw_elem, "I_sec", 0.0)),
                alpha=float(_field(raw_elem, "alpha", 0.0)),
                delta_T=float(_field(raw_elem, "delta_T", 0.0)),
                delta_L_free=float(
                    _field(
                        raw_elem,
                        "delta_L_free",
                        _field(raw_elem, "delta_L0", 0.0),
                    )
                ),
                effective_length_factor=float(
                    _field(raw_elem, "effective_length_factor", 1.0)
                ),
            )
        )

    raw_loads = _field(data, "loads", [])
    loads: list[NodalLoad] = []
    for raw_load in raw_loads:
        loads.append(
            NodalLoad(
                node_id=str(_field(raw_load, "node_id")),
                fx=float(_field(raw_load, "Fx", 0.0)),
                fy=float(_field(raw_load, "Fy", 0.0)),
            )
        )

    validate_inputs(nodes, elements)
    return ProblemData(nodes=nodes, elements=elements, loads=loads, source_path=path)


def _finite_or_none(value: float) -> float | None:
    value_f = float(value)
    return value_f if math.isfinite(value_f) else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4 alpha-degradation runner.")
    parser.add_argument("--problem", default="reference_problem.json")
    parser.add_argument("--alphas", default="1.0,0.9,0.8,0.7")
    parser.add_argument("--output-dir", default="PROJECT_DOCUMENTATION")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args(argv)

    alphas = [float(a) for a in args.alphas.split(",")]
    problem_path = _find_problem_path(args.problem)
    problem = load_problem(problem_path)
    operator = DamageOperator(
        nodes=problem.nodes,
        elements=problem.elements,
        loads=problem.loads,
    )
    profiles = operator.analyze_all(alphas=alphas)

    results_dict: dict[str, Any] = {
        "metadata": {
            "phase": "04_alpha_degradation",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script": _SCRIPT_NAME,
            "alphas": alphas,
            "problem_path": str(problem_path),
            "topology": {
                "nodes": len(problem.nodes),
                "members": len(problem.elements),
            },
        },
        "profiles": [],
    }

    for prof in profiles:
        prof_dict: dict[str, Any] = {
            "member_id": prof.member_id,
            "baseline_max_disp": prof.baseline_max_disp,
            "sensitivity_slope": prof.sensitivity_slope,
            "scf_alpha_min": prof.scf_alpha_min,
            "is_key_element": prof.is_key_element,
            "mechanism_detected_at": prof.mechanism_detected_at,
            "points": [
                {
                    "alpha": p.alpha,
                    "max_disp": _finite_or_none(p.max_disp),
                    "is_singular": p.is_singular,
                    "max_abs_force_delta": _max_force_delta(p, prof),
                    "error": p.error,
                }
                for p in prof.points
            ],
        }
        results_dict["profiles"].append(prof_dict)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "phase4_results.json"
    out_path.write_text(
        json.dumps(results_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = _build_validation_fragment(results_dict, profiles, args, alphas)
    frag_path = output_dir / "phase4_validation_log_fragment.md"
    frag_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report_path = None
    if args.write_report:
        phase_dir = output_dir / "phases" / "phase_04_alpha_degradation"
        phase_dir.mkdir(parents=True, exist_ok=True)
        report_path = phase_dir / "REPORT.md"
        _write_report(report_path, results_dict, args, profiles)

    print(f"Phase 4 raw results: {out_path}")
    print(f"Validation fragment: {frag_path}")
    if report_path:
        print(f"Report written: {report_path}")


def _max_force_delta(p: Any, prof: MemberSensitivityProfile) -> float | None:
    if p.is_singular:
        return None
    deltas = [
        abs(p.axial_forces.get(eid, 0.0) - prof.baseline_axial_forces.get(eid, 0.0))
        for eid in prof.baseline_axial_forces
    ]
    return max(deltas, default=0.0)


def _build_validation_fragment(
    results_dict: dict[str, Any],
    profiles: list[MemberSensitivityProfile],
    args: argparse.Namespace,
    alphas: list[float],
) -> list[str]:
    lines = [
        "## Phase 4: Alpha-Degradation Sensitivity (computed)",
        "",
        f"Execution date: {results_dict['metadata']['generated_utc']}",
        f"Computed by: `{_SCRIPT_NAME}`",
        f"Problem: `{args.problem}`",
        f"Alphas: {', '.join(str(a) for a in alphas)}",
        "",
        "| Member ID | Slope (SCF vs α) | SCF(α_min) | "
        "Key Element? | Mechanism at α | computed_by |",
        "|---|---|---|---|---|---|",
    ]
    for prof in profiles:
        slope_str = (
            f"{prof.sensitivity_slope:.4f}"
            if math.isfinite(prof.sensitivity_slope)
            else "NaN"
        )
        scf_str = (
            f"{prof.scf_alpha_min:.4f}" if math.isfinite(prof.scf_alpha_min) else "NaN"
        )
        key_str = "Yes" if prof.is_key_element else "No"
        mech_str = (
            f"{prof.mechanism_detected_at}"
            if prof.mechanism_detected_at is not None
            else "-"
        )
        lines.append(
            f"| {prof.member_id} | {slope_str} | {scf_str} | "
            f"{key_str} | {mech_str} | {_SCRIPT_NAME} |"
        )
    return lines


def _write_report(
    path: Path,
    results: dict[str, Any],
    args: argparse.Namespace,
    profiles: list[MemberSensitivityProfile],
) -> None:
    lines = [
        "# Phase 4: Alpha-Degradation Operator",
        "",
        "## Done",
        "- [x] Implemented `DamageOperator` with Geometric Scaling Rule "
        "(`A -> αA`, `I_sec -> α²I_sec`).",
        "- [x] Executed α-degradation on `reference_problem.json` for "
        "α ∈ {1.0, 0.9, 0.8, 0.7}.",
        "- [x] Implemented Key Element flag via numerical rank check at α → 0.",
        "- [x] Computed sensitivity slope (Linear Regression of SCF vs α) "
        "and SCF(α_min).",
        "",
        "## Technical decisions",
        "- Decision: Primary response metric is maximum nodal displacement "
        "magnitude. | Rationale: Always well-defined and monotonically sensitive "
        "to stiffness reduction. | Alternatives rejected: purely axial force "
        "ratio (undefined if baseline force is zero).",
        "- Decision: Key Element mechanism probe uses numerical matrix rank "
        "(`np.linalg.matrix_rank`) at α=1e-6. | Rationale: `np.linalg.solve` "
        "can be numerically unstable or silently return garbage for extremely "
        "ill-conditioned matrices instead of raising `LinAlgError`. Rank check "
        "is robust. | Alternatives rejected: relying solely on `LinAlgError`.",
        "- Decision: Phase 4 is purely deterministic. | Rationale: Isolates "
        "the structural sensitivity layer before coupling with Phase 2 "
        "reliability metrics in Phase 6.",
        "",
        "## Verification (real, run by script)",
        f"- Command run: `{' '.join(sys.argv)}`",
        f"- Execution date: {results['metadata']['generated_utc']}",
        "- Result: script completed and wrote raw JSON + validation fragment.",
        "",
        "## Computed metrics",
        "",
        "| Member ID | Slope (SCF vs α) | SCF(α_min) | Key Element? "
        "| Mechanism at α | computed_by |",
        "|---|---|---|---|---|---|",
    ]
    for prof in profiles:
        slope_str = (
            f"{prof.sensitivity_slope:.4f}"
            if math.isfinite(prof.sensitivity_slope)
            else "NaN"
        )
        scf_str = (
            f"{prof.scf_alpha_min:.4f}" if math.isfinite(prof.scf_alpha_min) else "NaN"
        )
        key_str = "Yes" if prof.is_key_element else "No"
        mech_str = (
            f"{prof.mechanism_detected_at}"
            if prof.mechanism_detected_at is not None
            else "-"
        )
        lines.append(
            f"| {prof.member_id} | {slope_str} | {scf_str} | "
            f"{key_str} | {mech_str} | {_SCRIPT_NAME} |"
        )
    lines.extend(
        [
            "",
            "## Numbers produced",
            "- See `PROJECT_DOCUMENTATION/phase4_results.json` and "
            "`PROJECT_DOCUMENTATION/phase4_validation_log_fragment.md`.",
            "",
            "## Stability check",
            "- [ ] Clean checkout + install + run reproduces this phase's output.",
            "",
            "## Handoff",
            "- Blocking items: None.",
            "- First command for next step: review `REPORT.md` and append "
            "the validation fragment to `VALIDATION_LOG.md`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
