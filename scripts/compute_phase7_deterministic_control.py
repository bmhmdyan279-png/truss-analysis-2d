"""Phase 7: Deterministic Control (Statically Determinate Truss)."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
for _candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    _candidate_str = str(_candidate)
    if _candidate_str not in sys.path:
        sys.path.insert(0, _candidate_str)

from truss_analysis.degradation import DamageOperator  # noqa: E402
from truss_analysis.model import Element, Node, validate_inputs  # noqa: E402
from truss_analysis.reliability_adapter import NodalLoad  # noqa: E402

_SCRIPT_NAME: Final = "scripts/compute_phase7_deterministic_control.py"


def load_problem(
    path: Path,
) -> tuple[list[Node], list[Element], list[NodalLoad]]:
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    nodes = [
        Node(
            id=str(n["id"]),
            x=float(n["x"]),
            y=float(n["y"]),
            is_support=bool(n.get("is_support", False)),
            support_dx=bool(n.get("support_dx", False)),
            support_dy=bool(n.get("support_dy", False)),
        )
        for n in data["nodes"]
    ]
    elements = [
        Element(
            id=str(e["id"]),
            node_i=str(e["node_i"]),
            node_j=str(e["node_j"]),
            E=float(e["E"]),
            A=float(e["A"]),
            I_sec=float(e.get("I_sec", 0.0)),
            alpha=float(e.get("alpha", 0.0)),
            delta_T=float(e.get("delta_T", 0.0)),
            delta_L_free=float(e.get("delta_L_free", e.get("delta_L0", 0.0))),
            effective_length_factor=float(e.get("effective_length_factor", 1.0)),
        )
        for e in data["elements"]
    ]
    loads = [
        NodalLoad(
            node_id=str(lf["node_id"]),
            fx=float(lf.get("Fx", 0.0)),
            fy=float(lf.get("Fy", 0.0)),
        )
        for lf in data.get("loads", [])
    ]
    validate_inputs(nodes, elements)
    return nodes, elements, loads


def main() -> None:
    problem_path = PROJECT_ROOT / "examples" / "determinate_control.json"
    nodes, elements, loads = load_problem(problem_path)

    operator = DamageOperator(nodes=nodes, elements=elements, loads=loads)
    profiles = operator.analyze_all(alphas=[1.0, 0.9, 0.8, 0.7])

    ts = datetime.now(timezone.utc).isoformat()
    print("## Phase 7: Deterministic Control (Statically Determinate Truss)")
    print("")
    print(f"Execution date: {ts}")
    print(f"Computed by: `{_SCRIPT_NAME}`")
    print("Problem: `examples/determinate_control.json`")
    print("")
    print(
        "| Member ID | SCF(α_min) | Slope | is_key_element "
        "| mechanism_detected_at | computed_by |"
    )
    print("|---|---|---|---|---|---|")

    all_key = True
    for prof in profiles:
        if not prof.is_key_element:
            all_key = False

        key_str = "Yes" if prof.is_key_element else "No"

        if prof.mechanism_detected_at is not None:
            mech_str = f"{prof.mechanism_detected_at}"
        else:
            mech_str = "-"

        scf_str = (
            f"{prof.scf_alpha_min:.4f}" if math.isfinite(prof.scf_alpha_min) else "NaN"
        )
        slope_str = (
            f"{prof.sensitivity_slope:.4f}"
            if math.isfinite(prof.sensitivity_slope)
            else "NaN"
        )
        print(
            f"| {prof.member_id} | {scf_str} | {slope_str} "
            f"| {key_str} | {mech_str} | {_SCRIPT_NAME} |"
        )

    print("")
    if all_key:
        msg = (
            "**Verification:** PASS - All members correctly flagged "
            "as Key Elements in statically determinate truss."
        )
    else:
        msg = (
            "**Verification:** FAIL - Some members were not flagged "
            "as Key Elements. Check numerical rank tolerance."
        )
    print(msg)


if __name__ == "__main__":
    main()
