"""Compute Phase 3 reference validation metrics for 2D truss reliability.

This script runs the Phase 2 Monte Carlo/MVFOSM engine on:
1. the local reference_problem.json, and
2. an optional literature/book problem JSON.

It writes raw JSON results and a VALIDATION_LOG fragment. It does not
invent reference beta values. If a reference-beta JSON is not supplied,
no literature difference is computed.
"""

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

from truss_analysis.model import (  # noqa: E402
    Element,
    Node,
    validate_inputs,
)
from truss_analysis.reliability import (  # noqa: E402
    ReliabilityEngine,
    ReliabilityReport,
)
from truss_analysis.reliability_adapter import (  # noqa: E402
    NodalLoad,
    TrussReliabilityModel,
)
from truss_analysis.uncertainty import (  # noqa: E402
    GumbelRV,
    LognormalRV,
    NormalRV,
    RandomVariable,
)

_MISSING: Final[object] = object()
_SCRIPT_NAME: Final = "scripts/compute_phase3_reference_validation.py"


@dataclass(frozen=True)
class ProblemData:
    nodes: list[Node]
    elements: list[Element]
    loads: list[NodalLoad]
    yield_stress_map: dict[str, float]
    yield_source: str
    variable_specs: list[dict[str, Any]]
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


def _finite_or_none(value: float) -> float | None:
    value_f = float(value)
    return value_f if math.isfinite(value_f) else None


def _md_float(value: float | None) -> str:
    if value is None:
        return "NaN"
    return f"{value:.6g}"


def _parse_sample_sizes(raw: str) -> list[int]:
    sizes: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        size = int(token)
        if size <= 0:
            raise ValueError("Sample sizes must be positive.")
        sizes.append(size)
    if not sizes:
        raise ValueError("At least one sample size is required.")
    return sorted(set(sizes))


def _choose_comparison_size(
    requested: int | None,
    sizes: list[int],
) -> int:
    if requested is None:
        return sizes[-1]
    if requested not in sizes:
        raise ValueError(
            "Comparison sample size must be one of: "
            f"{', '.join(str(size) for size in sizes)}"
        )
    return requested


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


def _global_yield_stress_pa(data: dict[str, Any]) -> float | None:
    fy_pa_raw = _field(data, "yield_stress_pa", None)
    if fy_pa_raw is not None:
        return float(fy_pa_raw)
    fy_mpa_raw = _field(data, "yield_stress_mpa", None)
    if fy_mpa_raw is not None:
        return float(fy_mpa_raw) * 1.0e6
    return None


def _element_yield_stress_pa(
    raw: dict[str, Any],
    global_fy: float | None,
) -> float | None:
    fy_pa_raw = _field(raw, "yield_stress_pa", None)
    if fy_pa_raw is not None:
        return float(fy_pa_raw)
    fy_mpa_raw = _field(raw, "yield_stress_mpa", None)
    if fy_mpa_raw is not None:
        return float(fy_mpa_raw) * 1.0e6
    return global_fy


def load_problem(
    path: Path,
    yield_stress_mpa: float | None,
) -> ProblemData:
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

    global_fy = _global_yield_stress_pa(data)
    yield_map: dict[str, float] = {}
    yield_source = "none"

    if yield_stress_mpa is not None:
        fy_pa = yield_stress_mpa * 1.0e6
        yield_map = {elem.id: fy_pa for elem in elements}
        yield_source = "cli_uniform_mpa"
    else:
        for raw_elem in raw_elements:
            elem_id = str(_field(raw_elem, "id"))
            elem_fy = _element_yield_stress_pa(raw_elem, global_fy)
            if elem_fy is not None:
                yield_map[elem_id] = elem_fy
                yield_source = "json"

    variable_specs = list(_field(data, "random_variables", []))

    return ProblemData(
        nodes=nodes,
        elements=elements,
        loads=loads,
        yield_stress_map=yield_map,
        yield_source=yield_source,
        variable_specs=variable_specs,
        source_path=path,
    )


def _variables_from_specs(
    specs: list[dict[str, Any]],
    seed_base: int,
) -> dict[str, RandomVariable]:
    variables: dict[str, RandomVariable] = {}
    for idx, raw in enumerate(specs):
        name = str(_field(raw, "name"))
        dist = str(_field(raw, "type")).strip().lower()
        mean = float(_field(raw, "mean"))
        seed = int(_field(raw, "seed", seed_base + idx + 1))

        std_raw = _field(raw, "std", None)
        cov_raw = _field(raw, "cov", None)

        std: float | None
        if std_raw is not None:
            std = float(std_raw)
        elif cov_raw is not None:
            std = abs(mean) * float(cov_raw)
        else:
            raise ValueError(f"Variable '{name}' needs either std or cov.")

        if dist == "normal":
            variables[name] = NormalRV(mean=mean, std=std, seed=seed)
        elif dist == "lognormal":
            if mean <= 0.0:
                raise ValueError(
                    f"Lognormal variable '{name}' must have positive mean."
                )
            variables[name] = LognormalRV(mean=mean, std=std, seed=seed)
        elif dist == "gumbel":
            variables[name] = GumbelRV(mean=mean, std=std, seed=seed)
        else:
            raise ValueError(f"Unsupported distribution: {dist}")

    return variables


def build_variables(
    problem: ProblemData,
    e_cov: float,
    a_cov: float,
    load_cov: float,
    seed_base: int,
) -> tuple[dict[str, RandomVariable], str]:
    if problem.variable_specs:
        return (
            _variables_from_specs(problem.variable_specs, seed_base),
            "problem_json",
        )

    variables: dict[str, RandomVariable] = {}

    for idx, elem in enumerate(problem.elements):
        if e_cov > 0.0:
            variables[f"E_{elem.id}"] = LognormalRV(
                mean=elem.E,
                cov=e_cov,
                seed=seed_base + idx + 1,
            )
        if a_cov > 0.0:
            variables[f"A_{elem.id}"] = NormalRV(
                mean=elem.A,
                cov=a_cov,
                seed=seed_base + 1000 + idx + 1,
            )

    for idx, load in enumerate(problem.loads):
        if load_cov <= 0.0:
            continue
        if abs(load.fx) > 0.0:
            variables[f"Fx_{idx}"] = GumbelRV(
                mean=load.fx,
                std=abs(load.fx) * load_cov,
                seed=seed_base + 2000 + idx + 1,
            )
        if abs(load.fy) > 0.0:
            variables[f"Fy_{idx}"] = GumbelRV(
                mean=load.fy,
                std=abs(load.fy) * load_cov,
                seed=seed_base + 3000 + idx + 1,
            )

    return variables, "auto_phase1_default"


def topology_summary(problem: ProblemData) -> dict[str, int]:
    reactions = sum(1 for node in problem.nodes if node.support_dx)
    reactions += sum(1 for node in problem.nodes if node.support_dy)
    n_nodes = len(problem.nodes)
    n_members = len(problem.elements)
    return {
        "nodes": n_nodes,
        "members": n_members,
        "reactions": reactions,
        "static_indeterminacy": n_members + reactions - 2 * n_nodes,
    }


def _report_to_records(report: ReliabilityReport) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stat in report.statistics:
        records.append(
            {
                "limit_state": stat.limit_state.value,
                "target_id": str(stat.target_id),
                "valid_samples": stat.valid_samples,
                "mean": _finite_or_none(stat.mean),
                "std": _finite_or_none(stat.std),
                "beta_hat": _finite_or_none(stat.beta_hat),
                "pf_approx": _finite_or_none(stat.pf_approx),
            }
        )
    return records


def run_case(
    name: str,
    path: Path,
    sample_sizes: list[int],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[int, ReliabilityReport]]:
    problem = load_problem(path, args.yield_stress_mpa)
    variables, variable_source = build_variables(
        problem,
        args.e_cov,
        args.a_cov,
        args.load_cov,
        args.seed_base,
    )

    model = TrussReliabilityModel(
        nodes=problem.nodes,
        elements=problem.elements,
        loads=problem.loads,
        yield_stress_map=problem.yield_stress_map,
    )
    engine = ReliabilityEngine(
        variables=variables,
        analyze_fn=model.create_analyze_fn(),
    )
    reports = engine.run_convergence(sample_sizes)

    results: list[dict[str, Any]] = []
    for size in sorted(reports):
        results.append(
            {
                "sample_size": size,
                "statistics": _report_to_records(reports[size]),
            }
        )

    try:
        rel_path = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        rel_path = str(path)

    case_result: dict[str, Any] = {
        "name": name,
        "problem_path": rel_path,
        "topology": topology_summary(problem),
        "yield_source": problem.yield_source,
        "random_variable_source": variable_source,
        "random_variable_count": len(variables),
        "random_variables": sorted(variables),
        "results": results,
    }
    return case_result, reports


def load_reference_beta(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    values: list[dict[str, Any]] = []
    for raw in _field(data, "values", []):
        values.append(
            {
                "limit_state": str(_field(raw, "limit_state")).lower(),
                "target_id": str(_field(raw, "target_id")),
                "beta": float(_field(raw, "beta")),
                "note": str(_field(raw, "note", "")),
            }
        )

    return {
        "source": str(_field(data, "source", "unknown")),
        "page": str(_field(data, "page", "")),
        "values": values,
    }


def compare_report(
    report: ReliabilityReport,
    reference_meta: dict[str, Any],
    case_name: str,
    sample_size: int,
) -> dict[str, Any]:
    stats_by_key = {
        (stat.limit_state.value, str(stat.target_id)): stat
        for stat in report.statistics
    }

    records: list[dict[str, Any]] = []
    for ref in reference_meta.get("values", []):
        limit_state = str(ref["limit_state"])
        target_id = str(ref["target_id"])
        key = (limit_state, target_id)
        stat = stats_by_key.get(key)

        computed_beta = _finite_or_none(stat.beta_hat) if stat is not None else None
        reference_beta = _finite_or_none(float(ref["beta"]))

        absolute = None
        relative = None
        if computed_beta is not None and reference_beta is not None:
            absolute = computed_beta - reference_beta
            if abs(reference_beta) > 1.0e-12:
                relative = absolute / reference_beta

        records.append(
            {
                "limit_state": limit_state,
                "target_id": target_id,
                "computed_beta": computed_beta,
                "reference_beta": reference_beta,
                "absolute_difference": absolute,
                "relative_difference": relative,
                "note": str(ref.get("note", "")),
            }
        )

    return {
        "case": case_name,
        "sample_size": sample_size,
        "source": str(reference_meta.get("source", "unknown")),
        "page": str(reference_meta.get("page", "")),
        "records": records,
    }


def _statistics_table(report: ReliabilityReport) -> list[str]:
    lines = [
        "| Limit State | Target ID | Mean(g) | Std(g) | Beta_Hat "
        "| P_f_approx | computed_by |",
        "|---|---|---|---|---|---|---|",
    ]
    for stat in report.statistics:
        mean = _md_float(_finite_or_none(stat.mean))
        std = _md_float(_finite_or_none(stat.std))
        beta = _md_float(_finite_or_none(stat.beta_hat))
        pf = _md_float(_finite_or_none(stat.pf_approx))
        lines.append(
            f"| {stat.limit_state.value} | {stat.target_id} | "
            f"{mean} | {std} | {beta} | {pf} | {_SCRIPT_NAME} |"
        )
    return lines


def _statistics_table_from_records(
    result: dict[str, Any],
) -> list[str]:
    lines = [
        "| Limit State | Target ID | Mean(g) | Std(g) | Beta_Hat "
        "| P_f_approx | computed_by |",
        "|---|---|---|---|---|---|---|",
    ]
    for stat in result.get("statistics", []):
        mean = _md_float(stat.get("mean"))
        std = _md_float(stat.get("std"))
        beta = _md_float(stat.get("beta_hat"))
        pf = _md_float(stat.get("pf_approx"))
        lines.append(
            f"| {stat.get('limit_state', '')} | "
            f"{stat.get('target_id', '')} | {mean} | {std} | {beta} "
            f"| {pf} | {_SCRIPT_NAME} |"
        )
    return lines


def _comparison_table(comparison: dict[str, Any]) -> list[str]:
    lines = [
        "| Limit State | Target ID | Computed beta | Reference beta "
        "| Abs diff | Rel diff | computed_by |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in comparison.get("records", []):
        computed = _md_float(record.get("computed_beta"))
        reference = _md_float(record.get("reference_beta"))
        absolute = _md_float(record.get("absolute_difference"))
        relative = _md_float(record.get("relative_difference"))
        lines.append(
            f"| {record['limit_state']} | {record['target_id']} | "
            f"{computed} | {reference} | {absolute} | {relative} "
            f"| {_SCRIPT_NAME} |"
        )
    return lines


def write_fragment(
    path: Path,
    reports_by_case: dict[str, dict[int, ReliabilityReport]],
    sample_sizes: list[int],
    comparison: dict[str, Any] | None,
    execution_date: str,
) -> None:
    size = comparison["sample_size"] if comparison else sample_sizes[-1]
    lines = [
        "## Phase 3: Reference Validation (computed)",
        "",
        f"Execution date: {execution_date}",
        f"Computed by: `{_SCRIPT_NAME}`",
        "",
    ]

    for case_name, reports in reports_by_case.items():
        report = reports.get(size)
        if report is None:
            continue
        lines.append(f"### {case_name} — sample size {size}")
        lines.append("")
        lines.extend(_statistics_table(report))
        lines.append("")

    if comparison is not None:
        lines.append("### Reference comparison")
        lines.append(f"Source: {comparison.get('source', 'unknown')}")
        if comparison.get("page"):
            lines.append(f"Page: {comparison['page']}")
        lines.append("")
        if comparison.get("records"):
            lines.extend(_comparison_table(comparison))
        else:
            lines.append("No comparison records found.")
    else:
        lines.append(
            "No external reference β JSON was provided; no literature "
            "difference computed."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    cases: dict[str, dict[str, Any]],
    comparison: dict[str, Any] | None,
    args: argparse.Namespace,
    sample_sizes: list[int],
    execution_date: str,
) -> None:
    size = comparison["sample_size"] if comparison else sample_sizes[-1]
    lines = [
        "# Phase 3: Reference Validation",
        "",
        "## Done",
        "- [x] Executed Phase 2 engine on `reference_problem.json`.",
    ]

    if "book_example" in cases:
        lines.append(
            "- [x] Executed Phase 2 engine on the supplied book/example problem."
        )
    else:
        lines.append(
            "- [ ] No book/example problem was supplied; topology "
            "comparison with the literature is limited."
        )

    if comparison is not None:
        lines.append(
            "- [x] Compared computed beta values with supplied reference beta JSON."
        )
    else:
        lines.append(
            "- [ ] No supplied reference beta JSON; quantitative "
            "literature comparison not computed."
        )

    lines.extend(
        [
            "",
            "## Technical decisions",
            "- Decision: Do not fabricate missing literature beta values. "
            "| Rationale: Integrity rules require computed_by evidence. "
            "| Alternatives rejected: using frame examples, inventing "
            "book data.",
            "- Decision: Randomize only variables that are reliably applied "
            "by the current adapter unless explicit random_variables are "
            "provided. | Rationale: The current `_apply_sample` parser does "
            "not reliably apply multi-underscore names such as delta_T_1. "
            "| Alternatives rejected: silently defining unused variables.",
            "- Decision: No predefined acceptance threshold is used. | "
            "Rationale: Phase 3 is record-and-analyze, not pass/fail by "
            "preset threshold.",
            "",
            "## Verification (real, run by script)",
            f"- Command run: `{' '.join(sys.argv)}`",
            f"- Execution date: {execution_date}",
            "- Result: script completed and wrote raw JSON + validation fragment.",
            "",
            "## Assumptions",
            f"- Sample sizes: {', '.join(str(s) for s in sample_sizes)}",
            f"- E CoV (automatic mode only): {args.e_cov}",
            f"- A CoV (automatic mode only): {args.a_cov}",
            f"- Load CoV (automatic mode only): {args.load_cov}",
            f"- Seed base: {args.seed_base}",
            f"- Yield stress override: {args.yield_stress_mpa}",
            "- Thermal and fabrication variables are deterministic unless "
            "explicitly supplied in `random_variables`.",
            "",
            "## Limitations",
            "- `reference_data_from_books.md` currently does not provide "
            "explicit member β values for the 6-member truss Example 5.1.",
            "- Frame β values in the extracted notes are not used because "
            "they are not truss member limit states.",
            "- Any difference must be interpreted as methodological/model "
            "difference, not as a preset pass/fail criterion.",
            "",
            "## Computed metrics",
            "",
        ]
    )

    for case_name, case in cases.items():
        lines.append(f"### Case: {case_name}")
        lines.append("")
        lines.append(f"Problem file: `{case.get('problem_path', '')}`")
        topo = case.get("topology", {})
        lines.append(
            "Topology: nodes="
            f"{topo.get('nodes', 'NaN')}, members="
            f"{topo.get('members', 'NaN')}, reactions="
            f"{topo.get('reactions', 'NaN')}, static indeterminacy="
            f"{topo.get('static_indeterminacy', 'NaN')}"
        )
        lines.append(f"Yield source: `{case.get('yield_source', 'none')}`")
        lines.append(
            f"Variable source: `{case.get('random_variable_source', 'unknown')}`"
        )
        lines.append(f"Random variables: {case.get('random_variable_count', 0)}")
        lines.append("")

        for result in case.get("results", []):
            if result.get("sample_size") == size:
                lines.extend(_statistics_table_from_records(result))
        lines.append("")

    if comparison is not None:
        lines.append("## Reference comparison")
        lines.append(f"Source: {comparison.get('source', 'unknown')}")
        if comparison.get("page"):
            lines.append(f"Page: {comparison['page']}")
        lines.append("")
        if comparison.get("records"):
            lines.extend(_comparison_table(comparison))
        else:
            lines.append("No comparison records found.")
    else:
        lines.append("## Reference comparison")
        lines.append(
            "No external reference β JSON was provided; no literature "
            "difference computed."
        )

    lines.extend(
        [
            "",
            "## Numbers produced",
            "- See `PROJECT_DOCUMENTATION/phase3_results.json` and "
            "`PROJECT_DOCUMENTATION/phase3_validation_log_fragment.md`.",
            "",
            "## Stability check",
            "- [ ] Clean checkout + install + run reproduces this phase's output.",
            "",
            "## Handoff",
            "- Blocking items: none created by this script; scientific "
            "gate decision remains human/document-based.",
            "- First command for next step: review `REPORT.md` and append "
            "the validation fragment to `VALIDATION_LOG.md`.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 3 reference validation runner.")
    parser.add_argument(
        "--reference-problem",
        default="reference_problem.json",
        help="Local system reference problem.",
    )
    parser.add_argument(
        "--book-json",
        default=None,
        help="Optional book/literature problem JSON with same schema.",
    )
    parser.add_argument(
        "--reference-beta-json",
        default=None,
        help="Optional JSON containing literature beta values.",
    )
    parser.add_argument(
        "--comparison-case",
        default="auto",
        choices=["auto", "reference_problem", "book_example"],
        help="Case to compare against reference beta values.",
    )
    parser.add_argument(
        "--samples",
        default="500,1000,5000",
        help="Comma-separated sample sizes.",
    )
    parser.add_argument(
        "--comparison-sample-size",
        type=int,
        default=None,
        help="Sample size used for beta comparison.",
    )
    parser.add_argument(
        "--yield-stress-mpa",
        type=float,
        default=None,
        help="Uniform yield stress override in MPa.",
    )
    parser.add_argument("--e-cov", type=float, default=0.05)
    parser.add_argument("--a-cov", type=float, default=0.02)
    parser.add_argument("--load-cov", type=float, default=0.15)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="PROJECT_DOCUMENTATION",
        help="Output directory for Phase 3 artifacts.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write REPORT.md under phases/phase_03_reference_validation.",
    )
    args = parser.parse_args(argv)

    if args.e_cov < 0.0 or args.a_cov < 0.0 or args.load_cov < 0.0:
        raise ValueError("CoV values must be non-negative.")

    sample_sizes = _parse_sample_sizes(args.samples)
    reference_path = _find_problem_path(args.reference_problem)

    cases: dict[str, dict[str, Any]] = {}
    reports_by_case: dict[str, dict[int, ReliabilityReport]] = {}

    case_result, reports = run_case(
        "reference_problem",
        reference_path,
        sample_sizes,
        args,
    )
    cases["reference_problem"] = case_result
    reports_by_case["reference_problem"] = reports

    if args.book_json is not None:
        book_path = _find_problem_path(args.book_json)
        case_result, reports = run_case(
            "book_example",
            book_path,
            sample_sizes,
            args,
        )
        cases["book_example"] = case_result
        reports_by_case["book_example"] = reports

    comparison: dict[str, Any] | None = None
    if args.reference_beta_json is not None:
        beta_path = _find_problem_path(args.reference_beta_json)
        reference_meta = load_reference_beta(beta_path)

        comparison_case = args.comparison_case
        if comparison_case == "auto":
            comparison_case = (
                "book_example"
                if "book_example" in reports_by_case
                else "reference_problem"
            )
        if comparison_case not in reports_by_case:
            raise ValueError(f"Comparison case not available: {comparison_case}")

        comparison_size = _choose_comparison_size(
            args.comparison_sample_size,
            sample_sizes,
        )
        comparison = compare_report(
            reports_by_case[comparison_case][comparison_size],
            reference_meta,
            comparison_case,
            comparison_size,
        )

    execution_date = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "metadata": {
            "phase": "03_reference_validation",
            "generated_utc": execution_date,
            "script": _SCRIPT_NAME,
            "sample_sizes": sample_sizes,
            "assumptions": {
                "e_cov": args.e_cov,
                "a_cov": args.a_cov,
                "load_cov": args.load_cov,
                "seed_base": args.seed_base,
                "yield_stress_mpa_override": args.yield_stress_mpa,
                "thermal_random": False,
                "fabrication_random": False,
                "service_limits": [],
                "adapter_note": (
                    "Automatic mode randomizes only E_<id>, A_<id>, "
                    "Fx_<idx>, and Fy_<idx>. Explicit random_variables "
                    "in problem JSON can override this."
                ),
            },
            "comparison": comparison,
        },
        "cases": cases,
    }

    out_path = output_dir / "phase3_results.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fragment_path = output_dir / "phase3_validation_log_fragment.md"
    write_fragment(
        fragment_path,
        reports_by_case,
        sample_sizes,
        comparison,
        execution_date,
    )

    report_path: Path | None = None
    if args.write_report:
        phase_dir = output_dir / "phases" / "phase_03_reference_validation"
        phase_dir.mkdir(parents=True, exist_ok=True)
        report_path = phase_dir / "REPORT.md"
        write_report(
            report_path,
            cases,
            comparison,
            args,
            sample_sizes,
            execution_date,
        )

    print(f"Phase 3 raw results: {out_path}")
    print(f"Validation fragment: {fragment_path}")
    if report_path is not None:
        print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
