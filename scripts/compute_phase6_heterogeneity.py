"""Phase 6: Compute heterogeneity index U and test H1 with Bootstrap."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from truss_analysis.heterogeneity import compute_heterogeneity
from truss_analysis.model import Element, Node
from truss_analysis.reliability import LimitState, ReliabilityEngine
from truss_analysis.reliability_adapter import NodalLoad, TrussReliabilityModel
from truss_analysis.uncertainty import (
    GumbelRV,
    LognormalRV,
    NormalRV,
    RandomVariable,
    load_distributions_config,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_problem(
    path: Path,
) -> tuple[list[Node], list[Element], list[NodalLoad]]:
    """Loads reference_problem.json into model objects."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

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
            delta_L_free=float(e.get("delta_L0", 0.0)),
            effective_length_factor=float(e.get("effective_length_factor", 1.0)),
        )
        for e in data["elements"]
    ]

    loads = [
        NodalLoad(str(ld["node_id"]), float(ld["Fx"]), float(ld["Fy"]))
        for ld in data.get("loads", [])
    ]

    return nodes, elements, loads


def build_random_variables(
    config: dict[str, Any],
    elements: list[Element],
    loads: list[NodalLoad],
) -> dict[str, RandomVariable]:
    """Creates RandomVariable instances using ACTUAL topology means + config CoV."""
    rv_map: dict[str, RandomVariable] = {}
    params = config.get("parameters", {})

    base_seed = 42
    e_cfg = params.get("E", {})
    a_cfg = params.get("A", {})
    f_cfg = params.get("F", {})

    for elem in elements:
        if e_cfg:
            rv_map[f"E_{elem.id}"] = LognormalRV(
                mean=elem.E,
                cov=float(e_cfg["cov"]),
                seed=base_seed + int(elem.id),
            )
        if a_cfg:
            rv_map[f"A_{elem.id}"] = NormalRV(
                mean=elem.A,
                cov=float(a_cfg["cov"]),
                seed=base_seed + int(elem.id) + 100,
            )

    if f_cfg:
        cov_f = float(f_cfg["cov"])
        for idx, load in enumerate(loads):
            if load.fx != 0.0:
                # Use absolute std explicitly to bypass uncertainty.py's
                # std = mean * cov bug for negative means
                fx_std = abs(load.fx) * cov_f
                rv_map[f"Fx_{idx}"] = GumbelRV(
                    mean=load.fx,
                    std=fx_std,
                    seed=base_seed + idx * 2 + 1000,
                )
            if load.fy != 0.0:
                fy_std = abs(load.fy) * cov_f
                rv_map[f"Fy_{idx}"] = GumbelRV(
                    mean=load.fy,
                    std=fy_std,
                    seed=base_seed + idx * 2 + 2000,
                )

    return rv_map


def get_critical_margins(report: Any) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Extracts critical safety margins (min of yield/buckling)."""
    margins: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}

    member_stats: dict[str, dict[LimitState, np.ndarray[Any, np.dtype[Any]]]] = {}
    for stat in report.statistics:
        if stat.limit_state in (LimitState.YIELD, LimitState.BUCKLING):
            mid = str(stat.target_id)
            if mid not in member_stats:
                member_stats[mid] = {}
            member_stats[mid][stat.limit_state] = stat.margins

    for mid, states in member_stats.items():
        gy = states.get(LimitState.YIELD)
        gb = states.get(LimitState.BUCKLING)

        if gy is None and gb is None:
            continue

        if gy is None and gb is not None:
            gc = gb
        elif gb is None and gy is not None:
            gc = gy
        else:
            assert gy is not None and gb is not None
            # Element-wise minimum, properly handling missing values
            gc = np.where(
                np.isnan(gb), gy, np.where(np.isnan(gy), gb, np.minimum(gy, gb))
            )

        margins[mid] = gc

    return margins


def main() -> None:
    """Executes Phase 6 heterogeneity computation."""
    parser = argparse.ArgumentParser(description="Phase 6: Heterogeneity and H1 Test")
    parser.add_argument(
        "--problem", type=str, default="examples/reference_problem.json"
    )
    parser.add_argument("--config", type=str, default="config/distributions.yaml")
    parser.add_argument(
        "--scf", type=str, default="PROJECT_DOCUMENTATION/phase4_results.json"
    )
    parser.add_argument(
        "--output", type=str, default="PROJECT_DOCUMENTATION/phase6_results.json"
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    logger.info("Loading problem definition and distributions...")
    prob_path = Path(args.problem)
    if not prob_path.exists():
        prob_path = Path("truss-analysis-2d/examples/reference_problem.json")

    nodes, elements, loads = load_problem(prob_path)
    config = load_distributions_config(args.config)

    yield_stress_map = {str(e.id): 240e6 for e in elements}
    rv_map = build_random_variables(config, elements, loads)

    logger.info("Initializing Truss Reliability Model...")
    model = TrussReliabilityModel(
        nodes=nodes, elements=elements, loads=loads, yield_stress_map=yield_stress_map
    )
    analyze_fn = model.create_analyze_fn()

    # Correct instantiation using positional/keyword arguments directly
    engine = ReliabilityEngine(
        variables=rv_map,
        analyze_fn=analyze_fn,
        service_limits=(),
    )

    logger.info(f"Running Monte Carlo Simulation with {args.samples} samples...")
    report = engine.run(args.samples)

    logger.info("Extracting critical margins...")
    critical_margins = get_critical_margins(report)

    logger.info("Loading SCF values from Phase 4...")
    p4_path = Path(args.scf)
    if not p4_path.exists():
        p4_path = Path("truss-analysis-2d/PROJECT_DOCUMENTATION/phase4_results.json")

    with open(p4_path, "r", encoding="utf-8") as f:
        p4_data = json.load(f)
    scf_values = {
        str(p["member_id"]): float(p["scf_alpha_min"]) for p in p4_data["profiles"]
    }

    logger.info("Computing heterogeneity and running Bootstrap...")
    result = compute_heterogeneity(
        margins=critical_margins,
        scf_values=scf_values,
        n_bootstrap=args.bootstrap,
        bootstrap_seed=2026,
    )

    output_data = {
        "metadata": {
            "phase": "06_bootstrap_consequence",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/compute_phase6_heterogeneity.py",
            "problem_path": str(prob_path.resolve()),
            "n_samples": args.samples,
            "n_bootstrap": args.bootstrap,
            "mc_seed": 42,
            "bootstrap_seed": 2026,
        },
        "results": {
            "u_empirical_mean": result.u_empirical_mean,
            "u_empirical_std": result.u_empirical_std,
            "u_empirical_quantiles": result.u_empirical_quantiles,
            "u_boot_mean_lower_95": result.u_boot_mean_lower_95,
            "cov_empirical": result.cov_empirical,
            "gini_empirical": result.gini_empirical,
            "h1_accepted": result.h1_accepted,
            "unstable_members": result.unstable_members,
            "mu_g": result.mu_g,
            "beta_hat": result.beta_hat,
            "scf_values": result.scf_values,
        },
        "warnings": result.warnings,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Phase 6 complete. Results saved to {out_path}")
    lower_ci = result.u_boot_mean_lower_95
    logger.info(
        f"H1 Accepted: {result.h1_accepted} (Lower 95% CI of Mean U: {lower_ci:.4f})"
    )


if __name__ == "__main__":
    main()
