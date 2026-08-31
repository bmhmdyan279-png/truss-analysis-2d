"""Compute Phase 2 reliability metrics for example1.json.

This script computes the first-order reliability index (beta_hat) for
yield, buckling, and serviceability limit states using Monte Carlo simulation.
It outputs a Markdown table ready to be appended to VALIDATION_LOG.md.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from truss_analysis.model import Element, Node  # noqa: E402
from truss_analysis.reliability import (  # noqa: E402
    Direction,
    ReliabilityEngine,
    ReliabilityReport,
    ServiceLimit,
)
from truss_analysis.reliability_adapter import (  # noqa: E402
    NodalLoad,
    TrussReliabilityModel,
)
from truss_analysis.uncertainty import (  # noqa: E402
    GumbelRV,
    LognormalRV,
    NormalRV,
    RandomVariable,  # این خط را اضافه کنید
)


def load_example1() -> tuple[list[Node], list[Element], list[NodalLoad]]:
    """Load example1.json and convert to model DTOs."""
    path = PROJECT_ROOT / "examples" / "example1.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    nodes = [
        Node(
            id=str(n["id"]),
            x=float(n["x"]),
            y=float(n["y"]),
            is_support=bool(n["is_support"]),
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
        NodalLoad(
            node_id=str(load["node_id"]),
            fx=float(load.get("Fx", 0.0)),
            fy=float(load.get("Fy", 0.0)),
        )
        for load in data["loads"]
    ]

    return nodes, elements, loads


def define_random_variables() -> (
    dict[str, RandomVariable]
):  # تغییر از object به RandomVariable
    """Define random variables based on Phase 1 default mapping."""
    return {
        "E_1": LognormalRV(mean=200.0e9, cov=0.05, seed=42),
        "E_2": LognormalRV(mean=200.0e9, cov=0.05, seed=43),
        "E_3": LognormalRV(mean=200.0e9, cov=0.05, seed=44),
        "A_1": NormalRV(mean=0.001, std=0.001 * 0.02, seed=45),
        "A_2": NormalRV(mean=0.002, std=0.002 * 0.02, seed=46),
        "A_3": NormalRV(mean=0.0015, std=0.0015 * 0.02, seed=47),
        "Fy_0": GumbelRV(mean=-5000.0, std=1000.0, seed=48),
        "delta_T_1": NormalRV(mean=50.0, std=50.0 * 0.15, seed=49),
    }


def print_markdown_table(
    report: ReliabilityReport, execution_date: str, n_samples: int
) -> None:
    """Print results formatted for VALIDATION_LOG.md."""
    print("\n### Phase 2: Reliability Metrics for example1.json\n")
    print(f"**Execution Date:** {execution_date}")
    print(f"**Sample Size:** {n_samples}")
    print("**Computed By:** `scripts/compute_phase2_metrics.py`\n")

    print("| Limit State | Target ID | Mean(g) | Std(g) | Beta_Hat | P_f_approx |")
    print("|-------------|-----------|---------|--------|----------|------------|")

    for stat in report.statistics:
        target_id = str(stat.target_id)
        mean_g = f"{stat.mean:.3e}" if np.isfinite(stat.mean) else "NaN"
        std_g = f"{stat.std:.3e}" if np.isfinite(stat.std) else "NaN"
        beta = f"{stat.beta_hat:.4f}" if np.isfinite(stat.beta_hat) else "NaN"
        pf = f"{stat.pf_approx:.3e}" if np.isfinite(stat.pf_approx) else "NaN"

        print(
            f"| {stat.limit_state.value} | {target_id} | "
            f"{mean_g} | {std_g} | {beta} | {pf} |"
        )


def main() -> None:
    """Main execution entry point."""
    nodes, elements, loads = load_example1()

    yield_stress_map = {
        "1": 250.0e6,
        "2": 250.0e6,
        "3": 250.0e6,
    }

    model = TrussReliabilityModel(
        nodes=nodes,
        elements=elements,
        loads=loads,
        yield_stress_map=yield_stress_map,
    )

    service_limits = [
        ServiceLimit(node_id="2", direction=Direction.MAGNITUDE, limit=0.01),
    ]

    variables = define_random_variables()

    engine = ReliabilityEngine(
        variables=variables,
        analyze_fn=model.create_analyze_fn(),
        service_limits=service_limits,
    )

    print("Running Monte Carlo simulation (10,000 samples)...")
    report = engine.run(n_samples=10_000)

    execution_date = date.today().isoformat()
    print_markdown_table(report, execution_date, report.sample_size)

    raw_data = {
        "metadata": {
            "example": "example1.json",
            "n_samples": report.sample_size,
            "execution_date": execution_date,
        },
        "statistics": [
            {
                "limit_state": stat.limit_state.value,
                "target_id": str(stat.target_id),
                "mean": stat.mean,
                "std": stat.std,
                "beta_hat": stat.beta_hat,
                "pf_approx": stat.pf_approx,
            }
            for stat in report.statistics
        ],
    }

    out_path = PROJECT_ROOT / "PROJECT_DOCUMENTATION" / "phase2_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    print(f"\nRaw data saved to: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
