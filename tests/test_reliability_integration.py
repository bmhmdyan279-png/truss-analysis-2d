"""Integration tests for Phase 2: Reliability engine with real truss models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from truss_analysis.model import Element, Node
from truss_analysis.reliability import (
    Direction,
    LimitState,
    ReliabilityEngine,
    ServiceLimit,
)
from truss_analysis.reliability_adapter import NodalLoad, TrussReliabilityModel
from truss_analysis.uncertainty import GumbelRV, LognormalRV, NormalRV


def _build_example1_model() -> tuple[list[Node], list[Element], list[NodalLoad]]:
    path = Path(__file__).resolve().parents[1] / "examples" / "example1.json"
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
        for load in data["loads"]  # تغییر l به load
    ]

    return nodes, elements, loads


def test_example1_monte_carlo_runs_without_crash() -> None:
    """Integration test: Run Monte Carlo on example1.json.

    Verifies the adapter correctly bridges the reliability engine and the FEM solver.
    No numeric reliability results are asserted here.
    """
    nodes, elements, loads = _build_example1_model()

    # Define random variables using the naming convention: {param}_{target_id}
    # Element 1: E is Lognormal
    # Element 2: A is Normal
    # Load index 0 (node 2): Fy is Gumbel (downward load, so mean is negative)
    variables = {
        "E_1": LognormalRV(mean=200.0e9, cov=0.05, seed=42),
        "A_2": NormalRV(mean=0.002, std=0.0001, seed=43),
        "Fy_0": GumbelRV(mean=-5000.0, std=1000.0, seed=44),
    }

    yield_stress_map = {"1": 250.0e6, "2": 250.0e6, "3": 250.0e6}

    model = TrussReliabilityModel(
        nodes=nodes,
        elements=elements,
        loads=loads,
        yield_stress_map=yield_stress_map,
    )

    service_limits = [
        ServiceLimit(node_id="2", direction=Direction.MAGNITUDE, limit=0.05),
    ]

    engine = ReliabilityEngine(
        variables=variables,
        analyze_fn=model.create_analyze_fn(),
        service_limits=service_limits,
    )

    report = engine.run(n_samples=50)

    assert report.sample_size == 50
    assert len(report.statistics) > 0

    yield_stat = report.get(LimitState.YIELD, "1")
    buckling_stat = report.get(LimitState.BUCKLING, "1")
    service_stat = report.get(LimitState.SERVICEABILITY, "node_2_magnitude")

    assert yield_stat is not None
    assert yield_stat.valid_samples == 50
    assert np.isfinite(yield_stat.beta_hat)

    assert buckling_stat is not None
    assert service_stat is not None
    assert service_stat.valid_samples == 50
    assert np.isfinite(service_stat.beta_hat)
