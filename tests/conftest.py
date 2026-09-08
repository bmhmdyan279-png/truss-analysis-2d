"""Pytest configuration — adds src/ to sys.path for import resolution.

Also provides the shared 21-topology suite used across the test suite:
18 statically indeterminate trusses (3 families x {4,6,8} panels x
{shallow,deep} depth ratios, panel = 4 m) plus 3 statically determinate
controls.  The campaign list below is the single place to update.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, NamedTuple

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from truss_analysis.model import Element, Node  # noqa: E402
from truss_analysis.topology_generator import (  # noqa: E402
    TopologyGenerator,
    generate_topology,
)

PANEL = 4.0
DEPTH_RATIOS = {"shallow": 0.15, "deep": 0.25}


class CampaignModel(NamedTuple):
    name: str
    nodes: list[Node]
    elements: list[Element]
    loads: dict[str, dict[str, float]]
    supports: dict[str, Any]


def model_objects(model: dict[str, Any]) -> CampaignModel:
    nodes = [
        Node(
            id=str(n["id"]),
            x=float(n["x"]),
            y=float(n["y"]),
            is_support=bool(n.get("is_support", False)),
            support_dx=bool(n.get("support_dx", False)),
            support_dy=bool(n.get("support_dy", False)),
        )
        for n in model["nodes"]
    ]
    elements = [
        Element(
            id=str(e["id"]),
            node_i=str(e["node_i"]),
            node_j=str(e["node_j"]),
            E=float(e["E"]),
            A=float(e["A"]),
        )
        for e in model["elements"]
    ]
    loads = {
        ld["node_id"]: {"Fx": ld["Fx"], "Fy": ld["Fy"]} for ld in model.get("loads", [])
    }
    supports = {s["node_id"]: s for s in model.get("supports", [])}
    return CampaignModel("", nodes, elements, loads, supports)


def campaign_spec() -> list[dict[str, Any]]:
    spec: list[dict[str, Any]] = []
    for family in ("warren", "pratt", "howe"):
        for n_panels in (4, 6, 8):
            for tag, ratio in DEPTH_RATIOS.items():
                span = PANEL * n_panels
                spec.append(
                    {
                        "name": f"{family}_{n_panels}_{tag}",
                        "model": generate_topology(
                            family,
                            n_panels=n_panels,
                            span=span,
                            height=ratio * span,
                        ),
                    }
                )
    for idx in (1, 2, 3):
        spec.append(
            {
                "name": f"control_{idx}",
                "model": TopologyGenerator.generate_determinate_control(index=idx),
            }
        )
    return spec


@pytest.fixture(scope="session")
def campaign() -> list[CampaignModel]:
    out = []
    for entry in campaign_spec():
        cm = model_objects(entry["model"])
        out.append(cm._replace(name=entry["name"]))
    return out


@pytest.fixture(scope="session")
def warren4_uniform() -> CampaignModel:
    """Warren-4 at depth ratio 0.1875 — the reference geometry for the
    measured CI range 0.086323 under uniform fields (CI is scale-invariant,
    so any similar geometry reproduces it; this one matches the reference
    model exactly)."""
    cm = model_objects(generate_topology("warren", n_panels=4, span=16.0, height=3.0))
    return cm._replace(name="warren_4_uniform")
