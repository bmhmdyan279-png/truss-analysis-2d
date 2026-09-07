"""Graph/numerical validation tests (prompt-05, T5)."""

from __future__ import annotations

import copy

import pytest
from truss_analysis.graph_validation import (
    TopologyValidationError,
    structural_report,
    validate_topology,
)
from truss_analysis.topology_generator import generate_topology

FAMILIES = ("warren", "pratt", "howe")


def _quad_model():
    """Four-node square ring: four members, no diagonal -> a mechanism."""
    nodes = [
        {
            "id": 1,
            "x": 0.0,
            "y": 0.0,
            "is_support": True,
            "support_dx": True,
            "support_dy": True,
        },
        {"id": 2, "x": 1.0, "y": 0.0, "is_support": True, "support_dy": True},
        {"id": 3, "x": 1.0, "y": 1.0, "is_support": False},
        {"id": 4, "x": 0.0, "y": 1.0, "is_support": False},
    ]
    template = {
        "A": 0.01,
        "E": 210e9,
        "alpha": 0.0,
        "delta_T": 0.0,
        "delta_L0": 0.0,
        "effective_length_factor": 1.0,
        "section_type": "idealised_square_hss",
        "I_sec": 1e-6,
    }
    elements = [
        {"id": 1, "node_i": 1, "node_j": 2, **template},
        {"id": 2, "node_i": 2, "node_j": 3, **template},
        {"id": 3, "node_i": 3, "node_j": 4, **template},
        {"id": 4, "node_i": 4, "node_j": 1, **template},
    ]
    loads = [
        {"node_id": 3, "Fx": 0.0, "Fy": -1.0},
        {"node_id": 4, "Fx": 0.0, "Fy": -1.0},
    ]
    return {
        "units": "SI",
        "temperature_change": 0.0,
        "nodes": nodes,
        "elements": elements,
        "loads": loads,
        "options": {},
    }


def test_mechanism_detected_by_svd_rank() -> None:
    model = _quad_model()
    report = structural_report(model)
    assert report.mechanism
    assert report.rank_k_ff < report.n_dof_free
    with pytest.raises(TopologyValidationError):
        validate_topology(model)
    # add the diagonal -> stable and determinate? m=5, r=3, j=4 -> 5+3-8=0
    fixed = copy.deepcopy(model)
    fixed["elements"].append(
        {**model["elements"][0], "id": 5, "node_i": 1, "node_j": 3}
    )
    report2 = structural_report(fixed)
    assert not report2.mechanism
    assert report2.indeterminacy == 0
    validate_topology(fixed)


def test_duplicate_and_self_loop_rejected() -> None:
    model = _quad_model()
    dup = copy.deepcopy(model)
    dup["elements"].append(copy.deepcopy(model["elements"][0]))
    dup["elements"][-1]["id"] = 99
    with pytest.raises(TopologyValidationError):
        validate_topology(dup)
    loop = copy.deepcopy(model)
    loop["elements"].append(
        {**model["elements"][0], "id": 77, "node_i": 3, "node_j": 3}
    )
    with pytest.raises(TopologyValidationError):
        validate_topology(loop)


def test_orphan_and_zero_length_and_disconnected_rejected() -> None:
    model = _quad_model()
    orphan = copy.deepcopy(model)
    orphan["nodes"].append({"id": 9, "x": 5.0, "y": 5.0, "is_support": False})
    with pytest.raises(TopologyValidationError):
        validate_topology(orphan)
    zero = copy.deepcopy(model)
    zero["nodes"][2]["x"] = zero["nodes"][1]["x"]
    zero["nodes"][2]["y"] = zero["nodes"][1]["y"]
    with pytest.raises(TopologyValidationError):
        validate_topology(zero)
    disc = copy.deepcopy(model)
    disc["elements"] = disc["elements"][:2]
    with pytest.raises(TopologyValidationError):
        validate_topology(disc)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n_panels", (4, 6, 8))
def test_campaign_reports_complete_and_healthy(family: str, n_panels: int) -> None:
    model = generate_topology(
        family, n_panels=n_panels, span=4.0 * n_panels, height=0.15 * 4.0 * n_panels
    )
    validate_topology(model)
    report = structural_report(model)
    assert report.connected
    assert not report.orphan_nodes
    assert not report.zero_length_members
    assert not report.duplicate_members
    assert not report.self_loops
    assert not report.mechanism
    assert report.rank_k_ff == report.n_dof_free
    assert report.cond_k_ff < 1e12
    assert not report.cond_warning
    assert report.symmetric
    # all three families as generated are statically determinate
    # (m + r = 2j): Warren 4n-1 members / 2n+1 nodes, Pratt & Howe
    # 4n-3 members / 2n nodes, both with r = 3.
    assert report.indeterminacy == 0, (family, n_panels)


@pytest.mark.parametrize("index", (1, 2, 3))
def test_controls_symmetric_and_determinate(index: int) -> None:
    from truss_analysis.topology_generator import TopologyGenerator

    model = TopologyGenerator.generate_determinate_control(index=index)
    report = structural_report(model)
    assert report.symmetric
    assert report.indeterminacy == 0
    assert report.rank_k_ff == report.n_dof_free
