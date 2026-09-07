"""Topology generator tests (prompt-05): load model, controls, determinism,
odd-panel stability (DR-020), docstring honesty (T4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from truss_analysis.graph_validation import structural_report, validate_topology
from truss_analysis.topology_generator import (
    TopologyGenerator,
    TrussConfig,
    TrussFamily,
    content_hash,
    generate_topology,
    model_to_json,
)

TOPO_SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "truss_analysis"
    / "topology_generator.py"
).read_text(encoding="utf-8")

FAMILIES = ("warren", "pratt", "howe")


def _loads_of(model):
    return {ld["node_id"]: ld for ld in model["loads"]}


# ----------------------------------------------------------------------
# T1 — load model
# ----------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n_panels", (4, 6, 8))
def test_load_model_family_independent(family: str, n_panels: int) -> None:
    model = generate_topology(
        family, n_panels=n_panels, span=4.0 * n_panels, height=0.15 * 4.0 * n_panels
    )
    loads = _loads_of(model)
    non_support = [n["id"] for n in model["nodes"] if not n["is_support"]]
    # every non-support node loaded, nothing else
    assert set(loads) == set(non_support)
    assert len(loads) / len(non_support) == 1.0
    # total demand equals the configured total, exactly
    total = sum(ld["Fy"] for ld in loads.values())
    assert total == pytest.approx(-100.0e3, rel=1e-12)
    # purely vertical
    assert all(ld["Fx"] == 0.0 for ld in loads.values())


def test_load_total_parameter_and_span_normalisation() -> None:
    small = generate_topology(
        "warren", n_panels=4, span=16.0, height=2.4, total_load=50.0e3
    )
    large = generate_topology(
        "warren", n_panels=8, span=32.0, height=4.8, total_load=50.0e3
    )
    for model in (small, large):
        total = sum(ld["Fy"] for ld in model["loads"])
        assert total == pytest.approx(-50.0e3)
    # same total demand for n=4 and n=8 -> comparable screening demand


@pytest.mark.parametrize("family", FAMILIES)
def test_no_hardcoded_magnitude(family: str) -> None:
    assert "10000" not in TOPO_SRC
    model = generate_topology(family, n_panels=4, span=16.0, height=2.4)
    # per-node shares derive from total_load / n_loaded, never a literal
    n_loaded = len([n for n in model["nodes"] if not n["is_support"]])
    for ld in model["loads"]:
        assert ld["Fy"] == pytest.approx(-100.0e3 / n_loaded, rel=1e-12)


# ----------------------------------------------------------------------
# T2 — section model wired into elements
# ----------------------------------------------------------------------
def test_elements_carry_idealised_hss() -> None:
    cfg = TrussConfig(TrussFamily.PRATT, n_panels=6, span=24.0, height=3.6)
    model = TopologyGenerator(cfg).generate()
    for elem in model["elements"]:
        assert elem["section_type"] == "idealised_square_hss"
        assert elem["I_sec"] == pytest.approx(cfg.i_sec, rel=1e-12)
        assert elem["A"] == cfg.area


# ----------------------------------------------------------------------
# T3 — three geometrically distinct determinate controls
# ----------------------------------------------------------------------
@pytest.mark.parametrize("index", (1, 2, 3))
def test_control_is_statically_determinate(index: int) -> None:
    model = TopologyGenerator.generate_determinate_control(index=index)
    j = len(model["nodes"])
    m = len(model["elements"])
    r = sum(
        (2 if (n.get("support_dx") and n.get("support_dy")) else 1)
        for n in model["nodes"]
        if n["is_support"]
    )
    assert m + r == 2 * j
    validate_topology(model)
    report = structural_report(model)
    assert report.indeterminacy == 0
    assert not report.mechanism


@pytest.mark.parametrize("index", (1, 2, 3))
def test_control_member_removal_is_mechanism(index: int) -> None:
    model = TopologyGenerator.generate_determinate_control(index=index)
    for elem in list(model["elements"]):
        reduced = {
            **model,
            "elements": [e for e in model["elements"] if e["id"] != elem["id"]],
        }
        report = structural_report(reduced)
        assert report.mechanism, (index, elem["id"])


def test_controls_geometrically_distinct() -> None:
    shapes = []
    for index in (1, 2, 3):
        model = TopologyGenerator.generate_determinate_control(index=index)
        shapes.append((len(model["nodes"]), len(model["elements"])))
    assert len(set(shapes)) == 3


def test_control_rejects_unknown_index() -> None:
    with pytest.raises(ValueError):
        TopologyGenerator.generate_determinate_control(index=4)


# ----------------------------------------------------------------------
# T4 — docstring honesty
# ----------------------------------------------------------------------
def test_legacy_lemma_claim_removed_and_load_model_documented() -> None:
    assert "Lemma 1 readiness" not in TOPO_SRC
    assert "Load model" in TOPO_SRC
    assert "uniform temperature field scales the **whole stiffness matrix**" in TOPO_SRC


# ----------------------------------------------------------------------
# T6 — deterministic serialisation
# ----------------------------------------------------------------------
def test_content_hash_stable_and_sensitive() -> None:
    a = generate_topology("howe", n_panels=6, span=24.0, height=3.6)
    b = generate_topology("howe", n_panels=6, span=24.0, height=3.6)
    assert model_to_json(a) == model_to_json(b)
    assert content_hash(a) == content_hash(b)
    c = generate_topology("howe", n_panels=6, span=24.0, height=3.7)
    assert content_hash(c) != content_hash(a)
    assert len(content_hash(a)) == 64


# ----------------------------------------------------------------------
# DR-020 — odd panel counts are stable now
# ----------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n_panels", (3, 5, 7, 9))
def test_odd_panel_counts_are_not_mechanisms(family: str, n_panels: int) -> None:
    model = generate_topology(
        family, n_panels=n_panels, span=4.0 * n_panels, height=0.2 * 4.0 * n_panels
    )
    validate_topology(model)  # raises on mechanism
    report = structural_report(model)
    assert not report.mechanism
    assert report.cond_k_ff < 1e12
    assert np.isfinite(report.cond_k_ff)
