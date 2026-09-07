"""Tests for TopologyGenerator — Phase 1 gate criteria.

Covers:
- All 18 indeterminate topologies (3 families × 3 panels × 2 depth ratios)
- 3 determinate controls
- Input validation
- Output schema conformance
"""

from __future__ import annotations

import pytest
from truss_analysis.graph_validation import validate_topology
from truss_analysis.topology_generator import (
    TopologyGenerator,
    TrussConfig,
    TrussFamily,
)


# ── Parametrized: all 18 indeterminate topologies ────────────────
@pytest.mark.parametrize("family", list(TrussFamily))
@pytest.mark.parametrize("n_panels", [4, 6, 8])
@pytest.mark.parametrize("height_ratio", [0.10, 0.15])
def test_generated_topology_passes_graph_validation(
    family: TrussFamily, n_panels: int, height_ratio: float
) -> None:
    span = 20.0
    height = span * height_ratio
    cfg = TrussConfig(family=family, n_panels=n_panels, span=span, height=height)
    model = TopologyGenerator(cfg).generate()

    # Must not raise
    validate_topology(model)

    # Schema conformance
    assert model["units"] == "SI"
    assert model["temperature_change"] == 0.0
    assert len(model["nodes"]) > 0
    assert len(model["elements"]) > 0
    assert len(model["loads"]) > 0
    assert "options" in model

    # Every element has required keys
    for elem in model["elements"]:
        assert "id" in elem
        assert "node_i" in elem
        assert "node_j" in elem
        assert "A" in elem
        assert "E" in elem
        assert "I_sec" in elem


# ── Determinate controls ─────────────────────────────────────────
def test_determinate_control_is_valid() -> None:
    model = TopologyGenerator.generate_determinate_control()
    validate_topology(model)
    assert len(model["nodes"]) == 3
    assert len(model["elements"]) == 3


# ── Input validation ─────────────────────────────────────────────
@pytest.mark.parametrize("bad_panels", [0, 1, -1])
def test_invalid_n_panels_raises(bad_panels: int) -> None:
    with pytest.raises(ValueError, match="n_panels"):
        TrussConfig(
            family=TrussFamily.WARREN,
            n_panels=bad_panels,
            span=10.0,
            height=2.0,
        )


@pytest.mark.parametrize("bad_span", [0.0, -5.0])
def test_invalid_span_raises(bad_span: float) -> None:
    with pytest.raises(ValueError, match="span"):
        TrussConfig(
            family=TrussFamily.WARREN,
            n_panels=4,
            span=bad_span,
            height=2.0,
        )


# ── Lemma 1 prerequisite: uniform section properties ─────────────
@pytest.mark.parametrize("family", list(TrussFamily))
def test_uniform_section_properties(family: TrussFamily) -> None:
    """All members must share identical A, E, I_sec (Lemma 1 prerequisite)."""
    cfg = TrussConfig(family=family, n_panels=4, span=16.0, height=2.0)
    model = TopologyGenerator(cfg).generate()
    areas = {e["A"] for e in model["elements"]}
    moduli = {e["E"] for e in model["elements"]}
    inertias = {e["I_sec"] for e in model["elements"]}
    assert len(areas) == 1, "A must be uniform"
    assert len(moduli) == 1, "E must be uniform"
    assert len(inertias) == 1, "I_sec must be uniform"
