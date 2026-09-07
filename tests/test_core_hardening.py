"""Prompt-7 tests: single DegradationOperator type + Gini extremes + viz smoke."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from truss_analysis.degradation import (
    DegradationOperator,
    MechanicalDegradation,
    ThermalDegradation,
    get_degradation_operator,
    registered_degradation_kinds,
)
from truss_analysis.heterogeneity import compute_bounded_metrics, gini_normalized
from truss_analysis.material.steel_eurocode import k_E as ssot_k_E
from truss_analysis.model import Element


def _elements():
    return [
        Element(id="a", node_i="1", node_j="2", E=210e9, A=0.01, I_sec=1e-6),
        Element(id="b", node_i="2", node_j="3", E=210e9, A=0.02, I_sec=2e-6),
    ]


def test_registry_has_exactly_two_kinds() -> None:
    assert set(registered_degradation_kinds()) == {"mechanical", "thermal"}
    assert issubclass(MechanicalDegradation, DegradationOperator)
    assert issubclass(ThermalDegradation, DegradationOperator)


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValueError, match="unknown degradation kind"):
        get_degradation_operator("magic", alpha=0.7)


def test_mechanical_operator_physics() -> None:
    op = get_degradation_operator("mechanical", target_id="a", alpha=0.7)
    out = op.apply(_elements())
    assert out[0].A == pytest.approx(0.7 * 0.01, rel=1e-12)
    assert out[0].I_sec == pytest.approx(0.49 * 1e-6, rel=1e-12)
    assert out[1].A == pytest.approx(0.02, rel=1e-12)


def test_thermal_operator_uses_ssot() -> None:
    temps = {"a": 600.0, "b": 20.0}
    op = get_degradation_operator("thermal", temps=temps)
    out = op.apply(_elements())
    assert out[0].E == pytest.approx(210e9 * 0.310, rel=1e-9)
    assert out[1].E == pytest.approx(210e9, rel=1e-12)
    assert ssot_k_E(600.0) == pytest.approx(0.310, rel=1e-9)


def test_gini_extremes_known_distributions() -> None:
    equal = [1.0] * 8
    metrics = compute_bounded_metrics(equal)
    assert metrics["gini"] == pytest.approx(0.0, abs=1e-12)
    n = 6
    winner = [0.0] * (n - 1) + [5.0]
    m2 = compute_bounded_metrics(winner)
    assert m2["gini"] == pytest.approx((n - 1) / n, rel=1e-12)
    assert gini_normalized(winner) == pytest.approx(1.0, rel=1e-12)
    assert gini_normalized(equal) == pytest.approx(0.0, abs=1e-12)
    assert gini_normalized([3.0]) == 0.0


def test_visualization_smoke() -> None:
    from truss_analysis import visualization as viz

    funcs = [f for f in dir(viz) if not f.startswith("_")]
    assert funcs  # module exposes a public surface
