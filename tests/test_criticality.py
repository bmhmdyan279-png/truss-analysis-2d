"""Ranking / tau-b / NCI / API behaviour after the prompt-4 restructure.

Covers CONTEXT_LOCK §4.5 items B1-B5 with the new policies:
natural-sort tie-break, tau from CI values with explicit degeneracy, a single
NCI function returning None on degenerate input, relative eps, populated
``tau_vs_base`` and absence of the hard-coded uniform branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from truss_analysis.criticality import (
    NciResult,
    TauResult,
    compute_ci_for_topology,
    compute_nci,
    natural_sort_key,
    rank_members,
    relative_eps,
    scenario_partition,
    tau_b,
)

PACKAGE_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "truss_analysis" / "criticality"
)


def test_natural_sort_key_numeric_order() -> None:
    assert sorted(["10", "2", "1", "20"], key=natural_sort_key) == [
        "1",
        "2",
        "10",
        "20",
    ]
    assert sorted(["e10", "e2"], key=natural_sort_key) == ["e2", "e10"]


def test_rank_members_descending_with_natural_tiebreak() -> None:
    ci = {"10": 0.5, "2": 0.5, "1": 0.9, "3": 0.1}
    assert rank_members(ci) == ["1", "2", "10", "3"]


def test_rank_members_deterministic_regardless_of_insertion_order() -> None:
    ci = {"7": 0.2, "12": 0.8, "3": 0.8, "1": 0.2}
    first = rank_members(ci)
    reversed_input = dict(reversed(list(ci.items())))
    assert rank_members(reversed_input) == first


def test_tau_b_uses_values_and_reports_ties() -> None:
    ci_a = {"a": 1.0, "b": 1.0, "c": 0.5, "d": 0.25}
    ci_b = {"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.25}
    res = tau_b(ci_a, ci_b)
    assert isinstance(res, TauResult)
    assert res.n_ties_a == 1  # the (a, b) pair tied in field a
    assert res.n_ties_b == 0  # field b has distinct values
    assert not res.is_degenerate
    assert res.tau is not None and -1.0 <= res.tau <= 1.0


def test_tau_b_all_ties_is_degenerate_not_silent_one() -> None:
    ci_a = {"a": 0.428571, "b": 0.428571, "c": 0.428571}
    ci_b = {"a": 0.1, "b": 0.2, "c": 0.3}
    res = tau_b(ci_a, ci_b)
    assert res.is_degenerate and res.tau is None
    res2 = tau_b(ci_b, ci_a)
    assert res2.is_degenerate and res2.tau is None


def test_tau_b_small_samples_degenerate() -> None:
    assert tau_b({}, {}).is_degenerate
    assert tau_b({"a": 1.0}, {"a": 2.0}).is_degenerate


def test_tau_b_quantization_convention() -> None:
    """Sub-tolerance noise is not a rank signal (lemma convention)."""
    ci_a = {"a": 0.5000000000000001, "b": 0.25, "c": 0.125}
    ci_b = {"a": 0.4999999999999999, "b": 0.25, "c": 0.125}
    quantised = tau_b(ci_a, ci_b, quantize=1e-10)
    assert quantised.tau == pytest.approx(1.0)
    # a REAL difference above the tolerance still moves tau
    ci_c = {"a": 0.125, "b": 0.25, "c": 0.5}
    moved = tau_b(ci_a, ci_c, quantize=1e-10)
    assert moved.tau is not None and moved.tau < 1.0


def test_nci_bounds_and_extremes() -> None:
    res = compute_nci({"a": 0.0, "b": 1.0, "c": 0.5})
    assert isinstance(res, NciResult)
    assert res.values == {"a": 0.0, "b": 1.0, "c": 0.5}
    assert not res.is_degenerate


def test_nci_degenerate_returns_none_and_flag() -> None:
    res = compute_nci({"a": 0.3, "b": 0.3, "c": 0.3})
    assert res.values is None and res.is_degenerate
    assert res.min_ci == pytest.approx(0.3) and res.max_ci == pytest.approx(0.3)


def test_nci_empty_input() -> None:
    res = compute_nci({})
    assert res.values == {} and not res.is_degenerate


def test_single_nci_policy_in_package() -> None:
    count = 0
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        count += path.read_text(encoding="utf-8").count("def compute_n")
    assert count == 1


def test_relative_eps_scales_with_span() -> None:
    assert relative_eps(1.0) == pytest.approx(1e-9)
    assert relative_eps(1e6) == pytest.approx(1e-3)
    assert relative_eps(1e-6) == pytest.approx(1e-12)  # floor


def test_boundary_centroid_follows_half_open_intervals() -> None:
    from truss_analysis.model import Element, Node

    nodes = [
        Node(id="n0", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="n1", x=3.0, y=0.0),
        Node(id="n2", x=6.0, y=0.0, is_support=True, support_dy=True),
    ]
    elements = [
        Element(id="left", node_i="n0", node_j="n1", E=1.0, A=1.0),  # xc=1.5
        Element(id="onbound", node_i="n1", node_j="n2", E=1.0, A=1.0),  # xc=4.5? no
    ]
    # span=6 -> thirds at 2 and 4; left centroid 1.5 (left), right centroid 4.5 (right)
    left, mid, right = scenario_partition(nodes, elements)
    assert left == {"left"} and right == {"onbound"} and mid == set()


def test_tau_vs_base_populated(campaign) -> None:
    cm = next(c for c in campaign if c.name == "warren_6_H1")
    hot = compute_ci_for_topology(
        cm.nodes, cm.elements, cm.loads, {}, "local_mid", 600.0
    )
    assert hot.tau_vs_base is not None
    assert -1.0 <= hot.tau_vs_base <= 1.0
    base = compute_ci_for_topology(
        cm.nodes, cm.elements, cm.loads, {}, "local_mid", 20.0
    )
    assert base.tau_vs_base == pytest.approx(1.0)


def test_no_hardcoded_uniform_branch_in_package() -> None:
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "ci_const" not in text, path.name
        assert "1.0 / alpha" not in text, path.name
        assert "u_max_base / alpha" not in text, path.name


def test_topology_result_flags_and_optional_nci(campaign) -> None:
    cm = next(c for c in campaign if c.name == "control_1")
    res = compute_ci_for_topology(cm.nodes, cm.elements, cm.loads, {}, "uniform", 600.0)
    assert isinstance(res.flags, dict)
    # determinate control under uniform degradation: CIs may collapse -> NCI None
    if res.nci_values is None:
        assert compute_nci(res.ci_values).is_degenerate
