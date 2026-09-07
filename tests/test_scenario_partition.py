"""Scenario geometry: exact partition, zero orphans, symmetry (prompt-04 §D).

Negative regressions from CONTEXT_LOCK §4.4 (measured on the pre-prompt-4
code): 10 orphan members overall and ``pratt_8`` heating only 7 of its 10
left-third members.  The centroid partition must now be exact everywhere.
"""

from __future__ import annotations

import pytest
from truss_analysis.criticality import (
    SCENARIOS,
    get_scenario_temperatures,
    member_centroids,
    scenario_partition,
)
from truss_analysis.criticality.scenarios import T_AMBIENT

# CONTEXT_LOCK §4.4 geometric thirds (left/mid/right member counts).
EXPECTED_THIRDS = {
    "warren_4_H1": (5, 5, 5),
    "warren_6_H1": (7, 9, 7),
    "warren_8_H1": (10, 11, 10),
    "pratt_4_H1": (3, 7, 3),
    "pratt_6_H1": (6, 9, 6),
    "pratt_8_H1": (10, 9, 10),
    "howe_4_H1": (3, 7, 3),
    "howe_6_H1": (6, 9, 6),
    "howe_8_H1": (10, 9, 10),
}

# CONTEXT_LOCK §4.4 orphan counts of the LEGACY criteria (must now be zero).
LEGACY_ORPHANS = {
    "warren_4_H1": 2,
    "warren_6_H1": 1,
    "warren_8_H1": 1,
    "pratt_8_H1": 3,
    "howe_8_H1": 3,
}


def test_partition_is_exact_on_campaign(campaign) -> None:
    for cm in campaign:
        left, mid, right = scenario_partition(cm.nodes, cm.elements)
        all_ids = {e.id for e in cm.elements}
        assert left | mid | right == all_ids, cm.name
        assert left & mid == set() and mid & right == set() and left & right == set()
        # zero orphans: every member belongs to exactly one third
        assert len(left) + len(mid) + len(right) == len(all_ids), cm.name


def test_third_counts_match_context_lock(campaign) -> None:
    for cm in campaign:
        if cm.name in EXPECTED_THIRDS:
            sizes = tuple(len(s) for s in scenario_partition(cm.nodes, cm.elements))
            assert sizes == EXPECTED_THIRDS[cm.name], (cm.name, sizes)


def test_legacy_orphans_are_gone(campaign) -> None:
    for cm in campaign:
        if cm.name in LEGACY_ORPHANS:
            assert LEGACY_ORPHANS[cm.name] > 0  # the regression baseline was real
            left, mid, right = scenario_partition(cm.nodes, cm.elements)
            orphans = len(left) + len(mid) + len(right) - len(cm.elements)
            assert orphans == 0, cm.name


def test_pratt8_local_left_heats_all_ten(campaign) -> None:
    cm = next(c for c in campaign if c.name == "pratt_8_H1")
    temps = get_scenario_temperatures(cm.nodes, cm.elements, "local_left", 600.0)
    hot = {eid for eid, t in temps.items() if t == 600.0}
    assert len(hot) == 10  # legacy code heated only 7 (30 % under-coverage)
    left, _, _ = scenario_partition(cm.nodes, cm.elements)
    assert hot == left


def test_local_right_scenario_exists_and_matches_partition(campaign) -> None:
    assert "local_right" in SCENARIOS
    cm = next(c for c in campaign if c.name == "warren_6_H1")
    _, _, right = scenario_partition(cm.nodes, cm.elements)
    temps = get_scenario_temperatures(cm.nodes, cm.elements, "local_right", 800.0)
    hot = {eid for eid, t in temps.items() if t == 800.0}
    cold = {eid for eid, t in temps.items() if t == T_AMBIENT}
    assert hot == right and cold == ({e.id for e in cm.elements} - right)


def test_symmetry_local_left_vs_local_right(campaign) -> None:
    """On a symmetric truss the mirrored scenario gives mirrored CIs."""
    from truss_analysis.criticality import compute_ci_for_topology

    cm = next(c for c in campaign if c.name == "warren_6_H1")
    centroids = member_centroids(cm.nodes, cm.elements)
    xs = [n.x for n in cm.nodes]
    total = min(xs) + max(xs)
    mirror = {}
    for eid, xc in centroids.items():
        target = total - xc
        for other, oc in centroids.items():
            if abs(oc - target) < 1e-9:
                mirror[eid] = other
                break
    assert len(mirror) == len(cm.elements)  # warren_6 is fully mirror-symmetric
    left_res = compute_ci_for_topology(
        cm.nodes, cm.elements, cm.loads, {}, "local_left", 600.0
    )
    right_res = compute_ci_for_topology(
        cm.nodes, cm.elements, cm.loads, {}, "local_right", 600.0
    )
    for eid, mirrored in mirror.items():
        assert left_res.ci_values[eid] == pytest.approx(
            right_res.ci_values[mirrored], abs=1e-10
        ), (eid, mirrored)
