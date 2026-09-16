"""Global negative controls — five automated falsification tests.

Each control is a case where a WRONG implementation would produce a
comfortable-looking number; the assertions here pin the physically forced
outcome instead:

NC1  statically determinate truss: every member is structurally
     indispensable — removing ANY member turns the structure into a
     mechanism (graph rank), and a full-removal perturbation (alpha = 0)
     flags EVERY member as mechanism with CI = +inf (never a silent finite
     number).  tau degenerates to None, never to a fake 1.0 or 0.0.
NC2  uniform thermal scenario: tau = 1 within 1e-8 and the CI vector is
     element-wise identical to the ambient one (uniform-field invariance, measured).
NC3  symmetric topology under mirrored local scenarios: local_left and
     local_right CI fields are mirror images (values equal on mirrored
     members; rank lists equal up to the identifier tie-break convention,
     which is why the rank comparison uses the quantised statistic).
NC4  alpha = 1.0 (no degradation): every CI must be EXACTLY 0.0 — the
     perturbation is the identity, so any nonzero value is fabricated.
NC5  zero load: CI is 0.0 with an explicit ``zero-displacement`` flag —
     never scattered NaN, never a silent ratio 0/0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from truss_analysis.criticality import compute_ci_for_topology, tau_b
from truss_analysis.graph_validation import structural_report
from truss_analysis.topology_generator import TopologyGenerator

# Negative controls construct degenerate and singular models on purpose; that is
# what makes them controls. scipy's LinAlgWarning from inside its own
# factorisation is expected, and the assertions are about the library's diagnosis
# of the same state.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.ConstantAlphaWarning"
    ),
    pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning"),
]

CONTROL_INDICES = (1, 2, 3)
SYMMETRIC_CASES = ("warren_4_shallow", "pratt_6_shallow", "howe_8_deep", "control_1")


def _model(campaign, name):
    return next(c for c in campaign if c.name == name)


# --------------------------------------------------------------------- NC1


@pytest.mark.parametrize("index", CONTROL_INDICES)
def test_nc1_any_member_removal_is_a_mechanism(index):
    model = TopologyGenerator.generate_determinate_control(index=index)
    report = structural_report(model)
    assert not report.mechanism  # intact control is stable
    assert report.indeterminacy == 0  # m + r = 2j exactly
    for k in range(len(model["elements"])):
        trimmed = dict(model)
        trimmed["elements"] = [e for j, e in enumerate(model["elements"]) if j != k]
        assert structural_report(trimmed).mechanism, (index, k)


@pytest.mark.parametrize("index", CONTROL_INDICES)
def test_nc1_alpha_zero_flags_every_member_as_mechanism(campaign, index):
    cm = _model(campaign, f"control_{index}")
    res = compute_ci_for_topology(
        cm.nodes, cm.elements, cm.loads, {}, "uniform", 600.0, alpha=0.0
    )
    assert len(res.flags) == len(cm.elements)
    assert all(v == "mechanism(singular)" for v in res.flags.values())
    assert all(v == math.inf for v in res.ci_values.values())
    assert math.isinf(res.u_max_perturbed_max)
    assert res.tau_vs_base is None  # degenerate comparison, never a number


def test_nc1_tau_b_rejects_nonfinite_fields():
    """The rank statistic must not crash or invent a value on inf fields."""
    a = {"1": math.inf, "2": math.inf, "3": 0.1}
    b = {"1": math.inf, "2": math.inf, "3": 0.1}
    res = tau_b(a, b, quantize=1e-10)
    assert res.is_degenerate
    assert res.tau is None


# --------------------------------------------------------------------- NC2


@pytest.mark.parametrize(
    "name", ["warren_6_shallow", "pratt_4_shallow", "howe_8_shallow"]
)
def test_nc2_uniform_scenario_tau_one(campaign, name):
    cm = _model(campaign, name)
    base = compute_ci_for_topology(cm.nodes, cm.elements, cm.loads, {}, "uniform", 20.0)
    hot = compute_ci_for_topology(cm.nodes, cm.elements, cm.loads, {}, "uniform", 600.0)
    assert hot.tau_vs_base is not None
    assert abs(hot.tau_vs_base - 1.0) <= 1e-8
    drift = max(abs(hot.ci_values[k] - base.ci_values[k]) for k in base.ci_values)
    assert drift < 1e-10


# --------------------------------------------------------------------- NC3


def _mirror_map(cm):
    """member id -> id of the member mirrored at the span centre."""
    nodes, elements = cm.nodes, cm.elements
    min_x = min(n.x for n in nodes)
    span = max(n.x for n in nodes) - min_x
    mapping = {}
    for e in elements:
        mirrored_ends = set()
        for nid in (e.node_i, e.node_j):
            node = next(n for n in nodes if n.id == nid)
            x_m = 2.0 * min_x + span - node.x
            best = min(nodes, key=lambda z: abs(z.x - x_m) + abs(z.y - node.y))
            assert abs(best.x - x_m) < 1e-9, "topology is not symmetric"
            mirrored_ends.add(best.id)
        match = [f.id for f in elements if {f.node_i, f.node_j} == mirrored_ends]
        assert len(match) == 1, "mirrored member not found"
        mapping[e.id] = match[0]
    assert len(set(mapping.values())) == len(mapping)  # bijection
    return mapping


@pytest.mark.parametrize("name", SYMMETRIC_CASES)
def test_nc3_local_left_right_are_mirror_images(campaign, name):
    cm = _model(campaign, name)
    mirror = _mirror_map(cm)
    left = compute_ci_for_topology(
        cm.nodes, cm.elements, cm.loads, {}, "local_left", 600.0
    )
    right = compute_ci_for_topology(
        cm.nodes, cm.elements, cm.loads, {}, "local_right", 600.0
    )
    scale = max(abs(v) for v in left.ci_values.values())
    for mid, mirrored in mirror.items():
        assert abs(left.ci_values[mid] - right.ci_values[mirrored]) <= 1e-12 * scale
    # rank-level mirror equality up to the identifier tie-break convention
    relabelled = {mirror[mid]: v for mid, v in right.ci_values.items()}
    res = tau_b(left.ci_values, relabelled, quantize=1e-10)
    assert not res.is_degenerate
    assert res.tau == pytest.approx(1.0, abs=1e-8)


# --------------------------------------------------------------------- NC4


@pytest.mark.parametrize(
    ("name", "scenario", "temperature"),
    [
        ("warren_6_shallow", "uniform", 600.0),
        ("pratt_8_deep", "local_left", 400.0),
        ("howe_4_shallow", "local_mid", 800.0),
    ],
)
def test_nc4_alpha_one_ci_exactly_zero(campaign, name, scenario, temperature):
    cm = _model(campaign, name)
    res = compute_ci_for_topology(
        cm.nodes, cm.elements, cm.loads, {}, scenario, temperature, alpha=1.0
    )
    assert all(v == 0.0 for v in res.ci_values.values())  # EXACT zeros
    assert res.flags == {}
    assert res.u_max_perturbed_max == pytest.approx(res.u_max_base, rel=0, abs=0)


# --------------------------------------------------------------------- NC5


@pytest.mark.parametrize("name", ["warren_4_shallow", "control_2"])
def test_nc5_zero_load_is_flagged_not_nan(campaign, name):
    cm = _model(campaign, name)
    empty = compute_ci_for_topology(cm.nodes, cm.elements, {}, {}, "uniform", 600.0)
    explicit_zero = compute_ci_for_topology(
        cm.nodes,
        cm.elements,
        {nid: {"Fx": 0.0, "Fy": 0.0} for nid in cm.loads},
        {},
        "uniform",
        600.0,
    )
    for res in (empty, explicit_zero):
        assert all(v == 0.0 for v in res.ci_values.values())
        assert all(np.isfinite(v) for v in res.ci_values.values())
        assert res.flags.get("base") == "zero-displacement"
        assert res.u_max_base == 0.0
        assert res.u_max_perturbed_max == 0.0
        assert res.tau_vs_base is None  # no ranking information without load
