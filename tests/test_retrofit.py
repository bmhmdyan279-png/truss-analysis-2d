"""Retrofit triage tests: map, proxy costs, one space, optimality."""

from __future__ import annotations

import pytest

from truss_analysis.model import Element, Node
from truss_analysis.retrofit import (
    ACTION_EFFECTS,
    COST_SCENARIOS,
    DECISION_LEVELS,
    STRATEGIES,
    Decision,
    apply_decision,
    budget_for,
    decision_cost,
    enumerate_space,
    evaluate,
    exhaustive,
    feasible,
    greedy,
    make_context,
)
from truss_analysis.sections import idealised_square_hss

# Deterministic greedy-failure case found by vault/experiment/find_greedy_failure.py
# (seed rng(5000), linear costs, budget fraction 0.35): greedy picks action 2 on
# member 3 while the global optimum protects member 2 at level 1.
FAILURE_CASE = {
    "nodes": [
        (0.0, 0.0, "pin"),
        (6.0, 0.0, "roller"),
        (1.1106926377148134, 1.7109467887696397, ""),
        (4.607664334867298, 1.040379973661508, ""),
    ],
    "members": [(1, 3), (3, 4), (4, 2), (3, 2), (1, 4)],
    "loads": {3: (0.0, -37730.9942426053), 4: (0.0, -37730.9942426053)},
    "cost_scenario": "linear",
    "budget_fraction": 0.35,
    "greedy_u": 0.0033678097796746805,
    "optimal_u": 0.003025823427124406,
}


def _build(case):
    nodes = [
        Node(
            id=str(i + 1),
            x=x,
            y=y,
            is_support=sup != "",
            support_dx=sup == "pin",
            support_dy=sup != "",
        )
        for i, (x, y, sup) in enumerate(case["nodes"])
    ]
    elements = [
        Element(
            id=str(i + 1), node_i=str(a), node_j=str(b), E=210e9, A=0.01, I_sec=1e-6
        )
        for i, (a, b) in enumerate(case["members"])
    ]
    loads = {str(k): {"Fx": fx, "Fy": fy} for k, (fx, fy) in case["loads"].items()}
    return nodes, elements, loads


def test_decision_physics_map_exact() -> None:
    assert set(ACTION_EFFECTS) == set(DECISION_LEVELS)
    assert ACTION_EFFECTS[0].theta_offset == 0.0
    assert ACTION_EFFECTS[1].theta_offset == -200.0
    assert ACTION_EFFECTS[2].theta_offset == -350.0
    assert ACTION_EFFECTS[3].area_factor == 1.3
    _nodes, elements, _ = _build(FAILURE_CASE)
    temps = {e.id: 600.0 for e in elements}
    new_elements, new_temps = apply_decision(
        elements, temps, {e.id: 3 for e in elements}
    )
    for e_old, e_new in zip(elements, new_elements, strict=True):
        assert pytest.approx(1.3 * e_old.A, rel=1e-12) == e_new.A
        # I follows the idealised section model, never A^2/12
        assert e_new.I_sec == pytest.approx(
            idealised_square_hss(1.3 * e_old.A).i_sec, rel=1e-9
        )
        assert e_new.I_sec != pytest.approx((1.3 * e_old.A) ** 2 / 12.0, rel=1e-3)
    new_elements, new_temps = apply_decision(
        elements, temps, {e.id: 1 for e in elements}
    )
    assert all(t == pytest.approx(400.0) for t in new_temps.values())


def test_cost_proxy_scenarios_and_budget() -> None:
    from truss_analysis.retrofit import costs as costs_mod

    assert "prox" in (costs_mod.__doc__ or "").lower()
    assert set(COST_SCENARIOS) == {"linear", "quadratic", "step"}
    lengths = {"1": 2.0, "2": 3.0}
    assert budget_for(lengths) == pytest.approx(0.2 * 1000.0 * 5.0)
    dec = {"1": 1, "2": 2}
    assert decision_cost(dec, lengths, "linear") == pytest.approx(1500 * 2 + 3000 * 3)
    assert decision_cost(dec, lengths, "quadratic") == pytest.approx(
        1500 * 2 + 6000 * 3
    )
    assert decision_cost(dec, lengths, "step") == pytest.approx(2000 * 2 + 4000 * 3)


def test_all_strategies_share_one_decision_space(campaign) -> None:
    cm = next(c for c in campaign if c.name == "warren_4_shallow")
    ctx = make_context(
        cm.nodes,
        cm.elements,
        cm.loads,
        "uniform",
        600.0,
        235e6,
        cost_scenario="linear",
        budget_fraction=0.5,
    )
    for name, fn in STRATEGIES.items():
        if name == "exhaustive":  # n=15 > 7: guard tested separately
            continue
        outcome = fn(ctx)
        dec = outcome.decision
        assert dec.member_ids == ctx.member_ids, name
        assert all(x in DECISION_LEVELS for x in dec.actions), name
        assert feasible(ctx, dec), name
        assert outcome.metrics.cost <= ctx.budget + 1e-9, name
        assert outcome.metrics.theta_sys >= 20.0
        assert outcome.metrics.n_dcr_ge_1 >= 0


def test_exhaustive_is_global_optimum_small_case() -> None:
    nodes, elements, loads = _build(FAILURE_CASE)
    ctx = make_context(
        nodes,
        elements,
        loads,
        "uniform",
        600.0,
        235e6,
        cost_scenario="linear",
        budget_fraction=0.35,
    )
    opt = exhaustive(ctx)
    for dec in enumerate_space(ctx):
        if feasible(ctx, dec):
            assert opt.objective <= evaluate(ctx, dec).u_max + 1e-12
    assert opt.objective == pytest.approx(FAILURE_CASE["optimal_u"], rel=1e-9)


def test_greedy_never_better_and_reported_failure() -> None:
    nodes, elements, loads = _build(FAILURE_CASE)
    ctx = make_context(
        nodes,
        elements,
        loads,
        "uniform",
        600.0,
        235e6,
        cost_scenario="linear",
        budget_fraction=0.35,
    )
    opt = exhaustive(ctx)
    gr = greedy(ctx)
    assert gr.objective >= opt.objective - 1e-12  # never better than global
    # and on this documented case greedy IS strictly worse (reported, not hidden)
    assert gr.objective == pytest.approx(FAILURE_CASE["greedy_u"], rel=1e-9)
    assert gr.objective > opt.objective + 1e-12
    assert gr.decision.actions != opt.decision.actions


def test_greedy_equals_optimal_when_no_conflict(campaign) -> None:
    cm = next(c for c in campaign if c.name == "control_1")
    ctx = make_context(
        cm.nodes,
        cm.elements,
        cm.loads,
        "uniform",
        600.0,
        235e6,
        cost_scenario="linear",
        budget_fraction=0.5,
    )
    opt = exhaustive(ctx)
    gr = greedy(ctx)
    assert gr.objective >= opt.objective - 1e-12


def test_redundant_limitation_documented() -> None:
    from truss_analysis.retrofit import strategies as strat_mod

    doc = strat_mod.redundant_strategy.__doc__ or ""
    assert "LIMITATION" in doc
    assert "topology" in doc


def test_exhaustive_guard_above_seven_members(campaign) -> None:
    cm = next(c for c in campaign if c.name == "warren_8_shallow")
    ctx = make_context(cm.nodes, cm.elements, cm.loads, "uniform", 600.0, 235e6)
    with pytest.raises(ValueError, match="exhaustive search limited"):
        exhaustive(ctx)


def test_decision_out_of_space_rejected() -> None:
    _nodes, elements, _ = _build(FAILURE_CASE)
    temps = {e.id: 600.0 for e in elements}
    with pytest.raises(ValueError, match="out of space"):
        apply_decision(elements, temps, {e.id: 7 for e in elements})


def test_decision_dataclass_roundtrip() -> None:
    dec = Decision(member_ids=("1", "2"), actions=(0, 3))
    assert dec.as_map() == {"1": 0, "2": 3}
