"""Retrofit strategies over ONE common decision space.

Fair-comparison rule: every strategy searches the SAME space
``x_i in {0,1,2,3}`` under the SAME proxy budget
``sum_i c_{x_i} L_i <= 0.2 C_base``.  Strategies that would need a topology
change (adding members) are documented as limited and use a cost-equivalent
in-space equivalent instead:

* ``redundant`` - "add a diagonal in the mid panel" changes the topology and
  therefore leaves the common space.  Its documented in-space equivalent
  upgrades the member with the highest redundancy participation (largest
  ``u_max`` increase when removed) to ``x=3``, whose proxy cost ``c_3 L``
  equals the cost assigned to the added member.

Performance metrics: ``u_max`` at the scenario temperature, the system
critical temperature ``theta*_sys``, and the count of members with
``DCR >= 1``.  The optimisation objective is ``u_max`` (minimised); the
other two are reported for every outcome.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..criticality.engine import (
    base_displacement,
    build_engine,
    total_load_vector,
)
from ..limitstates import dcr_field, system_critical_temperature
from ..model import Element, Node
from .actions import apply_decision
from .costs import budget_for, decision_cost

__all__ = [
    "STRATEGIES",
    "Decision",
    "RetrofitContext",
    "RetrofitMetrics",
    "RetrofitOutcome",
    "enumerate_space",
    "evaluate",
    "exhaustive",
    "feasible",
    "greedy",
    "make_context",
    "redundant_strategy",
    "robust_strategy",
    "stress_based_strategy",
]

_EXHAUSTIVE_MAX_MEMBERS = 7
_SYS_TEMP_GRID = tuple(range(20, 1201, 100))


@dataclass(frozen=True)
class RetrofitContext:
    """Everything a strategy needs; identical for all strategies."""

    nodes: Sequence[Node]
    elements: Sequence[Element]
    loads: Mapping[str, Mapping[str, float]]
    scenario: str
    t_target: float
    alpha: float
    f_y: float
    cost_scenario: str
    budget: float
    lengths: Mapping[str, float]
    member_ids: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    """A point of the common decision space."""

    member_ids: tuple[str, ...]
    actions: tuple[int, ...]

    def as_map(self) -> dict[str, int]:
        """Return the decision as a ``member_id -> action level`` mapping."""
        return dict(zip(self.member_ids, self.actions, strict=True))


@dataclass(frozen=True)
class RetrofitMetrics:
    """Performance metrics of one decision vector."""

    u_max: float
    theta_sys: float
    n_dcr_ge_1: int
    cost: float


@dataclass(frozen=True)
class RetrofitOutcome:
    """Decision + its metrics."""

    decision: Decision
    metrics: RetrofitMetrics

    @property
    def objective(self) -> float:
        """Return the optimisation objective (``u_max``, minimised)."""
        return self.metrics.u_max


def member_lengths(
    nodes: Sequence[Node], elements: Sequence[Element]
) -> dict[str, float]:
    """Return the geometric length [m] of every member, keyed by id."""
    out: dict[str, float] = {}
    nmap = {n.id: n for n in nodes}
    for e in elements:
        ni, nj = nmap[e.node_i], nmap[e.node_j]
        out[e.id] = float(np.hypot(nj.x - ni.x, nj.y - ni.y))
    return out


def make_context(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    scenario: str,
    t_target: float,
    f_y: float,
    alpha: float = 0.7,
    cost_scenario: str = "linear",
    budget_fraction: float = 0.2,
) -> RetrofitContext:
    """Build the shared strategy context (identical inputs for all strategies).

    The budget is computed once from the member lengths and the chosen
    proxy cost scenario so that every strategy is compared under the same
    constraint.
    """
    lengths = member_lengths(nodes, elements)
    return RetrofitContext(
        nodes=nodes,
        elements=elements,
        loads=loads,
        scenario=scenario,
        t_target=t_target,
        alpha=alpha,
        f_y=f_y,
        cost_scenario=cost_scenario,
        budget=budget_for(lengths, budget_fraction),
        lengths=lengths,
        member_ids=tuple(e.id for e in elements),
    )


def _scenario_temps(ctx: RetrofitContext) -> dict[str, float]:
    from ..criticality.scenarios import get_scenario_temperatures

    return get_scenario_temperatures(
        list(ctx.nodes), list(ctx.elements), ctx.scenario, ctx.t_target
    )


def feasible(ctx: RetrofitContext, decision: Decision) -> bool:
    """Return True when the decision's proxy cost fits the shared budget."""
    return decision_cost(decision.as_map(), ctx.lengths, ctx.cost_scenario) <= (
        ctx.budget
    )


def enumerate_space(ctx: RetrofitContext) -> Iterator[Decision]:
    """All ``4^n`` decision vectors (use only for small ``n``)."""
    for actions in itertools.product((0, 1, 2, 3), repeat=len(ctx.member_ids)):
        yield Decision(member_ids=ctx.member_ids, actions=actions)


def evaluate(ctx: RetrofitContext, decision: Decision) -> RetrofitMetrics:
    """Metrics of one decision: u_max at t_target, theta*_sys, #DCR>=1, cost."""
    temps = _scenario_temps(ctx)
    elements, temps = apply_decision(list(ctx.elements), temps, decision.as_map())
    setup = build_engine(list(ctx.nodes), elements, ctx.loads, temps)
    u = base_displacement(setup, total_load_vector(list(ctx.nodes), ctx.loads, setup))
    u_max = float(np.max(np.abs(u)))
    theta_sys = system_critical_temperature(
        list(ctx.nodes), elements, ctx.loads, ctx.f_y, temp_grid=_SYS_TEMP_GRID
    )
    states = dcr_field(list(ctx.nodes), elements, ctx.loads, temps, ctx.f_y)
    n_bad = sum(1 for s in states.values() if s.dcr >= 1.0)
    return RetrofitMetrics(
        u_max=u_max,
        theta_sys=theta_sys,
        n_dcr_ge_1=n_bad,
        cost=decision_cost(decision.as_map(), ctx.lengths, ctx.cost_scenario),
    )


def exhaustive(ctx: RetrofitContext) -> RetrofitOutcome:
    """Global optimum by full enumeration (``n <= 7`` members only)."""
    n = len(ctx.member_ids)
    if n > _EXHAUSTIVE_MAX_MEMBERS:
        msg = f"exhaustive search limited to n<={_EXHAUSTIVE_MAX_MEMBERS}, got {n}"
        raise ValueError(msg)
    best: RetrofitOutcome | None = None
    for decision in enumerate_space(ctx):
        if not feasible(ctx, decision):
            continue
        outcome = RetrofitOutcome(decision, evaluate(ctx, decision))
        if best is None or outcome.objective < best.objective - 1e-15:
            best = outcome
    if best is None:
        msg = "no feasible decision in space (budget too tight even for x=0?)"
        raise ValueError(msg)
    return best


def greedy(ctx: RetrofitContext) -> RetrofitOutcome:
    """Upgrade greedily: best ``delta_u_max`` per added proxy cost per step.

    After every reinforcement the state is re-evaluated through the rank-1
    engine (one factorisation serves all members), so the greedy loop sees
    redistribution effects.  Greedy can be strictly worse than the global
    optimum; when that happens it must be REPORTED, not hidden.
    """
    current = Decision(
        member_ids=ctx.member_ids, actions=tuple(0 for _ in ctx.member_ids)
    )
    best = RetrofitOutcome(current, evaluate(ctx, current))
    while True:
        candidate_best: RetrofitOutcome | None = None
        best_ratio = 0.0
        for idx in range(len(ctx.member_ids)):
            cur_x = current.actions[idx]
            for up in range(cur_x + 1, 4):
                actions = list(current.actions)
                actions[idx] = up
                trial = Decision(member_ids=ctx.member_ids, actions=tuple(actions))
                if not feasible(ctx, trial):
                    continue
                outcome = RetrofitOutcome(trial, evaluate(ctx, trial))
                added = outcome.metrics.cost - best.metrics.cost
                gain = best.objective - outcome.objective
                if added <= 0.0:
                    continue
                ratio = gain / added
                if gain > 1e-12 and ratio > best_ratio:
                    best_ratio = ratio
                    candidate_best = outcome
        if candidate_best is None:
            return best
        current = candidate_best.decision
        best = candidate_best


def _axial_stress_order(ctx: RetrofitContext) -> list[str]:
    """Return members ordered by ``|N_i| / A_i`` at t_target (descending)."""
    from ..limitstates import member_axial_forces

    temps = _scenario_temps(ctx)
    forces = member_axial_forces(list(ctx.nodes), list(ctx.elements), ctx.loads, temps)
    area = {e.id: e.A for e in ctx.elements}
    return sorted(ctx.member_ids, key=lambda m: abs(forces[m]) / area[m], reverse=True)


def robust_strategy(ctx: RetrofitContext) -> RetrofitOutcome:
    """Robust strategy: upgrade the single most-stressed member (|N|/A)."""
    order = _axial_stress_order(ctx)
    actions = dict.fromkeys(ctx.member_ids, 0)
    top = order[0]
    for level in (2, 1):
        trial = dict(actions)
        trial[top] = level
        dec = Decision(
            member_ids=ctx.member_ids, actions=tuple(trial[m] for m in ctx.member_ids)
        )
        if feasible(ctx, dec):
            actions = trial
            break
    dec = Decision(
        member_ids=ctx.member_ids, actions=tuple(actions[m] for m in ctx.member_ids)
    )
    return RetrofitOutcome(dec, evaluate(ctx, dec))


def redundant_strategy(ctx: RetrofitContext) -> RetrofitOutcome:
    """In-space equivalent of a topology-changing 'add redundancy' action.

    LIMITATION (documented): adding a diagonal changes the topology and
    leaves the common decision space, so a like-for-like comparison is
    impossible; the equivalent used here upgrades the member with the
    highest redundancy participation (largest u_max increase when the
    member is removed) to ``x=3``, whose proxy cost ``c_3 L`` equals the
    cost assigned to the added member.
    """
    temps = _scenario_temps(ctx)
    setup = build_engine(list(ctx.nodes), list(ctx.elements), ctx.loads, temps)
    u0 = base_displacement(setup, total_load_vector(list(ctx.nodes), ctx.loads, setup))
    base_umax = float(np.max(np.abs(u0)))
    participation: dict[str, float] = {}
    for e in ctx.elements:
        kept = [x for x in ctx.elements if x.id != e.id]
        try:
            s2 = build_engine(list(ctx.nodes), kept, ctx.loads, temps)
            u2 = base_displacement(
                s2, total_load_vector(list(ctx.nodes), ctx.loads, s2)
            )
            participation[e.id] = float(np.max(np.abs(u2))) - base_umax
        except Exception:  # removal causes a mechanism: maximal participation
            participation[e.id] = float("inf")
    top = max(ctx.member_ids, key=lambda m: participation[m])
    actions = dict.fromkeys(ctx.member_ids, 0)
    for level in (3, 2, 1):
        trial = dict(actions)
        trial[top] = level
        dec = Decision(
            member_ids=ctx.member_ids, actions=tuple(trial[m] for m in ctx.member_ids)
        )
        if feasible(ctx, dec):
            actions = trial
            break
    dec = Decision(
        member_ids=ctx.member_ids, actions=tuple(actions[m] for m in ctx.member_ids)
    )
    return RetrofitOutcome(dec, evaluate(ctx, dec))


def stress_based_strategy(ctx: RetrofitContext) -> RetrofitOutcome:
    """Stress-based strategy: light protection on members in stress order."""
    order = _axial_stress_order(ctx)
    actions = dict.fromkeys(ctx.member_ids, 0)
    for mid in order:
        trial = dict(actions)
        trial[mid] = 1
        dec = Decision(
            member_ids=ctx.member_ids, actions=tuple(trial[m] for m in ctx.member_ids)
        )
        if feasible(ctx, dec):
            actions = trial
        else:
            break
    dec = Decision(
        member_ids=ctx.member_ids, actions=tuple(actions[m] for m in ctx.member_ids)
    )
    return RetrofitOutcome(dec, evaluate(ctx, dec))


STRATEGIES: dict[str, Callable[[RetrofitContext], RetrofitOutcome]] = {
    "greedy": greedy,
    "exhaustive": exhaustive,
    "robust": robust_strategy,
    "redundant": redundant_strategy,
    "stress_based": stress_based_strategy,
}
