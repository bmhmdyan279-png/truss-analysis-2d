"""Retrofit triage with a documented decision -> physics map (prompt-06, C)."""

from __future__ import annotations

from .actions import (
    ACTION_EFFECTS,
    DECISION_ACTIONS,
    DECISION_LEVELS,
    ActionEffect,
    apply_decision,
)
from .costs import (
    COST_SCENARIOS,
    STEEL_COST_PROXY_PER_M,
    base_cost,
    budget_for,
    decision_cost,
)
from .strategies import (
    STRATEGIES,
    Decision,
    RetrofitContext,
    RetrofitMetrics,
    RetrofitOutcome,
    enumerate_space,
    evaluate,
    exhaustive,
    feasible,
    greedy,
    make_context,
    redundant_strategy,
    robust_strategy,
    stress_based_strategy,
)

__all__ = [
    "ACTION_EFFECTS",
    "COST_SCENARIOS",
    "DECISION_ACTIONS",
    "DECISION_LEVELS",
    "STEEL_COST_PROXY_PER_M",
    "STRATEGIES",
    "ActionEffect",
    "Decision",
    "RetrofitContext",
    "RetrofitMetrics",
    "RetrofitOutcome",
    "apply_decision",
    "base_cost",
    "budget_for",
    "decision_cost",
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
