"""Explicit PROXY cost models for retrofit triage (prompt-06, task 12).

All costs here are **proxies** proportional to member length, used only to
make strategy comparison possible at screening level (critic 2 in ``3.md``,
item 2).  They are NOT industrial cost models; the naming and docstrings say
so on purpose.

Three scenarios (prompt-06 task 12):

* ``linear``   — proposal §5.5 weights (0, 1500, 3000, 5000) per metre;
* ``quadratic``— super-linear protection costs (0, 1500, 6000, 13500) = 1500 x^2;
* ``step``     — stepped procurement costs (0, 2000, 4000, 8000).

Budget: ``sum_i c_{x_i} L_i <= 0.2 * C_base`` with ``C_base`` the proxy cost
of the unprotected steel structure (``STEEL_COST_PROXY_PER_M * sum L_i``).
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

__all__ = [
    "COST_SCENARIOS",
    "STEEL_COST_PROXY_PER_M",
    "base_cost",
    "budget_for",
    "decision_cost",
]

#: per-metre proxy weights per decision level, per cost scenario
COST_SCENARIOS: Dict[str, Tuple[float, float, float, float]] = {
    "linear": (0.0, 1500.0, 3000.0, 5000.0),
    "quadratic": (0.0, 1500.0, 6000.0, 13500.0),
    "step": (0.0, 2000.0, 4000.0, 8000.0),
}

#: proxy cost of unprotected steel per metre [currency/m]
STEEL_COST_PROXY_PER_M = 1000.0

BUDGET_FRACTION = 0.2


def base_cost(lengths: Mapping[str, float]) -> float:
    """Proxy cost of the unprotected structure: ``c_steel * sum(L_i)``."""
    return STEEL_COST_PROXY_PER_M * sum(lengths.values())


def budget_for(
    lengths: Mapping[str, float], fraction: float = BUDGET_FRACTION
) -> float:
    """Allowed retrofit budget: ``fraction * C_base``."""
    return fraction * base_cost(lengths)


def decision_cost(
    decision: Mapping[str, int],
    lengths: Mapping[str, float],
    scenario: str = "linear",
) -> float:
    """Proxy cost ``sum_i c_{x_i} L_i`` of a decision vector."""
    weights = COST_SCENARIOS[scenario]
    total = 0.0
    for mid, x in decision.items():
        total += weights[int(x)] * lengths[mid]
    return total


def scenario_names() -> Sequence[str]:
    return tuple(COST_SCENARIOS)
