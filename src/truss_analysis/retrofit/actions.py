"""Decision -> physics map (prompt-06, task 9).

The retrofit decision variable is ``x_i in {0, 1, 2, 3}`` for every member
and every strategy (prompt-06 task 10).  Each level maps to an exact physical
effect, documented here and asserted in tests:

===  ======  ==========================================================
 x_i  action  physical effect
===  ======  ==========================================================
 0    none    -
 1    fire protection 15 mm   theta_eff = max(20, theta_scen - 200)
 2    fire protection 30 mm   theta_eff = max(20, theta_scen - 350)
 3    section enlargement +30%  A -> 1.3 A, I from the prompt-5 section model
===  ======  ==========================================================

Note on ``x=3``: enlarging the area changes BOTH the axial stiffness and the
buckling capacity; ``I`` is recomputed through
:func:`truss_analysis.sections.idealised_square_hss` (the prompt-5 model),
never through the legacy ``A**2/12`` shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Dict, Mapping, Sequence, Tuple

from truss_analysis.model import Element
from truss_analysis.sections import idealised_square_hss

__all__ = [
    "ACTION_EFFECTS",
    "DECISION_ACTIONS",
    "DECISION_LEVELS",
    "ActionEffect",
    "apply_decision",
]

DECISION_LEVELS: Tuple[int, ...] = (0, 1, 2, 3)

DECISION_ACTIONS: Dict[int, str] = {
    0: "none",
    1: "fire protection 15 mm",
    2: "fire protection 30 mm",
    3: "section enlargement +30%",
}


@dataclass(frozen=True)
class ActionEffect:
    """Exact physical effect of one decision level."""

    theta_offset: float  # [degC] added to the scenario temperature
    area_factor: float  # multiplicative factor on A (and I via section model)


ACTION_EFFECTS: Dict[int, ActionEffect] = {
    0: ActionEffect(theta_offset=0.0, area_factor=1.0),
    1: ActionEffect(theta_offset=-200.0, area_factor=1.0),
    2: ActionEffect(theta_offset=-350.0, area_factor=1.0),
    3: ActionEffect(theta_offset=0.0, area_factor=1.3),
}

_AMBIENT = 20.0


def apply_decision(
    elements: Sequence[Element],
    temps: Mapping[str, float],
    decision: Mapping[str, int],
    thickness_ratio: float = 25.0,
) -> Tuple[list, Dict[str, float]]:
    """Return ``(elements', temps')`` with the decision applied.

    ``x=3`` recomputes ``I_sec`` through the prompt-5 idealised square HSS
    model at the enlarged area (same b/t ratio), so buckling capacity follows
    the same section physics as the base design.
    """
    new_elements = []
    for e in elements:
        x = int(decision.get(e.id, 0))
        if x not in DECISION_LEVELS:
            msg = f"decision for {e.id} out of space: {x}"
            raise ValueError(msg)
        effect = ACTION_EFFECTS[x]
        elem = e
        if effect.area_factor != 1.0:
            new_area = e.A * effect.area_factor
            new_i = idealised_square_hss(new_area, thickness_ratio).i_sec
            elem = dc_replace(e, A=new_area, I_sec=new_i)
        new_elements.append(elem)
    new_temps = {
        e.id: max(
            _AMBIENT,
            float(temps[e.id])
            + ACTION_EFFECTS[int(decision.get(e.id, 0))].theta_offset,
        )
        for e in elements
    }
    return new_elements, new_temps
