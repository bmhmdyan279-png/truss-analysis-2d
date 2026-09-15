"""Decision -> physics map for retrofit triage.

The retrofit decision variable is ``x_i in {0, 1, 2, 3}`` for every member
and every strategy.  Each level maps to an exact physical effect, documented
here and asserted in tests:

===  ======  ==========================================================
 x_i  action  physical effect
===  ======  ==========================================================
 0    none    -
 1    fire protection 15 mm   theta_eff = max(20, theta_scen - 200)
 2    fire protection 30 mm   theta_eff = max(20, theta_scen - 350)
 3    section enlargement +30%  A -> 1.3 A, I from the section model
===  ======  ==========================================================

Note on ``x=3``: enlarging the area changes BOTH the axial stiffness and the
buckling capacity; ``I`` is recomputed through
:func:`truss_analysis.sections.idealised_square_hss` (the idealised square
HSS model), never through a solid-section ``A**2/12`` shortcut.

Note on ``x=1``/``x=2`` (round-5 audit, C7-12): the fixed ``theta_offset``
values are a **first-order decision proxy**, not a thermal model. A real
protection layer's temperature drop depends on the section factor
``A_m/V``, the insulation thickness/conductivity and the exposure time
(see :mod:`truss_analysis.thermal.fire_curve` for the physics-based lumped
capacitance path). The offsets here answer the triage question "which
members deserve protection at all, under a common proxy budget?" -- they
deliberately do NOT claim a specific insulation design. At scenario
temperatures above ~700 degC a constant offset can misestimate the
protected temperature by more than 150 degC; treat the ranking, not the
absolute protected temperature, as the output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace

from ..model import Element
from ..sections import idealised_square_hss

__all__ = [
    "ACTION_EFFECTS",
    "DECISION_ACTIONS",
    "DECISION_LEVELS",
    "ActionEffect",
    "apply_decision",
]

DECISION_LEVELS: tuple[int, ...] = (0, 1, 2, 3)

DECISION_ACTIONS: dict[int, str] = {
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


ACTION_EFFECTS: dict[int, ActionEffect] = {
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
) -> tuple[list[Element], dict[str, float]]:
    """Return ``(elements', temps')`` with the decision applied.

    ``x=3`` recomputes ``I_sec`` through the idealised square HSS model at
    the enlarged area (same b/t ratio), so buckling capacity follows the
    same section physics as the base design.
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
