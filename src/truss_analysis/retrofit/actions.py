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
HSS model), never through a solid-section ``A**2/12`` shortcut.  The
width-to-thickness ratio used for the enlargement is *recovered from the
member's own* ``(A, I_sec)`` pair by :func:`member_thickness_ratio`, so a
model with mixed sections keeps its mixed sections through the retrofit
(round-6 audit C7; the ratio used to be hard-coded at 25 for every member).

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

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace

from ..exceptions import BucklingCheckWarning
from ..model import Element
from ..sections import (
    DEFAULT_THICKNESS_RATIO,
    idealised_square_hss,
    thickness_ratio_from_section,
)

__all__ = [
    "ACTION_EFFECTS",
    "DECISION_ACTIONS",
    "DECISION_LEVELS",
    "ActionEffect",
    "apply_decision",
    "member_thickness_ratio",
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


def member_thickness_ratio(
    element: Element,
    fallback: float = DEFAULT_THICKNESS_RATIO,
    warn: bool = True,
) -> float:
    """Recover a member's own ``b/t`` from its ``(A, I_sec)`` pair (C7).

    Enlarging a section has to preserve the wall slenderness the member was
    actually designed with.  Hard-coding ``b/t = 25`` -- what this function
    replaced -- silently over- or under-states the buckling capacity of every
    member whose real ratio differs, and the error goes the unsafe way for
    slender (high ``b/t``) sections: the enlarged ``I`` comes out too large,
    so the retrofit looks better than it is.

    ``I / A^2`` is a function of ``r = b/t`` alone for the idealised square
    HSS and is strictly monotone in ``r``, so the recovery is exact and
    unique; see :func:`truss_analysis.sections.thickness_ratio_from_section`.

    Parameters
    ----------
    element : Element
        Member whose section is to be characterised.
    fallback : float, default DEFAULT_THICKNESS_RATIO
        Ratio to use when the member carries no usable ``I_sec``.
    warn : bool, default True
        Issue a :class:`~truss_analysis.exceptions.BucklingCheckWarning`
        when the fallback is used, so an assumed ratio is never mistaken for
        a derived one.

    Returns
    -------
    float
        The recovered ``b/t``, or ``fallback`` if it cannot be recovered.
    """
    reason = ""
    if element.I_sec > 0.0:
        try:
            return thickness_ratio_from_section(element.A, element.I_sec)
        except ValueError:
            reason = "its section is not representable as an idealised square HSS"
    else:
        reason = "I_sec <= 0"
    if warn:
        warnings.warn(
            f"BucklingCheckWarning: member {element.id}: cannot recover b/t "
            f"from its section ({reason}); assuming b/t = {fallback:g} for the "
            "enlarged section. The resulting I_sec is an assumption, not a "
            "derivation -- supply a real I_sec to remove it.",
            BucklingCheckWarning,
            stacklevel=2,
        )
    return float(fallback)


def apply_decision(
    elements: Sequence[Element],
    temps: Mapping[str, float],
    decision: Mapping[str, int],
    thickness_ratio: float | None = None,
) -> tuple[list[Element], dict[str, float]]:
    """Return ``(elements', temps')`` with the decision applied.

    ``x=3`` recomputes ``I_sec`` through the idealised square HSS model at
    the enlarged area, so buckling capacity follows the same section physics
    as the base design.

    Parameters
    ----------
    elements : Sequence[Element]
        Base-design members.
    temps : Mapping[str, float]
        Scenario steel temperatures [degC] keyed by member id.
    decision : Mapping[str, int]
        Chosen action level per member id; absent members take level 0.
    thickness_ratio : float or None, optional
        ``None`` (the default) recovers each member's own ``b/t`` from its
        ``(A, I_sec)`` pair via :func:`member_thickness_ratio`, so a
        heterogeneous model keeps its heterogeneity through the retrofit.
        Pass an explicit value to force one ratio on every member -- useful
        for a parametric study, wrong for a real one.

    Returns
    -------
    tuple[list[Element], dict[str, float]]
        The modified members and their effective temperatures.

    Raises
    ------
    ValueError
        If a decision level is outside :data:`DECISION_LEVELS`.
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
            ratio = (
                member_thickness_ratio(e)
                if thickness_ratio is None
                else float(thickness_ratio)
            )
            new_i = idealised_square_hss(new_area, ratio).i_sec
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
