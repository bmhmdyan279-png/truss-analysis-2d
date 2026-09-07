"""Force-based limit states at elevated temperature (prompt-06, part A).

Closes CONTEXT_LOCK §4.5 B6: ``k_y(T)`` / ``f_y(T)`` were defined by the SSOT
but never consumed by any computational path.  This module puts them to work:

* Euler buckling capacity ``P_cr,i(T) = pi^2 E_i(T) I_i / (k L_i)^2`` for
  compression members only (``E_i(T) = k_E(T) E``).
* Axial yield capacity ``N_Rd,i(T) = k_y(T) f_y A_i / gamma_M,fi`` with
  ``gamma_M,fi = 1.0`` — EN 1993-1-2:2005 clause 2.3 NOTE ("the use of
  gamma_M,fi = 1.0 is recommended"), verified verbatim from the standard text.
* Demand-capacity ratio ``DCR_i(T) = |N_i(T)| / min(P_cr,i, N_Rd,i)`` — the
  minimum of buckling and yield because either can govern a fire-exposed
  steel truss member.
* Member critical temperature ``theta_cr,i`` (DCR = 1 crossing, root-found on
  the discrete reduction curves) and system critical temperature
  ``theta*_sys`` (highest scanned temperature with no member at DCR >= 1).
* Two-component CI per proposal §5.3:
  ``CI_i = max(u_ratio - 1, DCR_ratio - 1)`` with both components reported
  separately plus a ``governing`` field ("displacement" | "buckling" |
  "yield").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from truss_analysis.criticality.engine import (
    base_displacement,
    build_engine,
    ci_sweep,
    load_vector,
)
from truss_analysis.material.steel_eurocode import k_E as ssot_k_E
from truss_analysis.material.steel_eurocode import k_y as ssot_k_y
from truss_analysis.model import Element, Node
from truss_analysis.sections import euler_buckling_load

__all__ = [
    "GAMMA_M_FIRE",
    "ComponentCI",
    "Governing",
    "MemberLimitState",
    "TwoComponentResult",
    "ci_two_component",
    "dcr_field",
    "member_axial_forces",
    "member_critical_temperature",
    "system_critical_temperature",
    "yield_capacity",
]

#: EN 1993-1-2:2005 clause 2.3 NOTE: gamma_M,fi = 1.0 recommended (verified).
GAMMA_M_FIRE = 1.0

_DCR_BASE_TOL = 1e-12
_TEMP_GRID = tuple(range(20, 1201, 25))


class Governing(str, Enum):
    """Which physical limit state produced the CI of a member."""

    DISPLACEMENT = "displacement"
    BUCKLING = "buckling"
    YIELD = "yield"


@dataclass(frozen=True)
class MemberLimitState:
    """Force-based limit-state state of one member at one temperature."""

    member_id: str
    temperature: float
    axial_force: float
    compression: bool
    p_cr: Optional[float]
    n_rd: float
    dcr: float
    capacity_governing: Governing  # buckling | yield (which capacity is smaller)


@dataclass(frozen=True)
class ComponentCI:
    """Two-component CI with both components exposed (proposal §5.3)."""

    member_id: str
    ci: float
    u_component: float
    dcr_component: float
    governing: Governing


@dataclass(frozen=True)
class TwoComponentResult:
    """Container for a two-component CI sweep."""

    components: Dict[str, ComponentCI]
    ci_values: Dict[str, float]
    governing: Dict[str, str]
    u_max_base: float


def member_axial_forces(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
) -> Dict[str, float]:
    """Member axial forces [N] (tension positive) at the given temperatures."""
    setup = build_engine(nodes, elements, loads, temps)
    u = base_displacement(setup, load_vector(nodes, loads, setup.free_dofs))
    strains = setup.b_free @ u
    forces = setup.k_axial * strains
    return {eid: float(forces[i]) for i, eid in enumerate(setup.ids)}


def yield_capacity(area: float, f_y: float, temperature: float) -> float:
    """``N_Rd(T) = k_y(T) f_y A / gamma_M,fi`` [N]."""
    return float(ssot_k_y(temperature)) * f_y * area / GAMMA_M_FIRE


def _member_limit_state(
    member_id: str,
    temperature: float,
    axial_force: float,
    area: float,
    i_sec: float,
    length: float,
    youngs: float,
    k_factor: float,
    f_y: float,
) -> MemberLimitState:
    compression = axial_force < 0.0
    e_t = float(ssot_k_E(temperature)) * youngs
    p_cr: Optional[float] = (
        euler_buckling_load(i_sec, length, e_t, k_factor) if compression else None
    )
    n_rd = yield_capacity(area, f_y, temperature)
    capacity = min(p_cr, n_rd) if p_cr is not None else n_rd
    cap_gov = (
        Governing.BUCKLING if (p_cr is not None and p_cr <= n_rd) else Governing.YIELD
    )
    dcr = abs(axial_force) / capacity if capacity > 0.0 else float("inf")
    return MemberLimitState(
        member_id=member_id,
        temperature=temperature,
        axial_force=axial_force,
        compression=compression,
        p_cr=p_cr,
        n_rd=n_rd,
        dcr=dcr,
        capacity_governing=cap_gov,
    )


def dcr_field(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    f_y: float,
) -> Dict[str, MemberLimitState]:
    """DCR state of every member at the given member temperatures."""
    forces = member_axial_forces(nodes, elements, loads, temps)
    out: Dict[str, MemberLimitState] = {}
    for e in elements:
        ni = next(n for n in nodes if n.id == e.node_i)
        nj = next(n for n in nodes if n.id == e.node_j)
        length = float(np.hypot(nj.x - ni.x, nj.y - ni.y))
        out[e.id] = _member_limit_state(
            e.id,
            float(temps[e.id]),
            forces[e.id],
            e.A,
            e.I_sec,
            length,
            e.E,
            e.effective_length_factor,
            f_y,
        )
    return out


def _uniform_temps(elements: Sequence[Element], temperature: float) -> Dict[str, float]:
    return {e.id: float(temperature) for e in elements}


def member_critical_temperature(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    member_id: str,
    f_y: float,
    temp_grid: Sequence[float] = _TEMP_GRID,
) -> Optional[float]:
    """Smallest uniform temperature at which ``DCR_member >= 1``.

    Root-found by scanning the discrete reduction curves and linearly
    interpolating the DCR = 1 crossing between grid points.  ``None`` when the
    member never reaches DCR = 1 within [20, 1200] degC.
    """
    prev_t: Optional[float] = None
    prev_dcr: Optional[float] = None
    for t in temp_grid:
        states = dcr_field(nodes, elements, loads, _uniform_temps(elements, t), f_y)
        dcr = states[member_id].dcr
        if dcr >= 1.0:
            if prev_t is None or prev_dcr is None:
                return float(t)
            denom = dcr - prev_dcr
            frac = (1.0 - prev_dcr) / denom if denom > 0 else 0.0
            return float(prev_t + frac * (t - prev_t))
        prev_t, prev_dcr = float(t), dcr
    return None


def system_critical_temperature(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    f_y: float,
    temp_grid: Sequence[float] = _TEMP_GRID,
) -> float:
    """Highest scanned uniform temperature with no member at DCR >= 1.

    Grid-based by construction (DCR(T) is not monotone because forces
    redistribute as stiffnesses degrade); the resolution is the grid step.
    Returns ``temp_grid[0]`` when even the coldest scan already fails and
    ``temp_grid[-1]`` when nothing fails within the range.
    """
    last_safe = float(temp_grid[0])
    for t in temp_grid:
        states = dcr_field(nodes, elements, loads, _uniform_temps(elements, t), f_y)
        if any(s.dcr >= 1.0 for s in states.values()):
            return last_safe
        last_safe = float(t)
    return last_safe


def ci_two_component(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    alpha: float,
    f_y: float,
) -> TwoComponentResult:
    """Two-component CI: ``max(u_ratio - 1, DCR_ratio - 1)`` per member.

    The displacement component comes from the rank-1 engine sweep; the DCR
    component compares the perturbed member force (member i softened by
    ``alpha``) against the temperature-dependent capacity.  Both components
    and the governing limit state are reported (proposal §5.3).
    """
    setup = build_engine(nodes, elements, loads, temps)
    f_free = load_vector(nodes, loads, setup.free_dofs)
    u = base_displacement(setup, f_free)
    u_max_base = float(np.max(np.abs(u)))
    sweep = ci_sweep(setup, u, alpha)
    # Cold reference state (20 degC, same geometry/loads): the DCR component
    # compares against the COLD capacity-demand state so that temperature
    # degradation does NOT cancel out of the ratio (proposal §5.3 is silent
    # on the reference; this reading is the only one under which the
    # governing component can switch with temperature — recorded in
    # vault/reports/06_limitstates_uncertainty_retrofit.md).
    forces_cold = member_axial_forces(
        nodes, elements, loads, {e.id: 20.0 for e in elements}
    )

    components: Dict[str, ComponentCI] = {}
    ci_values: Dict[str, float] = {}
    governing: Dict[str, str] = {}
    for i, eid in enumerate(setup.ids):
        u_comp = sweep.ci_values[eid]
        elem = elements[i]
        length = _length(nodes, elem)
        state_cold = _member_limit_state(
            eid,
            20.0,
            forces_cold[eid],
            elem.A,
            elem.I_sec,
            length,
            elem.E,
            elem.effective_length_factor,
            f_y,
        )
        # perturbed axial force in member i: softened stiffness x compatibility
        k_pert = alpha * setup.k_axial[i]
        n_pert = float(k_pert * float(setup.b_free[i] @ sweep.u_pert[:, i]))
        state_pert = _member_limit_state(
            eid,
            float(temps[eid]),
            n_pert,
            elem.A,
            elem.I_sec,
            length,
            elem.E,
            elem.effective_length_factor,
            f_y,
        )
        if state_cold.dcr > _DCR_BASE_TOL:
            dcr_comp = state_pert.dcr / state_cold.dcr - 1.0
        else:
            dcr_comp = 0.0
        ci = max(u_comp, dcr_comp)
        if dcr_comp > u_comp:
            gov = state_pert.capacity_governing
        else:
            gov = Governing.DISPLACEMENT
        components[eid] = ComponentCI(
            member_id=eid,
            ci=ci,
            u_component=u_comp,
            dcr_component=dcr_comp,
            governing=gov,
        )
        ci_values[eid] = ci
        governing[eid] = gov.value
    return TwoComponentResult(
        components=components,
        ci_values=ci_values,
        governing=governing,
        u_max_base=u_max_base,
    )


def _length(nodes: Sequence[Node], elem: Element) -> float:
    ni = next(n for n in nodes if n.id == elem.node_i)
    nj = next(n for n in nodes if n.id == elem.node_j)
    return float(np.hypot(nj.x - ni.x, nj.y - ni.y))
