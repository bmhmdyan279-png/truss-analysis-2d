"""Force-based limit states at elevated temperature.

Consumes the temperature-dependent material reduction factors of
:mod:`truss_analysis.material` to build member and system limit states:

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
* Two-component criticality index:
  ``CI_i = max(u_ratio - 1, DCR_ratio - 1)`` with both components reported
  separately plus a ``governing`` field ("displacement" | "buckling" |
  "yield").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .criticality.engine import (
    base_displacement,
    build_engine,
    ci_sweep,
    load_vector,
)
from .material.steel_eurocode import k_E as eurocode_k_E
from .material.steel_eurocode import k_y as eurocode_k_y
from .model import Element, Node
from .sections import (
    LAMBDA_BAR_BUCKLING_LIMIT,
    buckling_reduction_factor,
    euler_buckling_load,
    non_dimensional_slenderness,
)

__all__ = [
    "DEFAULT_BUCKLING_CURVE",
    "GAMMA_M_FIRE",
    "BucklingModel",
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

#: Default flexural buckling curve. Curve ``c`` suits the thin-walled and
#: cold-formed hollow sections this library idealises, and is the
#: conservative choice among the common ones.
DEFAULT_BUCKLING_CURVE = "c"


class BucklingModel(str, Enum):
    """Compression-member capacity model used by the fire limit state.

    ``EUROCODE_CHI``
        EN 1993-1-2:2005 4.2.3.1: ``N_b,fi,theta,Rd = chi A f_y,theta /
        gamma_M,fi`` with ``chi`` from the buckling curve and the *fire*
        slenderness ``lambda_bar_theta = sqrt(A f_y,theta / N_cr)``. This is
        the default and the code-correct model. ``chi`` interpolates between
        the two physical limits, so it is ``1`` for a stocky member (yield
        governed) and tends to ``N_cr / (A f_y,theta)`` for a slender one
        (Euler governed).

    ``EULER_ONLY``
        The historical model, ``capacity = min(P_cr, N_Rd)``. Retained only so
        previously published numbers stay reproducible. It overestimates the
        capacity of intermediate-slenderness members -- by roughly 15% at
        ``lambda_bar ~ 2`` and far more near ``lambda_bar ~ 1`` -- because it
        ignores residual stresses and initial out-of-straightness. That makes
        the reported DCR and the critical temperature optimistic.
    """

    EUROCODE_CHI = "eurocode_chi"
    EULER_ONLY = "euler_only"


class Governing(str, Enum):
    """Which physical limit state produced the CI of a member."""

    DISPLACEMENT = "displacement"
    BUCKLING = "buckling"
    YIELD = "yield"


@dataclass(frozen=True)
class MemberLimitState:
    """Force-based limit-state state of one member at one temperature.

    Attributes
    ----------
    p_cr : float or None
        Euler elastic critical load ``N_cr = pi^2 E_theta I / (k L)^2`` [N],
        or ``None`` for a tension member. This is an *input* to the
        slenderness, not itself the design capacity.
    n_rd : float
        Yield (cross-section) resistance ``k_y,theta f_y A / gamma_M,fi`` [N].
    capacity : float
        The design resistance the DCR is actually formed against. Under
        :attr:`BucklingModel.EUROCODE_CHI` this is ``chi A f_y,theta /
        gamma_M,fi`` for compression and ``n_rd`` for tension.
    chi : float or None
        Flexural buckling reduction factor, or ``None`` for a tension member.
    lambda_bar : float or None
        Non-dimensional fire slenderness, or ``None`` for a tension member.
    capacity_governing : Governing
        ``YIELD`` when ``lambda_bar`` is at or below
        :data:`~truss_analysis.sections.LAMBDA_BAR_BUCKLING_LIMIT` (0.2), so
        buckling need not be considered; ``BUCKLING`` otherwise.
    """

    member_id: str
    temperature: float
    axial_force: float
    compression: bool
    p_cr: float | None
    n_rd: float
    dcr: float
    capacity_governing: Governing  # buckling | yield
    capacity: float = 0.0
    chi: float | None = None
    lambda_bar: float | None = None
    buckling_model: BucklingModel = BucklingModel.EUROCODE_CHI
    buckling_curve: str = DEFAULT_BUCKLING_CURVE


@dataclass(frozen=True)
class ComponentCI:
    """Two-component criticality index with both components exposed."""

    member_id: str
    ci: float
    u_component: float
    dcr_component: float
    governing: Governing


@dataclass(frozen=True)
class TwoComponentResult:
    """Container for a two-component CI sweep."""

    components: dict[str, ComponentCI]
    ci_values: dict[str, float]
    governing: dict[str, str]
    u_max_base: float


def member_axial_forces(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
) -> dict[str, float]:
    """Member axial forces [N] (tension positive) at the given temperatures."""
    setup = build_engine(nodes, elements, loads, temps)
    u = base_displacement(setup, load_vector(nodes, loads, setup.free_dofs))
    strains = setup.b_free @ u
    forces = setup.k_axial * strains
    return {eid: float(forces[i]) for i, eid in enumerate(setup.ids)}


def yield_capacity(area: float, f_y: float, temperature: float) -> float:
    """``N_Rd(T) = k_y(T) f_y A / gamma_M,fi`` [N]."""
    return float(eurocode_k_y(temperature)) * f_y * area / GAMMA_M_FIRE


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
    buckling_model: BucklingModel = BucklingModel.EUROCODE_CHI,
    buckling_curve: str = DEFAULT_BUCKLING_CURVE,
) -> MemberLimitState:
    """Build the fire limit state of one member at one temperature.

    Under the default :attr:`BucklingModel.EUROCODE_CHI` a compression member
    is checked against EN 1993-1-2:2005 4.2.3.1,

    .. code-block:: text

        lambda_bar_theta = sqrt(A * f_y,theta / N_cr),
        N_cr             = pi^2 * E_theta * I / (k L)^2,
        f_y,theta        = k_y,theta * f_y,      E_theta = k_E,theta * E,
        N_b,fi,theta,Rd  = chi * A * f_y,theta / gamma_M,fi.

    Because ``chi`` is derived from the same ``N_cr`` and the same
    ``f_y,theta``, this single expression covers both limits: it reduces to
    the yield resistance when the member is stocky and approaches the Euler
    load when it is slender. It therefore replaces the previous
    ``min(P_cr, N_Rd)`` approximation, which took the smaller of two
    asymptotes and so overestimated the capacity everywhere in between.

    ``lambda_bar_theta`` grows as the member heats, because ``k_E`` falls
    faster than ``k_y``. A member that is comfortably stocky at 20 degC can
    become buckling-governed at 600 degC, which the ``min()`` form could not
    express.
    """
    compression = axial_force < 0.0
    e_t = float(eurocode_k_E(temperature)) * youngs
    n_rd = yield_capacity(area, f_y, temperature)

    chi: float | None = None
    lambda_bar: float | None = None
    p_cr: float | None = None

    if compression:
        p_cr = euler_buckling_load(i_sec, length, e_t, k_factor)
        # Recover f_y,theta = k_y,theta * f_y from the yield resistance, since
        # n_rd = k_y,theta * f_y * A / gamma_M,fi by construction.
        f_y_theta = n_rd * GAMMA_M_FIRE / area
        lambda_bar = non_dimensional_slenderness(area, f_y_theta, p_cr)
        if buckling_model is BucklingModel.EUROCODE_CHI:
            chi = buckling_reduction_factor(lambda_bar, buckling_curve, fire=True)
            # N_b,fi,theta,Rd = chi * A * f_y,theta / gamma_M,fi == chi * n_rd
            capacity = chi * n_rd
        else:
            capacity = min(p_cr, n_rd)
    else:
        capacity = n_rd

    # Buckling need not be considered below lambda_bar = 0.2
    # (EN 1993-1-1:2005 6.3.1(4)); the member is yield-governed there.
    stocky = lambda_bar is not None and lambda_bar <= LAMBDA_BAR_BUCKLING_LIMIT
    if not compression or stocky:
        cap_gov = Governing.YIELD
    elif buckling_model is BucklingModel.EULER_ONLY and p_cr is not None:
        cap_gov = Governing.BUCKLING if p_cr <= n_rd else Governing.YIELD
    else:
        cap_gov = Governing.BUCKLING

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
        capacity=capacity,
        chi=chi,
        lambda_bar=lambda_bar,
        buckling_model=buckling_model,
        buckling_curve=buckling_curve,
    )


def dcr_field(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    f_y: float,
    buckling_model: BucklingModel = BucklingModel.EUROCODE_CHI,
    buckling_curve: str = DEFAULT_BUCKLING_CURVE,
) -> dict[str, MemberLimitState]:
    """DCR state of every member at the given member temperatures.

    Parameters
    ----------
    nodes, elements, loads, temps, f_y
        Model, loading, per-member temperature field [degC] and ambient yield
        strength ``f_y`` [Pa].
    buckling_model : BucklingModel, default EUROCODE_CHI
        Compression capacity model; see :class:`BucklingModel`.
    buckling_curve : str, default "c"
        Flexural buckling curve, only used by ``EUROCODE_CHI``.

    Returns
    -------
    dict[str, MemberLimitState]
        Limit state per member id.
    """
    forces = member_axial_forces(nodes, elements, loads, temps)
    out: dict[str, MemberLimitState] = {}
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
            buckling_model,
            buckling_curve,
        )
    return out


def _uniform_temps(elements: Sequence[Element], temperature: float) -> dict[str, float]:
    return {e.id: float(temperature) for e in elements}


def member_critical_temperature(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    member_id: str,
    f_y: float,
    temp_grid: Sequence[float] = _TEMP_GRID,
    buckling_model: BucklingModel = BucklingModel.EUROCODE_CHI,
    buckling_curve: str = DEFAULT_BUCKLING_CURVE,
) -> float | None:
    """Smallest uniform temperature at which ``DCR_member >= 1``.

    Root-found by scanning the discrete reduction curves and linearly
    interpolating the DCR = 1 crossing between grid points.  ``None`` when the
    member never reaches DCR = 1 within [20, 1200] degC.

    Because ``lambda_bar_theta`` grows as ``k_E`` falls faster than ``k_y``,
    the reported critical temperature depends on the compression capacity
    model; see :class:`BucklingModel`.
    """
    prev_t: float | None = None
    prev_dcr: float | None = None
    for t in temp_grid:
        states = dcr_field(
            nodes,
            elements,
            loads,
            _uniform_temps(elements, t),
            f_y,
            buckling_model,
            buckling_curve,
        )
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
    buckling_model: BucklingModel = BucklingModel.EUROCODE_CHI,
    buckling_curve: str = DEFAULT_BUCKLING_CURVE,
) -> float:
    """Highest scanned uniform temperature with no member at DCR >= 1.

    Grid-based by construction (DCR(T) is not monotone because forces
    redistribute as stiffnesses degrade); the resolution is the grid step.
    Returns ``temp_grid[0]`` when even the coldest scan already fails and
    ``temp_grid[-1]`` when nothing fails within the range.
    """
    last_safe = float(temp_grid[0])
    for t in temp_grid:
        states = dcr_field(
            nodes,
            elements,
            loads,
            _uniform_temps(elements, t),
            f_y,
            buckling_model,
            buckling_curve,
        )
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
    and the governing limit state are reported separately.
    """
    setup = build_engine(nodes, elements, loads, temps)
    f_free = load_vector(nodes, loads, setup.free_dofs)
    u = base_displacement(setup, f_free)
    u_max_base = float(np.max(np.abs(u)))
    sweep = ci_sweep(setup, u, alpha)
    # Cold reference state (20 degC, same geometry/loads): the DCR component
    # compares against the COLD capacity-demand state so that temperature
    # degradation does NOT cancel out of the ratio. This reference is the
    # only reading under which the governing component can switch with
    # temperature; see docs/theory.md for the limit-state definitions.
    forces_cold = member_axial_forces(
        nodes, elements, loads, {e.id: 20.0 for e in elements}
    )

    components: dict[str, ComponentCI] = {}
    ci_values: dict[str, float] = {}
    governing: dict[str, str] = {}
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
