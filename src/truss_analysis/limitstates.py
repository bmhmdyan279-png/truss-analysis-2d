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

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .criticality.engine import (
    MechanismError,
    base_displacement,
    build_engine,
    ci_sweep,
    load_vector,
    member_forces,
    total_load_vector,
)
from .criticality.scenarios import T_AMBIENT
from .exceptions import BucklingCheckWarning
from .material.steel_eurocode import k_E as eurocode_k_E
from .material.steel_eurocode import k_y as eurocode_k_y
from .model import Element, Node
from .sections import (
    LAMBDA_BAR_BUCKLING_LIMIT,
    buckling_reduction_factor,
    euler_buckling_load,
    non_dimensional_slenderness,
)
from .stability import linearized_buckling_load_factor

__all__ = [
    "DEFAULT_BUCKLING_CURVE",
    "GAMMA_M_FIRE",
    "BucklingModel",
    "ComponentCI",
    "CriticalTemperatureResult",
    "FailureMode",
    "Governing",
    "MemberLimitState",
    "TwoComponentResult",
    "UniformForceScan",
    "ci_two_component",
    "dcr_field",
    "member_axial_forces",
    "member_critical_temperature",
    "member_critical_temperature_detailed",
    "system_critical_temperature",
    "system_critical_temperature_detailed",
    "yield_capacity",
]

#: EN 1993-1-2:2005 clause 2.3 NOTE: gamma_M,fi = 1.0 recommended (verified).
GAMMA_M_FIRE = 1.0

_DCR_BASE_TOL = 1e-12
_TEMP_GRID = tuple(range(20, 1201, 25))

#: Bisection refinement of a bracketed DCR = 1 crossing: the interval is
#: halved on exact ``UniformForceScan.forces_at`` evaluations (O(m) each, no
#: refactorisation) until it is at most ``_BISECT_XTOL`` degC wide. The
#: returned temperature is the upper end, so ``DCR(theta) >= 1`` holds by
#: construction -- the reported crossing is conservative by at most the
#: tolerance instead of carrying a linear-interpolation error of the whole
#: grid step on a curved DCR(T).
_BISECT_XTOL = 1e-3
_BISECT_MAX_ITER = 60

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


class FailureMode(str, Enum):
    """What physical event produced a reported critical temperature.

    ``MATERIAL``
        A member limit state was reached: ``DCR(T) >= 1`` with finite
        stiffness (yield or buckling governed, per :class:`Governing`).

    ``STIFFNESS_COLLAPSE``
        The Eurocode material law lost all stiffness (``k_E(T) <= 0`` at the
        1200 degC table endpoint): the structure collapsed as a system,
        *without* any member necessarily reaching its own DCR = 1 first.
        Reporting such a temperature as an ordinary "critical temperature"
        is misleading in engineering reports -- the distinction the round-5
        audit asked for -- so the detailed scans label it explicitly.

    ``NONE``
        No failure occurred within the scanned temperature range.
    """

    MATERIAL = "material"
    STIFFNESS_COLLAPSE = "stiffness_collapse"
    NONE = "none"


@dataclass(frozen=True)
class CriticalTemperatureResult:
    """Critical temperature together with the mode of failure behind it.

    Attributes
    ----------
    theta : float or None
        Critical temperature [degC] with the exact same value and
        conventions as the legacy scalar-returning functions (member-level:
        the refined crossing or ``None`` when the member never fails inside
        the grid; system-level: the last safe grid temperature, never
        ``None``).
    failure_mode : FailureMode
        Which physical event the reported ``theta`` corresponds to.
    """

    theta: float | None
    failure_mode: FailureMode


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
    """Two-component criticality index with both components exposed.

    Attributes
    ----------
    ci : float
        Composite criticality ``max(u_component, dcr_component)`` — the pure
        damage counterfactual of ``docs/theory.md`` §5.1-5.2, deliberately
        independent of how hot the fire is.
    u_component : float
        ``max|u_pert| / max|u_base| - 1`` at the *same* temperature field.
    dcr_component : float
        Damage-conditional DCR ratio at the *same* temperature field,
        ``DCR_pert(T) / DCR_base(T) - 1``. The baseline is the undamaged
        structure in the given temperature field (``docs/theory.md`` §5.2),
        so this is exactly ``0`` for ``alpha = 1`` — no perturbation, no
        criticality, at any temperature.
    fire_component : float
        Fire-severity ratio for this member,
        ``DCR_base(T) / DCR_base(20 degC) - 1``, independent of ``alpha``.
        Reported explicitly so thermal degradation stays visible without
        being smuggled into the perturbation criticality.
    dcr_combined : float
        The explicit combination
        ``(1 + dcr_component)(1 + fire_component) - 1
        = DCR_pert(T) / DCR_base(20 degC) - 1`` — identical to the legacy
        (<= 2.6.0) ``dcr_component``, which referenced the COLD state and
        therefore reported undamaged members as critical at temperature.
        For triage contexts that want fire severity and damage sensitivity
        in one number, combine through this field, never through ``ci``.
    governing : Governing
        Limit state that produced the larger of the two *damage* components.
    """

    member_id: str
    ci: float
    u_component: float
    dcr_component: float
    governing: Governing
    fire_component: float = 0.0
    dcr_combined: float = 0.0


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
    """Member axial forces [N] (tension positive) at the given temperatures.

    The solve carries the **full thermal demand**: the temperature field both
    degrades stiffness (``k_e(T) = k_E(T) E A / L``) and, through restrained
    expansion, loads the structure via the equivalent nodal forces
    ``B^T diag(k(T)) dL_pre``.  The reported force is the mechanical one,
    ``N_e = k_e(T) (b_e . u - dL_pre,e)``.  In a redundant structure a heated
    member therefore develops real compression even under no external load —
    the demand the DCR must see.  Elements without ``alpha`` /
    ``delta_L_free`` reduce exactly to the previous stiffness-degradation-only
    result.
    """
    setup = build_engine(nodes, elements, loads, temps)
    u = base_displacement(setup, total_load_vector(nodes, loads, setup))
    forces = member_forces(setup, u)
    return {eid: float(forces[i]) for i, eid in enumerate(setup.ids)}


@dataclass(frozen=True)
class UniformForceScan:
    """One factorisation serving a whole uniform-temperature force scan.

    Under a **uniform** field every stiffness scales by the same factor,
    ``K(T) = k_E(T) K_0``, and the thermal right-hand side is affine in
    ``T``, so the member forces at any grid temperature follow from three
    solves against the *ambient* factorisation:

    .. code-block:: text

        u(T)  = z_m / k_E(T) + (T - T_0) z_alpha + z_free
        N_e(T) = k_E(T) k0_e ( b_e.u(T) - alpha_e (T - T_0) L_e - dL_free,e )

    with ``z_m = K_0^-1 F_mech``, ``z_alpha = K_0^-1 B^T (k0 alpha L)`` and
    ``z_free = K_0^-1 B^T (k0 dL_free)``.  Expanding gives the closed form
    used in :meth:`forces_at`, which is exact — no interpolation, no
    approximation — and identical to calling :func:`member_axial_forces` at
    each temperature.  The critical-temperature scans
    (:func:`member_critical_temperature`, :func:`system_critical_temperature`)
    used to rebuild and refactorise the engine at all 48 grid points; they now
    build this scan once and evaluate ``O(m)`` arithmetic per point, which is
    the difference between ``O(grid * n^3)`` and ``O(n^3)`` on the hot paths
    of the retrofit triage.

    Scope limit
    -----------
    Exact solver for the **special case of a uniform scalar stiffness
    degradation only**: one shared factor ``s(T)`` must scale every member's
    modulus (``E_e(T) = s(T) E_e`` -- one material law, one temperature for
    all members). Per-member materials, protection or temperature histories
    break ``K(T) = s(T) K_0``, and those scenarios must fall back to the
    per-point :func:`~truss_analysis.criticality.engine.build_engine`
    rebuild. This is not a generic thermal-scan engine.

    Attributes
    ----------
    ids : tuple[str, ...]
        Member ids, in element order.
    k0 : np.ndarray
        Ambient axial stiffnesses ``E A / L`` [N/m], shape ``(m,)``.
    el_mech : np.ndarray
        ``b_e . z_m`` — mechanical elongation from external loads, ``(m,)``.
    el_alpha : np.ndarray
        ``b_e . z_alpha`` — elongation per unit ``(T - T_0)`` from restrained
        thermal expansion, ``(m,)``.
    el_free : np.ndarray
        ``b_e . z_free`` — elongation from fabrication strains, ``(m,)``.
    alpha_lengths : np.ndarray
        ``alpha_e * L_e`` [m/degC], ``(m,)``.
    delta_l_free : np.ndarray
        Free length change ``delta_L_free,e`` [m], ``(m,)``.
    """

    ids: tuple[str, ...]
    k0: np.ndarray
    el_mech: np.ndarray
    el_alpha: np.ndarray
    el_free: np.ndarray
    alpha_lengths: np.ndarray
    delta_l_free: np.ndarray

    @classmethod
    def build(
        cls,
        nodes: Sequence[Node],
        elements: Sequence[Element],
        loads: Mapping[str, Mapping[str, float]],
    ) -> UniformForceScan:
        """Factorise the ambient state once and prepare the three solves."""
        temps0 = {e.id: T_AMBIENT for e in elements}
        setup = build_engine(nodes, elements, loads, temps0)
        # At T_AMBIENT the Eurocode reduction is exactly 1, so k_axial is the
        # ambient stiffness and dl_pre is the pure fabrication term.
        f_mech = load_vector(nodes, loads, setup.free_dofs)
        z_m = base_displacement(setup, f_mech)
        el_mech = setup.b_free @ z_m

        node_idx = {n.id: i for i, n in enumerate(nodes)}
        lengths = np.array(
            [
                float(
                    np.hypot(
                        nodes[node_idx[e.node_j]].x - nodes[node_idx[e.node_i]].x,
                        nodes[node_idx[e.node_j]].y - nodes[node_idx[e.node_i]].y,
                    )
                )
                for e in elements
            ]
        )
        alpha_arr = np.array([e.alpha for e in elements], dtype=float)
        alpha_lengths = alpha_arr * lengths

        q_alpha = setup.b_free.T @ (setup.k_axial * alpha_lengths)
        el_alpha = setup.b_free @ base_displacement(setup, q_alpha)
        q_free = setup.b_free.T @ (setup.k_axial * setup.dl_pre)
        el_free = setup.b_free @ base_displacement(setup, q_free)
        return cls(
            ids=setup.ids,
            k0=setup.k_axial.copy(),
            el_mech=el_mech,
            el_alpha=el_alpha,
            el_free=el_free,
            alpha_lengths=alpha_lengths,
            delta_l_free=setup.dl_pre.copy(),
        )

    def forces_at(self, temperature: float) -> np.ndarray:
        """Exact member forces [N] at a uniform temperature, shape ``(m,)``.

        Raises
        ------
        MechanismError
            If ``k_E(T) <= 0`` (the Eurocode table reaches 0 at 1200 degC):
            the structure has no stiffness left, which the per-point engine
            build reported the same way.
        """
        s = float(eurocode_k_E(temperature))
        if s <= 0.0:
            msg = (
                f"stiffness matrix singular (mechanism) in base state: "
                f"k_E({temperature} degC) = {s}"
            )
            raise MechanismError(msg)
        dt = float(temperature) - T_AMBIENT
        # b.u(T) = el_mech / s + dt * el_alpha + el_free
        elong = self.el_mech / s + dt * self.el_alpha + self.el_free
        dl_pre = self.alpha_lengths * dt + self.delta_l_free
        forces: np.ndarray = np.asarray(s * self.k0 * (elong - dl_pre), dtype=float)
        return forces

    def forces_dict_at(self, temperature: float) -> dict[str, float]:
        """`forces_at` keyed by member id."""
        forces = self.forces_at(temperature)
        return {eid: float(forces[i]) for i, eid in enumerate(self.ids)}


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
            # EULER_ONLY: historical model that overestimates capacity of
            # intermediate-slenderness members (~15% at lambda_bar ~ 2).
            # This is optimistic and should be flagged to users.
            warnings.warn(
                "BucklingModel.EULER_ONLY ignores residual stresses and initial "
                "out-of-straightness, overestimating capacity of intermediate-"
                "slenderness members. The reported DCR and critical temperature "
                "are optimistic. Use EUROCODE_CHI for code-compliant assessment.",
                UserWarning,
                stacklevel=2,
            )
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


def _limit_states_from_forces(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    forces: Mapping[str, float],
    temps: Mapping[str, float],
    f_y: float,
    buckling_model: BucklingModel = BucklingModel.EUROCODE_CHI,
    buckling_curve: str = DEFAULT_BUCKLING_CURVE,
) -> dict[str, MemberLimitState]:
    """Per-member limit states from precomputed forces and temperatures.

    The capacity side of :func:`dcr_field` factored out, so the uniform-
    temperature scans can reuse it with forces from a
    :class:`UniformForceScan` (one factorisation for the whole grid) instead
    of rebuilding the engine at every grid point.
    """
    out: dict[str, MemberLimitState] = {}
    node_by_id = {n.id: n for n in nodes}
    unassessable: list[str] = []
    for e in elements:
        ni = node_by_id[e.node_i]
        nj = node_by_id[e.node_j]
        length = float(np.hypot(nj.x - ni.x, nj.y - ni.y))
        state = _member_limit_state(
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
        out[e.id] = state
        # A compressed member without a second moment of area has no
        # computable buckling capacity: lambda_bar -> inf, chi -> 0 and the
        # DCR is +inf. That verdict is conservative but it is NOT a limit
        # state -- it is a missing input, and the static-report path
        # (postprocess.calculate_buckling) already refuses to pass such a
        # member silently. The fire chain must be equally loud, or a model
        # built with the Element default ``I_sec = 0.0`` reports every
        # compression member as failed at ambient with no explanation
        # (round-5 audit, C7 conceptual-3).
        if state.compression and e.I_sec <= 0.0:
            unassessable.append(e.id)
    if unassessable:
        warnings.warn(
            f"buckling capacity not assessable for {len(unassessable)} "
            f"compressed member(s) with I_sec <= 0 "
            f"({', '.join(unassessable[:5])}"
            f"{'...' if len(unassessable) > 5 else ''}); their DCR is "
            "reported as +inf. Set a real second moment of area to assess "
            "them.",
            BucklingCheckWarning,
            stacklevel=2,
        )
    return out


def dcr_field(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    f_y: float,
    buckling_model: BucklingModel = BucklingModel.EUROCODE_CHI,
    buckling_curve: str = DEFAULT_BUCKLING_CURVE,
    check_system_stability: bool = True,
) -> dict[str, MemberLimitState]:
    """DCR state of every member at the given member temperatures.

    The demand forces include restrained thermal expansion (see
    :func:`member_axial_forces`), so a heated member of a redundant truss is
    checked against the compression it really develops, not only against the
    redistribution of the mechanical loads.

    Parameters
    ----------
    nodes, elements, loads, temps, f_y
        Model, loading, per-member temperature field [degC] and ambient yield
        strength ``f_y`` [Pa].
    buckling_model : BucklingModel, default EUROCODE_CHI
        Compression capacity model; see :class:`BucklingModel`.
    buckling_curve : str, default "c"
        Flexural buckling curve, only used by ``EUROCODE_CHI``.
    check_system_stability : bool, default True
        If True, compute the linearised system buckling load factor
        ``lambda_cr``. When ``lambda_cr < 1``, all compression members
        receive a DCR adjustment to reflect system-level instability risk
        (A1: connect lambda_cr to dcr_field).

    Returns
    -------
    dict[str, MemberLimitState]
        Limit state per member id. When ``check_system_stability=True`` and
        ``lambda_cr < 1``, compression members have their DCR increased by
        the factor ``1 / lambda_cr`` to account for system instability.
    """
    forces = member_axial_forces(nodes, elements, loads, temps)
    result = _limit_states_from_forces(
        nodes, elements, forces, temps, f_y, buckling_model, buckling_curve
    )

    # A1: Connect lambda_cr to dcr_field for system stability
    if check_system_stability:
        try:
            # warn_shallow=False: this is a member-level code check that
            # borrows lambda_cr for a system-stability DCR adjustment.  The
            # shallow-geometry advisory belongs to a deliberate stability
            # study, not to every dcr_field() call on every model shape.
            buckling_result = linearized_buckling_load_factor(
                nodes, elements, loads, temps, warn_shallow=False
            )
            lambda_cr = buckling_result.lambda_cr

            # If lambda_cr < 1, the system is unstable at current load level
            # Adjust DCR for compression members to reflect this
            if lambda_cr < 1.0:
                stability_factor = 1.0 / lambda_cr
                for ls in result.values():
                    if ls.compression and ls.p_cr is not None:
                        # Scale DCR by stability factor
                        # This ensures DCR > 1 when system is unstable
                        object.__setattr__(ls, "dcr", ls.dcr * stability_factor)
        except MechanismError:
            # Base state already unstable - all compression members should fail
            for ls in result.values():
                if ls.compression:
                    object.__setattr__(ls, "dcr", float("inf"))

    return result


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
    """Scalar facade of :func:`member_critical_temperature_detailed`.

    Returns exactly the same temperature (bit-for-bit); callers that need
    to know *why* the member failed -- limit state reached, or the system
    collapsing at the ``k_E = 0`` table endpoint -- should use the detailed
    variant and read :attr:`CriticalTemperatureResult.failure_mode`.
    """
    return member_critical_temperature_detailed(
        nodes,
        elements,
        loads,
        member_id,
        f_y,
        temp_grid,
        buckling_model,
        buckling_curve,
    ).theta


def member_critical_temperature_detailed(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    member_id: str,
    f_y: float,
    temp_grid: Sequence[float] = _TEMP_GRID,
    buckling_model: BucklingModel = BucklingModel.EUROCODE_CHI,
    buckling_curve: str = DEFAULT_BUCKLING_CURVE,
) -> CriticalTemperatureResult:
    """Smallest uniform temperature at which ``DCR_member >= 1``.

    Root-found in two stages: the grid locates the first bracket
    ``[T_k, T_k+1]`` with ``DCR(T_k) < 1 <= DCR(T_k+1)``, then bisection on
    exact :meth:`UniformForceScan.forces_at` evaluations (``O(m)`` each --
    no refactorisation, no interpolation of a curved ``DCR(T)``) narrows the
    crossing to ``_BISECT_XTOL`` degC.  The upper end of the final bracket
    is returned, so ``DCR(theta) >= 1`` holds by construction: the report is
    conservative by at most the tolerance.  If ``DCR(T)`` crosses several
    times inside one grid cell, the crossing found is the one bisection
    converges to inside the FIRST failing cell -- the grid resolution still
    defines which cell that is.  ``theta`` is ``None`` (with
    ``failure_mode = NONE``) when the member never reaches DCR = 1 within
    the grid range.

    ``failure_mode`` distinguishes the two ways the scan can end in
    failure: ``MATERIAL`` when a genuine limit state (``DCR >= 1`` at
    finite stiffness) was crossed, and ``STIFFNESS_COLLAPSE`` when the
    crossing is the ``k_E = 0`` table endpoint itself -- the structure lost
    all stiffness without the member's own DCR necessarily reaching 1 (a
    lightly loaded member in a redundant truss).  Reporting ~1200 degC as
    that member's "critical temperature" without the label would put a
    system-collapse temperature into engineering output as if it were a
    material limit state (round-5 audit, C8-5).

    A grid point where the material law has lost all stiffness
    (``k_E(T) <= 0``, the Eurocode endpoint at 1200 degC) is treated as
    ``DCR = +inf`` -- failure by collapse -- instead of propagating
    :class:`MechanismError` out of the middle of the scan.

    Because ``lambda_bar_theta`` grows as ``k_E`` falls faster than ``k_y``,
    the reported critical temperature depends on the compression capacity
    model; see :class:`BucklingModel`.

    The whole grid is served by ONE ambient factorisation through
    :class:`UniformForceScan`: at each evaluation the exact member forces
    (including restrained thermal expansion) are ``O(m)`` arithmetic, not a
    fresh ``O(n^3)`` engine build.
    """
    scan = UniformForceScan.build(nodes, elements, loads)
    if member_id not in scan.ids:
        msg = f"member_critical_temperature: unknown member {member_id!r}"
        raise KeyError(msg)

    def dcr_at(temp: float) -> tuple[float, bool]:
        """``(DCR, collapsed)`` at one temperature."""
        try:
            forces = scan.forces_at(temp)
        except MechanismError:
            # k_E(T) <= 0: zero stiffness is structural collapse, i.e. the
            # failure side of the DCR = 1 crossing, not a scan error.
            return float("inf"), True
        temps_t = {eid: temp for eid in scan.ids}
        state = _limit_states_from_forces(
            nodes,
            elements,
            {eid: float(forces[i]) for i, eid in enumerate(scan.ids)},
            temps_t,
            f_y,
            buckling_model,
            buckling_curve,
        )[member_id]
        return state.dcr, False

    prev_t: float | None = None
    for t in temp_grid:
        t_f = float(t)
        dcr_t, collapsed = dcr_at(t_f)
        if dcr_t >= 1.0:
            if prev_t is None:
                return CriticalTemperatureResult(
                    t_f,
                    FailureMode.STIFFNESS_COLLAPSE
                    if collapsed
                    else FailureMode.MATERIAL,
                )
            lo, hi = prev_t, t_f  # DCR(lo) < 1 <= DCR(hi)
            hi_collapsed = collapsed
            for _ in range(_BISECT_MAX_ITER):
                if hi - lo <= _BISECT_XTOL:
                    break
                mid = 0.5 * (lo + hi)
                dcr_mid, collapsed_mid = dcr_at(mid)
                if dcr_mid >= 1.0:
                    hi = mid
                    hi_collapsed = collapsed_mid
                else:
                    lo = mid
            return CriticalTemperatureResult(
                hi,
                FailureMode.STIFFNESS_COLLAPSE
                if hi_collapsed
                else FailureMode.MATERIAL,
            )
        prev_t = t_f
    return CriticalTemperatureResult(None, FailureMode.NONE)


def system_critical_temperature(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    f_y: float,
    temp_grid: Sequence[float] = _TEMP_GRID,
    buckling_model: BucklingModel = BucklingModel.EUROCODE_CHI,
    buckling_curve: str = DEFAULT_BUCKLING_CURVE,
) -> float:
    """Scalar facade of :func:`system_critical_temperature_detailed`.

    Returns exactly the same temperature (bit-for-bit); the detailed
    variant additionally labels whether the scan stopped on a member limit
    state (:attr:`FailureMode.MATERIAL`) or on the zero-stiffness table
    endpoint (:attr:`FailureMode.STIFFNESS_COLLAPSE`), or found no failure
    in range (:attr:`FailureMode.NONE`).
    """
    return system_critical_temperature_detailed(
        nodes,
        elements,
        loads,
        f_y,
        temp_grid,
        buckling_model,
        buckling_curve,
    ).theta  # type: ignore[return-value]  # never None for the system scan


def system_critical_temperature_detailed(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    f_y: float,
    temp_grid: Sequence[float] = _TEMP_GRID,
    buckling_model: BucklingModel = BucklingModel.EUROCODE_CHI,
    buckling_curve: str = DEFAULT_BUCKLING_CURVE,
) -> CriticalTemperatureResult:
    """First loss of acceptability along the monotone heating path.

    Precisely: the largest grid temperature ``T`` such that EVERY scanned
    point up to and including ``T`` is safe (no member at ``DCR >= 1``).
    ``DCR(T)`` is **not** monotone -- forces redistribute as stiffnesses
    degrade -- so a grid point above a failure can look safe again; such
    post-failure "recovery" does NOT make the structure acceptable, because
    a standard fire only heats: once some member crosses ``DCR = 1`` at
    ``T*``, it has failed at ``T*`` whatever redistribution does above it.
    The scan therefore stops at the first failing grid point and returns the
    last safe one. (Pre-2.7 the docstring said "highest scanned temperature
    with no member at DCR >= 1", which read over the WHOLE grid and
    contradicted the first-failure algorithm -- the round-4 audit asked for
    the definition to be made explicit; this is it.)

    A grid point where the material law has lost all stiffness
    (``k_E(T) <= 0``, the Eurocode endpoint at 1200 degC) counts as failure
    by collapse at that point instead of raising :class:`MechanismError` out
    of the middle of the scan.

    Returns ``temp_grid[0]`` when even the coldest scan already fails (or
    collapses) and ``temp_grid[-1]`` when nothing fails within the range;
    the resolution is the grid step.  The detailed variant labels the
    stopping event: ``MATERIAL`` (first member DCR >= 1),
    ``STIFFNESS_COLLAPSE`` (the ``k_E = 0`` endpoint) or ``NONE`` (no
    failure inside the grid).

    Like :func:`member_critical_temperature`, the grid is evaluated from a
    single :class:`UniformForceScan` factorisation, which is what keeps the
    retrofit triage (a ``theta_sys`` per candidate decision) affordable.
    """
    scan = UniformForceScan.build(nodes, elements, loads)
    last_safe = float(temp_grid[0])
    for t in temp_grid:
        try:
            forces = scan.forces_at(t)
        except MechanismError:
            # k_E(T) <= 0: zero-stiffness endpoint == collapse == failure.
            return CriticalTemperatureResult(last_safe, FailureMode.STIFFNESS_COLLAPSE)
        temps_t = {eid: float(t) for eid in scan.ids}
        states = _limit_states_from_forces(
            nodes,
            elements,
            {eid: float(forces[i]) for i, eid in enumerate(scan.ids)},
            temps_t,
            f_y,
            buckling_model,
            buckling_curve,
        )
        if any(s.dcr >= 1.0 for s in states.values()):
            return CriticalTemperatureResult(last_safe, FailureMode.MATERIAL)
        last_safe = float(t)
    return CriticalTemperatureResult(last_safe, FailureMode.NONE)


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
    component is the *damage-conditional* ratio against the undamaged
    structure **in the same temperature field** (``docs/theory.md`` §5.2):

    .. code-block:: text

        dcr_component = DCR_pert(T) / DCR_base(T) - 1

    Both states share temperature, geometry and loads, so the component is
    exactly ``0`` at ``alpha = 1`` — the exposing test of the round-4 audit:
    an unperturbed member carries no perturbation criticality, however hot
    the fire is. Fire severity (degradation of the DCR of the *undamaged*
    member relative to the cold structure) is reported separately as
    ``fire_component = DCR_base(T) / DCR_base(20 degC) - 1``, and the
    explicit product ``dcr_combined = (1 + dcr)(1 + fire) - 1`` reproduces
    the legacy (<= 2.6.0) cold-referenced ``dcr_component`` for triage
    contexts that want the two effects in one number. The pre-2.7 composite
    hid the fire term inside the perturbation criticality, which made
    ``governing`` and the ranking respond to the fire even at ``alpha = 1``.

    The base state carries the full thermal demand (equivalent nodal forces
    from restrained expansion), and the perturbed force is the *mechanical*
    one, ``N = alpha k_i (b_i . u_pert - dL_pre,i)`` — consistent with
    :func:`member_axial_forces` and with the rank-1 numerator in
    :func:`~truss_analysis.criticality.engine.ci_sweep`.
    """
    setup = build_engine(nodes, elements, loads, temps)
    f_free = total_load_vector(nodes, loads, setup)
    u = base_displacement(setup, f_free)
    u_max_base = float(np.max(np.abs(u)))
    sweep = ci_sweep(setup, u, alpha)
    forces_base = member_forces(setup, u)
    # Cold reference state (20 degC, same geometry/loads) for the EXPLICIT
    # fire-severity component only. The damage component below references
    # the undamaged structure in the same temperature field, so temperature
    # degradation can never leak into the perturbation criticality.
    forces_cold = member_axial_forces(
        nodes, elements, loads, {e.id: T_AMBIENT for e in elements}
    )

    components: dict[str, ComponentCI] = {}
    ci_values: dict[str, float] = {}
    governing: dict[str, str] = {}
    for i, eid in enumerate(setup.ids):
        u_comp = sweep.ci_values[eid]
        elem = elements[i]
        length = _length(nodes, elem)
        t_e = float(temps[eid])
        state_base = _member_limit_state(
            eid,
            t_e,
            float(forces_base[i]),
            elem.A,
            elem.I_sec,
            length,
            elem.E,
            elem.effective_length_factor,
            f_y,
        )
        state_cold = _member_limit_state(
            eid,
            T_AMBIENT,
            forces_cold[eid],
            elem.A,
            elem.I_sec,
            length,
            elem.E,
            elem.effective_length_factor,
            f_y,
        )
        # perturbed axial force in member i: softened stiffness x the
        # MECHANICAL elongation (total minus imposed), matching the rank-1
        # numerator convention of the engine
        k_pert = alpha * setup.k_axial[i]
        n_pert = float(
            k_pert * (float(setup.b_free[i] @ sweep.u_pert[:, i]) - setup.dl_pre[i])
        )
        state_pert = _member_limit_state(
            eid,
            t_e,
            n_pert,
            elem.A,
            elem.I_sec,
            length,
            elem.E,
            elem.effective_length_factor,
            f_y,
        )
        # Damage component: same-temperature DCR ratio. Zero-capacity states
        # (k_y/k_E collapsed at extreme T) give base DCR = inf; the ratio is
        # then undefined and reported as 0 — the fire component already says
        # "capacity gone", and a mechanism shows up through u_comp = inf.
        if _DCR_BASE_TOL < state_base.dcr < float("inf"):
            dcr_comp = state_pert.dcr / state_base.dcr - 1.0
        else:
            dcr_comp = 0.0
        if state_cold.dcr > _DCR_BASE_TOL:
            fire_comp = state_base.dcr / state_cold.dcr - 1.0
        else:
            fire_comp = 0.0
        combined = (1.0 + dcr_comp) * (1.0 + fire_comp) - 1.0
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
            fire_component=fire_comp,
            dcr_combined=combined,
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
    node_by_id = {n.id: n for n in nodes}
    ni = node_by_id[elem.node_i]
    nj = node_by_id[elem.node_j]
    return float(np.hypot(nj.x - ni.x, nj.y - ni.y))
