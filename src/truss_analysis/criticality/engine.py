"""Exact rank-1 criticality engine (Sherman-Morrison) and the CI pipeline.

Mathematics (verified against independent full re-solves in the test suite)
--------------------------------------------------------------------------
Reducing the axial stiffness of a single truss member is a **rank-1 update**
of the global stiffness matrix, because ``k_i = (E_i A_i / L_i) b_i b_i^T``
with ``b_i`` the member compatibility vector:

    K_pert = K + Delta_i b_i b_i^T,        Delta_i = (alpha - 1) k_i
    (K + Delta b b^T)^-1 F = u - Delta (K^-1 b)(b^T u) / (1 + Delta b^T K^-1 b)

Factorising ``K_ff`` once and solving for the whole compatibility matrix
``B = [b_1 ... b_n]`` (``Z = K^-1 B``) yields every perturbed state in one
vectorised pass:

    d_i = b_i^T Z_i,  f_i = b_i^T u,  coef_i = Delta_i f_i / (1 + Delta_i d_i)
    U_pert = u - Z diag(coef)          -> column i = perturbed state of member i
    CI_i   = max|U_pert[:, i]| / max|u| - 1

Validity limit (explicit)
-----------------------------------------
The rank-1 path is exact **only for single-member perturbations**.  For
simultaneous multi-member perturbations (retrofit studies) use
:func:`perturb_multi` (Woodbury rank-r) or a full re-solve; applying the
rank-1 formula member-by-member to a multi-member change is wrong and no
public function here does it.

Thermal / fabrication demand (equivalent forces)
------------------------------------------------
The temperature field does two physically distinct things to a member: it
degrades its stiffness (``k_e(T) = k_E(T) E A / L``) and, through restrained
expansion, it loads the structure.  The imposed elongation

    dL_pre,e = alpha_e (T_e - T_AMBIENT) L_e + delta_L_free,e

enters the right-hand side as the equivalent nodal force vector
``B^T diag(k(T)) dL_pre`` — the same term :mod:`truss_analysis.assembly`
builds for ``run()`` — and the member force is the *mechanical* one,

    N_e = k_e(T) (b_e . u - dL_pre,e).

Omitting either half made the whole DCR / theta_sys / CI / retrofit chain
blind to thermal stress in redundant structures: ``run()`` heated a
restrained member into real compression while the demand chain saw only the
stiffness degradation.  With the equivalent forces on the RHS the rank-1
numerator changes too.  Softening member ``i`` by ``alpha`` changes *both*
``K`` and its thermal force ``k_i dL_i b_i``, so the perturbed solve is

    u' = u - Delta_i (f_i - dL_i) / (1 + Delta_i d_i) * z_i
       = u - Delta_i (N_i / k_i) / (1 + Delta_i d_i) * z_i,

i.e. the numerator is the member's mechanical force, not its total
elongation.  For ``dL_pre = 0`` (elements without ``alpha`` /
``delta_L_free``) every formula reduces exactly to the previous one, so
mechanical-only analyses are bit-for-bit unchanged.

Numerical guard
-------------------------------
``1 + Delta_i d_i -> 0`` means the perturbed structure is (near) a mechanism.
Such members are flagged and routed to a full brute-force solve; if that
solve is singular the member CI is ``+inf`` with a ``mechanism`` flag.  The
engine never emits a silent finite number for a guarded member.  The guard
threshold is **condition-scaled**: ``denom`` inherits a round-off level of
about ``cond(K_ff) * eps`` from ``z = K_ff^-1 b``, so a fixed ``1e-8`` would
misjudge healthy but ill-conditioned structures (``cond ~ 1e10`` has round-off
``~ 2e-6``).  :func:`build_engine` therefore measures ``cond`` from the LU
factors it already computed (LAPACK ``dgecon``, ``O(n^2)``) and stores
``guard_tol = min(max(1e-8, 100 * cond * eps), 1e-4)`` on the setup;
:func:`ci_sweep` uses it unless the caller overrides it.

No deep copying of element containers anywhere in this package:
perturbed states are built from the rank-1/rank-r formulas
or from fresh dataclass instances via :func:`dataclasses.replace`.

Scale and memory ceiling
-------------------------------
This engine is **dense**: :func:`build_engine` materialises the free-free
stiffness ``(n_free x n_free)`` and its inverse action on every member
(``z`` is ``(n_free x n_members)``), and :func:`ci_sweep` holds one
perturbed-displacement column per member (``u_pert`` is
``(n_free x n_members)``).  Memory therefore grows as ``O(n_free *
n_members)``, not ``O(nnz)``.  Concretely, a 10k-member truss with ~20k
free DOFs needs ~1.6 GB for ``u_pert`` alone.  The *assembly* and *solve*
layers (:mod:`truss_analysis.assembly`, :mod:`truss_analysis.solver`) do
support CSR-sparse storage and scale far beyond this; the criticality /
retrofit / uncertainty tools that sit on this dense rank-1 engine do not.
Until a blocked ``ci_sweep`` and a sparse rank-1 update land, treat the
criticality engine as scoped to small/medium models (the round-5 audit's
C8 performance finding) -- the physics is exact, the footprint is the
constraint.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import LinAlgWarning, lapack, lu_factor, lu_solve

from ..exceptions import ConstantAlphaWarning, IllConditionedPerturbationWarning
from ..material.steel_eurocode import FloatOrArray, effective_alpha, thermal_strain
from ..material.steel_eurocode import k_E as eurocode_k_E
from ..model import Element, Node, fixed_dof_indices
from .indices import NciResult, compute_nci
from .ranking import TauResult, rank_members, tau_b
from .scenarios import T_AMBIENT, get_scenario_temperatures

logger = logging.getLogger(__name__)

__all__ = [
    "GUARD_COND_COEF",
    "GUARD_TOL",
    "GUARD_TOL_MAX",
    "PERTURB_COND_WARN",
    "PERTURB_RESID_WARN",
    "CiSweep",
    "EngineSetup",
    "MechanismError",
    "TopologyResult",
    "base_displacement",
    "brute_force_ci",
    "build_engine",
    "ci_sweep",
    "compute_ci_for_topology",
    "imposed_load_vector",
    "load_vector",
    "member_forces",
    "member_matrices",
    "perturb_multi",
    "prestress_lengths",
    "setup_indeterminacy",
    "total_load_vector",
]

GUARD_TOL = 1e-8
#: Coefficient of the condition-scaled part of the adaptive guard tolerance
#: (``GUARD_COND_COEF * cond(K_ff) * eps``); see :func:`_guard_tolerance`.
GUARD_COND_COEF = 100.0
#: Hard cap on the adaptive guard tolerance. Members below the tol are routed
#: to an exact brute-force solve, so a loose tol costs time, never accuracy;
#: the cap keeps an extreme ``cond`` estimate from routing the whole sweep.
GUARD_TOL_MAX = 1e-4
_SINGULAR_TOL = 1e-12

#: Condition number of the Woodbury core above which
#: :func:`perturb_multi` issues an
#: :exc:`~truss_analysis.exceptions.IllConditionedPerturbationWarning`.
#: ``1e10`` leaves ~6 trustworthy digits in double precision -- enough for
#: ranking, not enough to quote as an answer (round-6 audit C2).
PERTURB_COND_WARN = 1e10
#: Backward (relative residual) error above which the same warning fires.
#: This is the *measured* quality of the solve rather than an estimate of
#: how bad the matrix could be, so it catches cores that are ill-conditioned
#: in a direction the right-hand side happens to excite.
PERTURB_RESID_WARN = 1e-8


class MechanismError(RuntimeError):
    """Raised when a stiffness matrix is singular (mechanism detected)."""


@dataclass(frozen=True)
class TopologyResult:
    """Immutable result of one (topology, scenario, temperature, alpha) run."""

    topology_id: str
    scenario: str
    temperature: float
    alpha: float
    ci_values: dict[str, float]
    nci_values: dict[str, float] | None
    ranks: list[str]
    top_5: list[str]
    tau_vs_base: float | None
    u_max_base: float
    u_max_perturbed_max: float
    flags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineSetup:
    """Factorised base state plus the rank-1 machinery for all members."""

    ids: tuple[str, ...]
    free_dofs: tuple[int, ...]
    lu: tuple[np.ndarray, np.ndarray]
    b_free: np.ndarray  # (nE, ndof_free)
    k_axial: np.ndarray  # (nE,)
    z: np.ndarray  # (ndof_free, nE) = K_ff^-1 B
    d: np.ndarray  # (nE,) = b_i^T Z_i
    dl_pre: np.ndarray  # (nE,) imposed elongation alpha*(T-T0)*L + delta_L_free
    guard_tol: float = GUARD_TOL
    """float: condition-scaled guard tolerance for the rank-1 denominator.

    Computed by :func:`_guard_tolerance` from the LU factors at build time;
    :func:`ci_sweep` uses it when the caller does not override ``guard_tol``.
    """


@dataclass(frozen=True)
class CiSweep:
    """Full CI sweep over all members in one vectorised pass."""

    ci_values: dict[str, float]
    u_pert: np.ndarray  # (ndof_free, nE); +inf columns mark mechanisms
    flagged: dict[str, str]
    u_max_perturbed_max: float


def free_dof_indices(nodes: Sequence[Node]) -> tuple[int, ...]:
    """Return the unconstrained global DOF indices for ``nodes``.

    Delegates the support semantics to :func:`truss_analysis.model.fixed_dof_indices`,
    the single definition shared with the assembler. Duplicating the rule here
    would let the criticality engine and the static solver disagree about
    which DOFs are fixed -- an error no equivalence test on ``K`` would catch,
    because both paths would be wrong together.
    """
    fixed = set(fixed_dof_indices(list(nodes)))
    return tuple(d for d in range(2 * len(nodes)) if d not in fixed)


def member_matrices(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    k_scale: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility vectors ``b`` (nE x ndof) and axial stiffnesses ``k``."""
    node_idx = {n.id: i for i, n in enumerate(nodes)}
    n_dof = 2 * len(nodes)
    b = np.zeros((len(elements), n_dof))
    k = np.zeros(len(elements))
    for i, e in enumerate(elements):
        ii = node_idx[e.node_i]
        jj = node_idx[e.node_j]
        dx = nodes[jj].x - nodes[ii].x
        dy = nodes[jj].y - nodes[ii].y
        length = float(np.hypot(dx, dy))
        c, s = dx / length, dy / length
        scale = 1.0 if k_scale is None else float(k_scale[e.id])
        k[i] = scale * e.E * e.A / length
        b[i, (2 * ii, 2 * ii + 1, 2 * jj, 2 * jj + 1)] = (-c, -s, c, s)
    return b, k


#: Member temperature [degC] above which a constant ``alpha`` is reported.
#:
#: Measured, not chosen for roundness.  Against the EN 1993-1-2 elongation curve
#: ``eps_th(T)`` a constant ``alpha = 1.2e-5`` understates the imposed strain by
#: 3.9% at 100 degC and 12.3% at 400 degC.  150 degC is where the error first
#: exceeds 5%, which is the band below which a linear approximation of a thermal
#: strain is ordinarily accepted without comment; above it the gap is large
#: enough to move a utilisation ratio and has to be said out loud.
ALPHA_CONSTANCY_LIMIT = 150.0


def constant_alpha_understatement(theta: float, alphas: Sequence[float]) -> float:
    """Percent by which a constant-``alpha`` strain understates ``eps_th(T)``.

    Computed from the ``alpha`` values actually used, so the number in the
    warning is about the caller's model rather than about a nominal 1.2e-5: a
    user who already put a secant coefficient into ``Element.alpha`` gets told
    the truth about their own input instead of being warned about somebody
    else's.

    Parameters
    ----------
    theta : float
        Steel temperature [degC].
    alphas : Sequence[float]
        The expansion coefficients the constant path used [1/K].

    Returns
    -------
    float
        Understatement in percent, ``0.0`` when the curve cannot be evaluated
        (no temperature rise) or the imposed strain is already larger than the
        curve's, in which case nothing is being understated.
    """
    if not alphas:
        return 0.0
    curve = float(thermal_strain(theta, T_AMBIENT))
    if curve <= 0.0:
        return 0.0
    linear = float(np.mean(np.asarray(alphas, dtype=float))) * (theta - T_AMBIENT)
    shortfall = 1.0 - linear / curve
    return max(0.0, shortfall * 100.0)


def prestress_lengths(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    temps: Mapping[str, float] | None = None,
    use_effective_alpha: bool = False,
) -> np.ndarray:
    """Imposed (thermal + fabrication) elongation per member, shape ``(nE,)``.

    .. code-block:: text

        dL_pre,e = alpha_e * (T_e - T_AMBIENT) * L_e + delta_L_free,e

    This is the same quantity :func:`truss_analysis.assembly._imposed_nodal_forces`
    builds the equivalent nodal forces from, expressed on the temperature
    *field* the engine works with: ``temps`` maps member id to absolute steel
    temperature [degC] and ``T_e - T_AMBIENT`` is the temperature change from
    the stress-free reference state (20 degC, :data:`scenarios.T_AMBIENT`).
    ``Element.delta_T`` is deliberately NOT read here — in the fire chain the
    ``temps`` mapping is the authoritative temperature input, and honouring
    both would double-count the thermal strain.  When ``temps`` is ``None``
    only the fabrication term survives.

    Parameters
    ----------
    nodes : Sequence[Node]
        Model nodes (supply the member lengths).
    elements : Sequence[Element]
        Model elements (supply ``alpha`` and ``delta_L_free``).
    temps : Mapping[str, float] or None, optional
        Member id -> steel temperature [degC].
    use_effective_alpha : bool, default False
        Replace each member's constant ``alpha`` with the **secant** coefficient
        :func:`~truss_analysis.material.steel_eurocode.effective_alpha` at that
        member's own temperature, so the imposed strain follows the EN 1993-1-2
        elongation curve ``eps_th(T)`` exactly instead of a straight line through
        the origin.  ``False`` keeps the historical constant-``alpha`` behaviour
        **bit for bit** -- changing the default would silently move every result
        ever published with this library, which is not a change to make quietly.

        With ``alpha = 1.2e-5`` the constant path understates the imposed strain
        by 3.9% at 100 degC, 12.3% at 400, 17.1% at 600 and 19.4% at 700.  In a
        restrained member the thermal force is proportional to that strain, so a
        hot fire analysis run on the default reports demand that is low by the
        same percentage against an unchanged capacity -- an un-conservative DCR.
        A member whose ``alpha`` is exactly zero is left alone under either
        setting: zero is an explicit statement that the member does not expand,
        not an unfilled default, so "correcting" it to the Eurocode curve would
        invert what the model says.

        A :class:`~truss_analysis.exceptions.ConstantAlphaWarning` is issued
        when a supplied field exceeds :data:`ALPHA_CONSTANCY_LIMIT` and this flag
        is not set, so the choice is made deliberately rather than inherited.

    Returns
    -------
    np.ndarray
        ``dL_pre`` per member [m], in element order.

    Warns
    -----
    ConstantAlphaWarning
        When ``temps`` contains a member above :data:`ALPHA_CONSTANCY_LIMIT` and
        ``use_effective_alpha`` is ``False``.  Members whose ``alpha`` is
        exactly zero are excluded from the count and from the quoted
        percentage: they impose no thermal strain, so including them would
        both inflate the member count and dilute the mean ``alpha`` the
        percentage is computed from (a single ``alpha = 0`` tie rod was
        measured to move the quoted shortfall from 17.1% to 58.6% at 600 degC).
    """
    node_idx = {n.id: i for i, n in enumerate(nodes)}
    out = np.zeros(len(elements))
    thetas: list[float] = []
    alphas: list[float] = []
    for i, e in enumerate(elements):
        ii = node_idx[e.node_i]
        jj = node_idx[e.node_j]
        length = float(np.hypot(nodes[jj].x - nodes[ii].x, nodes[jj].y - nodes[ii].y))
        delta_t = 0.0 if temps is None else float(temps[e.id]) - T_AMBIENT
        if temps is None:
            alpha = e.alpha
        elif use_effective_alpha and e.alpha != 0.0:
            # Replace the ambient coefficient with the secant coefficient of the
            # EN 1993-1-2 elongation curve at this member's own temperature.
            #
            # A member whose ``alpha`` is exactly zero is skipped deliberately.
            # Zero is not an unfilled default here: it is an explicit statement
            # that the member does not expand -- a tie rod modelled as fixed in
            # length, or a fabrication-only element whose whole imposed
            # elongation is ``delta_L_free``.  Overriding it would turn "this
            # member cannot expand" into "this member expands per Eurocode",
            # which is the opposite of what the model says.
            alpha = float(effective_alpha(float(temps[e.id]), T_AMBIENT))
        else:
            alpha = e.alpha
            # A member whose ``alpha`` is exactly zero contributes no thermal
            # strain at all, so it is not "being built from a constant alpha"
            # and must not enter the warning's statistics.  Counting it does
            # two kinds of damage at once: it inflates the reported member
            # count, and -- worse -- it drags ``np.mean(alphas)`` towards
            # zero, which makes :func:`constant_alpha_understatement` report
            # an ever-larger shortfall the more non-expanding members the
            # model has.  Measured on a two-member model at 600 degC, one
            # ``alpha = 0`` tie rod moved the quoted number from the correct
            # 17.1% to 58.6%: a warning that overstates the error by 3.4x on
            # a member that provably has none.
            if float(temps[e.id]) > ALPHA_CONSTANCY_LIMIT and alpha != 0.0:
                thetas.append(float(temps[e.id]))
                alphas.append(alpha)
        out[i] = alpha * delta_t * length + e.delta_L_free

    if thetas:
        worst = max(thetas)
        warnings.warn(
            f"{len(thetas)} member(s) reach {worst:.0f} degC but the imposed "
            "strain is being built from a constant alpha: against the EN "
            "1993-1-2 elongation curve eps_th(T) that understates the thermal "
            f"strain by about {constant_alpha_understatement(worst, alphas):.1f}% "
            "at this temperature, and in a restrained member the force is low by "
            "the same amount -- an un-conservative DCR. Pass "
            "use_effective_alpha=True to follow the curve exactly, or set each "
            "Element.alpha to effective_alpha(T) yourself. The default is "
            "unchanged so that previously published results stay reproducible.",
            ConstantAlphaWarning,
            stacklevel=2,
        )
    return out


_TINY_DENOM = 1e-300


def setup_indeterminacy(setup: EngineSetup) -> int:
    """Return the static indeterminacy of a factorised setup: ``m - n_free``.

    Identical to ``m + r - 2j`` from
    :func:`truss_analysis.graph_validation.static_indeterminacy`, because
    ``n_free = 2j - r`` by construction -- the setup simply does not carry
    the reaction count, while the free-DOF count is exactly what its own
    factorisation was built on.  Deriving the redundancy from the factorised
    object rather than re-walking the model is what keeps this check
    consistent with the matrix that will actually be solved.

    Parameters
    ----------
    setup : EngineSetup
        Factorised base state.

    Returns
    -------
    int
        Degrees of static redundancy; ``< 0`` means the base model is
        already under-braced.
    """
    # m - n_free, which equals m + r - 2j because n_free = 2j - r.
    return int(len(setup.ids) - len(setup.free_dofs))


def _check_lu(
    lu: tuple[np.ndarray, np.ndarray], context: str = "in base state"
) -> None:
    lu_mat, _piv = lu  # scipy packs L and U into one matrix; second item is pivots
    diag = np.abs(np.diag(lu_mat))
    if diag.size == 0:
        # No free DOFs at all (every node fully restrained): nothing to
        # factorise, nothing that can be a mechanism among free DOFs. The
        # member forces of such a model are pure imposed-strain forces,
        # N = -k * dL_pre, which the engine still reports correctly.
        return
    scale = max(float(np.max(diag)), 1.0)
    if float(np.min(diag)) < _SINGULAR_TOL * scale:
        msg = f"stiffness matrix singular (mechanism) {context}"
        raise MechanismError(msg)


def _guard_tolerance(k_ff: np.ndarray, lu: tuple[np.ndarray, np.ndarray]) -> float:
    """Condition-scaled tolerance for the rank-1 denominator guard.

    The guarded quantity ``denom = 1 + Delta_i d_i`` is formed from
    ``z = K_ff^-1 b``; its floating-point error level is roughly
    ``cond(K_ff) * eps``. A *fixed* threshold of ``GUARD_TOL = 1e-8`` is far
    below that error level for ill-conditioned-but-healthy structures (at
    ``cond ~ 1e10`` the round-off alone is ``~ 2e-6``), so such a structure
    can have perfectly sound members routed to the brute-force fallback — or,
    with no fallback given, wrongly reported as mechanisms. The tolerance
    therefore scales with a reciprocal-condition estimate obtained from the
    LU factors that already exist (LAPACK ``dgecon``, ``O(n^2)``, no extra
    factorisation):

    .. code-block:: text

        tol = min(max(GUARD_TOL, GUARD_COND_COEF * cond_est * eps), GUARD_TOL_MAX)

    Two safety nets bound the estimate:

    * ``GUARD_TOL_MAX`` caps the routing damage of an extreme ``cond_est``:
      a member below the tol is re-solved exactly, so the cap trades a little
      brute-force work against guard thrashing, never accuracy.
    * ``denom`` is mathematically bounded below by ``alpha`` (since
      ``0 < k_i d_i <= 1``), so for any ``alpha >= GUARD_TOL_MAX`` no healthy
      member can ever be flagged, whatever the conditioning.
    """
    n = k_ff.shape[0]
    if n == 0:
        return GUARD_TOL
    try:
        anorm = float(np.abs(k_ff).sum(axis=0).max())
        if not np.isfinite(anorm) or anorm <= 0.0:
            return GUARD_TOL
        rcond, info = lapack.dgecon(lu[0], anorm, "1")
        rcond = float(rcond)
        if info != 0 or not np.isfinite(rcond) or rcond <= 0.0:
            # rcond == 0 means "singular to working precision"; _check_lu has
            # already rejected that case, so treat it as maximally guarded.
            return GUARD_TOL_MAX if rcond == 0.0 else GUARD_TOL
        cond_est = min(1.0 / rcond, 1.0 / float(np.finfo(float).eps) ** 2)
    except (ValueError, TypeError, OverflowError, lapack.LapackError):
        return GUARD_TOL
    tol = GUARD_COND_COEF * cond_est * float(np.finfo(float).eps)
    return float(min(max(GUARD_TOL, tol), GUARD_TOL_MAX))


def build_engine(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float] | None = None,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = eurocode_k_E,
    use_effective_alpha: bool = False,
) -> EngineSetup:
    """Factorise the (thermally degraded) base state for rank-1 sweeps.

    ``temps`` maps member id -> steel temperature [degC]; ``E_i`` is scaled by
    ``k_e_func(T_i)`` (EN 1993-1-2 material model by default).  The imposed
    elongation ``dl_pre`` (thermal expansion + fabrication) is computed from
    the same field via :func:`prestress_lengths` and stored on the setup, so
    every consumer — :func:`total_load_vector`, :func:`member_forces`,
    :func:`ci_sweep` — sees one consistent demand state.  ``loads`` is kept in
    the signature for API symmetry but the force vector is built by
    :func:`total_load_vector` at solve time.

    ``use_effective_alpha`` is passed straight to :func:`prestress_lengths` and
    has no other effect: the stiffness degradation ``k_E(T) E`` is identical
    either way, so the flag changes only how much thermal strain the imposed
    elongation carries.  See that function for the measured size of the
    difference and for why the default is the constant-``alpha`` path.
    """
    del loads  # force vector assembled at solve time
    k_scale = (
        {e.id: float(k_e_func(temps[e.id])) for e in elements}
        if temps is not None
        else {e.id: 1.0 for e in elements}
    )
    b, k = member_matrices(nodes, elements, k_scale)
    free = free_dof_indices(nodes)
    k_ff = np.einsum("i,ip,iq->pq", k, b, b)[np.ix_(free, free)]
    lu = lu_factor(k_ff)
    _check_lu(lu)
    b_free = b[:, list(free)]
    z = lu_solve(lu, b_free.T)
    d = np.einsum("ip,pi->i", b_free, z)
    return EngineSetup(
        ids=tuple(e.id for e in elements),
        free_dofs=free,
        lu=lu,
        b_free=b_free,
        k_axial=k,
        z=z,
        d=d,
        dl_pre=prestress_lengths(
            nodes, elements, temps, use_effective_alpha=use_effective_alpha
        ),
        guard_tol=_guard_tolerance(k_ff, lu),
    )


def load_vector(
    nodes: Sequence[Node],
    loads: Mapping[str, Mapping[str, float]],
    free: Sequence[int],
) -> np.ndarray:
    """Mechanical load vector restricted to the free DOFs."""
    f = np.zeros(2 * len(nodes))
    node_idx = {n.id: i for i, n in enumerate(nodes)}  # built once, not per load
    for node_id, load in loads.items():
        idx = node_idx[str(node_id)]
        f[2 * idx] += float(load.get("Fx", 0.0))
        f[2 * idx + 1] += float(load.get("Fy", 0.0))
    return f[list(free)]


def imposed_load_vector(setup: EngineSetup) -> np.ndarray:
    """Equivalent nodal forces from imposed strain, on the free DOFs.

    ``F_th = B^T diag(k(T)) dL_pre`` — a member with imposed elongation
    ``dL_pre`` and fixed nodes pushes on the structure with the self-
    equilibrated pair ``k * dL_pre``; restricted here to the free DOFs the
    engine solves on.  Identical physics to the ``F_ext`` term
    :func:`truss_analysis.assembly.assemble_global_matrices` hands to
    ``run()``, built from the same ``k_axial`` and ``dl_pre`` the setup
    carries, so the fire chain and the static solver cannot drift apart.
    """
    imposed: np.ndarray = setup.b_free.T @ (setup.k_axial * setup.dl_pre)
    return imposed


def total_load_vector(
    nodes: Sequence[Node],
    loads: Mapping[str, Mapping[str, float]],
    setup: EngineSetup,
) -> np.ndarray:
    """Full demand right-hand side (mechanical + imposed) on the free DOFs.

    This — not :func:`load_vector` alone — is what every base-state solve in
    the fire chain must use; see the module docstring.
    """
    total: np.ndarray = np.asarray(
        load_vector(nodes, loads, setup.free_dofs) + imposed_load_vector(setup),
        dtype=float,
    )
    return total


def member_forces(setup: EngineSetup, u: np.ndarray) -> np.ndarray:
    """Mechanical axial forces ``N_e = k_e (b_e . u - dL_pre,e)``, shape ``(nE,)``.

    Tension positive.  The single definition of member force for the whole
    criticality / limit-state chain: the elongation a member *stores
    elastically* is its total elongation minus the imposed part, and only
    that produces force.  With ``dL_pre = 0`` this is the previous
    ``k * (b . u)``.
    """
    forces: np.ndarray = setup.k_axial * (setup.b_free @ u - setup.dl_pre)
    return forces


def base_displacement(setup: EngineSetup, f_free: np.ndarray) -> np.ndarray:
    """Return the base (unperturbed) displacement field on the free DOFs."""
    u: np.ndarray = lu_solve(setup.lu, f_free)
    return u


def _solve_perturbed_full(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    alpha: float,
    member_index: int,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = eurocode_k_E,
) -> np.ndarray:
    """Solve the reference system with one member additionally scaled by alpha.

    Mirrors the rank-1 physics exactly: the perturbed member's stiffness is
    scaled by ``alpha`` **and** its equivalent thermal force with it (the
    imposed elongation itself is a geometric input and does not scale).
    """
    k_scale = {e.id: float(k_e_func(temps[e.id])) for e in elements}
    k_scale[elements[member_index].id] *= alpha
    b, k = member_matrices(nodes, elements, k_scale)
    free = free_dof_indices(nodes)
    k_ff = np.einsum("i,ip,iq->pq", k, b, b)[np.ix_(free, free)]
    lu = lu_factor(k_ff)
    _check_lu(lu)
    b_free = b[:, list(free)]
    dl_pre = prestress_lengths(nodes, elements, temps)
    f_free = load_vector(nodes, loads, free) + b_free.T @ (k * dl_pre)
    u_pert: np.ndarray = lu_solve(lu, f_free)
    return u_pert


def ci_sweep(
    setup: EngineSetup,
    u: np.ndarray,
    alpha: float,
    guard_tol: float | None = None,
    brute_column: Callable[[int], np.ndarray] | None = None,
) -> CiSweep:
    """Vectorised CI sweep: column i of ``u_pert`` = state with member i hit.

    The rank-1 numerator is the member's *mechanical* elongation
    ``f_i - dL_pre,i = N_i / k_i``, not its total elongation: softening
    member ``i`` scales its thermal equivalent force ``k_i dL_i b_i`` by the
    same ``alpha`` as its stiffness, and the two effects combine into exactly
    this form (derivation in the module docstring).  ``u`` must have been
    solved against :func:`total_load_vector` for the same setup.

    ``guard_tol`` defaults to ``setup.guard_tol`` — the condition-scaled
    tolerance measured from this setup's LU factors (see
    :func:`_guard_tolerance`); passing a float overrides it.
    """
    delta = (alpha - 1.0) * setup.k_axial
    denom = 1.0 + delta * setup.d
    f = setup.b_free @ u - setup.dl_pre
    coef = np.zeros_like(delta)
    # None -> the condition-scaled tolerance measured on this setup's LU
    # factors; an explicit float still overrides (tests, sensitivity studies).
    tol = setup.guard_tol if guard_tol is None else float(guard_tol)
    ok = np.abs(denom) >= tol
    coef[ok] = delta[ok] * f[ok] / denom[ok]
    u_pert = u[:, None] - setup.z * coef[None, :]
    flagged: dict[str, str] = {}
    for i in np.where(~ok)[0]:
        eid = setup.ids[i]
        if brute_column is None:
            msg = f"member {eid} near mechanism but no brute-force fallback given"
            raise MechanismError(msg)
        flagged[eid] = "guard:brute-force"
        try:
            u_pert[:, i] = brute_column(int(i))
        except MechanismError:
            flagged[eid] = "mechanism(singular)"
            u_pert[:, i] = np.inf
    if u.size == 0:
        # Every DOF restrained: no displacement response to rank, so every CI
        # is exactly zero (the member forces are pure imposed-strain forces,
        # which the force-based indices downstream still measure).
        return CiSweep(
            ci_values={eid: 0.0 for eid in setup.ids},
            u_pert=u_pert,
            flagged=flagged,
            u_max_perturbed_max=0.0,
        )
    u_max_base = float(np.max(np.abs(u)))
    finite = np.isfinite(u_pert).all(axis=0)
    col_max = np.where(
        finite,
        np.max(np.abs(np.where(np.isfinite(u_pert), u_pert, 0.0)), axis=0),
        np.inf,
    )
    ci_values = {
        setup.ids[i]: float(col_max[i] / u_max_base - 1.0)
        for i in range(len(setup.ids))
    }
    u_max_perturbed_max = float(np.max(col_max)) if len(col_max) else 0.0
    return CiSweep(ci_values, u_pert, flagged, u_max_perturbed_max)


def perturb_multi(
    setup: EngineSetup,
    u: np.ndarray,
    indices: Sequence[int],
    alphas: Sequence[float],
    condition_warning: bool = True,
) -> np.ndarray:
    """Woodbury rank-r update for simultaneous multi-member perturbations.

    ``K_pert = K + B_S Delta B_S^T`` with diagonal ``Delta``; used for retrofit
    studies where the rank-1 formula does not apply (see module docstring).
    The right-hand side carries the perturbed members' thermal equivalent
    forces too, which -- exactly as in the rank-1 case -- replaces the total
    elongation ``f_S`` with the mechanical one ``f_S - dL_S`` in the core
    solve.  ``u`` must come from :func:`total_load_vector`.

    Parameters
    ----------
    setup : EngineSetup
        Factorised base state from :func:`build_engine`.
    u : numpy.ndarray
        Base displacement on the free DOFs, from :func:`base_displacement`
        applied to :func:`total_load_vector`.
    indices : Sequence[int]
        Positional member indices perturbed simultaneously.
    alphas : Sequence[float]
        Stiffness multipliers, one per entry of ``indices``.  ``alpha = 1``
        is rejected (a no-op that would divide by zero); ``alpha <= 0``
        removes the member outright.
    condition_warning : bool, default True
        Issue an
        :exc:`~truss_analysis.exceptions.IllConditionedPerturbationWarning`
        when the Woodbury core is ill-conditioned or the solve fails its
        residual check.  Set to ``False`` inside tight loops that screen the
        result themselves.

    Returns
    -------
    numpy.ndarray
        Perturbed displacement on the free DOFs.

    Raises
    ------
    ValueError
        If any ``alpha == 1``.
    MechanismError
        If the simultaneous removals exceed the system's static
        indeterminacy (guaranteed mechanism, detected combinatorially before
        any factorisation), or if the Woodbury core is singular to working
        precision.

    Warns
    -----
    IllConditionedPerturbationWarning
        If ``cond(core) > PERTURB_COND_WARN`` or the backward error exceeds
        ``PERTURB_RESID_WARN``.
    """
    idx = list(indices)
    alpha_arr = np.asarray(alphas, dtype=float)
    deltas = (alpha_arr - 1.0) * setup.k_axial[idx]
    if np.any(deltas == 0.0):
        msg = "perturb_multi: alpha=1.0 members are no-ops; drop them"
        raise ValueError(msg)

    # --- C3: redundancy screen, before any linear algebra -----------------
    # Removing a member outright (alpha <= 0) takes one bar away without
    # changing the DOF count, so the perturbed assembly has redundancy
    # (m - k) - n_free.  When that goes negative the structure is *certain*
    # to be a mechanism -- no factorisation needed to know it -- and saying
    # so in engineering terms beats letting a near-zero pivot surface as a
    # LinAlgError or, worse, as a finite but meaningless answer.
    n_removed = int(np.count_nonzero(alpha_arr <= 0.0))
    redundancy = setup_indeterminacy(setup)
    if n_removed > redundancy:
        members = ", ".join(setup.ids[i] for i in idx)
        msg = (
            f"perturb_multi: {n_removed} member(s) removed ({members}) from a "
            f"system with only {redundancy} degree(s) of static "
            f"indeterminacy; the perturbed assembly is a mechanism "
            f"({redundancy - n_removed} < 0 redundancy remains)"
        )
        raise MechanismError(msg)

    z_s = setup.z[:, idx]
    g_ss = setup.b_free[idx] @ z_s
    core = np.diag(1.0 / deltas) + g_ss
    f_s = setup.b_free[idx] @ u - setup.dl_pre[idx]
    # The Woodbury core is singular exactly when the simultaneously perturbed
    # structure is (or is numerically indistinguishable from) a mechanism.
    # Screen it with the same relative criterion the base-state solve uses,
    # so the library raises its contract error instead of a bare LinAlgError
    # -- or, worse, a silent finite answer from an ~1e-17 pivot (round-5
    # audit: deleting two members of a determinate triangle left the 2x2
    # core mathematically singular but numerically solvable garbage).
    members = ", ".join(setup.ids[i] for i in idx)
    with warnings.catch_warnings():
        # lu_factor emits LinAlgWarning on an exactly-zero pivot; the screen
        # below turns that case into the library's own MechanismError, so the
        # raw warning would only be noise on top of a handled condition.
        warnings.simplefilter("ignore", LinAlgWarning)
        lu = lu_factor(core, check_finite=False)
    _check_lu(lu, f"after simultaneous perturbation of member(s) {members}")
    core_sol = lu_solve(lu, f_s)

    # --- C2: measured quality of the Woodbury solve -----------------------
    # _check_lu only rejects cores that are singular to working precision.
    # A core with cond ~ 1e13 passes that gate and still returns digits that
    # are mostly noise, and the caller (retrofit triage, multi-member
    # criticality) would rank members on it.  The core is r x r with r the
    # number of simultaneously perturbed members -- small enough that an
    # exact condition number and an explicit residual are both cheap.
    if condition_warning:
        cond_core = float(np.linalg.cond(core))
        residual = core @ core_sol - f_s
        denom = max(
            float(np.linalg.norm(core, 1)) * float(np.linalg.norm(core_sol, np.inf)),
            float(np.linalg.norm(f_s, np.inf)),
            _TINY_DENOM,
        )
        rel_resid = float(np.linalg.norm(residual, np.inf) / denom)
        if cond_core > PERTURB_COND_WARN or rel_resid > PERTURB_RESID_WARN:
            warnings.warn(
                f"Woodbury core for "
                f"member(s) {members} is ill-conditioned "
                f"(cond = {cond_core:.3e}, relative residual = "
                f"{rel_resid:.3e}); the perturbed displacements may carry "
                "fewer trustworthy digits than the ranking assumes. Verify "
                "with brute_force_ci() or split the perturbation.",
                IllConditionedPerturbationWarning,
                stacklevel=2,
            )

    perturbed: np.ndarray = u - z_s @ core_sol
    return perturbed


def brute_force_ci(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    alpha: float,
    tol: float = 1e-9,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = eurocode_k_E,
) -> tuple[dict[str, float], float, float]:
    """Compute CI values by brute force: one full factorisation per member.

    Reference path used to verify the rank-1 engine.

    Returns
    -------
    tuple[dict[str, float], float, float]
        ``(ci_values, u_max_base, u_max_perturbed_max)``.
    """
    k_scale = {e.id: float(k_e_func(temps[e.id])) for e in elements}
    b, k = member_matrices(nodes, elements, k_scale)
    free = free_dof_indices(nodes)
    lu = lu_factor(np.einsum("i,ip,iq->pq", k, b, b)[np.ix_(free, free)])
    _check_lu(lu)
    b_free = b[:, list(free)]
    dl_pre = prestress_lengths(nodes, elements, temps)
    f_free = load_vector(nodes, loads, free) + b_free.T @ (k * dl_pre)
    u = lu_solve(lu, f_free)
    if u.size == 0:
        # Every DOF restrained: no displacement response, every CI is zero
        # (same convention as ci_sweep).
        return {e.id: 0.0 for e in elements}, 0.0, 0.0
    u_max_base = float(np.max(np.abs(u)))
    ci: dict[str, float] = {}
    u_max_pert = 0.0
    for i, e in enumerate(elements):
        if u_max_base < tol:
            ci[e.id] = 0.0
            continue
        u_p = _solve_perturbed_full(nodes, elements, loads, temps, alpha, i, k_e_func)
        m = float(np.max(np.abs(u_p)))
        u_max_pert = max(u_max_pert, m)
        ci[e.id] = m / u_max_base - 1.0
    return ci, u_max_base, u_max_pert


def compute_ci_for_topology(
    nodes: list[Node],
    elements: list[Element],
    loads: Mapping[str, Mapping[str, float]],
    supports: Mapping[str, object],
    scenario: str,
    t_target: float,
    alpha: float = 0.7,
    tol: float = 1e-9,
) -> TopologyResult:
    """Run the full CI pipeline on the rank-1 engine (no special-cased scenarios).

    The uniform scenario travels the same numerical path as every other
    scenario: uniform-temperature invariance of the ranking is a measured
    outcome, not an input.  ``tau_vs_base`` is populated with tau-b between
    the CI field at ``t_target`` and the one at :data:`T_AMBIENT`.
    """
    del supports  # boundary conditions live on the Node flags
    temps = get_scenario_temperatures(nodes, elements, scenario, t_target)
    setup = build_engine(nodes, elements, loads, temps)
    f_free = total_load_vector(nodes, loads, setup)
    u = base_displacement(setup, f_free)
    u_max_base = float(np.max(np.abs(u)))

    flags: dict[str, str] = {}
    if u_max_base < tol:
        ci_values = {e.id: 0.0 for e in elements}
        flags["base"] = "zero-displacement"
        u_max_perturbed_max = 0.0
    else:
        brute = lambda i: _solve_perturbed_full(  # noqa: E731
            nodes, elements, loads, temps, alpha, i
        )
        sweep = ci_sweep(setup, u, alpha, brute_column=brute)
        ci_values = sweep.ci_values
        u_max_perturbed_max = sweep.u_max_perturbed_max
        flags.update(sweep.flagged)

    nci: NciResult = compute_nci(ci_values)
    ranks = rank_members(ci_values)

    tau_vs_base: float | None = None
    if u_max_base >= tol:
        # tau is a discrete rank statistic: compare at the rank tolerance
        # (1e-10) so sub-tolerance float noise on mirror pairs is not counted
        # as rank instability (see ranking.tau_b docstring).
        if float(t_target) == T_AMBIENT:
            tau_res: TauResult = tau_b(ci_values, ci_values, quantize=1e-10)
        else:
            temps_base = get_scenario_temperatures(nodes, elements, scenario, T_AMBIENT)
            setup0 = build_engine(nodes, elements, loads, temps_base)
            u0 = base_displacement(setup0, total_load_vector(nodes, loads, setup0))
            if float(np.max(np.abs(u0))) >= tol:
                # the cold reference sweep takes the SAME guard routing as
                # the hot one: with alpha near 0 on a determinate truss every
                # cold member can be near-mechanism, so the brute-force
                # fallback must flag those members instead of raising
                brute0 = lambda i: _solve_perturbed_full(  # noqa: E731
                    nodes, elements, loads, temps_base, alpha, i
                )
                sweep0 = ci_sweep(setup0, u0, alpha, brute_column=brute0)
                tau_res = tau_b(ci_values, sweep0.ci_values, quantize=1e-10)
            else:
                tau_res = TauResult(None, 0, 0, 0, None, True)
        tau_vs_base = tau_res.tau

    return TopologyResult(
        topology_id=f"{scenario}_{t_target}C",
        scenario=scenario,
        temperature=float(t_target),
        alpha=alpha,
        ci_values=ci_values,
        nci_values=nci.values,
        ranks=ranks,
        top_5=ranks[:5],
        tau_vs_base=tau_vs_base,
        u_max_base=u_max_base,
        u_max_perturbed_max=u_max_perturbed_max,
        flags=flags,
    )
