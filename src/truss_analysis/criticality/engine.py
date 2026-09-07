"""Exact rank-1 criticality engine (Sherman–Morrison) and the CI pipeline.

Mathematics (verified in CONTEXT_LOCK §4.9 and by ``test_engine_equivalence``)
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

Validity limit (explicit, prompt-04 §A1)
-----------------------------------------
The rank-1 path is exact **only for single-member perturbations**.  For
simultaneous multi-member perturbations (retrofit studies) use
:func:`perturb_multi` (Woodbury rank-r) or a full re-solve; applying the
rank-1 formula member-by-member to a multi-member change is wrong and no
public function here does it.

Numerical guard (prompt-04 §A1)
-------------------------------
``1 + Delta_i d_i -> 0`` means the perturbed structure is (near) a mechanism.
Such members are flagged and routed to a full brute-force solve; if that
solve is singular the member CI is ``+inf`` with a ``mechanism`` flag.  The
engine never emits a silent finite number for a guarded member.

No deep copying of element containers anywhere in this package
(prompt-04 §A4): perturbed states are built from the rank-1/rank-r formulas
or from fresh dataclass instances via :func:`dataclasses.replace`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from truss_analysis.material.steel_eurocode import FloatOrArray
from truss_analysis.material.steel_eurocode import k_E as ssot_k_E
from truss_analysis.model import Element, Node

from .indices import NciResult, compute_nci
from .ranking import TauResult, rank_members, tau_b
from .scenarios import T_AMBIENT, get_scenario_temperatures

logger = logging.getLogger(__name__)

__all__ = [
    "CiSweep",
    "EngineSetup",
    "GUARD_TOL",
    "MechanismError",
    "TopologyResult",
    "base_displacement",
    "brute_force_ci",
    "build_engine",
    "ci_sweep",
    "compute_ci_for_topology",
    "load_vector",
    "member_matrices",
    "perturb_multi",
]

GUARD_TOL = 1e-8
_SINGULAR_TOL = 1e-12


class MechanismError(RuntimeError):
    """Raised when a stiffness matrix is singular (mechanism detected)."""


@dataclass(frozen=True)
class TopologyResult:
    """Immutable result of one (topology, scenario, temperature, alpha) run."""

    topology_id: str
    scenario: str
    temperature: float
    alpha: float
    ci_values: Dict[str, float]
    nci_values: Optional[Dict[str, float]]
    ranks: List[str]
    top_5: List[str]
    tau_vs_base: Optional[float]
    u_max_base: float
    u_max_perturbed_max: float
    flags: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineSetup:
    """Factorised base state plus the rank-1 machinery for all members."""

    ids: Tuple[str, ...]
    free_dofs: Tuple[int, ...]
    lu: Tuple[np.ndarray, np.ndarray]
    b_free: np.ndarray  # (nE, ndof_free)
    k_axial: np.ndarray  # (nE,)
    z: np.ndarray  # (ndof_free, nE) = K_ff^-1 B
    d: np.ndarray  # (nE,) = b_i^T Z_i


@dataclass(frozen=True)
class CiSweep:
    """Full CI sweep over all members in one vectorised pass."""

    ci_values: Dict[str, float]
    u_pert: np.ndarray  # (ndof_free, nE); +inf columns mark mechanisms
    flagged: Dict[str, str]
    u_max_perturbed_max: float


def free_dof_indices(nodes: Sequence[Node]) -> Tuple[int, ...]:
    fixed = set()
    for i, node in enumerate(nodes):
        if node.is_support:
            if node.support_dx:
                fixed.add(2 * i)
            if node.support_dy:
                fixed.add(2 * i + 1)
    return tuple(d for d in range(2 * len(nodes)) if d not in fixed)


def member_matrices(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    k_scale: Mapping[str, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
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


def _check_lu(lu: Tuple[np.ndarray, np.ndarray]) -> None:
    lu_mat, _piv = lu  # scipy packs L and U into one matrix; second item is pivots
    diag = np.abs(np.diag(lu_mat))
    scale = max(float(np.max(diag)), 1.0)
    if float(np.min(diag)) < _SINGULAR_TOL * scale:
        msg = "stiffness matrix singular (mechanism) in base state"
        raise MechanismError(msg)


def build_engine(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float] | None = None,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = ssot_k_E,
) -> EngineSetup:
    """Factorise the (thermally degraded) base state for rank-1 sweeps.

    ``temps`` maps member id -> steel temperature [degC]; ``E_i`` is scaled by
    ``k_e_func(T_i)`` (EN 1993-1-2 SSOT by default).  ``loads`` is kept in the
    signature for API symmetry but the force vector is built by
    :func:`load_vector` at solve time.
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
    )


def load_vector(
    nodes: Sequence[Node],
    loads: Mapping[str, Mapping[str, float]],
    free: Sequence[int],
) -> np.ndarray:
    """Mechanical load vector restricted to the free DOFs."""
    f = np.zeros(2 * len(nodes))
    for node_id, load in loads.items():
        idx = {n.id: i for i, n in enumerate(nodes)}[str(node_id)]
        f[2 * idx] += float(load.get("Fx", 0.0))
        f[2 * idx + 1] += float(load.get("Fy", 0.0))
    return f[list(free)]


def base_displacement(setup: EngineSetup, f_free: np.ndarray) -> np.ndarray:
    """Base (unperturbed) displacement field on the free DOFs."""
    return lu_solve(setup.lu, f_free)


def _solve_perturbed_full(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    alpha: float,
    member_index: int,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = ssot_k_E,
) -> np.ndarray:
    """Reference solve with member ``member_index`` additionally scaled by alpha."""
    k_scale = {e.id: float(k_e_func(temps[e.id])) for e in elements}
    k_scale[elements[member_index].id] *= alpha
    b, k = member_matrices(nodes, elements, k_scale)
    free = free_dof_indices(nodes)
    k_ff = np.einsum("i,ip,iq->pq", k, b, b)[np.ix_(free, free)]
    lu = lu_factor(k_ff)
    _check_lu(lu)
    return lu_solve(lu, load_vector(nodes, loads, free))


def ci_sweep(
    setup: EngineSetup,
    u: np.ndarray,
    alpha: float,
    guard_tol: float = GUARD_TOL,
    brute_column: Callable[[int], np.ndarray] | None = None,
) -> CiSweep:
    """Vectorised CI sweep: column i of ``u_pert`` = state with member i hit."""
    delta = (alpha - 1.0) * setup.k_axial
    denom = 1.0 + delta * setup.d
    f = setup.b_free @ u
    coef = np.zeros_like(delta)
    ok = np.abs(denom) >= guard_tol
    coef[ok] = delta[ok] * f[ok] / denom[ok]
    u_pert = u[:, None] - setup.z * coef[None, :]
    flagged: Dict[str, str] = {}
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
) -> np.ndarray:
    """Woodbury rank-r update for simultaneous multi-member perturbations.

    ``K_pert = K + B_S Delta B_S^T`` with diagonal ``Delta``; used for retrofit
    studies where the rank-1 formula does not apply (see module docstring).
    """
    idx = list(indices)
    deltas = (np.asarray(alphas, dtype=float) - 1.0) * setup.k_axial[idx]
    if np.any(deltas == 0.0):
        msg = "perturb_multi: alpha=1.0 members are no-ops; drop them"
        raise ValueError(msg)
    z_s = setup.z[:, idx]
    g_ss = setup.b_free[idx] @ z_s
    core = np.diag(1.0 / deltas) + g_ss
    f_s = setup.b_free[idx] @ u
    return u - z_s @ np.linalg.solve(core, f_s)


def brute_force_ci(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    alpha: float,
    tol: float = 1e-9,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = ssot_k_E,
) -> Tuple[Dict[str, float], float, float]:
    """Reference implementation: one full factorisation per member.

    Returns ``(ci_values, u_max_base, u_max_perturbed_max)``.
    """
    k_scale = {e.id: float(k_e_func(temps[e.id])) for e in elements}
    b, k = member_matrices(nodes, elements, k_scale)
    free = free_dof_indices(nodes)
    lu = lu_factor(np.einsum("i,ip,iq->pq", k, b, b)[np.ix_(free, free)])
    _check_lu(lu)
    f_free = load_vector(nodes, loads, free)
    u = lu_solve(lu, f_free)
    u_max_base = float(np.max(np.abs(u)))
    ci: Dict[str, float] = {}
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
    nodes: List[Node],
    elements: List[Element],
    loads: Dict,
    supports: Dict,
    scenario: str,
    t_target: float,
    alpha: float = 0.7,
    tol: float = 1e-9,
) -> TopologyResult:
    """Full CI pipeline on the rank-1 engine (no special-cased scenarios).

    The uniform scenario travels the same numerical path as every other
    scenario (prompt-04 §B5): Lemma 1 is a measured outcome, not an input.
    ``tau_vs_base`` is populated with tau-b between the CI field at
    ``t_target`` and the one at :data:`T_AMBIENT` (prompt-04 §C12).
    """
    del supports  # boundary conditions live on the Node flags
    temps = get_scenario_temperatures(nodes, elements, scenario, t_target)
    setup = build_engine(nodes, elements, loads, temps)
    f_free = load_vector(nodes, loads, setup.free_dofs)
    u = base_displacement(setup, f_free)
    u_max_base = float(np.max(np.abs(u)))

    flags: Dict[str, str] = {}
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

    tau_vs_base: Optional[float] = None
    if u_max_base >= tol:
        # tau is a discrete rank statistic: compare at the lemma tolerance
        # (1e-10) so sub-tolerance float noise on mirror pairs is not counted
        # as rank instability (see ranking.tau_b docstring).
        if float(t_target) == T_AMBIENT:
            tau_res: TauResult = tau_b(ci_values, ci_values, quantize=1e-10)
        else:
            temps_base = get_scenario_temperatures(nodes, elements, scenario, T_AMBIENT)
            setup0 = build_engine(nodes, elements, loads, temps_base)
            u0 = base_displacement(setup0, load_vector(nodes, loads, setup0.free_dofs))
            if float(np.max(np.abs(u0))) >= tol:
                sweep0 = ci_sweep(setup0, u0, alpha)
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
