"""OpenSeesPy reference bridge — independent solver, no synthetic fallbacks.

Scope of the comparison (explicit, to preempt the circularity objection)
------------------------------------------------------------------------
The OpenSees ``Truss`` element has **no temperature dependence of its own**.
Every member therefore receives an explicit ``uniaxialMaterial Elastic`` with
``E_i(T) = k_E(T_i) * E_i`` built from the same single-source reduction curves that the
internal engine uses.  The comparison consequently validates the
*structural* path — assembly, boundary conditions, solution, the single-member
stiffness-perturbation (criticality) sweep and its ranking — against an
independent, widely used solver, given an identical modulus field.  It does
NOT re-validate the reduction curves themselves; those are checked against
the published table and independent literature separately (level 2 of the
validation suite).

The criticality sweep is reproduced in OpenSees the hard way: **n+1 fully
independent models** (baseline plus one model per member with that member's
modulus additionally scaled by ``alpha``).  Comparing those against the
internal rank-1 (Sherman-Morrison) sweep is the strongest equivalence
witness available: exact linear algebra on one side, an external black-box
finite-element solver on the other.

Honesty rule
------------
There is **no mock mode**.  When OpenSeesPy is not installed, every bridge
function raises :class:`OpenseesUnavailableError`; test callers skip
explicitly via ``pytest.importorskip``.  Fabricated "reference" numbers are
worse than no reference at all.

Sign conventions (measured, not assumed)
----------------------------------------
``ops.basicForce`` on a ``Truss`` element returns the axial force with
tension positive — identical to the internal post-processing convention.
Displacements are read with ``ops.nodeDisp`` in global X/Y order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..criticality.engine import (
    base_displacement,
    build_engine,
    ci_sweep,
    member_forces,
    member_matrices,
    prestress_lengths,
    total_load_vector,
)
from ..material.steel_eurocode import FloatOrArray
from ..material.steel_eurocode import k_E as eurocode_k_E
from ..model import Element, Node
from .metrics import RhoBranch, classify_rho, rank_correlation

try:  # pragma: no cover - environment dependent
    from openseespy import opensees as _ops
except ImportError:  # pragma: no cover - environment dependent
    _ops = None
_HAS_OPENSEES = _ops is not None

RHO_QUANTIZE: float = 1e-10

__all__ = [
    "RHO_QUANTIZE",
    "CiRankingComparison",
    "OpenseesSolveError",
    "OpenseesUnavailableError",
    "StateComparison",
    "ci_sweep_in_opensees",
    "compare_ci_ranking",
    "compare_state",
    "opensees_available",
    "opensees_version",
    "solve_truss_in_opensees",
]


class OpenseesUnavailableError(RuntimeError):
    """Raised when a reference solve is requested without OpenSeesPy."""


class OpenseesSolveError(RuntimeError):
    """Raised when the OpenSees static analysis reports a non-zero status."""


def opensees_available() -> bool:
    """Whether the optional OpenSeesPy dependency can be imported."""
    return _HAS_OPENSEES


def opensees_version() -> str:
    """Installed OpenSeesPy distribution version (or ``'absent'``)."""
    if not _HAS_OPENSEES:
        return "absent"
    from importlib.metadata import PackageNotFoundError, version

    try:
        return str(version("openseespy"))
    except PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"


def _require_ops() -> Any:
    if _ops is None:
        msg = (
            "OpenSeesPy is not installed; the reference bridge refuses to "
            "fabricate results. Install 'openseespy' (Linux additionally "
            "needs libquadmath/libblas/liblapack/libgfortran) or skip the "
            "reference level explicitly."
        )
        raise OpenseesUnavailableError(msg)
    return _ops


@dataclass(frozen=True)
class OpenseesSolution:
    """One linear static solve of the pin-jointed model in OpenSees."""

    node_ids: tuple[str, ...]
    disp: NDArray[np.float64]  # (2*nN,) global DOF order of ``node_ids``
    member_forces: dict[str, float]  # tension positive
    analyze_status: int


def solve_truss_in_opensees(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    k_scale: Mapping[str, float] | None = None,
    imposed_nodal_forces: NDArray[np.float64] | None = None,
) -> OpenseesSolution:
    """Build and solve the truss in OpenSeesPy (one fresh model per call).

    ``k_scale`` maps member id -> multiplier on its Young's modulus (use it
    to pass ``k_E(T_i)`` explicitly; OpenSees applies no thermal reduction
    by itself).

    ``imposed_nodal_forces`` is an optional full-DOF vector of equivalent
    nodal forces from imposed (thermal/fabrication) strain — the reference
    side of what :func:`truss_analysis.criticality.engine.imposed_load_vector`
    builds internally.  It is applied as ordinary nodal loads in the pattern.
    Note the consequence for member forces: OpenSees then reports
    ``basicForce = k * (b . u)``, the *total*-elongation force, while the
    internal engine reports the mechanical one ``k * (b . u - dL_pre)``.
    :func:`compare_state` applies exactly that offset when it compares.
    """
    ops = _require_ops()
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    tag_of_node: dict[str, int] = {}
    for pos, node in enumerate(nodes):
        tag = pos + 1
        tag_of_node[node.id] = tag
        ops.node(tag, float(node.x), float(node.y))
    for node in nodes:
        if node.is_support and (node.support_dx or node.support_dy):
            ops.fix(
                tag_of_node[node.id],
                1 if node.support_dx else 0,
                1 if node.support_dy else 0,
            )
    for pos, elem in enumerate(elements):
        mat_tag = pos + 1
        scale = 1.0 if k_scale is None else float(k_scale[elem.id])
        ops.uniaxialMaterial("Elastic", mat_tag, float(elem.E) * scale)
        ops.element(
            "Truss",
            mat_tag,
            tag_of_node[elem.node_i],
            tag_of_node[elem.node_j],
            float(elem.A),
            mat_tag,
        )
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    nodal = np.zeros(2 * len(nodes))
    for node_id, load in loads.items():
        i = tag_of_node[str(node_id)] - 1
        nodal[2 * i] += float(load.get("Fx", 0.0))
        nodal[2 * i + 1] += float(load.get("Fy", 0.0))
    if imposed_nodal_forces is not None:
        nodal = nodal + np.asarray(imposed_nodal_forces, dtype=float)
    for pos, node in enumerate(nodes):
        fx, fy = float(nodal[2 * pos]), float(nodal[2 * pos + 1])
        if fx != 0.0 or fy != 0.0:
            ops.load(tag_of_node[node.id], fx, fy)
    ops.system("BandSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    status = int(ops.analyze(1))
    if status != 0:
        msg = f"OpenSees analyze() returned {status} (singular or failed)"
        raise OpenseesSolveError(msg)
    disp = np.zeros(2 * len(nodes))
    for node in nodes:
        tag = tag_of_node[node.id]
        disp[2 * (tag - 1)] = float(ops.nodeDisp(tag, 1))
        disp[2 * (tag - 1) + 1] = float(ops.nodeDisp(tag, 2))
    forces: dict[str, float] = {}
    for pos, elem in enumerate(elements):
        forces[elem.id] = float(ops.basicForce(pos + 1)[0])
    return OpenseesSolution(
        node_ids=tuple(n.id for n in nodes),
        disp=disp,
        member_forces=forces,
        analyze_status=status,
    )


@dataclass(frozen=True)
class OpenseesCiSweep:
    """CI field from n+1 independent OpenSees models (no rank-1 shortcut)."""

    ci_values: dict[str, float]
    u_max_base: float
    u_max_perturbed_max: float
    n_solves: int


def _imposed_nodal_vector(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    temps: Mapping[str, float],
    k_scale: Mapping[str, float],
) -> NDArray[np.float64]:
    """Full-DOF equivalent nodal forces ``B^T diag(k_scale*k0) dL_pre``.

    The OpenSees-side mirror of the engine's ``imposed_load_vector``:
    the same imposed elongations, the same (scaled) stiffnesses, assembled on
    the full DOF map instead of the free one.  ``k_scale`` must be the field
    the solve uses, so a perturbed member's thermal force scales with its
    perturbed stiffness exactly as in the rank-1 formula.
    """
    b, k = member_matrices(nodes, elements, k_scale)
    dl_pre = prestress_lengths(nodes, elements, temps)
    imposed: NDArray[np.float64] = b.T @ (k * dl_pre)
    return imposed


def ci_sweep_in_opensees(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    alpha: float = 0.7,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = eurocode_k_E,
) -> OpenseesCiSweep:
    """Run the reference criticality sweep: baseline + one degraded member."""
    base_scale = {e.id: float(k_e_func(temps[e.id])) for e in elements}
    base_imposed = _imposed_nodal_vector(nodes, elements, temps, base_scale)
    base = solve_truss_in_opensees(nodes, elements, loads, base_scale, base_imposed)
    u_max_base = float(np.max(np.abs(base.disp)))
    if u_max_base <= 0.0:
        msg = "zero baseline displacement: the CI ratio is undefined"
        raise ValueError(msg)
    ci: dict[str, float] = {}
    u_max_pert_max = 0.0
    for elem in elements:
        scale = dict(base_scale)
        scale[elem.id] *= float(alpha)
        imposed = _imposed_nodal_vector(nodes, elements, temps, scale)
        pert = solve_truss_in_opensees(nodes, elements, loads, scale, imposed)
        m = float(np.max(np.abs(pert.disp)))
        u_max_pert_max = max(u_max_pert_max, m)
        ci[elem.id] = m / u_max_base - 1.0
    return OpenseesCiSweep(ci, u_max_base, u_max_pert_max, len(elements) + 1)


def _internal_state(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    k_e_func: Callable[[FloatOrArray], FloatOrArray],
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Full-DOF displacement and member forces from the internal engine.

    ``build_engine`` receives the *temperature field* and applies
    ``k_e_func`` itself — passing pre-computed scale factors here would
    silently solve the wrong (undegraded) state (found by the level-4 smoke
    test: u_max off by exactly 1/k_E, forces identical because a uniform
    E-scaling leaves forces of a loaded truss invariant).

    The base state is solved against :func:`total_load_vector` (mechanical +
    restrained-expansion equivalent forces) and the reported forces are the
    mechanical ones from :func:`member_forces` — the same demand state the
    DCR chain sees.
    """
    setup = build_engine(nodes, elements, loads, temps, k_e_func)
    u_free = base_displacement(setup, total_load_vector(nodes, loads, setup))
    u_full = np.zeros(2 * len(nodes))
    u_full[list(setup.free_dofs)] = u_free
    forces_arr = member_forces(setup, u_free)
    forces = {eid: float(forces_arr[i]) for i, eid in enumerate(setup.ids)}
    return u_full, forces


@dataclass(frozen=True)
class StateComparison:
    """Node-by-node / member-by-member comparison of one physical state."""

    temperature_field: dict[str, float]
    u_max_internal: float
    u_max_opensees: float
    rel_err_u_max: float
    max_abs_node_err: float  # [m], over all nodal DOFs
    max_rel_node_err: float  # scaled by max|u| of the internal solution
    max_rel_force_err: float  # scaled by max|N| of the internal solution
    nodal_disp_internal: dict[str, list[float]]
    nodal_disp_opensees: dict[str, list[float]]
    forces_internal: dict[str, float]
    forces_opensees: dict[str, float]


def compare_state(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = eurocode_k_E,
) -> StateComparison:
    """Solve the identical degraded state in both solvers and compare.

    Both sides carry the same physical demand: the OpenSees model receives
    the restrained-expansion equivalent forces as nodal loads, and its
    ``basicForce`` readings (total-elongation forces ``k b.u``) are converted
    to the mechanical convention ``k (b.u - dL_pre)`` the internal engine
    reports, so the force comparison is like-for-like even for heated
    redundant structures.
    """
    k_scale = {e.id: float(k_e_func(temps[e.id])) for e in elements}
    u_int, f_int = _internal_state(nodes, elements, loads, temps, k_e_func)
    imposed = _imposed_nodal_vector(nodes, elements, temps, k_scale)
    sol = solve_truss_in_opensees(nodes, elements, loads, k_scale, imposed)

    # basicForce = k * (b . u) = N_mech + k * dL_pre  ->  subtract the offset.
    _b, k_arr = member_matrices(nodes, elements, k_scale)
    dl_pre = prestress_lengths(nodes, elements, temps)
    forces_ops = {
        e.id: float(sol.member_forces[e.id]) - float(k_arr[i] * dl_pre[i])
        for i, e in enumerate(elements)
    }

    u_scale = float(np.max(np.abs(u_int)))
    f_scale = max(abs(v) for v in f_int.values()) if f_int else 0.0
    abs_node_err = float(np.max(np.abs(u_int - sol.disp))) if len(u_int) else 0.0
    rel_node = abs_node_err / u_scale if u_scale > 0.0 else 0.0
    force_errs = [
        abs(f_int[e.id] - forces_ops[e.id]) / f_scale if f_scale > 0.0 else 0.0
        for e in elements
    ]
    u_max_ops = float(np.max(np.abs(sol.disp)))
    rel_umax = abs(u_max_ops - u_scale) / u_scale if u_scale > 0.0 else float("nan")
    disp_int: dict[str, list[float]] = {}
    disp_ops: dict[str, list[float]] = {}
    for i, node in enumerate(nodes):
        disp_int[node.id] = [float(u_int[2 * i]), float(u_int[2 * i + 1])]
        disp_ops[node.id] = [float(sol.disp[2 * i]), float(sol.disp[2 * i + 1])]
    return StateComparison(
        temperature_field=dict(temps),
        u_max_internal=u_scale,
        u_max_opensees=u_max_ops,
        rel_err_u_max=float(rel_umax),
        max_abs_node_err=abs_node_err,
        max_rel_node_err=float(rel_node),
        max_rel_force_err=float(max(force_errs) if force_errs else 0.0),
        nodal_disp_internal=disp_int,
        nodal_disp_opensees=disp_ops,
        forces_internal=f_int,
        forces_opensees=forces_ops,
    )


@dataclass(frozen=True)
class CiRankingComparison:
    """Rank-1 internal sweep vs the n+1-model OpenSees reference sweep."""

    alpha: float
    ci_internal: dict[str, float]
    ci_opensees: dict[str, float]
    rho: float
    branch: RhoBranch
    max_abs_ci_diff: float
    max_rel_ci_diff: float  # scaled by max|CI| of the internal sweep
    u_max_base_internal: float
    u_max_base_opensees: float
    n_opensees_solves: int
    extra: dict[str, Any] = field(default_factory=dict)


def compare_ci_ranking(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    alpha: float = 0.7,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = eurocode_k_E,
) -> CiRankingComparison:
    """Compare criticality fields and classify the rank correlation."""
    setup = build_engine(nodes, elements, loads, temps, k_e_func)
    u_free = base_displacement(setup, total_load_vector(nodes, loads, setup))
    sweep = ci_sweep(setup, u_free, alpha)
    u_max_int = float(np.max(np.abs(u_free)))
    ref = ci_sweep_in_opensees(nodes, elements, loads, temps, alpha, k_e_func)
    # Tie-noise convention (same as ranking.tau_b): two exact
    # solvers agree on the CI field to ~1e-13, but symmetric topologies
    # carry near-tied CI pairs whose ORDER is pure floating-point noise.
    # The ranked quantity is therefore computed on values quantised at
    # 1e-10; the raw (unquantised) Spearman rho is kept in ``extra``.
    rho = rank_correlation(sweep.ci_values, ref.ci_values, quantize=RHO_QUANTIZE)
    rho_raw = rank_correlation(sweep.ci_values, ref.ci_values)
    branch = classify_rho(rho)
    ci_scale = max(abs(v) for v in sweep.ci_values.values()) or 1.0
    diffs = [abs(sweep.ci_values[eid] - ref.ci_values[eid]) for eid in sweep.ci_values]
    return CiRankingComparison(
        alpha=float(alpha),
        ci_internal=dict(sweep.ci_values),
        ci_opensees=dict(ref.ci_values),
        rho=rho,
        branch=branch,
        max_abs_ci_diff=float(max(diffs)) if diffs else 0.0,
        max_rel_ci_diff=float(max(diffs) / ci_scale) if diffs else 0.0,
        u_max_base_internal=u_max_int,
        u_max_base_opensees=ref.u_max_base,
        n_opensees_solves=ref.n_solves,
        extra={"rho_raw_unquantised": rho_raw, "rho_quantize": RHO_QUANTIZE},
    )
