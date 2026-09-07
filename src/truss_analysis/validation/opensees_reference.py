"""OpenSeesPy reference bridge — independent solver, no synthetic fallbacks.

Scope of the comparison (explicit, to preempt the circularity objection)
------------------------------------------------------------------------
The OpenSees ``Truss`` element has **no temperature dependence of its own**.
Every member therefore receives an explicit ``uniaxialMaterial Elastic`` with
``E_i(T) = k_E(T_i) * E_i`` built from the same reduction-curve SSOT that the
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
internal rank-1 (Sherman–Morrison) sweep is the strongest equivalence
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

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..criticality.engine import (
    base_displacement,
    build_engine,
    ci_sweep,
    load_vector,
)
from ..material.steel_eurocode import FloatOrArray
from ..material.steel_eurocode import k_E as ssot_k_E
from ..model import Element, Node
from .metrics import RhoBranch, classify_rho, rank_correlation

try:  # pragma: no cover - environment dependent
    from openseespy import opensees as _ops  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    _ops = None
_HAS_OPENSEES = _ops is not None

RHO_QUANTIZE: float = 1e-10

__all__ = [
    "CiRankingComparison",
    "RHO_QUANTIZE",
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

    node_ids: Tuple[str, ...]
    disp: NDArray[np.float64]  # (2*nN,) global DOF order of ``node_ids``
    member_forces: Dict[str, float]  # tension positive
    analyze_status: int


def solve_truss_in_opensees(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    k_scale: Optional[Mapping[str, float]] = None,
) -> OpenseesSolution:
    """Build and solve the truss in OpenSeesPy (one fresh model per call).

    ``k_scale`` maps member id -> multiplier on its Young's modulus (use it
    to pass ``k_E(T_i)`` explicitly; OpenSees applies no thermal reduction
    by itself).
    """
    ops = _require_ops()
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    tag_of_node: Dict[str, int] = {}
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
    for node_id, load in loads.items():
        ops.load(
            tag_of_node[str(node_id)],
            float(load.get("Fx", 0.0)),
            float(load.get("Fy", 0.0)),
        )
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
    forces: Dict[str, float] = {}
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

    ci_values: Dict[str, float]
    u_max_base: float
    u_max_perturbed_max: float
    n_solves: int


def ci_sweep_in_opensees(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    alpha: float = 0.7,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = ssot_k_E,
) -> OpenseesCiSweep:
    """Reference criticality sweep: baseline + one degraded member per model."""
    base_scale = {e.id: float(k_e_func(temps[e.id])) for e in elements}
    base = solve_truss_in_opensees(nodes, elements, loads, base_scale)
    u_max_base = float(np.max(np.abs(base.disp)))
    if u_max_base <= 0.0:
        msg = "zero baseline displacement: the CI ratio is undefined"
        raise ValueError(msg)
    ci: Dict[str, float] = {}
    u_max_pert_max = 0.0
    for elem in elements:
        scale = dict(base_scale)
        scale[elem.id] *= float(alpha)
        pert = solve_truss_in_opensees(nodes, elements, loads, scale)
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
) -> Tuple[NDArray[np.float64], Dict[str, float]]:
    """Full-DOF displacement and member forces from the internal engine.

    ``build_engine`` receives the *temperature field* and applies
    ``k_e_func`` itself — passing pre-computed scale factors here would
    silently solve the wrong (undegraded) state (found by the level-4 smoke
    test: u_max off by exactly 1/k_E, forces identical because a uniform
    E-scaling leaves forces of a loaded truss invariant).
    """
    setup = build_engine(nodes, elements, loads, temps, k_e_func)
    u_free = base_displacement(setup, load_vector(nodes, loads, setup.free_dofs))
    u_full = np.zeros(2 * len(nodes))
    u_full[list(setup.free_dofs)] = u_free
    strains = setup.b_free @ u_free
    forces_arr = setup.k_axial * strains
    forces = {eid: float(forces_arr[i]) for i, eid in enumerate(setup.ids)}
    return u_full, forces


@dataclass(frozen=True)
class StateComparison:
    """Node-by-node / member-by-member comparison of one physical state."""

    temperature_field: Dict[str, float]
    u_max_internal: float
    u_max_opensees: float
    rel_err_u_max: float
    max_abs_node_err: float  # [m], over all nodal DOFs
    max_rel_node_err: float  # scaled by max|u| of the internal solution
    max_rel_force_err: float  # scaled by max|N| of the internal solution
    nodal_disp_internal: Dict[str, List[float]]
    nodal_disp_opensees: Dict[str, List[float]]
    forces_internal: Dict[str, float]
    forces_opensees: Dict[str, float]


def compare_state(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = ssot_k_E,
) -> StateComparison:
    """Solve the identical degraded state in both solvers and compare."""
    k_scale = {e.id: float(k_e_func(temps[e.id])) for e in elements}
    u_int, f_int = _internal_state(nodes, elements, loads, temps, k_e_func)
    sol = solve_truss_in_opensees(nodes, elements, loads, k_scale)
    u_scale = float(np.max(np.abs(u_int)))
    f_scale = max(abs(v) for v in f_int.values()) if f_int else 0.0
    abs_node_err = float(np.max(np.abs(u_int - sol.disp))) if len(u_int) else 0.0
    rel_node = abs_node_err / u_scale if u_scale > 0.0 else 0.0
    force_errs = [
        abs(f_int[e.id] - sol.member_forces[e.id]) / f_scale if f_scale > 0.0 else 0.0
        for e in elements
    ]
    u_max_ops = float(np.max(np.abs(sol.disp)))
    rel_umax = abs(u_max_ops - u_scale) / u_scale if u_scale > 0.0 else float("nan")
    disp_int: Dict[str, List[float]] = {}
    disp_ops: Dict[str, List[float]] = {}
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
        forces_opensees=dict(sol.member_forces),
    )


@dataclass(frozen=True)
class CiRankingComparison:
    """Rank-1 internal sweep vs the n+1-model OpenSees reference sweep."""

    alpha: float
    ci_internal: Dict[str, float]
    ci_opensees: Dict[str, float]
    rho: float
    branch: RhoBranch
    max_abs_ci_diff: float
    max_rel_ci_diff: float  # scaled by max|CI| of the internal sweep
    u_max_base_internal: float
    u_max_base_opensees: float
    n_opensees_solves: int
    extra: Dict[str, Any] = field(default_factory=dict)


def compare_ci_ranking(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float],
    alpha: float = 0.7,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = ssot_k_E,
) -> CiRankingComparison:
    """Compare criticality fields and classify the rank correlation."""
    setup = build_engine(nodes, elements, loads, temps, k_e_func)
    u_free = base_displacement(setup, load_vector(nodes, loads, setup.free_dofs))
    sweep = ci_sweep(setup, u_free, alpha)
    u_max_int = float(np.max(np.abs(u_free)))
    ref = ci_sweep_in_opensees(nodes, elements, loads, temps, alpha, k_e_func)
    # Tie-noise convention (same as ranking.tau_b, DL-028): two exact
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
