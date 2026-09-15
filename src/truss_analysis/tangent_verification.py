"""Exact-tangent verification of the linearised stiffness operator.

The library's static and stability chains are *first-order*: the elastic
assembler builds ``K_E`` from the undeformed geometry and
:func:`truss_analysis.stability.geometric_stiffness` adds the load-dependent
softening/stiffening term ``K_G = sum_e (N_e / L_e) g_e g_e^T``.  That
linearisation is only an approximation of the true tangent of the internal
force operator, and the round-6 audit asked for the approximation to be
*measured* rather than asserted.

This module supplies both halves of that measurement.

**1. The exact operator.**  :func:`internal_force` evaluates the
geometrically-exact internal force of the pin-jointed assembly at an
arbitrary displacement state -- current member length, current direction
cosines, engineering elongation measured from the stress-free length
(including thermal and fabrication prestrain):

.. code-block:: text

    N_e(u) = (k_E(T_e) E_e A_e / L_e^0) * ( |x_e(u)| - L_e^0 - dL_pre,e )
    P(u)   = sum_e N_e(u) * b_e(u),        b_e(u) = [-n_e; n_e]

**2. Its exact tangent.**  Differentiating ``P`` gives the classical
two-term decomposition

.. code-block:: text

    K_T(u) = dP/du = sum_e (E A / L^0) b_e b_e^T  +  sum_e (N_e / L_e) g_e g_e^T
             ^ material (elastic) part        ^ geometric part

with ``g_e = [s, -c, -s, c]`` the transverse (chord-rotation) extractor
built from the *current* direction cosines.  :func:`exact_tangent_stiffness`
assembles it; :func:`verify_tangent_stiffness` checks it against a central
finite difference of :func:`internal_force`, which is the strongest
available oracle -- an error there means the operator or its derivative is
wrong, not merely that a formula looks plausible.

**3. The linearisation gap.**  :func:`verify_linearization_convergence`
quantifies how well the library's ``K_E + K_G`` reproduces ``K_T`` and, more
importantly, that the gap closes at *first order* in the load: halving the
demand halves the relative difference.  That is the precise sense in which
:mod:`truss_analysis.stability` is a linearised theory, and it converts the
claim from a docstring sentence into a measured rate.

Scope: small strains, elastic material (temperature-degraded through
``k_E(T)`` when a field is supplied), pin-jointed members.  No plasticity,
no member bending -- the same boundary the rest of the library declares.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .assembly import MemberGeometry, member_geometry
from .criticality.engine import (
    base_displacement,
    build_engine,
    member_forces,
    prestress_lengths,
    total_load_vector,
)
from .material.steel_eurocode import FloatOrArray
from .material.steel_eurocode import k_E as eurocode_k_E
from .model import Element, Node
from .stability import geometric_stiffness

__all__ = [
    "LinearizationCheck",
    "TangentCheck",
    "exact_tangent_stiffness",
    "internal_force",
    "linearized_tangent_stiffness",
    "verify_linearization_convergence",
    "verify_tangent_stiffness",
]

_NDArrayF = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# shared geometric helpers
# ---------------------------------------------------------------------------


def _coordinates(nodes: Sequence[Node]) -> _NDArrayF:
    """Return nodal coordinates as an ``(n, 2)`` array, in DOF-map order."""
    coords: _NDArrayF = np.array([[n.x, n.y] for n in nodes], dtype=float).reshape(
        -1, 2
    )
    return coords


def _axial_stiffness(
    geom: MemberGeometry,
    elements: Sequence[Element],
    temps: Mapping[str, float] | None,
    k_e_func: Callable[[FloatOrArray], FloatOrArray],
) -> _NDArrayF:
    """``k_E(T) E A / L`` per member -- the material stiffness of the dyad."""
    base = np.array([float(e.E) * float(e.A) for e in elements], dtype=float)
    base = base / geom.lengths
    if temps is None:
        return np.asarray(base, dtype=float)
    t_field = np.array([float(temps[e.id]) for e in elements], dtype=float)
    scale = np.asarray(k_e_func(t_field), dtype=float)
    return np.asarray(base * np.broadcast_to(scale, base.shape), dtype=float)


def _deformed_state(
    geom: MemberGeometry,
    coords: _NDArrayF,
    u: _NDArrayF,
) -> tuple[_NDArrayF, _NDArrayF]:
    """Return current member length and unit direction at displacement ``u``.

    Returns
    -------
    lengths : numpy.ndarray
        Deformed member lengths ``|x_e(u)|``, shape ``(m,)``.
    unit : numpy.ndarray
        Current direction cosines ``[c_e, s_e]``, shape ``(m, 2)``.
    """
    disp = np.asarray(u, dtype=float).reshape(-1, 2)
    du = disp[geom.node_j_idx] - disp[geom.node_i_idx]
    chord = (coords[geom.node_j_idx] - coords[geom.node_i_idx]) + du
    lengths = np.linalg.norm(chord, axis=1)
    safe = np.where(lengths > 0.0, lengths, 1.0)
    return lengths, chord / safe[:, None]


def _dyad_matrix(
    geom: MemberGeometry,
    n_dof: int,
    vectors: _NDArrayF,
) -> _NDArrayF:
    """Scatter ``(m, 4)`` local vectors onto an ``(m, n_dof)`` operator."""
    out = np.zeros((geom.n_members, n_dof), dtype=float)
    out[np.arange(geom.n_members)[:, None], geom.dofs] = vectors
    return np.asarray(out, dtype=float)


def _elongation_vectors(geom: MemberGeometry, n_dof: int) -> _NDArrayF:
    """Undeformed axial compatibility vectors ``b = [-c, -s, c, s]``."""
    c, s = geom.cosines, geom.sines
    local = np.stack([-c, -s, c, s], axis=1)
    return np.asarray(_dyad_matrix(geom, n_dof, local), dtype=float)


# ---------------------------------------------------------------------------
# 1. the exact internal force operator
# ---------------------------------------------------------------------------


def internal_force(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    u: _NDArrayF,
    temps: Mapping[str, float] | None = None,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = eurocode_k_E,
) -> _NDArrayF:
    """Geometrically-exact internal force vector ``P(u)`` [N], shape ``(2n,)``.

    Every member is evaluated at its *current* length and orientation, so the
    result is the true nonlinear residual of the pin-jointed assembly -- the
    operator whose derivative the linearised solver approximates.  Imposed
    elongation (thermal expansion plus fabrication ``delta_L_free``) shifts
    the stress-free length exactly as
    :func:`~truss_analysis.criticality.engine.prestress_lengths` defines it,
    so a restrained heated member carries compression at ``u = 0``.

    Parameters
    ----------
    nodes, elements : Sequence[Node], Sequence[Element]
        Model.  Node ordering defines the global DOF map.
    u : numpy.ndarray
        Nodal displacement field, shape ``(2n,)`` or ``(n, 2)`` [m].
    temps : Mapping[str, float] or None, optional
        Member id -> steel temperature [degC]; ``None`` keeps ``k_E = 1`` and
        drops the thermal part of the imposed elongation.
    k_e_func : Callable, optional
        EN 1993-1-2 stiffness reduction factor ``k_E(T)``.

    Returns
    -------
    numpy.ndarray
        Internal force vector, shape ``(2n,)``; tension positive.

    Notes
    -----
    Restrained DOFs are *not* zeroed: the returned vector contains the
    reaction-equivalent entries too, which is what a finite-difference
    tangent check needs in order to see every column of ``K_T``.
    """
    geom = member_geometry(list(nodes), list(elements))
    n_dof = 2 * len(nodes)
    coords = _coordinates(nodes)
    lengths, unit = _deformed_state(geom, coords, np.asarray(u, dtype=float))
    dl_pre = prestress_lengths(nodes, elements, temps)
    axial = _axial_stiffness(geom, list(elements), temps, k_e_func)

    forces = axial * (lengths - geom.lengths - dl_pre)
    local = np.concatenate([-unit, unit], axis=1)  # (m, 4): [-n, n]
    p = np.zeros(n_dof, dtype=float)
    np.add.at(p, geom.dofs.reshape(-1), (forces[:, None] * local).reshape(-1))
    return p


# ---------------------------------------------------------------------------
# 2. the exact tangent and its finite-difference oracle
# ---------------------------------------------------------------------------


def exact_tangent_stiffness(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    u: _NDArrayF,
    temps: Mapping[str, float] | None = None,
    free_dofs: Sequence[int] | None = None,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = eurocode_k_E,
) -> _NDArrayF:
    """Exact tangent stiffness ``K_T = dP/du`` at state ``u``, dense.

    .. code-block:: text

        K_T(u) = sum_e (k_E E A / L^0) b_e(u) b_e(u)^T
               + sum_e (N_e(u) / L_e(u)) g_e(u) g_e(u)^T

    Both dyads use the *deformed* geometry, which is what distinguishes this
    from :func:`linearized_tangent_stiffness`.  The first term is the
    material (elastic) contribution; the second is the geometric one and is
    exactly the operator :func:`~truss_analysis.stability.geometric_stiffness`
    assembles from the undeformed geometry.

    Parameters
    ----------
    nodes, elements : Sequence[Node], Sequence[Element]
        Model.
    u : numpy.ndarray
        Displacement state at which the tangent is evaluated.
    temps : Mapping[str, float] or None, optional
        Member steel temperatures [degC].
    free_dofs : Sequence[int] or None, optional
        Restrict the matrix to these global DOFs (the ``K_ff`` convention).
        ``None`` returns the full ``(2n, 2n)`` matrix.
    k_e_func : Callable, optional
        EN 1993-1-2 stiffness reduction factor ``k_E(T)``.

    Returns
    -------
    numpy.ndarray
        Symmetric tangent stiffness [N/m], shape ``(2n, 2n)`` or
        ``(len(free_dofs), len(free_dofs))``.
    """
    geom = member_geometry(list(nodes), list(elements))
    n_dof = 2 * len(nodes)
    coords = _coordinates(nodes)
    lengths, unit = _deformed_state(geom, coords, np.asarray(u, dtype=float))
    dl_pre = prestress_lengths(nodes, elements, temps)
    axial = _axial_stiffness(geom, list(elements), temps, k_e_func)

    forces = axial * (lengths - geom.lengths - dl_pre)
    b_local = np.concatenate([-unit, unit], axis=1)
    # Transverse extractor on the *current* chord, ordered [u_i, v_i, u_j, v_j]:
    #     g = [s, -c, -s, c]  with (c, s) = (n_x, n_y)
    # i.e. g . u is the relative displacement perpendicular to the member --
    # bit-for-bit the vector `stability.member_geometric_vectors` builds from
    # the undeformed geometry, and orthogonal to b = [-c, -s, c, s] by
    # construction (b . g = -cs + sc - cs + sc = 0).
    g_local = np.concatenate(
        [unit[:, [1]], -unit[:, [0]], -unit[:, [1]], unit[:, [0]]], axis=1
    )

    b_mat = _dyad_matrix(geom, n_dof, b_local)
    g_mat = _dyad_matrix(geom, n_dof, g_local)
    k_t = np.einsum("e,ei,ej->ij", axial, b_mat, b_mat)
    k_t += np.einsum("e,ei,ej->ij", forces / lengths, g_mat, g_mat)

    if free_dofs is not None:
        idx = np.asarray(list(free_dofs), dtype=int)
        k_t = np.asarray(k_t[np.ix_(idx, idx)], dtype=float)
    return np.asarray(k_t, dtype=float)


def linearized_tangent_stiffness(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    forces: Mapping[str, float],
    temps: Mapping[str, float] | None = None,
    free_dofs: Sequence[int] | None = None,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = eurocode_k_E,
) -> _NDArrayF:
    """Assemble the library's first-order tangent: ``K_E`` plus ``K_G(N)``.

    ``K_E`` is built from the *undeformed* geometry, which is the whole
    difference from :func:`exact_tangent_stiffness`.

    This is what the solver and the stability module actually use, assembled
    here from the same shared geometry so the comparison in
    :func:`verify_linearization_convergence` cannot be rigged by a second,
    divergent implementation of ``K_E``.

    Parameters
    ----------
    nodes, elements : Sequence[Node], Sequence[Element]
        Model.
    forces : Mapping[str, float]
        Member axial forces [N] of the base state, keyed by element id;
        tension positive.  Members absent from the mapping are zero-force.
    temps : Mapping[str, float] or None, optional
        Member steel temperatures [degC] (degrade ``E`` through ``k_E(T)``).
    free_dofs : Sequence[int] or None, optional
        Restrict the matrix to these global DOFs.
    k_e_func : Callable, optional
        EN 1993-1-2 stiffness reduction factor ``k_E(T)``.

    Returns
    -------
    numpy.ndarray
        ``K_E + K_G`` [N/m], dense.
    """
    geom = member_geometry(list(nodes), list(elements))
    n_dof = 2 * len(nodes)
    axial = _axial_stiffness(geom, list(elements), temps, k_e_func)
    b_mat = _elongation_vectors(geom, n_dof)
    k_e = np.einsum("e,ei,ej->ij", axial, b_mat, b_mat)
    k_g = np.asarray(geometric_stiffness(nodes, elements, forces))
    k_lin = k_e + k_g
    if free_dofs is not None:
        idx = np.asarray(list(free_dofs), dtype=int)
        k_lin = np.asarray(k_lin[np.ix_(idx, idx)], dtype=float)
    return np.asarray(k_lin, dtype=float)


@dataclass(frozen=True)
class TangentCheck:
    """Outcome of the finite-difference tangent verification.

    Attributes
    ----------
    passed : bool
        ``max_rel_error <= tol``.
    max_rel_error : float
        Largest per-column relative error
        ``||K_fd[:, j] - K_T[:, j]|| / max(||K_T[:, j]||, scale_floor)``.
    max_abs_error : float
        Largest absolute entrywise difference ``max |K_fd - K_T|`` [N/m].
    scale : float
        Reference norm ``||K_T||_F`` the relative error is measured against.
    n_dof : int
        Number of DOFs the check covered.
    rel_error : numpy.ndarray
        Per-column relative error, shape ``(n_dof,)``.
    k_analytical : numpy.ndarray
        Analytical tangent from :func:`exact_tangent_stiffness`.
    k_fd : numpy.ndarray
        Central-difference tangent of :func:`internal_force`.
    state_norm : float
        ``||u||`` of the displacement state the tangent was taken at; a
        verification at ``state_norm = 0`` exercises no geometric term.
    """

    passed: bool
    max_rel_error: float
    max_abs_error: float
    scale: float
    n_dof: int
    rel_error: _NDArrayF
    k_analytical: _NDArrayF
    k_fd: _NDArrayF
    state_norm: float


def verify_tangent_stiffness(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    u: _NDArrayF,
    temps: Mapping[str, float] | None = None,
    *,
    dofs: Sequence[int] | None = None,
    epsilon: float = 1e-6,
    tol: float = 1e-6,
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = eurocode_k_E,
) -> TangentCheck:
    """Finite-difference check of :func:`exact_tangent_stiffness`.

    Column ``j`` of the tangent is approximated by the central difference of
    the internal force,

    .. code-block:: text

        K_fd[:, j] = [ P(u + eps e_j) - P(u - eps e_j) ] / (2 eps)

    which converges at ``O(eps^2)`` in truncation and ``O(machine/eps)`` in
    round-off; ``epsilon = 1e-6`` sits near the optimum for double precision
    and a stiffness of order ``1e8`` N/m.

    Parameters
    ----------
    nodes, elements : Sequence[Node], Sequence[Element]
        Model.
    u : numpy.ndarray
        Displacement state to linearise about.  Use a *non-zero* state (e.g.
        the solved base displacement of a loaded model) if the geometric
        part of the tangent is to be exercised at all.
    temps : Mapping[str, float] or None, optional
        Member steel temperatures [degC].
    dofs : Sequence[int] or None, optional
        DOF columns to perturb; ``None`` covers every DOF (``2n`` calls of
        the force operator per column pair).
    epsilon : float, default 1e-6
        Central-difference step [m].
    tol : float, default 1e-6
        Acceptance threshold on ``max_rel_error``.
    k_e_func : Callable, optional
        EN 1993-1-2 stiffness reduction factor ``k_E(T)``.

    Returns
    -------
    TangentCheck
        Verdict plus both matrices, so a failure can be diagnosed without
        re-running the sweep.
    """
    node_list = list(nodes)
    element_list = list(elements)
    n_dof = 2 * len(node_list)
    u_arr = np.asarray(u, dtype=float).reshape(-1)[:n_dof].copy()
    columns = list(range(n_dof)) if dofs is None else [int(d) for d in dofs]

    k_analytical = exact_tangent_stiffness(
        node_list, element_list, u_arr, temps, k_e_func=k_e_func
    )
    k_fd = np.zeros_like(k_analytical)
    for j in columns:
        step = np.zeros(n_dof, dtype=float)
        step[j] = epsilon
        p_plus = internal_force(node_list, element_list, u_arr + step, temps, k_e_func)
        p_minus = internal_force(node_list, element_list, u_arr - step, temps, k_e_func)
        k_fd[:, j] = (p_plus - p_minus) / (2.0 * epsilon)

    scale = float(np.linalg.norm(k_analytical))
    scale_floor = scale if scale > 0.0 else 1.0
    col_norm = np.linalg.norm(k_analytical, axis=0)
    denom = np.where(col_norm > 1e-12 * scale_floor, col_norm, scale_floor)
    rel_error = np.linalg.norm(k_fd - k_analytical, axis=0) / denom
    # Untested columns keep a zero residual by construction; only report the
    # ones actually perturbed so a partial sweep is not silently "perfect".
    if dofs is not None:
        mask = np.zeros(n_dof, dtype=bool)
        mask[columns] = True
        rel_error = np.where(mask, rel_error, 0.0)

    max_rel = float(np.max(rel_error)) if rel_error.size else 0.0
    return TangentCheck(
        passed=bool(max_rel <= tol),
        max_rel_error=max_rel,
        max_abs_error=float(np.max(np.abs(k_fd - k_analytical))) if n_dof else 0.0,
        scale=scale,
        n_dof=n_dof,
        rel_error=rel_error,
        k_analytical=k_analytical,
        k_fd=k_fd,
        state_norm=float(np.linalg.norm(u_arr)),
    )


# ---------------------------------------------------------------------------
# 3. the linearisation gap, measured
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearizationCheck:
    """Measured agreement between ``K_E + K_G`` and the exact tangent.

    Attributes
    ----------
    scales : tuple[float, ...]
        Load multipliers applied to ``loads`` (descending by construction).
    rel_gap : tuple[float, ...]
        ``||K_T_exact(u_s) - (K_E + K_G(N_s))||_F / ||K_E||_F`` per scale.
    observed_orders : tuple[float, ...]
        Local convergence order ``log2(gap_s / gap_{s/2})`` between successive
        scales; ``~1.0`` confirms the first-order claim.
    order_estimate : float
        Least-squares slope of ``log gap`` against ``log scale`` over the
        whole sweep -- the single number to quote in a report.
    first_order : bool
        ``order_estimate`` lies in ``[0.75, 1.35]``, i.e. the gap genuinely
        closes linearly with the demand rather than saturating.
    reference_scale : float
        ``||K_E||_F`` used to normalise the gap.
    """

    scales: tuple[float, ...]
    rel_gap: tuple[float, ...]
    observed_orders: tuple[float, ...]
    order_estimate: float
    first_order: bool
    reference_scale: float


def verify_linearization_convergence(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float] | None = None,
    *,
    scales: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    k_e_func: Callable[[FloatOrArray], FloatOrArray] = eurocode_k_E,
) -> LinearizationCheck:
    """Measure how fast the library's first-order tangent approaches truth.

    For each load multiplier ``s`` the linear base state is solved with the
    production engine (so the axial forces come from the same demand path the
    DCR chain uses), the exact tangent is assembled at that displacement, and
    the relative Frobenius gap to ``K_E + K_G`` is recorded.  Because
    ``u_s = s u_1`` for a linear solve, a correct first-order theory must
    give ``gap(s) ~ s``; the fitted slope is returned as
    :attr:`LinearizationCheck.order_estimate`.

    Parameters
    ----------
    nodes, elements : Sequence[Node], Sequence[Element]
        Model.
    loads : Mapping[str, Mapping[str, float]]
        Mechanical nodal loads ``{node_id: {"Fx": ..., "Fy": ...}}`` [N].
    temps : Mapping[str, float] or None, optional
        Member steel temperatures [degC].  A thermal field contributes an
        imposed force state that does *not* scale with ``s``; it is therefore
        held fixed and excluded from the gap normalisation, which is why the
        sweep is most informative for the purely mechanical case.
    scales : Sequence[float], optional
        Descending load multipliers.  At least two are required.
    k_e_func : Callable, optional
        EN 1993-1-2 stiffness reduction factor ``k_E(T)``.

    Returns
    -------
    LinearizationCheck
        Gaps, local orders and the fitted order.

    Raises
    ------
    ValueError
        If fewer than two scales are supplied or any scale is non-positive.
    """
    ordered = sorted((float(s) for s in scales), reverse=True)
    if len(ordered) < 2:
        msg = "verify_linearization_convergence needs at least two scales"
        raise ValueError(msg)
    if any(s <= 0.0 for s in ordered):
        msg = f"load scales must be positive, got {ordered}"
        raise ValueError(msg)

    node_list = list(nodes)
    element_list = list(elements)
    n_dof = 2 * len(node_list)
    k_e_ref = linearized_tangent_stiffness(
        node_list, element_list, {}, temps, k_e_func=k_e_func
    )
    reference = float(np.linalg.norm(k_e_ref))
    if reference <= 0.0:  # pragma: no cover - a model with no stiffness
        reference = 1.0

    gaps: list[float] = []
    for s in ordered:
        scaled = {
            nid: {k: s * float(v) for k, v in comp.items()}
            for nid, comp in loads.items()
        }
        setup = build_engine(node_list, element_list, scaled, temps, k_e_func)
        free = list(setup.free_dofs)
        u_free = base_displacement(setup, total_load_vector(node_list, scaled, setup))
        n_mem = member_forces(setup, u_free)
        forces = {e.id: float(n_mem[i]) for i, e in enumerate(element_list)}

        u_full = np.zeros(n_dof, dtype=float)
        u_full[free] = u_free
        k_exact = exact_tangent_stiffness(
            node_list, element_list, u_full, temps, k_e_func=k_e_func
        )
        k_lin = linearized_tangent_stiffness(
            node_list, element_list, forces, temps, k_e_func=k_e_func
        )
        gaps.append(float(np.linalg.norm(k_exact - k_lin)) / reference)

    orders = tuple(
        float(np.log2(gaps[i] / gaps[i + 1]))
        for i in range(len(gaps) - 1)
        if gaps[i + 1] > 0.0 and gaps[i] > 0.0
    )
    log_s = np.log(np.asarray(ordered, dtype=float))
    log_g = np.log(np.maximum(np.asarray(gaps, dtype=float), 1e-300))
    slope = float(np.polyfit(log_s, log_g, 1)[0]) if len(ordered) > 1 else 0.0

    return LinearizationCheck(
        scales=tuple(ordered),
        rel_gap=tuple(gaps),
        observed_orders=orders,
        order_estimate=slope,
        first_order=bool(0.75 <= slope <= 1.35),
        reference_scale=reference,
    )
