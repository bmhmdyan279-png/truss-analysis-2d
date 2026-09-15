"""Linearised stability: geometric stiffness and system buckling loads.

The library's static chain is first-order: member forces come from
``K_E u = F`` with the *elastic* stiffness only. Axial forces nevertheless
change a pin-jointed assembly's resistance to transverse deformation --
compression softens it, tension stiffens it. That second-order effect is
carried by the **geometric stiffness**

.. code-block:: text

    K_G = sum_e (N_e / L_e) g_e g_e^T,      g_e . u = transverse relative
                                            displacement of member e

with ``g_e = [s, -c, -s, c]`` on the member's DOFs (``c``, ``s`` the
direction cosines). ``g_e`` is everywhere orthogonal to the elongation
vector ``b_e = [-c, -s, c, s]`` of the elastic dyad ``K_E = sum_e k_e b_e
b_e^T``: the two rank-1 contributions act on the axial and the transverse
relative displacement respectively. ``u^T K_G u = sum_e N_e L_e phi_e^2``
with ``phi_e`` the chord rotation, i.e. twice the second-order work of the
axial forces -- the classical potential-energy statement (round-5 audit,
C2-A: the missing ``K_G`` was the top scientific gap).

:func:`linearized_buckling_load_factor` solves the linearised bifurcation
problem around the **prestressed base state**:

.. code-block:: text

    [ K_E + K_G(N_imposed) + lambda * K_G(N_mech) ] u = 0

where ``N_imposed`` are the temperature/fabrication (eigenstrain) forces
that stay fixed and ``N_mech`` the forces from the mechanical load pattern
that ``lambda`` amplifies. ``lambda_cr`` is the smallest positive root,
computed through the symmetric whitened eigenproblem
``C v = nu v`` with ``C = L^-1 (-K_G(N_mech)) L^-T`` and
``LL^T = K_E + K_G(N_imposed)``; then ``lambda_cr = 1 / nu_max``.

Scope -- stated plainly, per the round-5 audit's physics-boundary demand:

* **System bifurcation of the pin-jointed assembly**, not member code
  checks. The member-level Euler/chi capacity model in
  :mod:`truss_analysis.limitstates` remains the EN 1993 verification path;
  ``lambda_cr`` answers the *structural* question "how far can the
  mechanical load be amplified before the current force state loses
  stability?". The two are complementary: a truss whose compressed chords
  are code-safe member-by-member can still bifurcate as a system.
* **Linearised (tangent-stiffness) criterion.** For shallow systems the
  true collapse is a limit point (snap-through) that the linearised factor
  approximates from the base configuration; the two-bar toggle test in
  ``tests/test_stability.py`` pins the closed form and shows the
  agreement is ``O(theta_0^2)`` for shallow geometries. Post-buckling
  paths, imperfection sensitivity and nonlinear Newton-Raphson load
  stepping remain out of scope (roadmap).
* **Small strains, elastic materials.** ``K_E`` carries whatever ``E`` the
  elements have (temperature-degraded ``k_E(T) E`` when a temperature
  field is supplied); no plasticity.

All geometry comes from :class:`truss_analysis.assembly.MemberGeometry` --
the single source the assembler itself uses -- so this module cannot drift
from the stiffness assembly by a round-off.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.linalg import LinAlgError, cholesky, eigh, solve_triangular

from .assembly import MemberGeometry, member_geometry
from .criticality.engine import (
    MechanismError,
    base_displacement,
    build_engine,
    load_vector,
    member_forces,
    total_load_vector,
)
from .exceptions import AmbiguousModeWarning, ShallowSystemWarning
from .model import Element, Node

__all__ = [
    "BucklingResult",
    "geometric_stiffness",
    "linearized_buckling_load_factor",
    "member_geometric_vectors",
]


def member_geometric_vectors(
    geom: MemberGeometry,
    n_dof: int,
) -> np.ndarray:
    """Transverse compatibility vectors ``g`` of every member, shape ``(m, n_dof)``.

    ``g_e . u`` is the relative displacement of member ``e``'s two nodes
    perpendicular to its axis (the chord-rotation displacement ``phi L``).
    Built from the shared :class:`~truss_analysis.assembly.MemberGeometry`
    cosines/sines/DOF map -- the same single source the elastic dyads use.
    """
    g = np.zeros((geom.n_members, n_dof), dtype=float)
    for e in range(geom.n_members):
        c = float(geom.cosines[e])
        s = float(geom.sines[e])
        g[e, geom.dofs[e]] = (s, -c, -s, c)
    return g


def geometric_stiffness(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    forces: Mapping[str, float],
    free_dofs: Sequence[int] | None = None,
    sparse: bool = False,
) -> np.ndarray | sp.csr_matrix:
    """Assemble ``K_G = sum_e (N_e / L_e) g_e g_e^T``.

    Parameters
    ----------
    nodes, elements : Sequence[Node], Sequence[Element]
        Model geometry (the same objects the elastic assembler consumes).
    forces : Mapping[str, float]
        Member axial forces [N] keyed by element id; tension positive.
        Members absent from the mapping are treated as zero-force (they
        contribute nothing to ``K_G``).
    free_dofs : Sequence[int] or None, optional
        Restrict the matrix to these global DOFs (the usual ``K_ff``
        convention). ``None`` returns the full ``(2n, 2n)`` matrix.
    sparse : bool, default False
        Return a ``scipy.sparse.csr_matrix`` assembled through COO with
        C-level duplicate summation, mirroring
        :func:`truss_analysis.assembly.assemble_global_matrices_sparse`.

    Returns
    -------
    np.ndarray or scipy.sparse.csr_matrix
        The geometric stiffness matrix [N/m]; symmetric and indefinite in
        general.  With tension-positive ``N_e``, a tensioned member
        contributes a positive semi-definite dyad (it *stiffens* the
        transverse direction) and a compressed member a negative
        semi-definite one (it *softens* it) -- the physical origin of
        buckling.

    Raises
    ------
    KeyError
        If ``forces`` contains ids that are not model members.
    """
    known = {e.id for e in elements}
    unknown = sorted(set(forces) - known)
    if unknown:
        msg = f"geometric_stiffness: unknown member id(s) {unknown}"
        raise KeyError(msg)

    geom = member_geometry(list(nodes), list(elements))
    n_dof = 2 * len(nodes)
    g = member_geometric_vectors(geom, n_dof)
    n_vec = np.array([float(forces.get(e.id, 0.0)) for e in elements], dtype=float)
    coeff = n_vec / geom.lengths  # N_e / L_e

    if free_dofs is not None:
        g = g[:, list(free_dofs)]
        n_out = len(free_dofs)
    else:
        n_out = n_dof

    if sparse:
        # Mirror the elastic sparse path: build (row, col, val) triplets for
        # every dyad and let COO sum duplicates in C. Each dyad uses the
        # member's LOCAL 4-vector gathered at its own DOFs; when the output
        # is restricted to ``free_dofs``, entries whose DOF is restrained
        # are masked out of the triplet lists.
        if free_dofs is None:
            idx_local = geom.dofs  # (m, 4) output-space indices
            g_local = np.take_along_axis(g, geom.dofs, axis=1)
        else:
            pos = {int(d): i for i, d in enumerate(free_dofs)}
            idx_local = np.array(
                [[pos.get(int(d), -1) for d in row] for row in geom.dofs]
            )
            safe = np.maximum(idx_local, 0)
            g_local = np.where(
                idx_local >= 0,
                g[np.arange(g.shape[0])[:, None], safe],
                0.0,
            )
        r = np.repeat(idx_local, 4, axis=1)
        c_idx = np.tile(idx_local, (1, 4))
        vals = np.einsum("e,ei,ej->eij", coeff, g_local, g_local).reshape(
            geom.n_members, 16
        )
        keep = (r >= 0) & (c_idx >= 0)
        K: sp.csr_matrix = sp.coo_matrix(
            (vals[keep], (r[keep], c_idx[keep])), shape=(n_out, n_out)
        ).tocsr()
        return K

    dense: np.ndarray = np.einsum("e,ei,ej->ij", coeff, g, g)
    return dense


@dataclass(frozen=True)
class BucklingResult:
    """Outcome of the linearised bifurcation analysis.

    Attributes
    ----------
    lambda_cr : float
        Smallest positive load factor amplifying the *mechanical* load
        pattern (imposed thermal/fabrication forces held fixed) at which
        the tangent stiffness loses positive definiteness. ``inf`` when no
        bifurcation exists under load amplification (e.g. every member is
        in tension or there is no mechanical load).
    mode : numpy.ndarray
        Buckling mode on the full DOF vector ``(2n,)``, unit 2-norm, fixed
        DOFs exactly zero. Arbitrary sign (eigenvectors are).
    modes : list[numpy.ndarray]
        All computed buckling modes (up to the number requested or all
        positive eigenvalues). The first element equals ``mode``. Present
        to support multiplicity detection and mode-shape exploration.
    multiplicity : int
        Number of eigenvalues within numerical tolerance of the critical
        one. ``multiplicity > 1`` indicates repeated buckling loads and
        triggers :exc:`~truss_analysis.exceptions.AmbiguousModeWarning`.
    n_compressed : int
        Number of members carrying compression in the base state beyond the
        library's force classification band.
    base_forces : dict[str, float]
        Base-state member forces [N] (mechanical + imposed) the bifurcation
        was linearised about.
    """

    lambda_cr: float
    mode: npt.NDArray[np.float64]
    modes: list[npt.NDArray[np.float64]]
    multiplicity: int
    n_compressed: int
    base_forces: dict[str, float]


def linearized_buckling_load_factor(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float] | None = None,
    n_modes: int = 1,
) -> BucklingResult:
    """Smallest positive bifurcation load factor of the prestressed assembly.

    Solves ``[K_E + K_G(N_imposed) + lambda K_G(N_mech)] u = 0`` for the
    smallest ``lambda > 0`` -- see the module docstring for the full
    statement and the scope limits. With ``temps`` supplied the elastic
    stiffness is the fire-degraded ``k_E(T) E`` and the imposed forces are
    the restrained-thermal eigenstrain forces, both from the *same*
    :func:`~truss_analysis.criticality.engine.build_engine` setup the whole
    fire chain uses, so the bifurcation analysis cannot disagree with the
    DCR chain about the base state.

    Parameters
    ----------
    nodes, elements : Sequence[Node], Sequence[Element]
        Model.
    loads : Mapping[str, Mapping[str, float]]
        Mechanical nodal loads ``{node_id: {"Fx": ..., "Fy": ...}}`` [N] --
        the load pattern ``lambda`` amplifies.
    temps : Mapping[str, float] or None, optional
        Member steel temperatures [degC]; ``None`` is the ambient
        (undegraded, no thermal strain) case.
    n_modes : int, default 1
        Number of buckling modes to compute. If ``n_modes > 1``, all
        computed modes are returned in :attr:`BucklingResult.modes` and
        multiplicity of the critical eigenvalue is detected.

    Returns
    -------
    BucklingResult
        ``lambda_cr``, the mode shape(s) and the base state it refers to.
        If multiple eigenvalues are within numerical tolerance of the
        critical one, :attr:`BucklingResult.multiplicity` > 1 and an
        :exc:`~truss_analysis.exceptions.AmbiguousModeWarning` is issued.

    Raises
    ------
    MechanismError
        If the prestressed base state ``K_E + K_G(N_imposed)`` is not
        positive definite: the structure is *already* at or past a critical
        point before any load amplification (or is a mechanism outright),
        and a load factor is undefined.

    Notes
    -----
    A :exc:`~truss_analysis.exceptions.ShallowSystemWarning` is issued (not
    raised) when the overall rise/span ratio is below 0.1: for such systems
    the true collapse is a snap-through limit point and the linearised
    factor can significantly underestimate the demand at failure.
    """
    setup = build_engine(nodes, elements, loads, temps)
    free = list(setup.free_dofs)
    if not free:
        msg = (
            "linearized_buckling_load_factor: every DOF is restrained; "
            "there is no stability problem to solve"
        )
        raise MechanismError(msg)

    # Check for shallow geometry (A4: ShallowSystemWarning)
    node_coords = np.array([[n.x, n.y] for n in nodes])
    x_span = float(np.ptp(node_coords[:, 0]))
    y_rise = float(np.ptp(node_coords[:, 1]))
    if x_span > 0 and y_rise / x_span < 0.1:
        warnings.warn(
            f"ShallowSystemWarning: rise/span ratio = {y_rise / x_span:.3f} < 0.1; "
            "linearised buckling may significantly underestimate snap-through load. "
            "Consider nonlinear geometric analysis for shallow systems.",
            ShallowSystemWarning,
            stacklevel=2,
        )

    n_dof = 2 * len(nodes)
    geom = member_geometry(list(nodes), list(elements))
    g_all = member_geometric_vectors(geom, n_dof)
    g_free = g_all[:, free]

    # Base-state demand: total, mechanical-only, and the imposed remainder.
    u_total = base_displacement(setup, total_load_vector(nodes, loads, setup))
    n_total = member_forces(setup, u_total)
    u_mech = base_displacement(setup, load_vector(nodes, loads, free))
    n_mech = setup.k_axial * (setup.b_free @ u_mech)
    n_imposed = n_total - n_mech

    # Tangent stiffness of the prestressed base state.
    k_e_ff = np.einsum("i,ip,iq->pq", setup.k_axial, setup.b_free, setup.b_free)
    coeff_th = n_imposed / geom.lengths
    a_mat = k_e_ff + np.einsum("e,ei,ej->ij", coeff_th, g_free, g_free)
    try:
        chol = cholesky(a_mat, lower=True, check_finite=False)
    except LinAlgError as exc:
        msg = (
            "prestressed base state is not positive definite: the imposed "
            "(thermal/fabrication) force state alone has reached or passed "
            "a critical point, so no load factor is defined"
        )
        raise MechanismError(msg) from exc

    # Load-pattern geometric stiffness, negated so that lambda > 0 is the
    # physically meaningful amplification of the mechanical demand.
    coeff_mech = -(n_mech / geom.lengths)
    b_mat = np.einsum("e,ei,ej->ij", coeff_mech, g_free, g_free)

    # Symmetric whitening: C = L^-1 B L^-T, nu = 1/lambda.
    x = solve_triangular(chol, b_mat, lower=True, check_finite=False)
    c_mat = solve_triangular(chol, x.T, lower=True, check_finite=False).T
    c_mat = 0.5 * (c_mat + c_mat.T)  # kill round-off asymmetry

    # Compute eigenvalues/eigenvectors
    nu, vecs = eigh(c_mat)

    # Find positive eigenvalues (corresponding to positive lambda)
    pos_mask = nu > 1e-14
    if not np.any(pos_mask):
        lam_cr = float("inf")
        mode_free = np.zeros(len(free))
        modes_list = [mode_free.copy()]
        multiplicity = 1
    else:
        nu_pos = nu[pos_mask]
        vecs_pos = vecs[:, pos_mask]

        # Sort by eigenvalue (ascending) to get smallest lambda first
        sort_idx = np.argsort(nu_pos)
        nu_sorted = nu_pos[sort_idx]
        vecs_sorted = vecs_pos[:, sort_idx]

        # Critical eigenvalue is the largest nu (smallest lambda = 1/nu)
        nu_max = float(nu_sorted[-1])
        lam_cr = 1.0 / nu_max

        # Detect multiplicity: count eigenvalues within tolerance of nu_max
        eig_tol = 1e-10 * max(1.0, abs(nu_max))
        close_to_critical = np.abs(nu_sorted - nu_max) < eig_tol
        multiplicity = int(np.sum(close_to_critical))

        # Build all mode shapes (transform back from whitened space)
        modes_list = []
        for i in range(len(nu_sorted)):
            mode_free_i = solve_triangular(
                chol.T, vecs_sorted[:, i], check_finite=False
            )
            norm = float(np.linalg.norm(mode_free_i))
            if norm > 0.0:
                mode_free_i = mode_free_i / norm
            modes_list.append(mode_free_i)

        # Use the critical mode (last one, corresponding to nu_max)
        mode_free = modes_list[-1].copy()

        # Issue warning if multiplicity > 1 (A2: AmbiguousModeWarning)
        if multiplicity > 1:
            warnings.warn(
                f"AmbiguousModeWarning: {multiplicity} buckling modes share "
                f"the critical load factor (lambda_cr = {lam_cr:.6g}). The "
                "returned mode is implementation-dependent; any linear "
                "combination of these modes is equally valid.",
                AmbiguousModeWarning,
                stacklevel=2,
            )

    mode = np.zeros(n_dof)
    mode[free] = mode_free

    zero_band = 1e-12  # relative to E A, the library-wide strain-zero band
    compressed = int(
        np.count_nonzero(n_total < -zero_band * setup.k_axial * geom.lengths)
    )
    return BucklingResult(
        lambda_cr=float(lam_cr),
        mode=np.asarray(mode, dtype=float),
        modes=modes_list,
        multiplicity=multiplicity,
        n_compressed=compressed,
        base_forces={e.id: float(n_total[i]) for i, e in enumerate(elements)},
    )
