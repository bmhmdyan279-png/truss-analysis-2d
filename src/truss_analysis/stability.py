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
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.linalg import LinAlgError, cholesky, eigh, solve_triangular
from scipy.sparse.linalg import ArpackError, LinearOperator, SuperLU, eigsh, splu

from .assembly import MemberGeometry, member_geometry
from .criticality.engine import (
    MechanismError,
    base_displacement,
    build_engine,
    load_vector,
    member_forces,
    total_load_vector,
)
from .exceptions import (
    AmbiguousModeWarning,
    EigenConvergenceError,
    InputIgnoredWarning,
    ShallowSystemWarning,
)
from .model import Element, Node, fixed_dof_indices

__all__ = [
    "CODE_IMPERFECTION_AMPLITUDES",
    "DEFAULT_IMPERFECTION_AMPLITUDES",
    "EIGEN_MODE_PADDING",
    "SPARSE_EIGEN_THRESHOLD",
    "BucklingResult",
    "EigenSolver",
    "ImperfectionStudy",
    "geometric_stiffness",
    "imperfection_sensitivity",
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
    n_vec = np.array([float(forces.get(e.id, 0.0)) for e in elements], dtype=float)
    coeff = n_vec / geom.lengths  # N_e / L_e
    n_out = n_dof if free_dofs is None else len(free_dofs)

    if sparse:
        # Mirror the elastic sparse path: build (row, col, val) triplets for
        # every dyad and let COO sum duplicates in C.  Each dyad is formed
        # from the member's LOCAL 4-vector ``g_e = [s, -c, -s, c]`` -- built
        # straight from the shared cosines rather than gathered out of the
        # full ``(m, n_dof)`` matrix, which is what
        # :func:`member_geometric_vectors` scatters them into -- so the two
        # paths cannot disagree by an indexing slip.  When the output is
        # restricted to ``free_dofs``, entries whose DOF is restrained are
        # masked out of the triplet lists.
        _b_local, g_local = _local_dyad_vectors(geom)
        if free_dofs is None:
            idx_local = geom.dofs  # (m, 4) output-space indices
        else:
            idx_local = _free_dof_positions(n_dof, list(free_dofs))[geom.dofs]
            g_local = np.where(idx_local >= 0, g_local, 0.0)
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

    g = member_geometric_vectors(geom, n_dof)
    if free_dofs is not None:
        g = g[:, list(free_dofs)]
    dense: np.ndarray = np.einsum("e,ei,ej->ij", coeff, g, g)
    return dense


# ---------------------------------------------------------------------------
# eigen-path selection (C1: large-model scalability)
# ---------------------------------------------------------------------------

#: Free-DOF count at which ``eigen_solver="auto"`` switches from the dense
#: whitened LAPACK path to sparse Lanczos.
#:
#: Set from measurement, not from taste.  On a Pratt truss (4 members per
#: panel, ``n_free ~ 4 * n_panels``) the two paths were timed against each
#: other and agree to ``< 2e-13`` relative on ``lambda_cr`` throughout:
#:
#: .. code-block:: text
#:
#:     n_free     33     65    121    181    241    481    801   1281   2001
#:     sparse/dense  1.7x  3.6x  0.42x  0.88x  0.52x  0.40x  0.36x  0.29x  0.29x
#:
#: (entries < 1 mean the sparse path is faster).  Below ~100 free DOFs the
#: fixed cost of the sparse factorisation and the ARPACK setup dominates and
#: the dense path wins; from ~120 upward the sparse path is ahead, reaching
#: ~3.5x by 1300 DOFs and holding there.  400 is chosen rather than 120
#: because the dense path still buys two things the sparse one does not: the
#: *full* spectrum rather than the leading Ritz pairs, and a verdict whose
#: arithmetic is LAPACK's rather than SuperLU's -- both paths now decide
#: positive-definiteness unconditionally (Cholesky failure on the dense side,
#: Sylvester inertia of the ``U`` diagonal on the sparse side, see
#: :func:`_sparse_inertia`), so the choice is about spectrum and provenance,
#: not about safety.  For a model that solves in well under a second, the
#: dense path is the better trade.  Memory is the other driver: the dense
#: path holds three ``n x n`` matrices, 8n^2 bytes each, which is 96 MB at
#: n = 2000 and 2.4 GB at n = 10000.  Override per call with
#: ``eigen_solver="dense"`` / ``"sparse"``.
SPARSE_EIGEN_THRESHOLD = 400

#: Extra eigenpairs requested beyond ``n_modes``.  The padding exists so that
#: multiplicity detection has room: with ``n_modes = 1`` the caller still
#: needs to *see* the second eigenvalue to know whether the critical one is
#: repeated.
EIGEN_MODE_PADDING = 2

#: Relative eigenvalue tolerance for declaring two load factors equal.
_EIGEN_MULT_RTOL = 1e-10

#: Absolute floor below which an eigenvalue ``nu`` is treated as zero (and
#: therefore as "no bifurcation in this direction").
_NU_POSITIVE_FLOOR = 1e-14


def _definiteness_tolerance(n: int, a_scale: float) -> float:
    """Below this, ``lambda_min`` is numerically indistinguishable from zero.

    The dense path takes its verdict from a Cholesky *failure*, whose
    effective threshold is ``lambda_min <~ n * eps * ||A||``.  The sparse path
    has an actual number, so it needs the same bound written down explicitly
    -- otherwise the two paths disagree about what counts as a mechanism, and
    a legitimately soft-but-stable base state (a slender truss with a very
    low first eigenvalue, measured at ``lambda_min ~ 0.5`` N/m against
    ``||A||_F ~ 1.6e10``) gets rejected on one path and accepted on the
    other.

    ``a_scale`` is the Frobenius norm, an upper bound on ``||A||_2``, so the
    bound errs slightly toward declaring instability.  That is the correct
    direction for a safety-critical verdict.
    """
    return max(int(n), 1) * float(np.finfo(float).eps) * max(float(a_scale), 1.0)


#: Relative rise below which a system counts as shallow (A4).
SHALLOW_RISE_SPAN = 0.1

#: Relative drop in ``lambda_cr`` at the largest probed amplitude above which
#: :func:`imperfection_sensitivity` reports the system as imperfection
#: sensitive (B4).  5% is an engineering judgement, not a code limit, and is
#: stated here so it can be argued with.
IMPERFECTION_SENSITIVE_DROP = 0.05

#: EN 1993-1-1 Table 5.1 equivalent initial bow imperfections ``e0/L`` for
#: buckling curves a0, a, b, c and d -- the code's own band of *member*
#: out-of-straightness, from L/350 (curve a0) to L/150 (curve d).  Used here
#: as the realistic range for a *system* geometric imperfection: transferring
#: a member tolerance to a whole-structure initial shape is an approximation,
#: but it is the approximation a designer actually has data for, and it is an
#: order of magnitude tighter than "2% of the bounding box".
CODE_IMPERFECTION_AMPLITUDES: tuple[float, ...] = (
    1.0 / 350.0,
    1.0 / 300.0,
    1.0 / 250.0,
    1.0 / 200.0,
    1.0 / 150.0,
)

#: Default imperfection amplitudes, as fractions of ``reference_length``.
#:
#: The round-7 audit measured the previous default ``(0.001, 0.005, 0.01,
#: 0.02)`` on a 24 m truss: its largest amplitude is an initial out-of-
#: straightness of **0.48 m**, several times larger than any code-based
#: tolerance, and because the sensitivity verdict was taken from
#: ``max(relative_drop)`` the *verdict itself* was decided by the least
#: realistic amplitude in the set.  The series now spans the code band as
#: well as the exploratory tail, so the trend and the verdict come from
#: amplitudes a designer could defend.  Four amplitudes, as before -- eight
#: solves per study, since both imperfection signs are probed.
#:
#: * ``0.001``  -- small enough for the local gradient fit (see
#:   :attr:`~truss_analysis.stability.ImperfectionStudy.normalized_gradient`);
#: * ``1/350``  -- the least conservative EN 1993-1-1 Table 5.1 value;
#: * ``1/200``  -- the most conservative common Table 5.1 value, and the
#:   amplitude that drives the verdict by default;
#: * ``0.02``   -- exploratory tail, reported but not decisive.
DEFAULT_IMPERFECTION_AMPLITUDES: tuple[float, ...] = (
    0.001,
    1.0 / 350.0,
    1.0 / 200.0,
    0.02,
)

EigenSolver = Literal["auto", "dense", "sparse"]
"""Eigen-solve strategy for :func:`linearized_buckling_load_factor`.

``"auto"`` picks by size (:data:`SPARSE_EIGEN_THRESHOLD`), ``"dense"`` forces
the whitened LAPACK path, ``"sparse"`` forces Lanczos.  The path actually
taken is reported on :attr:`BucklingResult.solver_path`, so a forced
``"sparse"`` request that had to fall back (ARPACK needs ``k < n``) is
visible rather than silent.
"""


def _local_dyad_vectors(
    geom: MemberGeometry,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-member 4-vectors ``b = [-c, -s, c, s]`` and ``g = [s, -c, -s, c]``.

    Built straight from the shared :class:`~truss_analysis.assembly.MemberGeometry`
    cosines, so the sparse assembly cannot drift from the dense one.
    """
    c = np.asarray(geom.cosines, dtype=float)
    s = np.asarray(geom.sines, dtype=float)
    b_local = np.stack([-c, -s, c, s], axis=1)
    g_local = np.stack([s, -c, -s, c], axis=1)
    return b_local, g_local


def _free_dof_positions(n_dof: int, free: Sequence[int]) -> npt.NDArray[np.intp]:
    """Global-DOF -> position-in-``free`` lookup vector (``-1`` if restrained)."""
    inv = np.full(n_dof, -1, dtype=np.intp)
    free_arr = np.asarray(free, dtype=np.intp)
    inv[free_arr] = np.arange(free_arr.size, dtype=np.intp)
    return inv


def _dyad_sum_sparse(
    geom: MemberGeometry,
    n_dof: int,
    free: Sequence[int],
    coeffs: npt.NDArray[np.float64],
    local: npt.NDArray[np.float64],
) -> sp.csr_matrix:
    """``sum_e coeffs[e] * local[e] local[e]^T`` on the free DOFs, in CSR.

    The ``O(nnz)`` counterpart of the ``einsum`` used by the dense path: only
    the 4x4 block of each member is ever materialised, so memory stays linear
    in the member count instead of quadratic in the DOF count.  Entries whose
    DOF is restrained are dropped from the triplet lists, matching
    :func:`geometric_stiffness` with ``sparse=True``.

    The global-to-free DOF map is applied by fancy indexing rather than a
    Python comprehension over ``geom.dofs``: at the sizes this path exists
    for, the comprehension is a measurable fraction of the total cost.
    """
    idx = _free_dof_positions(n_dof, free)[geom.dofs]
    vals_local = np.where(idx >= 0, local, 0.0)
    rows = np.repeat(idx, 4, axis=1)
    cols = np.tile(idx, (1, 4))
    vals = np.einsum("e,ei,ej->eij", coeffs, vals_local, vals_local).reshape(
        len(idx), 16
    )
    keep = (rows >= 0) & (cols >= 0)
    n_out = len(free)
    return sp.coo_matrix(
        (vals[keep], (rows[keep], cols[keep])), shape=(n_out, n_out)
    ).tocsr()


def _factor_sparse_base(
    a_sparse: sp.csr_matrix,
) -> tuple[LinearOperator, SuperLU]:
    """Sparse ``LU`` of the prestressed base state, wrapped as an operator.

    One factorisation serves three consumers -- the inertia verdict, the
    positive-definiteness probe and the Lanczos eigensolve -- because all
    three need ``A^{-1}``: the probe through shift-invert with ``sigma = 0``
    (which is exactly ``A^{-1}``), the eigensolve as the ``M``-preconditioner
    of the generalised problem, and the inertia read off the same ``U``
    factor.  Factoring twice would roughly double the sparse path's cost.

    ``permc_spec="MMD_AT_PLUS_A"`` with ``SymmetricMode`` is SuperLU's
    fill-reducing ordering for symmetric patterns, which is what a stiffness
    matrix is.  ``diag_pivot_thresh=0.0`` is *load-bearing* here rather than
    merely a stability preference: it forces SuperLU to take the diagonal
    entry as the pivot, so the factorisation is the symmetric
    ``P A P^T = L U`` whose ``U`` diagonal carries the inertia (see
    :func:`_sparse_inertia`).  A threshold-pivoted factorisation would
    permute rows independently of columns and the sign count would mean
    nothing.

    Returns
    -------
    tuple[scipy.sparse.linalg.LinearOperator, scipy.sparse.linalg.SuperLU]
        The ``A^{-1}`` operator and the factorisation it was built from.

    Raises
    ------
    MechanismError
        If ``A`` is exactly singular: the base state has lost a degree of
        stability before any load amplification.
    """
    n = int(a_sparse.shape[0])
    try:
        lu = splu(
            a_sparse.tocsc(),
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
    except RuntimeError as exc:
        msg = (
            "prestressed base state is singular: the imposed "
            "(thermal/fabrication) force state alone has reached a critical "
            f"point, so no load factor is defined ({exc})"
        )
        raise MechanismError(msg) from exc
    return LinearOperator((n, n), matvec=lu.solve, dtype=float), lu


def _sparse_inertia(lu: SuperLU) -> tuple[int, int]:
    """Count the negative and zero pivots of a symmetric ``splu`` factor.

    By **Sylvester's law of inertia** the number of negative (respectively
    zero) diagonal entries of ``U`` in a symmetric factorisation
    ``P A P^T = L U`` equals the number of negative (respectively zero)
    eigenvalues of ``A``, regardless of the fill-reducing permutation.  This
    is an *unconditional* verdict: it does not depend on an iterative solver
    converging, on a tolerance, or on where the spectrum happens to sit.

    Why this exists -- the round-7 audit's most important finding.  The
    shift-invert probe in :func:`_sparse_smallest_eigenvalue` looks for the
    eigenvalue of *smallest magnitude*, which is the right question at the
    loss-of-definiteness crossing (``lambda_min -> 0``) and the wrong one
    past it.  On a deeply indefinite base state -- a temperature near
    collapse, where ``K_E`` has degraded towards zero while
    ``K_G(N_imposed)`` has driven it negative, giving a spectrum such as
    ``{-5, +1, +2}`` -- the probe returns ``+1``, the tolerance test passes,
    and ``eigsh`` is then handed an indefinite ``M`` for what is documented
    as a symmetric positive-definite generalised problem.  ARPACK does not
    validate that assumption; it returns a number.  A safety-critical verdict
    that depends on the user *guessing* to pass ``eigen_solver="dense"`` is
    exactly the silent failure mode the library's physics-boundary policy
    forbids.

    The inertia read costs nothing: ``U`` is already in hand from the
    factorisation the path needs anyway.

    Parameters
    ----------
    lu : scipy.sparse.linalg.SuperLU
        A factorisation produced by :func:`_factor_sparse_base`, i.e. with
        ``SymmetricMode=True`` and ``diag_pivot_thresh=0.0``.

    Returns
    -------
    tuple[int, int]
        ``(n_negative, n_zero)`` -- the counts of negative and exactly-zero
        pivots of ``U``.  ``A`` is positive definite iff both are zero.
    """
    diag_u = np.asarray(lu.U.diagonal(), dtype=float)
    return int(np.count_nonzero(diag_u < 0.0)), int(np.count_nonzero(diag_u == 0.0))


def _sparse_smallest_eigenvalue(
    a_sparse: sp.csr_matrix, a_inv: LinearOperator
) -> float:
    """Eigenvalue of the base state nearest zero, by shift-invert Lanczos.

    With ``sigma = 0`` the transformed operator is ``A^{-1}``, whose largest
    eigenvalue is ``1/lambda_min`` -- so ``which="LM"`` returns
    ``lambda_min`` for a positive definite ``A``, and returns an eigenvalue
    straddling zero for one that is not.  This converges in a handful of
    iterations because ``1/lambda_min`` is spectrally well separated from the
    rest of ``A^{-1}``'s eigenvalues.

    Shifting far *below* the spectrum instead -- the ``sigma = -||A||_F``
    choice that looks more obviously rigorous -- is numerically useless here:
    every transformed eigenvalue then clusters at ``1/||A||_F`` and Lanczos
    needs thousands of iterations to separate them.  Measured, not assumed:
    the far shift failed to converge at 1121 free DOFs where ``sigma = 0``
    converges in under ten iterations.

    Scope, stated plainly: this probe detects the *loss-of-definiteness
    crossing* (``lambda_min -> 0``), which is the physically relevant
    failure and the one the dense path's Cholesky attempt also catches.  A
    base state that is already deeply indefinite -- far past the crossing,
    with no eigenvalue near zero -- can return a positive value here, so
    this function is **not** a positive-definiteness test on its own.  The
    unconditional verdict comes from :func:`_sparse_inertia`, which the
    caller applies *before* this probe runs; by the time control reaches
    here the matrix is already known to be positive definite and the probe
    is reporting a magnitude, not deciding a verdict.
    """
    n = int(a_sparse.shape[0])
    if n == 0:
        return float("inf")
    if n == 1:
        return float(a_sparse[0, 0])
    try:
        vals = eigsh(
            a_sparse,
            k=1,
            sigma=0.0,
            OPinv=a_inv,
            which="LM",
            return_eigenvectors=False,
        )
    except ArpackError as exc:
        msg = (
            "the sparse positive-definiteness probe of the prestressed base "
            f"state did not converge ({exc}); retry with eigen_solver='dense'"
        )
        raise EigenConvergenceError(msg) from exc
    arr = np.asarray(vals, dtype=float)
    return float(np.min(arr)) if arr.size else float("inf")


def _dense_spectrum(
    a_mat: npt.NDArray[np.float64],
    b_mat: npt.NDArray[np.float64],
    k_want: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Whitened LAPACK path: the top ``k_want`` eigenpairs, ``nu`` descending.

    ``C = L^-1 B L^-T`` with ``LL^T = A`` turns the generalised problem
    ``B x = nu A x`` into a standard symmetric one; the eigenvectors come
    back in physical coordinates through ``x = L^-T v``.  Only the leading
    ``k_want`` pairs are computed (``subset_by_index``), which is what makes
    ``n_modes`` mean something rather than being a silently ignored argument.

    Raises
    ------
    scipy.linalg.LinAlgError
        If ``A`` is not positive definite; the caller converts it to
        :class:`MechanismError`.
    """
    n = int(a_mat.shape[0])
    chol = cholesky(a_mat, lower=True, check_finite=False)
    x = solve_triangular(chol, b_mat, lower=True, check_finite=False)
    c_mat = solve_triangular(chol, x.T, lower=True, check_finite=False).T
    c_mat = 0.5 * (c_mat + c_mat.T)  # kill round-off asymmetry

    k = max(1, min(k_want, n))
    if k >= n:
        nu, vecs_w = eigh(c_mat)
    else:
        nu, vecs_w = eigh(c_mat, subset_by_index=[n - k, n - 1])
    order = np.argsort(np.asarray(nu, dtype=float))[::-1]
    nu_out = np.asarray(nu, dtype=float)[order]
    physical = solve_triangular(
        chol.T, np.asarray(vecs_w, dtype=float)[:, order], check_finite=False
    )
    return nu_out, np.asarray(physical, dtype=float)


def _sparse_spectrum(
    a_sparse: sp.csr_matrix,
    b_sparse: sp.csr_matrix,
    a_inv: LinearOperator,
    k_want: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Lanczos path: the top ``k_want`` eigenpairs of ``B x = nu A x``.

    No whitening, no dense matrix, no full spectrum: the sparse factorisation
    of ``A`` (already computed by :func:`_factor_sparse_base`) is handed to
    ARPACK as the ``M``-preconditioner and only the leading Ritz pairs are
    formed.  ``which="LA"`` asks for the largest *algebraic* ``nu``, which is
    ``1/lambda_cr`` -- not the same request as "smallest magnitude", and
    asking for the wrong end of the spectrum is the easy way to turn a
    scalability fix into a wrong answer.

    Raises
    ------
    EigenConvergenceError
        If ARPACK fails to converge.  A buckling load factor is
        safety-critical, so an unconverged Ritz value is never returned as if
        it were an answer.
    """
    n = int(a_sparse.shape[0])
    k = max(1, min(k_want, n - 1))
    try:
        nu, vecs = eigsh(b_sparse, k=k, M=a_sparse, Minv=a_inv, which="LA")
    except ArpackError as exc:
        msg = (
            f"Lanczos did not converge on the {n}-DOF bifurcation problem "
            f"after requesting {k} eigenpair(s) ({exc}); retry with "
            "eigen_solver='dense' or a coarser model"
        )
        raise EigenConvergenceError(msg) from exc
    order = np.argsort(np.asarray(nu, dtype=float))[::-1]
    return (
        np.asarray(nu, dtype=float)[order],
        np.asarray(vecs, dtype=float)[:, order],
    )


def _select_sparse_path(eigen_solver: EigenSolver, n_free: int, k_want: int) -> bool:
    """Decide whether the Lanczos path can and should be used."""
    if eigen_solver not in ("auto", "dense", "sparse"):
        msg = (
            "eigen_solver must be one of 'auto', 'dense', 'sparse'; got "
            f"{eigen_solver!r}"
        )
        raise ValueError(msg)
    if eigen_solver == "dense":
        return False
    # ARPACK requires k < n; with no room for even one Ritz pair the dense
    # path is the only option, whatever was requested.
    if n_free <= k_want + 1:
        return False
    if eigen_solver == "sparse":
        return True
    return n_free >= SPARSE_EIGEN_THRESHOLD


def _rise_span_ratio(nodes: Sequence[Node]) -> float:
    """Smaller-over-larger bounding-box extent, i.e. depth/span.

    Orientation-robust: a vertical truss and the same truss rotated by 90
    degrees get the same number, which a plain ``ptp(y) / ptp(x)`` does not.

    A *degenerate* bounding box -- every node on one line, as for a single
    column or a perfectly flat chord -- has no meaningful depth/span ratio,
    and returning ``0.0`` there would flag every column check in
    :mod:`truss_analysis.limitstates` as a shallow system.  Such a model is
    reported as ``inf`` (never shallow); if the collinearity is real rather
    than a single member, the eigen-solve itself declares the mechanism,
    which is the correct diagnosis.
    """
    coords = np.array([[n.x, n.y] for n in nodes], dtype=float).reshape(-1, 2)
    if coords.shape[0] == 0:
        return float("inf")
    extents = np.ptp(coords, axis=0)
    longest = float(np.max(extents))
    shortest = float(np.min(extents))
    if longest <= 0.0 or shortest <= 0.0:
        return float("inf")
    return shortest / longest


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
        Critical buckling mode on the full DOF vector ``(2n,)``, unit
        2-norm, fixed DOFs exactly zero. Arbitrary sign (eigenvectors are).
        Equals ``modes[0]`` whenever :attr:`modes` is non-empty.  When
        ``lambda_cr`` is ``inf`` there is no critical mode and this field is
        the **zero vector**: it keeps a fixed shape so callers need not
        branch on ``None``, and its zero norm is the signal -- that is what
        :func:`imperfection_sensitivity` tests before refusing to sweep an
        imperfection along a direction that does not exist.
    modes : list[numpy.ndarray]
        The ``n_modes`` leading buckling modes on the free DOFs, ordered by
        *descending* ``nu`` -- i.e. by *ascending* ``lambda_cr``, so
        ``modes[0]`` is the critical one and equals ``mode``.  Each is unit
        2-norm.  Fewer than ``n_modes`` entries appear when the spectrum has
        fewer positive eigenvalues, and the list is **empty** when
        ``lambda_cr`` is ``inf``: absence is reported as absence rather than
        as a placeholder zero vector, so ``zip(modes, load_factors)`` cannot
        pair a non-mode with a non-load.
    load_factors : tuple[float, ...]
        Load factor belonging to each entry of :attr:`modes`, in the same
        order and the same length (so ``load_factors[0] == lambda_cr`` and
        the sequence is non-decreasing).  Empty exactly when :attr:`modes`
        is, i.e. when ``lambda_cr`` is ``inf``.  Every entry is finite: only
        eigenvalues with ``nu > _NU_POSITIVE_FLOOR`` are retained, so a
        non-positive ``nu`` -- which would map to an infinite or negative
        load factor -- never reaches this tuple and is instead reflected in
        ``lambda_cr = inf``.
    multiplicity : int
        Number of computed eigenvalues within
        :data:`_EIGEN_MULT_RTOL` (relative) of the critical one.
        ``multiplicity > 1`` indicates repeated buckling loads and triggers
        :exc:`~truss_analysis.exceptions.AmbiguousModeWarning`.  It is a
        *lower* bound: only the eigenpairs actually computed are compared,
        so raise ``n_modes`` if a higher multiplicity must be resolved.
        ``0`` when ``lambda_cr`` is ``inf`` -- there is no critical
        eigenvalue to be multiple of.
    n_compressed : int
        Number of members carrying compression in the base state beyond the
        library's force classification band.
    base_forces : dict[str, float]
        Base-state member forces [N] (mechanical + imposed) the bifurcation
        was linearised about.
    solver_path : str
        Which eigen-solve actually ran: ``"dense-whitened"`` or
        ``"sparse-lanczos"``.  Reported rather than implied, because
        ``eigen_solver="sparse"`` silently cannot be honoured on a model too
        small for ARPACK and the caller is entitled to know.
    n_free_dof : int
        Number of free DOFs the eigenproblem was posed on.
    """

    lambda_cr: float
    mode: npt.NDArray[np.float64]
    modes: list[npt.NDArray[np.float64]]
    load_factors: tuple[float, ...]
    multiplicity: int
    n_compressed: int
    base_forces: dict[str, float]
    solver_path: str = "dense-whitened"
    n_free_dof: int = 0


def linearized_buckling_load_factor(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float] | None = None,
    n_modes: int = 1,
    eigen_solver: EigenSolver = "auto",
    warn_shallow: bool = True,
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
        Number of leading buckling modes to return in
        :attr:`BucklingResult.modes`, ordered by ascending ``lambda_cr``.
        The solver internally requests ``n_modes + EIGEN_MODE_PADDING``
        eigenpairs so that multiplicity of the critical load can be detected
        even at ``n_modes = 1``.
    eigen_solver : {"auto", "dense", "sparse"}, default "auto"
        ``"dense"`` is the whitened LAPACK path (full accuracy, ``O(n^3)``
        time and ``O(n^2)`` memory); ``"sparse"`` is Lanczos on the CSR
        matrices (``O(nnz)`` memory, only the leading eigenpairs);
        ``"auto"`` switches at :data:`SPARSE_EIGEN_THRESHOLD` free DOFs.
        The path taken is reported on
        :attr:`BucklingResult.solver_path`.
    warn_shallow : bool, default True
        Issue :exc:`~truss_analysis.exceptions.ShallowSystemWarning` when the
        depth/span ratio is below :data:`SHALLOW_RISE_SPAN`.  Set to
        ``False`` inside sweeps (e.g.
        :func:`imperfection_sensitivity`) that would otherwise emit the same
        warning once per amplitude.

    Returns
    -------
    BucklingResult
        ``lambda_cr``, the leading mode shape(s), the multiplicity of the
        critical eigenvalue and the base state they refer to.

    Raises
    ------
    MechanismError
        If there are no free DOFs, or if the prestressed base state
        ``K_E + K_G(N_imposed)`` is not positive definite: the structure is
        *already* at or past a critical point before any load amplification
        (or is a mechanism outright), and a load factor is undefined.
    ValueError
        If ``n_modes < 1`` or ``eigen_solver`` is not a recognised strategy.
    EigenConvergenceError
        If the sparse Lanczos path fails to converge.  An unconverged
        buckling load is not returned as an answer.

    Notes
    -----
    When ``warn_shallow`` is set and the depth/span ratio is below 0.1, a
    :exc:`~truss_analysis.exceptions.ShallowSystemWarning` is issued (not
    raised): for such systems the true collapse is a snap-through limit
    point and the linearised factor can significantly underestimate the
    demand at failure.  Use :func:`imperfection_sensitivity` to quantify how
    much of that optimism is geometry-driven.
    """
    if int(n_modes) < 1:
        msg = f"n_modes must be >= 1, got {n_modes}"
        raise ValueError(msg)
    n_modes = int(n_modes)

    setup = build_engine(nodes, elements, loads, temps)
    free = list(setup.free_dofs)
    if not free:
        msg = (
            "linearized_buckling_load_factor: every DOF is restrained; "
            "there is no stability problem to solve"
        )
        raise MechanismError(msg)

    # A4: shallow-geometry screen.  Orientation-robust depth/span ratio.
    ratio = _rise_span_ratio(nodes)
    if warn_shallow and ratio < SHALLOW_RISE_SPAN:
        warnings.warn(
            f"ShallowSystemWarning: depth/span ratio = {ratio:.3f} < "
            f"{SHALLOW_RISE_SPAN}; linearised buckling may significantly "
            "underestimate the snap-through load. Use "
            "imperfection_sensitivity() or a geometrically nonlinear analysis "
            "for shallow systems.",
            ShallowSystemWarning,
            stacklevel=2,
        )

    n_dof = 2 * len(nodes)
    n_free = len(free)
    geom = member_geometry(list(nodes), list(elements))

    # Base-state demand: total, mechanical-only, and the imposed remainder.
    u_total = base_displacement(setup, total_load_vector(nodes, loads, setup))
    n_total = member_forces(setup, u_total)
    u_mech = base_displacement(setup, load_vector(nodes, loads, free))
    n_mech = setup.k_axial * (setup.b_free @ u_mech)
    n_imposed = n_total - n_mech

    coeff_th = n_imposed / geom.lengths
    # Load-pattern geometric stiffness, negated so that lambda > 0 is the
    # physically meaningful amplification of the mechanical demand.
    coeff_mech = -(n_mech / geom.lengths)

    k_want = min(n_free, n_modes + EIGEN_MODE_PADDING)
    use_sparse = _select_sparse_path(eigen_solver, n_free, k_want)

    if use_sparse:
        b_local, g_local = _local_dyad_vectors(geom)
        a_sparse = _dyad_sum_sparse(geom, n_dof, free, setup.k_axial, b_local)
        a_sparse = a_sparse + _dyad_sum_sparse(geom, n_dof, free, coeff_th, g_local)
        b_sparse = _dyad_sum_sparse(geom, n_dof, free, coeff_mech, g_local)
        a_inv, lu = _factor_sparse_base(a_sparse)
        a_scale = float(sp.linalg.norm(a_sparse, "fro"))
        # Unconditional verdict first: Sylvester inertia from the factor's U
        # diagonal.  The shift-invert probe below detects the *crossing*
        # (lambda_min -> 0) but is blind to a base state already deep past
        # it, so it is a diagnostic here, never the gate (round-7 audit).
        n_negative, n_zero = _sparse_inertia(lu)
        if n_zero:
            msg = (
                "prestressed base state is singular: the imposed "
                "(thermal/fabrication) force state alone has reached a "
                f"critical point ({n_zero} exactly-zero pivot(s) in the "
                "symmetric factorisation), so no load factor is defined"
            )
            raise MechanismError(msg)
        if n_negative:
            msg = (
                "prestressed base state is not positive definite: the imposed "
                "(thermal/fabrication) force state alone has passed a critical "
                f"point ({n_negative} of {n_free} negative eigenvalue(s), by "
                "Sylvester inertia of the symmetric factorisation), so no "
                "load factor is defined"
            )
            raise MechanismError(msg)
        lam_min = _sparse_smallest_eigenvalue(a_sparse, a_inv)
        pd_tol = _definiteness_tolerance(n_free, a_scale)
        if lam_min <= pd_tol:
            msg = (
                "prestressed base state is not positive definite: the imposed "
                "(thermal/fabrication) force state alone has reached or passed "
                f"a critical point (lambda_min = {lam_min:.6g} <= the "
                f"numerical zero bound {pd_tol:.6g} for ||A||_F = "
                f"{a_scale:.6g}), so no load factor is defined"
            )
            raise MechanismError(msg)
        nu, modes_free = _sparse_spectrum(a_sparse, b_sparse, a_inv, k_want)
        solver_path = "sparse-lanczos"
    else:
        g_free = member_geometric_vectors(geom, n_dof)[:, free]
        k_e_ff = np.einsum("i,ip,iq->pq", setup.k_axial, setup.b_free, setup.b_free)
        a_mat = k_e_ff + np.einsum("e,ei,ej->ij", coeff_th, g_free, g_free)
        b_mat = np.einsum("e,ei,ej->ij", coeff_mech, g_free, g_free)
        try:
            nu, modes_free = _dense_spectrum(a_mat, b_mat, k_want)
        except LinAlgError as exc:
            msg = (
                "prestressed base state is not positive definite: the imposed "
                "(thermal/fabrication) force state alone has reached or "
                "passed a critical point, so no load factor is defined"
            )
            raise MechanismError(msg) from exc
        solver_path = "dense-whitened"

    # --- critical value, multiplicity and modes ---------------------------
    positive = nu > _NU_POSITIVE_FLOOR
    if not np.any(positive):
        # No bifurcation under load amplification.  The container reports
        # *absence* rather than a placeholder: an empty ``modes`` list and an
        # empty ``load_factors`` tuple, so a caller that zips the two gets
        # nothing to misread.  The previous shape returned
        # ``modes=[zeros]`` with ``load_factors=(inf,)``, which paired a zero
        # vector with an infinite load factor and invited exactly the reading
        # "here is the buckling mode, it buckles at infinity" -- a mode shape
        # that is not a mode shape, for a load that is not a load.  ``mode``
        # stays the zero vector because it is a fixed-shape field and
        # :func:`imperfection_sensitivity` relies on its norm being zero to
        # detect that there is no bifurcation direction to imperil.
        lam_cr = float("inf")
        modes_list: list[npt.NDArray[np.float64]] = []
        load_factors: tuple[float, ...] = ()
        multiplicity = 0
    else:
        nu_pos = nu[positive]
        modes_pos = modes_free[:, positive]
        nu_max = float(nu_pos[0])  # nu is sorted descending by construction
        lam_cr = 1.0 / nu_max

        eig_tol = _EIGEN_MULT_RTOL * max(1.0, abs(nu_max))
        multiplicity = int(np.count_nonzero(np.abs(nu_pos - nu_max) < eig_tol))

        modes_list = []
        factors: list[float] = []
        for i in range(min(n_modes, nu_pos.shape[0])):
            vec = np.asarray(modes_pos[:, i], dtype=float)
            norm = float(np.linalg.norm(vec))
            modes_list.append(vec / norm if norm > 0.0 else vec)
            factors.append(float(1.0 / nu_pos[i]) if nu_pos[i] > 0.0 else float("inf"))
        load_factors = tuple(factors)

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
    if modes_list:
        mode[free] = modes_list[0]

    zero_band = 1e-12  # relative to E A, the library-wide strain-zero band
    compressed = int(
        np.count_nonzero(n_total < -zero_band * setup.k_axial * geom.lengths)
    )
    return BucklingResult(
        lambda_cr=float(lam_cr),
        mode=np.asarray(mode, dtype=float),
        modes=modes_list,
        load_factors=load_factors,
        multiplicity=multiplicity,
        n_compressed=compressed,
        base_forces={e.id: float(n_total[i]) for i, e in enumerate(elements)},
        solver_path=solver_path,
        n_free_dof=n_free,
    )


#: Ceiling on how many amplitudes the local gradient fit consumes.
#:
#: A first-order sensitivity is a derivative at ``eps -> 0``.  Fitting it over
#: a series that spans a decade does not produce a derivative; it produces a
#: secant dominated by the largest amplitudes, where the response has already
#: curved away.  Measured on the closed-form two-bar toggle: the global
#: least-squares slope over ``(0.001, 1/350, 1/200, 0.02)`` is ``-42.3``
#: against a true derivative of ``-119.9`` -- a 65% error, in the
#: *unconservative* direction, for a field whose docstring had always
#: described it as ``d(lambda/lambda_0)/d(eps)``.  Restricting the fit to the
#: code band brings that to 0.003%.
LOCAL_GRADIENT_POINTS = 4

#: Degree ceiling of the local gradient fit.  Cubic is the highest degree the
#: default series can support once the exact ``eps = 0`` anchor is included,
#: and it already resolves the toggle's curvature to four significant figures.
LOCAL_GRADIENT_DEGREE = 3


def _select_local_amplitudes(
    amplitudes: npt.NDArray[np.float64],
    usable: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Pick the amplitudes a *derivative at zero* may legitimately be fitted on.

    The selection rule is physical, not arbitrary: only amplitudes inside the
    EN 1993-1-1 Table 5.1 out-of-straightness band
    (:data:`CODE_IMPERFECTION_AMPLITUDES`) describe a geometry a built
    structure could actually have, and a derivative estimated from
    amplitudes beyond that band is an extrapolation of a curve the code never
    claimed to describe.  The smallest usable amplitude is always kept, so a
    series that starts inside the band stays inside it.

    Returns
    -------
    tuple[numpy.ndarray, bool]
        The selected amplitudes, ascending, and whether the selection stayed
        inside the code band.  ``False`` means no probed amplitude was inside
        the band and the fit fell back to the smallest
        :data:`LOCAL_GRADIENT_POINTS` -- which the caller turns into a
        warning, because a first-order sensitivity quoted from
        un-code-like amplitudes should not look like one quoted from
        code-like ones.
    """
    amps = np.asarray(amplitudes, dtype=float)[usable]
    if amps.size == 0:
        return amps, True
    amps = np.sort(amps)
    band_ceiling = max(CODE_IMPERFECTION_AMPLITUDES)
    in_band = amps <= band_ceiling
    if not bool(np.any(in_band)):
        return amps[: min(amps.size, LOCAL_GRADIENT_POINTS)], False
    n_in_band = int(np.count_nonzero(in_band))
    selected = amps[in_band][: min(n_in_band, LOCAL_GRADIENT_POINTS)]
    # always keep the smallest amplitude, even if it sits below the band
    if selected[0] > amps[0]:
        selected = np.concatenate(([amps[0]], selected))
    return selected, True


def _local_gradient(
    amplitudes: npt.NDArray[np.float64],
    ratios: npt.NDArray[np.float64],
) -> tuple[float, tuple[float, ...], bool]:
    """First-order sensitivity ``d(lambda/lambda_0)/d(eps)`` at ``eps -> 0``.

    Fits a polynomial of degree ``min(LOCAL_GRADIENT_DEGREE, n - 1)`` to the
    amplitudes :func:`_select_local_amplitudes` keeps, **plus the exact anchor
    ``(0, 1)``** -- the perfect-geometry ratio is 1 by definition, so it is a
    free data point that raises the supportable degree by one without another
    eigensolve -- and returns the fit's derivative at ``eps = 0``.

    Parameters
    ----------
    amplitudes, ratios : numpy.ndarray
        Probed amplitudes and the corresponding ``lambda_cr(eps)/lambda_cr(0)``.
        Non-finite ratios (an imperfect geometry that lost stability outright)
        are dropped.

    Returns
    -------
    tuple[float, tuple[float, ...], bool]
        The derivative at zero (``nan`` when fewer than two usable amplitudes
        remain), the amplitudes actually fitted -- so the number carries its
        own provenance rather than being an unverifiable scalar -- and whether
        those amplitudes lay inside the EN 1993-1-1 Table 5.1 band.
    """
    amps = np.asarray(amplitudes, dtype=float)
    vals = np.asarray(ratios, dtype=float)
    usable = np.isfinite(vals)
    selected, in_band = _select_local_amplitudes(amps, usable)
    if selected.size < 2:
        return float("nan"), tuple(float(x) for x in selected), in_band
    order = np.argsort(amps[usable])
    usable_amps = amps[usable][order]
    usable_vals = vals[usable][order]
    idx = np.searchsorted(usable_amps, selected)
    fitted_a = np.concatenate(([0.0], selected))
    fitted_y = np.concatenate(([1.0], usable_vals[idx]))
    degree = min(LOCAL_GRADIENT_DEGREE, fitted_a.size - 1)
    coeffs = np.polyfit(fitted_a, fitted_y, degree)
    # np.polyfit returns the highest degree first, so the linear coefficient --
    # the derivative at zero -- is the second-to-last entry.
    return float(coeffs[-2]), tuple(float(x) for x in selected), in_band


# ---------------------------------------------------------------------------
# B4: imperfection sensitivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImperfectionStudy:
    """How much a geometric imperfection erodes the bifurcation load.

    Attributes
    ----------
    amplitudes : tuple[float, ...]
        Imperfection amplitudes probed, as fractions of
        :attr:`reference_length`.
    lambda_cr : tuple[float, ...]
        The *adverse* linearised load factor at each amplitude -- the smaller
        of the two imperfection signs (see the class Notes).  Same order as
        :attr:`amplitudes`.
    lambda_cr_positive : tuple[float, ...]
        Load factor with the imperfection applied along ``+mode``.
    lambda_cr_negative : tuple[float, ...]
        Load factor with the imperfection applied along ``-mode``.
    reference_lambda_cr : float
        Load factor of the perfect geometry (``amplitude = 0``).
    reference_length : float
        Length [m] the amplitudes are fractions of; the largest bounding-box
        extent unless overridden.
    imperfection_mode : numpy.ndarray
        Unit-norm mode shape applied as the imperfection, on the full DOF
        vector ``(2n,)``.
    relative_drop : tuple[float, ...]
        ``1 - lambda_cr(eps) / lambda_cr(0)`` per amplitude.  ``nan`` where
        the perfect-geometry factor is infinite (no bifurcation to erode).
    normalized_gradient : float
        First-order sensitivity ``d(lambda/lambda_0)/d(eps)`` **evaluated at
        ``eps = 0``**: the derivative at zero of a degree-``<= 3`` polynomial
        fitted to the probed amplitudes that lie inside the EN 1993-1-1
        Table 5.1 out-of-straightness band, anchored at the exact point
        ``(0, 1)``.  Negative means the imperfection is destabilising.
        ``nan`` when the reference factor is infinite or fewer than two
        usable amplitudes remain.

        This field used to be the global least-squares slope over *every*
        probed amplitude, which is a secant, not a derivative, and was wrong
        by a factor of 2.7 on the closed-form two-bar toggle (-42.3 against a
        true -119.9) -- wrong in the unconservative direction, for a quantity
        whose docstring had always claimed it was the first-order
        sensitivity.  The secant is still available, honestly named, as
        :attr:`secant_gradient`.
    secant_gradient : float
        Global least-squares slope of ``lambda_cr(eps) / lambda_cr(0)`` over
        **all** usable probed amplitudes -- the average erosion per unit
        imperfection across the whole sweep, not a derivative at any point.
        Kept because it is a legitimate summary of the sweep; kept *separate*
        because it is not :attr:`normalized_gradient` and calling it that was
        the defect.  ``nan`` under the same conditions.
    gradient_amplitudes : tuple[float, ...]
        The amplitudes :attr:`normalized_gradient` was actually fitted on, so
        the derivative carries its provenance instead of being an
        unverifiable scalar.  Empty when the gradient is ``nan``.
    gradient_in_code_band : bool
        ``True`` when :attr:`gradient_amplitudes` all lie within
        :data:`CODE_IMPERFECTION_AMPLITUDES`.  ``False`` means no probed
        amplitude was code-like and the derivative was extrapolated from
        larger ones -- in which case an
        :exc:`~truss_analysis.exceptions.InputIgnoredWarning`-family notice
        was issued at sweep time.
    verdict_amplitude : float
        The amplitude that drives :attr:`imperfection_sensitive`: the largest
        probed amplitude inside the code band, or the largest probed
        amplitude at all when none is inside it.  ``nan`` when the reference
        factor is infinite.
    relative_drop_at_verdict : float
        ``relative_drop`` evaluated at :attr:`verdict_amplitude``.  This, not
        ``max(relative_drop)``, is what the verdict compares against
        :data:`IMPERFECTION_SENSITIVE_DROP`.  ``nan`` under the same
        conditions.
    imperfection_sensitive : bool
        ``True`` when the erosion at :attr:`verdict_amplitude` exceeds
        :data:`IMPERFECTION_SENSITIVE_DROP`.  A system that is imperfection
        sensitive is one whose linearised bifurcation load should *not* be
        used as a design capacity without a nonlinear check -- the practical
        answer to the round-5 audit's objection that ``lambda_cr`` alone
        hides mode-shape sensitivity.

        The verdict is deliberately *not* ``max(relative_drop) > threshold``.
        Taking the maximum lets the least realistic amplitude in the sweep
        decide the answer: on a 24 m truss the old default's largest
        amplitude was an initial out-of-straightness of 0.48 m, several times
        any code tolerance, so the flag was being set by a geometry no one
        would build.  Anchoring the verdict inside the code band makes it a
        statement about a real structure.

    Notes
    -----
    Each point is a fresh *linearised* analysis on the imperfect geometry,
    not a nonlinear limit-point search: the imperfect system still has a
    tangent stiffness that loses definiteness at a computable factor, and
    the trend of that factor with amplitude is the engineering signal.  Full
    arc-length post-buckling remains out of scope (see the module
    docstring).

    **Both imperfection signs are probed and the adverse one reported.**  An
    eigenvector's sign is arbitrary, so "+the critical mode" carries no
    physical meaning on its own, and for an asymmetric post-buckling path the
    two signs are *not* equivalent.  On a shallow toggle, for instance,
    pushing the apex along one sign deepens the arch and ``lambda_cr``
    roughly quadruples with the rise, while the opposite sign flattens it and
    the reserve collapses.  Reporting only the favourable sign would be the
    single most dangerous possible output of this function, so
    :attr:`lambda_cr` is ``min(lambda_cr_positive, lambda_cr_negative)``
    elementwise and both are kept for diagnosis.
    """

    amplitudes: tuple[float, ...]
    lambda_cr: tuple[float, ...]
    lambda_cr_positive: tuple[float, ...]
    lambda_cr_negative: tuple[float, ...]
    reference_lambda_cr: float
    reference_length: float
    imperfection_mode: npt.NDArray[np.float64]
    relative_drop: tuple[float, ...]
    normalized_gradient: float
    secant_gradient: float
    gradient_amplitudes: tuple[float, ...]
    gradient_in_code_band: bool
    verdict_amplitude: float
    relative_drop_at_verdict: float
    imperfection_sensitive: bool


def _perturbed_nodes(
    nodes: Sequence[Node],
    mode: npt.NDArray[np.float64],
    amplitude: float,
    reference_length: float,
) -> list[Node]:
    """Copy ``nodes`` with the *free* coordinates displaced along ``mode``.

    A geometric imperfection is a deviation of the built structure from its
    nominal geometry.  A support is not part of that deviation: moving a
    restrained node moves the ground, and the resulting analysis would be
    solving a **support-settlement** problem while reporting it as an
    imperfection-sensitivity study -- two different physics with two
    different demand paths.

    Modes returned by :func:`linearized_buckling_load_factor` already have
    exactly-zero entries on the restrained DOFs, so for the default
    ``mode=None`` this mask is a no-op and costs nothing.  It matters for a
    caller-supplied ``mode=``, which is an arbitrary vector: the restrained
    components are zeroed here rather than consumed as settlements.
    :func:`imperfection_sensitivity` checks the masked-out magnitude once, up
    front, and warns if it was not negligible.
    """
    disp = np.asarray(mode, dtype=float).reshape(-1, 2) * (amplitude * reference_length)
    restrained = set(fixed_dof_indices(list(nodes)))
    if restrained:
        mask = np.array(
            [dof not in restrained for dof in range(disp.shape[0] * 2)], dtype=bool
        ).reshape(-1, 2)
        disp = np.where(mask, disp, 0.0)
    return [
        Node(
            id=nd.id,
            x=float(nd.x + disp[i, 0]),
            y=float(nd.y + disp[i, 1]),
            is_support=nd.is_support,
            support_dx=nd.support_dx,
            support_dy=nd.support_dy,
        )
        for i, nd in enumerate(nodes)
    ]


def imperfection_sensitivity(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    temps: Mapping[str, float] | None = None,
    *,
    amplitudes: Sequence[float] = DEFAULT_IMPERFECTION_AMPLITUDES,
    reference_length: float | None = None,
    mode: npt.NDArray[np.float64] | None = None,
    n_modes: int = 1,
    eigen_solver: EigenSolver = "auto",
) -> ImperfectionStudy:
    """Measure how a geometric imperfection erodes ``lambda_cr`` (B4).

    The perfect geometry is solved once; its critical mode is then imposed on
    the *coordinates* at a series of amplitudes and the bifurcation analysis
    repeated on each imperfect geometry:

    .. code-block:: text

        x_i <- x_i + eps * L_ref * phi_i

    A bifurcation load that barely moves with ``eps`` belongs to a system
    whose post-critical reserve is real; one that collapses with ``eps``
    belongs to a system where the perfect-geometry factor is an upper bound
    that the built structure will not achieve.  That distinction is exactly
    what a linearised analysis cannot make on its own.

    Parameters
    ----------
    nodes, elements : Sequence[Node], Sequence[Element]
        Model.
    loads : Mapping[str, Mapping[str, float]]
        Mechanical nodal loads [N], the pattern ``lambda`` amplifies.
    temps : Mapping[str, float] or None, optional
        Member steel temperatures [degC].
    amplitudes : Sequence[float], optional
        Imperfection amplitudes as fractions of ``reference_length``.  Must
        be positive and non-degenerate; the default series spans
        :data:`DEFAULT_IMPERFECTION_AMPLITUDES`.
    reference_length : float or None, optional
        Length [m] the amplitudes are fractions of.  ``None`` uses the
        largest bounding-box extent of the model, which makes ``eps = 0.01``
        mean "1% of the structure's overall size".
    mode : numpy.ndarray or None, optional
        Imperfection shape on the full DOF vector ``(2n,)``.  ``None`` uses
        the perfect geometry's critical mode -- the worst-case direction for
        a symmetric bifurcation.
    n_modes : int, default 1
        Modes requested per imperfect solve (only the critical one is used).
    eigen_solver : {"auto", "dense", "sparse"}, default "auto"
        Passed through to :func:`linearized_buckling_load_factor`.

    Returns
    -------
    ImperfectionStudy
        The sweep, its first-order gradient and the sensitivity verdict.

    Raises
    ------
    ValueError
        If ``amplitudes`` is empty or contains a non-positive value, or if
        ``reference_length`` is not positive.
    MechanismError
        If the perfect geometry -- or an imperfect one -- is already at or
        past a critical point.

    Notes
    -----
    Very large amplitudes can make two nodes coincide or invert a member;
    that surfaces as an
    :class:`~truss_analysis.exceptions.AssemblyError` from the geometry
    check rather than being silently clamped.  Keep ``eps * L_ref`` well
    below the smallest member length.
    """
    amps = [float(a) for a in amplitudes]
    if not amps:
        msg = "imperfection_sensitivity: amplitudes must not be empty"
        raise ValueError(msg)
    if any(a <= 0.0 for a in amps):
        msg = f"imperfection amplitudes must be positive, got {amps}"
        raise ValueError(msg)

    coords = np.array([[n.x, n.y] for n in nodes], dtype=float).reshape(-1, 2)
    extents = np.ptp(coords, axis=0) if coords.size else np.zeros(2)
    default_length = float(np.max(extents)) if extents.size else 0.0
    ref_length = default_length if reference_length is None else float(reference_length)
    if ref_length <= 0.0:
        msg = (
            "imperfection_sensitivity: reference_length must be positive "
            f"(model extent is {default_length:g}); pass it explicitly"
        )
        raise ValueError(msg)

    base = linearized_buckling_load_factor(
        nodes,
        elements,
        loads,
        temps,
        n_modes=n_modes,
        eigen_solver=eigen_solver,
        warn_shallow=False,
    )
    shape = base.mode if mode is None else np.asarray(mode, dtype=float)
    if shape.shape[0] != 2 * len(nodes):
        msg = (
            f"imperfection mode must have {2 * len(nodes)} entries, got "
            f"{shape.shape[0]}"
        )
        raise ValueError(msg)
    shape_norm = float(np.linalg.norm(shape))
    if shape_norm <= 0.0:
        msg = (
            "imperfection_sensitivity: the imperfection shape is identically "
            "zero. The perfect geometry has no bifurcation mode to imperil "
            "(lambda_cr = inf, e.g. an all-tension load case); pass an "
            "explicit `mode=` to define the imperfection direction."
        )
        raise ValueError(msg)
    shape = shape / shape_norm

    # A caller-supplied imperfection shape is an arbitrary vector: unlike a
    # mode returned by the solver it may carry components on restrained DOFs.
    # :func:`_perturbed_nodes` masks those out, because moving a support is a
    # settlement analysis rather than an imperfection study -- but masking
    # input the user believes controls the physics is exactly what
    # :class:`~truss_analysis.exceptions.InputIgnoredWarning` exists to
    # report, so say it once here rather than 2 * len(amplitudes) times inside
    # the sweep.
    if mode is not None:
        restrained = np.array(sorted(fixed_dof_indices(list(nodes))), dtype=int)
        if restrained.size:
            masked_norm = float(np.linalg.norm(shape[restrained]))
            if masked_norm > 1e-12:
                warnings.warn(
                    "imperfection_sensitivity: the supplied mode carries "
                    f"{masked_norm:.3e} of its unit norm on restrained DOFs "
                    f"{restrained.tolist()}; those components were zeroed. "
                    "Perturbing a support coordinate is a support-settlement "
                    "analysis, not a geometric imperfection, and the two have "
                    "different demand paths.",
                    InputIgnoredWarning,
                    stacklevel=2,
                )

    lam0 = base.lambda_cr
    lam_plus: list[float] = []
    lam_minus: list[float] = []
    for eps in amps:
        signs: list[float] = []
        for sign in (1.0, -1.0):
            imperfect = _perturbed_nodes(nodes, sign * shape, eps, ref_length)
            res = linearized_buckling_load_factor(
                imperfect,
                elements,
                loads,
                temps,
                n_modes=n_modes,
                eigen_solver=eigen_solver,
                warn_shallow=False,
            )
            signs.append(res.lambda_cr)
        lam_plus.append(signs[0])
        lam_minus.append(signs[1])
    # adverse of the two signs; inf loses to any finite value by construction
    lambdas = [min(a, b) for a, b in zip(lam_plus, lam_minus, strict=True)]

    amp_arr = np.asarray(amps, dtype=float)
    finite_ref = np.isfinite(lam0)
    if finite_ref:
        drops = tuple(
            float(1.0 - lam / lam0) if np.isfinite(lam) else float("nan")
            for lam in lambdas
        )
        ratios = np.array(
            [lam / lam0 if np.isfinite(lam) else np.nan for lam in lambdas], dtype=float
        )
        usable = np.isfinite(ratios)

        # The derivative at zero, fitted only where the code band reaches.
        gradient, fitted, in_band = _local_gradient(amp_arr, ratios)
        if not in_band and usable.any():
            warnings.warn(
                "imperfection_sensitivity: none of the probed amplitudes "
                f"{tuple(round(float(a), 6) for a in amp_arr)} lies inside the "
                "EN 1993-1-1 Table 5.1 out-of-straightness band "
                f"[{min(CODE_IMPERFECTION_AMPLITUDES):.5f}, "
                f"{max(CODE_IMPERFECTION_AMPLITUDES):.5f}]; the reported "
                "first-order gradient is extrapolated from amplitudes larger "
                "than any a built structure would carry. Add a smaller "
                "amplitude to make it a local measurement.",
                InputIgnoredWarning,
                stacklevel=2,
            )

        # The secant over the whole sweep, kept and named as what it is.
        if int(np.count_nonzero(usable)) >= 2:
            secant = float(np.polyfit(amp_arr[usable], ratios[usable], 1)[0])
        else:
            secant = float("nan")

        # The verdict comes from the largest *code-like* amplitude probed, so
        # an exploratory tail cannot decide a safety flag on its own.
        band_ceiling = max(CODE_IMPERFECTION_AMPLITUDES)
        in_band_mask = usable & (amp_arr <= band_ceiling)
        if bool(np.any(in_band_mask)):
            verdict_amp = float(np.max(amp_arr[in_band_mask]))
        else:
            verdict_amp = (
                float(np.max(amp_arr[usable])) if bool(usable.any()) else float("nan")
            )
        drop_at_verdict = float("nan")
        for a, d in zip(amp_arr, drops, strict=True):
            if float(a) == verdict_amp:
                drop_at_verdict = float(d)
                break
        sensitive = bool(
            np.isfinite(drop_at_verdict)
            and drop_at_verdict > IMPERFECTION_SENSITIVE_DROP
        )
    else:
        drops = tuple(float("nan") for _ in lambdas)
        gradient = float("nan")
        fitted = ()
        in_band = True
        secant = float("nan")
        verdict_amp = float("nan")
        drop_at_verdict = float("nan")
        sensitive = False

    return ImperfectionStudy(
        amplitudes=tuple(amps),
        lambda_cr=tuple(float(x) for x in lambdas),
        lambda_cr_positive=tuple(float(x) for x in lam_plus),
        lambda_cr_negative=tuple(float(x) for x in lam_minus),
        reference_lambda_cr=float(lam0),
        reference_length=ref_length,
        imperfection_mode=np.asarray(shape, dtype=float),
        relative_drop=drops,
        normalized_gradient=gradient,
        secant_gradient=secant,
        gradient_amplitudes=fitted,
        gradient_in_code_band=bool(in_band),
        verdict_amplitude=verdict_amp,
        relative_drop_at_verdict=drop_at_verdict,
        imperfection_sensitive=sensitive,
    )
