"""Assembly: global stiffness matrix and force vectors.

The truss stiffness matrix has the exact member-wise outer-product form

.. code-block:: text

    K = sum_e k_e * b_e b_e^T,     k_e = E_e A_e / L_e
    b_e = [-c_e, -s_e, c_e, s_e]

so every element contributes a rank-1 block spread over its four DOFs. That
identity is what makes a fully vectorised COO assembly possible: the row
indices, column indices and values for the whole model are three numpy arrays
built without a single Python-level loop over the 4x4 block, and
``coo_matrix(...).tocsr()`` performs the duplicate-index summation in C.

Two storage layouts are offered from one formulation:

``sparse=False`` (default)
    Dense ``(2n, 2n)`` array. Best for the small and medium models this
    library targets, and what every existing caller receives.

``sparse=True``
    ``scipy.sparse.csr_matrix``. Memory drops from ``O(n^2)`` to ``O(nnz)``
    with ``nnz ~ 9n`` for a truss, and :func:`truss_analysis.solver.solve`
    hands it to SuperLU. A 20 000-node model needs ~13 GB as a dense ``K``
    and a few tens of megabytes as CSR.

Geometry (lengths, direction cosines, axial stiffness, DOF indices) is
computed once by :func:`member_geometry` and shared by both paths, so the two
layouts cannot drift apart — they are the same numbers in two containers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .exceptions import AssemblyError
from .model import Element, Node, fixed_dof_indices
from .numerics import DEFAULT_TOLERANCES

__all__ = [
    "MemberGeometry",
    "assemble_global_matrices",
    "assemble_global_matrices_sparse",
    "member_geometry",
]


@dataclass(frozen=True)
class MemberGeometry:
    """Per-member geometric and stiffness quantities, computed once.

    Lengths and direction cosines were previously recomputed independently in
    the assembler, the post-processor, the sensitivity validator, the
    limit-state module and the degradation module. Computing them once and
    passing the result around removes that duplication and the risk of two
    modules disagreeing about a member's length by a round-off.

    Attributes
    ----------
    lengths : np.ndarray
        Member lengths ``L`` [m], shape ``(m,)``.
    cosines : np.ndarray
        Direction cosines ``c = dx / L``, shape ``(m,)``.
    sines : np.ndarray
        Direction sines ``s = dy / L``, shape ``(m,)``.
    k_axial : np.ndarray
        Axial stiffness ``k = E A / L`` [N/m], shape ``(m,)``.
    dofs : np.ndarray
        Integer global DOF indices per member, shape ``(m, 4)``, ordered
        ``[u_i, v_i, u_j, v_j]``.
    node_i_idx : np.ndarray
        Positional index of each member's start node, shape ``(m,)``.
    node_j_idx : np.ndarray
        Positional index of each member's end node, shape ``(m,)``.
    """

    lengths: np.ndarray
    cosines: np.ndarray
    sines: np.ndarray
    k_axial: np.ndarray
    dofs: np.ndarray
    node_i_idx: np.ndarray
    node_j_idx: np.ndarray

    @property
    def n_members(self) -> int:
        """Number of members described."""
        return int(self.lengths.shape[0])

    def compatibility(self) -> np.ndarray:
        """Return the compatibility matrix ``B``, shape ``(m, n_dof)``.

        Row ``e`` holds ``b_e``, the member elongation operator: applying it to
        the global displacement vector gives the total elongation of member
        ``e``. The stiffness matrix is exactly ``B^T diag(k) B``, which is the
        identity used by the assembly tests to verify this module
        independently of its own implementation.
        """
        n_dof = int(self.dofs.max()) + 1 if self.n_members else 0
        b = np.zeros((self.n_members, n_dof))
        signs = np.array([-1.0, -1.0, 1.0, 1.0])
        for e in range(self.n_members):
            b[e, self.dofs[e, 0]] = signs[0] * self.cosines[e]
            b[e, self.dofs[e, 1]] = signs[1] * self.sines[e]
            b[e, self.dofs[e, 2]] = signs[2] * self.cosines[e]
            b[e, self.dofs[e, 3]] = signs[3] * self.sines[e]
        return b

    def element_blocks(self) -> np.ndarray:
        """Return every member's ``4x4`` stiffness block, shape ``(m, 4, 4)``.

        Built as the rank-1 outer product ``k_e * b_e b_e^T`` restricted to the
        member's own four DOFs, which is algebraically identical to the
        classical ``c^2 / cs / s^2`` layout but computed without a loop.
        """
        c = self.cosines[:, None]
        s = self.sines[:, None]
        # Local 4-vector [-c, -s, c, s] per member, shape (m, 4).
        b = np.concatenate([-c, -s, c, s], axis=1)
        blocks = self.k_axial[:, None, None] * b[:, :, None] * b[:, None, :]
        return np.asarray(blocks, dtype=float)


def member_geometry(
    nodes: list[Node] | tuple[Node, ...],
    elements: list[Element] | tuple[Element, ...],
) -> MemberGeometry:
    """Compute lengths, direction cosines, stiffness and DOF maps for a model.

    Parameters
    ----------
    nodes : list[Node]
        Model nodes; node ordering defines the global DOF map
        (node ``i`` owns DOFs ``2i`` and ``2i+1``).
    elements : list[Element]
        Model elements.

    Returns
    -------
    MemberGeometry
        Frozen container of the shared geometric quantities.

    Raises
    ------
    AssemblyError
        If an element references a missing node, or has a length below
        ``DEFAULT_TOLERANCES.length_tol`` (where ``k = EA/L`` diverges and the
        direction cosines are undefined).
    """
    node_map: dict[str, int] = {node.id: i for i, node in enumerate(nodes)}
    coords = np.array([[node.x, node.y] for node in nodes], dtype=float).reshape(-1, 2)

    m = len(elements)
    i_idx = np.empty(m, dtype=int)
    j_idx = np.empty(m, dtype=int)
    for e_i, elem in enumerate(elements):
        if elem.node_i not in node_map or elem.node_j not in node_map:
            raise AssemblyError(f"Element {elem.id} references non-existent nodes")
        i_idx[e_i] = node_map[elem.node_i]
        j_idx[e_i] = node_map[elem.node_j]

    if m == 0:
        empty = np.empty(0, dtype=float)
        return MemberGeometry(
            lengths=empty,
            cosines=empty,
            sines=empty,
            k_axial=empty,
            dofs=np.empty((0, 4), dtype=int),
            node_i_idx=i_idx,
            node_j_idx=j_idx,
        )

    delta = coords[j_idx] - coords[i_idx]
    lengths = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)

    tol = DEFAULT_TOLERANCES.length_tol
    bad = np.nonzero(lengths < tol)[0]
    if bad.size:
        raise AssemblyError(
            f"Element {elements[int(bad[0])].id} has zero or negative length"
        )

    cosines = delta[:, 0] / lengths
    sines = delta[:, 1] / lengths

    e_arr = np.array([elem.E for elem in elements], dtype=float)
    a_arr = np.array([elem.A for elem in elements], dtype=float)
    k_axial = e_arr * a_arr / lengths

    dofs = np.empty((m, 4), dtype=int)
    dofs[:, 0] = 2 * i_idx
    dofs[:, 1] = 2 * i_idx + 1
    dofs[:, 2] = 2 * j_idx
    dofs[:, 3] = 2 * j_idx + 1

    return MemberGeometry(
        lengths=lengths,
        cosines=cosines,
        sines=sines,
        k_axial=k_axial,
        dofs=dofs,
        node_i_idx=i_idx,
        node_j_idx=j_idx,
    )


def _imposed_nodal_forces(
    elements: list[Element] | tuple[Element, ...],
    geom: MemberGeometry,
    n_dof: int,
) -> np.ndarray:
    """Equivalent nodal force vector from imposed (thermal/fabrication) strain.

    A member with imposed elongation ``delta_L_prestress`` and no nodal
    displacement exerts a self-equilibrated pair of axial forces
    ``k * delta_L_prestress`` on its end nodes. Those forces belong in the
    *total* right-hand side but not in the mechanical one, which is why the
    assembler keeps ``F_ext`` and ``F_mechanical`` separate: the energy check
    needs the mechanical part alone to apply the Clapeyron identity.
    """
    F_imposed = np.zeros(n_dof)
    if geom.n_members == 0:
        return F_imposed

    alpha = np.array([elem.alpha for elem in elements], dtype=float)
    delta_T = np.array([elem.delta_T for elem in elements], dtype=float)
    delta_L_free = np.array([elem.delta_L_free for elem in elements], dtype=float)

    delta_L_prestress = alpha * delta_T * geom.lengths + delta_L_free
    # Members with no imposed strain contribute nothing; skipping them keeps
    # the common purely mechanical case free of avoidable work.
    active = delta_L_prestress != 0.0
    if not np.any(active):
        return F_imposed

    f_axial = geom.k_axial * delta_L_prestress
    idx = np.nonzero(active)[0]
    for e in idx:
        i = int(geom.node_i_idx[e])
        j = int(geom.node_j_idx[e])
        fx = f_axial[e] * geom.cosines[e]
        fy = f_axial[e] * geom.sines[e]
        F_imposed[2 * i] -= fx
        F_imposed[2 * i + 1] -= fy
        F_imposed[2 * j] += fx
        F_imposed[2 * j + 1] += fy
    return F_imposed


def assemble_global_matrices_sparse(
    nodes: list[Node],
    elements: list[Element],
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, list[int]]:
    """Assemble the global system in CSR sparse storage.

    Identical physics and identical numbers to
    :func:`assemble_global_matrices`, in a layout whose memory cost is
    ``O(nnz)`` instead of ``O(n^2)``. The stiffness blocks are formed as
    rank-1 outer products and summed by ``coo_matrix`` in C, so there is no
    Python-level loop over the 4x4 entries.

    Parameters
    ----------
    nodes : list[Node]
        Model nodes; node ordering defines the global DOF map.
    elements : list[Element]
        Model elements, including imposed-elongation fields.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray, list[int]]
        ``(K, F_ext, F_mechanical, fixed_dofs)`` with ``K`` of shape
        ``(2n, 2n)`` in CSR format.

    Raises
    ------
    AssemblyError
        If an element references a missing node or has zero length.
    """
    n = len(nodes)
    n_dof = 2 * n
    geom = member_geometry(nodes, elements)

    if geom.n_members:
        blocks = geom.element_blocks()
        dofs = geom.dofs
        # Spread the (m, 4, 4) blocks into flat COO triplets: for block entry
        # (ii, jj) the row is dofs[ii] and the column is dofs[jj].
        rows = np.broadcast_to(dofs[:, :, None], (geom.n_members, 4, 4)).ravel()
        cols = np.broadcast_to(dofs[:, None, :], (geom.n_members, 4, 4)).ravel()
        vals = blocks.ravel()
        K: sp.csr_matrix = sp.coo_matrix(
            (vals, (rows, cols)), shape=(n_dof, n_dof)
        ).tocsr()
    else:
        K = sp.csr_matrix((n_dof, n_dof))

    F_imposed = _imposed_nodal_forces(elements, geom, n_dof)
    F_ext = F_imposed
    F_mechanical = np.zeros(n_dof)
    fixed_dofs = fixed_dof_indices(nodes)
    return K, F_ext, F_mechanical, fixed_dofs


def assemble_global_matrices(
    nodes: list[Node],
    elements: list[Element],
    *,
    sparse: bool = False,
) -> tuple[np.ndarray | sp.csr_matrix, np.ndarray, np.ndarray, list[int]]:
    """Assemble the global stiffness matrix and force vectors.

    Parameters
    ----------
    nodes : list[Node]
        Model nodes; node ordering defines the global DOF map
        (node ``i`` owns DOFs ``2i`` and ``2i+1``).
    elements : list[Element]
        Model elements, including thermal (``alpha``, ``delta_T``) and
        fabrication (``delta_L_free``) imposed-elongation fields.
    sparse : bool, default False
        Return ``K`` as a :class:`scipy.sparse.csr_matrix` instead of a dense
        array. Delegates to :func:`assemble_global_matrices_sparse`; the force
        vectors and DOF list are dense either way.

    Returns
    -------
    tuple[np.ndarray or scipy.sparse.csr_matrix, np.ndarray, np.ndarray, list[int]]
        ``K`` of shape ``(2n, 2n)``, ``F_ext`` (equivalent nodal forces from
        imposed elongations; mechanical loads are added by the caller),
        ``F_mechanical`` (zero on entry) and ``fixed_dofs``.

    Raises
    ------
    AssemblyError
        If an element references a missing node or has zero length.
    """
    if sparse:
        return assemble_global_matrices_sparse(nodes, elements)

    n = len(nodes)
    n_dof = 2 * n
    geom = member_geometry(nodes, elements)

    K = np.zeros((n_dof, n_dof))
    # One vectorised 4x4 scatter per member replaces the previous nested
    # 16-iteration Python loop over the same block.
    blocks = geom.element_blocks()
    for e in range(geom.n_members):
        K[np.ix_(geom.dofs[e], geom.dofs[e])] += blocks[e]

    F_ext = _imposed_nodal_forces(elements, geom, n_dof)
    F_mechanical = np.zeros(n_dof)

    # Boundary conditions. The node -> constrained-DOF map is defined once, in
    # model.fixed_dof_indices, and shared with the criticality engine so the
    # two numerical paths can never disagree about the supports.
    fixed_dofs = fixed_dof_indices(nodes)

    return K, F_ext, F_mechanical, fixed_dofs
