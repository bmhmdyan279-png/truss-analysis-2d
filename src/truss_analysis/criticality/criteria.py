"""Multi-criteria criticality indices.

The single scalar reported by :func:`truss_analysis.criticality.engine.ci_sweep`
is

.. code-block:: text

    CI_i = max|U_pert[:, i]| / max|u| - 1

that is, the relative change in the **largest nodal displacement magnitude**
when member ``i`` loses stiffness. It is cheap, well defined and exactly
computable through the rank-1 update, which is why it is the library's default.

It is also a *scalar summary of a vector field*, and summaries lose information.
Two members can be equally damaging in quite different ways:

.. code-block:: text

    member A: max|U| rises 20%, forces barely redistribute
    member B: max|U| rises  2%, one neighbouring member's force rises 80%,
              total strain energy rises 30%

``CI_u`` ranks A far above B, even though B is the member whose loss drives
another member past its capacity. For a fire-triage or retrofit study that is
the wrong answer, because what fails a structure is usually a *force* limit
state rather than a displacement one.

Perturbed forces need the perturbed stiffness
---------------------------------------------
Softening member ``i`` changes the stiffness that member's own force is
computed from. Writing ``e_j = b_j . u_pert[:, i] - dL_pre,j`` for the
*mechanical* elongation of member ``j`` in the perturbed state (total
elongation minus the imposed thermal/fabrication part — only the mechanical
one carries force),

.. code-block:: text

    N_pert[j, i] = k_j          * e_j      for j != i
    N_pert[i, i] = alpha * k_i  * e_i      for the perturbed member itself

Using the unperturbed ``k_i`` for the perturbed member is wrong by a factor of
``1/alpha``: because ``N = alpha k_i e_i`` holds, the elongation grows to
``N / (alpha k_i)``, so multiplying it back by ``k_i`` returns ``N / alpha``
rather than ``N``. At ``alpha = 0.5`` that is exactly a factor of two, and it
shows up as a spurious doubling of the force index even in a statically
*determinate* truss, where softening a member cannot change any force at all.
That determinate case is asserted in the test suite precisely because it
catches this error.

``CI_E`` is the change in **mechanical strain energy**
``0.5 sum_e k_e (b_e . u - dL_pre,e)^2`` — the energy stored in elastic
elongation, not the total thermomechanical potential; imposed strain that a
member sheds into the structure is not "absorbed" by it.

Indices
-------
===============  ====================================================
Index            Definition
===============  ====================================================
``CI_u``         ``max|U_pert| / max|u| - 1`` (the existing measure)
``CI_N_max``     ``max|N_pert| / max|N_base| - 1`` over all members
``CI_N_self``    ``|N_pert[i, i]| / |N_base[i]| - 1`` for member ``i`` itself
``CI_E``         ``E_pert / E_base - 1`` for the whole structure
``CI_R``         ``max|R_pert| / max|R_base| - 1`` over constrained DOFs
===============  ====================================================

``CI_N_self`` deserves a note: perturbing member ``i`` changes its own force
too, and for a redundant structure that change measures how much load the
member was actually carrying. A member with ``CI_N_self ~ 0`` is one whose
force is insensitive to its own stiffness, i.e. a member the structure does
not really need.

Reactions under a rank-1 perturbation
-------------------------------------
``R = (K U)[fixed] - F_ext[fixed]`` — the supports carry whatever the
stiffness pulls minus what is applied directly at the constrained DOFs
(mechanical loads placed on support nodes and the imposed thermal/fabrication
equivalent forces; :class:`ReactionInfluence` carries that vector as
``f_ext_fixed``). The perturbed stiffness is ``K + Delta_i b_i b_i^T`` with
``Delta_i = (alpha - 1) k_i`` and the perturbed member's thermal force scales
with it, so

.. code-block:: text

    R_pert[:, i] = K[fixed, free] u_pert[:, i] - F_ext[fixed]
                 + Delta_i * b_i[fixed] * (b_i[free] . u_pert[:, i] - dL_pre,i)

The second line is the perturbed member's own contribution to the supports and
must not be dropped; :class:`ReactionInfluence` carries ``b_i[fixed]`` for
exactly that purpose.

No approximation is introduced anywhere above. Every index is an exact
function of the same exact rank-1 perturbed field the existing CI uses, so the
validity limit and mechanism guard of
:mod:`truss_analysis.criticality.engine` carry over unchanged: these indices
are exact for **single-member** perturbations only.

Memory
------
Forming ``N_pert`` materialises an ``nE x nE`` matrix (one column of member
forces per perturbed member). At 1 000 members that is 8 MB; at 10 000 it is
800 MB. :func:`multi_criteria_ci` therefore evaluates in blocks of
:data:`FORCE_BLOCK_SIZE` members and never holds the whole matrix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..model import Element, Node, fixed_dof_indices
from .engine import CiSweep, EngineSetup, free_dof_indices, member_matrices

__all__ = [
    "FORCE_BLOCK_SIZE",
    "MultiCriteriaResult",
    "ReactionInfluence",
    "multi_criteria_ci",
    "reaction_influence",
]

#: Members evaluated per block when forming the perturbed force field. Chosen
#: so an ``nE x FORCE_BLOCK_SIZE`` working matrix stays around 8 MB.
FORCE_BLOCK_SIZE = 512


@dataclass(frozen=True)
class ReactionInfluence:
    """Everything needed to turn a perturbed displacement field into reactions.

    Attributes
    ----------
    k_fixed_free : np.ndarray
        ``K[fixed, free]``, shape ``(n_fixed, ndof_free)``. Maps free
        displacements to support reactions for the *unperturbed* structure.
    b_fixed : np.ndarray
        Member compatibility vectors restricted to the constrained DOFs,
        shape ``(nE, n_fixed)``. Needed for the rank-1 correction term.
    k_axial : np.ndarray
        Member axial stiffnesses, shape ``(nE,)``.
    fixed_dofs : tuple[int, ...]
        The constrained global DOF indices, in ascending order, so a caller can
        map a row of ``k_fixed_free`` back to a node and direction.
    f_ext_fixed : np.ndarray or None
        External force already applied *directly* on the constrained DOFs,
        shape ``(n_fixed,)``: mechanical loads placed on support nodes plus
        the imposed (thermal/fabrication) equivalent nodal forces. Reactions
        are the residual ``R = (K u)[fixed] - f_ext_fixed``; omitting this
        term is only correct when nothing loads the supports, which a heated
        redundant structure violates by construction. ``None`` (the default
        for callers that pass no loads/temps) is treated as zero.
    """

    k_fixed_free: np.ndarray
    b_fixed: np.ndarray
    k_axial: np.ndarray
    fixed_dofs: tuple[int, ...]
    f_ext_fixed: np.ndarray | None = None


@dataclass(frozen=True)
class MultiCriteriaResult:
    """All criticality indices for one member, from one perturbed field.

    Attributes
    ----------
    member_id : str
        Member the perturbation was applied to.
    ci_displacement : float
        ``CI_u``: relative change in ``max|U|``. Identical to the value
        :func:`~truss_analysis.criticality.engine.ci_sweep` reports, kept here
        so a caller can compare indices without joining two structures.
    ci_force_max : float
        ``CI_N_max``: relative change in the largest member force magnitude
        anywhere in the structure. This is the index that catches a member
        whose loss overloads a *neighbour* while barely moving the structure.
    ci_force_self : float
        ``CI_N_self``: relative change in the perturbed member's own force,
        compared by magnitude so a sign reversal is not read as a reduction.
    ci_energy : float
        ``CI_E``: relative change in total strain energy.
    ci_reaction : float
        ``CI_R``: relative change in the largest support reaction magnitude.
        ``nan`` when no reaction influence data was supplied.
    governing : str
        Name of the index with the largest value, i.e. the response quantity
        through which this member is most critical. ``"none"`` when no index is
        finite, ``"mechanism"`` when the member was guarded.
    composite : float
        Maximum of the five indices: a single conservative summary for ranking
        when the caller does not want to choose a response quantity. It never
        understates an individual effect, which is the right default for triage.
    flagged : str
        Mechanism/guard flag carried over from the sweep; empty when the member
        was handled by the exact rank-1 path.
    """

    member_id: str
    ci_displacement: float
    ci_force_max: float
    ci_force_self: float
    ci_energy: float
    ci_reaction: float
    governing: str
    composite: float
    flagged: str = ""

    def as_dict(self) -> dict[str, float | str]:
        """Return a JSON-friendly view of every index."""
        return {
            "member_id": self.member_id,
            "ci_displacement": self.ci_displacement,
            "ci_force_max": self.ci_force_max,
            "ci_force_self": self.ci_force_self,
            "ci_energy": self.ci_energy,
            "ci_reaction": self.ci_reaction,
            "governing": self.governing,
            "composite": self.composite,
            "flagged": self.flagged,
        }


_INDEX_NAMES = (
    ("ci_displacement", "displacement"),
    ("ci_force_max", "force_max"),
    ("ci_force_self", "force_self"),
    ("ci_energy", "energy"),
    ("ci_reaction", "reaction"),
)


def reaction_influence(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    k_scale: Mapping[str, float] | None = None,
    *,
    loads: Mapping[str, Mapping[str, float]] | None = None,
    temps: Mapping[str, float] | None = None,
) -> ReactionInfluence:
    """Build the constrained-DOF blocks needed for the reaction index.

    Assembled from the same rank-1 member dyads the engine uses,
    ``K = sum_e k_e b_e b_e^T``, so it cannot disagree with the stiffness the
    sweep was computed against.  ``K[fixed, free]`` is formed **directly** as
    ``(b[fixed] * k)^T @ b[free]`` — an ``O(m * n_fixed * n_free)`` product —
    instead of materialising the dense ``(2n, 2n)`` global matrix first and
    slicing it: the dense intermediate is exactly the ``O(n^2)`` memory wall
    the sparse assembly path exists to remove, and rebuilding it here would
    negate that for every CI sweep with reactions.

    Parameters
    ----------
    nodes : Sequence[Node]
        Model nodes.
    elements : Sequence[Element]
        Model elements.
    k_scale : Mapping[str, float] or None, optional
        Per-member stiffness multipliers (e.g. ``k_E(T)`` for a thermal state).
        Must match what was passed to
        :func:`~truss_analysis.criticality.engine.build_engine`, or the
        reactions will not correspond to the sweep.
    loads : Mapping or None, optional
        Nodal mechanical loads; the part applied on *support* nodes enters
        ``f_ext_fixed`` and must be subtracted from ``(K u)[fixed]`` to get
        the true reactions.
    temps : Mapping[str, float] or None, optional
        Member temperature field the sweep was built with; together with the
        element ``alpha`` / ``delta_L_free`` it defines the imposed equivalent
        nodal forces whose fixed-DOF part also enters ``f_ext_fixed``.

    Returns
    -------
    ReactionInfluence
        ``K[fixed, free]``, ``b[fixed]``, the axial stiffnesses, the
        constrained DOF indices and the direct fixed-DOF load.
    """
    from .engine import prestress_lengths

    b, k = member_matrices(nodes, elements, k_scale)
    fixed = fixed_dof_indices(list(nodes))
    free = list(free_dof_indices(nodes))
    n_members = len(elements)
    if not fixed:
        return ReactionInfluence(
            k_fixed_free=np.zeros((0, len(free))),
            b_fixed=np.zeros((n_members, 0)),
            k_axial=k,
            fixed_dofs=(),
            f_ext_fixed=np.zeros(0),
        )
    b_fixed = b[:, fixed]  # (nE, n_fixed)
    b_free = b[:, free]  # (nE, n_free)
    # K[fixed, free] = sum_e k_e b_e[fixed] b_e[free]^T — no dense (2n, 2n).
    k_fixed_free = (b_fixed * k[:, None]).T @ b_free

    f_ext = np.zeros(2 * len(nodes))
    if loads:
        node_idx = {n.id: i for i, n in enumerate(nodes)}
        for node_id, load in loads.items():
            i = node_idx[str(node_id)]
            f_ext[2 * i] += float(load.get("Fx", 0.0))
            f_ext[2 * i + 1] += float(load.get("Fy", 0.0))
    dl_pre = prestress_lengths(nodes, elements, temps)
    imposed_fixed = b_fixed.T @ (k * dl_pre)
    return ReactionInfluence(
        k_fixed_free=k_fixed_free,
        b_fixed=b_fixed,
        k_axial=k,
        fixed_dofs=tuple(fixed),
        f_ext_fixed=f_ext[fixed] + imposed_fixed,
    )


def multi_criteria_ci(
    setup: EngineSetup,
    u_base: np.ndarray,
    sweep: CiSweep,
    alpha: float,
    reactions: ReactionInfluence | None = None,
) -> dict[str, MultiCriteriaResult]:
    """Derive displacement, force, energy and reaction indices from one sweep.

    Parameters
    ----------
    setup : EngineSetup
        Factorised base state from
        :func:`~truss_analysis.criticality.engine.build_engine`; supplies
        ``b_free``, ``k_axial`` and the member ids.
    u_base : np.ndarray
        Base displacement on the free DOFs, shape ``(ndof_free,)``.
    sweep : CiSweep
        Perturbed displacement field from
        :func:`~truss_analysis.criticality.engine.ci_sweep`. Columns marked
        ``+inf`` (mechanism) propagate ``+inf`` through every index rather
        than being silently dropped.
    alpha : float
        Stiffness retention factor the sweep was run at. Required so the
        perturbed member's own force and energy use ``alpha * k_i`` rather
        than ``k_i``; see the module docstring.
    reactions : ReactionInfluence or None, optional
        From :func:`reaction_influence`, needed for ``CI_R``. ``CI_R`` is
        ``nan`` when omitted, so callers that do not need it pay nothing.

    Returns
    -------
    dict[str, MultiCriteriaResult]
        One result per member, keyed by member id, in element order.
    """
    ids = setup.ids
    n_members = len(ids)
    u_pert = sweep.u_pert
    k_axial = setup.k_axial

    # ---- base quantities, computed once for the whole sweep ---------------
    # MECHANICAL elongation: total elongation minus the imposed (thermal /
    # fabrication) part -- only the mechanical part carries force and stores
    # strain energy, matching engine.member_forces and the assembler.
    elong_base = setup.b_free @ u_base - setup.dl_pre  # (nE,)
    n_base = k_axial * elong_base
    n_base_max = float(np.max(np.abs(n_base))) if n_members else 0.0
    energy_base = float(0.5 * np.sum(k_axial * elong_base**2))
    u_base_max = float(np.max(np.abs(u_base))) if u_base.size else 0.0

    want_reaction = reactions is not None and reactions.k_fixed_free.size > 0
    if want_reaction and reactions is not None:
        f_ext_fixed = (
            reactions.f_ext_fixed
            if reactions.f_ext_fixed is not None
            else np.zeros(reactions.k_fixed_free.shape[0])
        )
        r_base = reactions.k_fixed_free @ u_base - f_ext_fixed
        reaction_base_max = float(np.max(np.abs(r_base)))
    else:
        f_ext_fixed = np.zeros(0)
        reaction_base_max = 0.0

    results: dict[str, MultiCriteriaResult] = {}
    diagonal = np.arange(n_members)

    for start in range(0, n_members, FORCE_BLOCK_SIZE):
        stop = min(start + FORCE_BLOCK_SIZE, n_members)
        width = stop - start
        block = u_pert[:, start:stop]  # (ndof_free, width)
        finite_cols = np.isfinite(block).all(axis=0)
        # Replace the +inf mechanism columns so the matrix products stay
        # finite; those columns are overwritten below and never reported.
        safe_block = np.where(np.isfinite(block), block, 0.0)

        elong = setup.b_free @ safe_block - setup.dl_pre[:, None]  # (nE, width)

        # Per-column stiffness: identical to k_axial except in the diagonal
        # slot, where the perturbed member carries alpha * k_i.
        k_pert = np.repeat(k_axial[:, None], width, axis=1)
        k_pert[diagonal[start:stop], np.arange(width)] *= alpha

        n_pert = k_pert * elong  # (nE, width)
        force_max_col = np.max(np.abs(n_pert), axis=0)
        energy_col = 0.5 * np.sum(k_pert * elong**2, axis=0)
        force_self_col = np.abs(n_pert[diagonal[start:stop], np.arange(width)])
        disp_max_col = np.max(np.abs(safe_block), axis=0)

        if want_reaction and reactions is not None:
            # R_pert = K[fixed,free] u_pert - f_ext[fixed]
            #          + Delta_i * b_i[fixed] * (b_i[free] . u_pert - dL_pre,i)
            # (the rank-1 correction carries the MECHANICAL elongation of the
            # perturbed member, because its thermal equivalent force scales by
            # the same alpha as its stiffness -- see the engine docstring)
            r_pert = reactions.k_fixed_free @ safe_block - f_ext_fixed[:, None]
            delta_i = (alpha - 1.0) * k_axial[start:stop]  # (width,)
            b_fixed_block = reactions.b_fixed[start:stop]  # (width, n_fixed)
            self_elong = elong[diagonal[start:stop], np.arange(width)]  # (width,)
            r_pert = r_pert + (b_fixed_block.T * (delta_i * self_elong)[None, :])
            reaction_col = np.max(np.abs(r_pert), axis=0)
        else:
            reaction_col = None

        for offset, j in enumerate(range(start, stop)):
            eid = ids[j]
            flagged = sweep.flagged.get(eid, "")

            if not finite_cols[offset]:
                # Mechanism: every response quantity diverges. Reporting inf
                # rather than a finite number is the point of the guard.
                results[eid] = MultiCriteriaResult(
                    member_id=eid,
                    ci_displacement=float("inf"),
                    ci_force_max=float("inf"),
                    ci_force_self=float("inf"),
                    ci_energy=float("inf"),
                    ci_reaction=float("inf"),
                    governing="mechanism",
                    composite=float("inf"),
                    flagged=flagged or "mechanism(singular)",
                )
                continue

            ci_reaction = (
                _ratio(float(reaction_col[offset]), reaction_base_max)
                if reaction_col is not None
                else float("nan")
            )
            values = {
                "ci_displacement": _ratio(float(disp_max_col[offset]), u_base_max),
                "ci_force_max": _ratio(float(force_max_col[offset]), n_base_max),
                "ci_force_self": _ratio(
                    float(force_self_col[offset]), abs(float(n_base[j]))
                ),
                "ci_energy": _ratio(float(energy_col[offset]), energy_base),
                "ci_reaction": ci_reaction,
            }

            finite_values = [v for v in values.values() if np.isfinite(v)]
            composite = max(finite_values) if finite_values else float("nan")
            governing = "none"
            if finite_values:
                best = max(finite_values)
                for key, name in _INDEX_NAMES:
                    if values[key] == best:
                        governing = name
                        break

            results[eid] = MultiCriteriaResult(
                member_id=eid,
                ci_displacement=values["ci_displacement"],
                ci_force_max=values["ci_force_max"],
                ci_force_self=values["ci_force_self"],
                ci_energy=values["ci_energy"],
                ci_reaction=values["ci_reaction"],
                governing=governing,
                composite=composite,
                flagged=flagged,
            )

    return results


def _ratio(value: float, base: float) -> float:
    """Relative change ``value / base - 1``, with a zero base handled honestly."""
    if base == 0.0:
        # No base response to compare against: any non-zero perturbed value is
        # an unbounded relative change, and zero means nothing happened.
        return 0.0 if value == 0.0 else float("inf")
    return value / base - 1.0
