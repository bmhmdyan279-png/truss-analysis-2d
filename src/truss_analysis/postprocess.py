"""Post-processing: element forces, reactions, equilibrium and buckling."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from .exceptions import BucklingCheckWarning, LargeDisplacementWarning, ShallowSystemWarning
from .model import Element, Node
from .sections import euler_buckling_load

#: Geometric zero-length threshold [m]. Members shorter than this cannot
#: carry a meaningful axial stiffness (``k = EA/L`` diverges) and are skipped
#: rather than divided by.
_ZERO_LENGTH_TOL = 1e-12

#: Relative band used to classify an axial force as ``"Zero"``. The force is
#: compared against the member's own yield-scale reference ``E*A`` so the
#: classification is invariant to the unit system and to model size; an
#: absolute newton threshold would call a 1e-6 N force "Tension" in a
#: kilonewton model and a 1e3 N force "Zero" in a giganewton one.
_FORCE_ZERO_REL = 1e-12


def calculate_element_forces(
    nodes: list[Node],
    elements: list[Element],
    U: np.ndarray,
) -> tuple[list[dict[str, Any]], float, float]:
    """Compute axial forces, strain energy and prestress work per element.

    The mechanical elongation is the total elongation from nodal
    displacements minus the imposed (thermal + fabrication) elongation;
    only the mechanical part produces axial force.

    Parameters
    ----------
    nodes : list[Node]
        Model nodes (ordering defines the DOF map).
    elements : list[Element]
        Model elements.
    U : np.ndarray
        Global displacement vector, shape ``(2n,)``.

    Returns
    -------
    tuple[list[dict[str, Any]], float, float]
        ``(results, strain_energy, prestress_work)`` where ``results`` holds
        one dict per element with keys ``id``, ``N``, ``delta_L_mech``,
        ``delta_L_prestress`` and ``status`` (``"Tension"``,
        ``"Compression"``, ``"Zero"`` or ``"ZERO_LENGTH"``).

    Notes
    -----
    Per-element quantities:

    - ``delta_L_total``: elongation from nodal displacements
    - ``delta_L_thermal = alpha * delta_T * L``
    - ``delta_L_prestress = delta_L_thermal + delta_L_free``
    - ``delta_L_mech = delta_L_total - delta_L_prestress``
    - ``N = (E A / L) * delta_L_mech`` (positive = tension)
    """
    results: list[dict[str, Any]] = []
    strain_energy = 0.0
    prestress_work = 0.0
    node_map = {node.id: i for i, node in enumerate(nodes)}

    for elem in elements:
        i = node_map[elem.node_i]
        j = node_map[elem.node_j]
        dx = nodes[j].x - nodes[i].x
        dy = nodes[j].y - nodes[i].y
        L = np.sqrt(dx**2 + dy**2)

        if L < _ZERO_LENGTH_TOL:
            results.append({"id": elem.id, "N": 0.0, "status": "ZERO_LENGTH"})
            continue

        c = dx / L
        s = dy / L

        # Nodal displacements
        ui, vi = U[2 * i], U[2 * i + 1]
        uj, vj = U[2 * j], U[2 * j + 1]

        # Total elongation from nodal displacements
        delta_L_total = (uj - ui) * c + (vj - vi) * s

        # Thermal/fabrication elongation
        delta_L_thermal = elem.alpha * elem.delta_T * L
        delta_L_prestress = delta_L_thermal + elem.delta_L_free

        # Mechanical elongation (what causes stress)
        delta_L_mech = delta_L_total - delta_L_prestress

        # Axial stiffness
        k = elem.E * elem.A / L

        # Axial force (positive = tension)
        N = k * delta_L_mech

        # Strain energy (mechanical only)
        strain_energy += 0.5 * k * delta_L_mech**2

        # Prestress work
        prestress_work += k * delta_L_prestress * delta_L_mech

        # Sign classification via the *axial strain* N/(E*A) rather than an
        # absolute newton cut-off, so the verdict is invariant to unit system
        # and model size: a 1e-6 N force is "Tension" in a millinewton model
        # and indistinguishable from zero in a giganewton one.
        ea = elem.E * elem.A
        axial_strain = N / ea if ea > 0.0 else 0.0
        if axial_strain > _FORCE_ZERO_REL:
            status = "Tension"
        elif axial_strain < -_FORCE_ZERO_REL:
            status = "Compression"
        else:
            status = "Zero"
        results.append(
            {
                "id": elem.id,
                "N": float(N),
                "delta_L_mech": float(delta_L_mech),
                "delta_L_prestress": float(delta_L_prestress),
                "status": status,
            }
        )

    return results, float(strain_energy), float(prestress_work)


def imposed_strain_energy(nodes: list[Node], elements: list[Element]) -> float:
    """Characteristic elastic energy of the imposed (eigen) strain field.

    Returns ``sum(0.5 * k * delta_L_prestress**2)`` over all elements, where
    ``delta_L_prestress = alpha * delta_T * L + delta_L_free`` is the imposed
    elongation and ``k = E A / L`` the axial stiffness.

    This quantity is *not* an energy the structure necessarily stores: for a
    freely expanding bar it is entirely relieved and the stored strain energy
    is zero. Its role here is to provide a physically meaningful **reference
    scale** for the energy balance in :func:`truss_analysis.solver.check_energy`.

    Motivation
    ----------
    Imposed-strain problems can make every term of the Clapeyron balance
    vanish analytically (free thermal expansion: ``W_mech = U_strain =
    W_prestress = 0``). A purely *relative* residual test then divides
    round-off by round-off and reports a 100% error. An *absolute* joule
    threshold would be dimensionally arbitrary and would behave differently
    for millimetre and kilometre models. Anchoring the round-off floor to
    this characteristic energy keeps the test scale-free and unit-agnostic.

    Parameters
    ----------
    nodes : list[Node]
        Model nodes (ordering defines the DOF map).
    elements : list[Element]
        Model elements.

    Returns
    -------
    float
        Non-negative characteristic energy in joules. Zero when the model
        carries no imposed strain.
    """
    node_map = {node.id: i for i, node in enumerate(nodes)}
    total = 0.0
    for elem in elements:
        i = node_map[elem.node_i]
        j = node_map[elem.node_j]
        dx = nodes[j].x - nodes[i].x
        dy = nodes[j].y - nodes[i].y
        L = float(np.sqrt(dx**2 + dy**2))
        if L < _ZERO_LENGTH_TOL:
            continue
        delta_L_prestress = elem.alpha * elem.delta_T * L + elem.delta_L_free
        k = elem.E * elem.A / L
        total += 0.5 * k * delta_L_prestress**2
    return float(total)


def calculate_reactions(
    nodes: list[Node],
    K: np.ndarray,
    U: np.ndarray,
    F_ext: np.ndarray,
    fixed_dofs: list[int],
) -> dict[str, dict[str, float]]:
    """Compute support reactions from ``R = K U - F_ext`` at constrained DOFs.

    Parameters
    ----------
    nodes : list[Node]
        Model nodes.
    K : np.ndarray
        Global stiffness matrix.
    U : np.ndarray
        Global displacement vector.
    F_ext : np.ndarray
        Global external force vector (mechanical + thermal).
    fixed_dofs : list[int]
        Indices of constrained degrees of freedom.

    Returns
    -------
    dict[str, dict[str, float]]
        Map ``node_id -> {"Fx": ..., "Fy": ...}``; a component is ``0.0``
        when the corresponding DOF is not constrained.
    """
    R = K @ U - F_ext
    fixed = set(fixed_dofs)
    reactions: dict[str, dict[str, float]] = {}
    for i, node in enumerate(nodes):
        dx_fixed = 2 * i in fixed
        dy_fixed = 2 * i + 1 in fixed
        if dx_fixed or dy_fixed:
            reactions[node.id] = {
                "Fx": float(R[2 * i]) if dx_fixed else 0.0,
                "Fy": float(R[2 * i + 1]) if dy_fixed else 0.0,
            }
    return reactions


def check_equilibrium(
    nodes: list[Node],
    reactions: dict[str, dict[str, float]],
    applied_loads: list[dict[str, Any]],
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Check global static equilibrium: sum(Fx) = sum(Fy) = sum(M) = 0.

    Force and moment residuals are tested against **separate** scales. They
    have different dimensions — ``[N]`` versus ``[N m]`` — so a single
    tolerance cannot serve both. The moment scale is the force scale times a
    characteristic length taken from the model's bounding box, which makes the
    test invariant to the size of the structure.

    An earlier revision scaled the moment residual by ``limit * max(1, ref)``,
    i.e. by the force reference *twice*. The resulting bound had units of
    force squared: on a 1000 m bridge it was orders of magnitude too tight
    and rejected a perfectly equilibrated solution, while on a millimetre
    model it was so loose that a genuine imbalance would pass.

    Parameters
    ----------
    nodes : list[Node]
        Model nodes (used for moment arms and the characteristic length).
    reactions : dict[str, dict[str, float]]
        Support reactions as returned by :func:`calculate_reactions`.
    applied_loads : list[dict[str, Any]]
        Applied nodal loads with keys ``node_id``, ``Fx``, ``Fy``.
    tol : float, default 1e-6
        Relative tolerance applied to the force and moment scales.

    Returns
    -------
    dict[str, Any]
        ``{"sum_fx", "sum_fy", "sum_m", "is_valid"}`` with the raw residual
        sums and the scaled pass/fail verdict, plus the diagnostic keys
        ``"ref_force"``, ``"ref_moment"``, ``"length_char"``,
        ``"force_limit"`` and ``"moment_limit"`` used to reach it.
    """
    coords = {node.id: (node.x, node.y) for node in nodes}
    sum_fx = sum_fy = sum_m = 0.0

    # Sum reactions
    for nid, rec in reactions.items():
        x, y = coords[nid]
        sum_fx += rec["Fx"]
        sum_fy += rec["Fy"]
        sum_m += x * rec["Fy"] - y * rec["Fx"]

    # Sum applied loads
    for lf in applied_loads:
        x, y = coords[str(lf["node_id"])]
        sum_fx += lf["Fx"]
        sum_fy += lf["Fy"]
        sum_m += x * lf["Fy"] - y * lf["Fx"]

    # Characteristic length: the bounding-box diagonal, i.e. the scale of the
    # moment arms in this model. Taking it from the geometry rather than a
    # constant is what makes the moment bound dimensionally [N m].
    if nodes:
        xs = [node.x for node in nodes]
        ys = [node.y for node in nodes]
        length_char = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    else:
        length_char = 0.0

    # Reference magnitudes built from the model itself, so both bounds are
    # relative and unit-agnostic.
    ref_force = 0.0
    for lf in applied_loads:
        ref_force += abs(lf["Fx"]) + abs(lf["Fy"])
    for rec in reactions.values():
        ref_force += abs(rec["Fx"]) + abs(rec["Fy"])
    ref_moment = ref_force * length_char

    # A load-free model has zero references and, exactly, zero residuals; the
    # absolute fallback only absorbs round-off in that degenerate case.
    force_limit = tol * ref_force if ref_force > 0.0 else tol
    moment_limit = tol * ref_moment if ref_moment > 0.0 else tol

    return {
        "sum_fx": float(sum_fx),
        "sum_fy": float(sum_fy),
        "sum_m": float(sum_m),
        "ref_force": float(ref_force),
        "ref_moment": float(ref_moment),
        "length_char": float(length_char),
        "force_limit": float(force_limit),
        "moment_limit": float(moment_limit),
        "is_valid": bool(
            abs(sum_fx) <= force_limit
            and abs(sum_fy) <= force_limit
            and abs(sum_m) <= moment_limit
        ),
    }


def calculate_buckling(
    nodes: list[Node],
    elements: list[Element],
    results: list[dict[str, Any]],
    tol: float = 1e-12,
) -> list[dict[str, Any]]:
    """Report elastic buckling utilisation for compressed members.

    Uses the Euler elastic critical load with the member's effective-length
    factor,

    .. code-block:: text

        P_cr = pi^2 E I / (K L)^2

    delegating to :func:`truss_analysis.sections.euler_buckling_load` so that
    this check, the fire limit-state check in
    :mod:`truss_analysis.limitstates` and the section utilities cannot drift
    apart. ``K`` defaults to 1.0 (pin-ended), which is the usual assumption
    for a pin-jointed truss member; a member with rotational restraint should
    carry ``K < 1``.

    Members that cannot be assessed are reported explicitly rather than
    defaulting to a pass. A compressed member whose ``I_sec`` was never set
    previously came back as ``P_cr = None, safe = True`` — missing data
    interpreted as safety. It now returns ``status = "unknown"`` with
    ``safe = False`` and emits a warning, because an unevaluated member is not
    a verified one.

    Parameters
    ----------
    nodes : list[Node]
        Model nodes.
    elements : list[Element]
        Model elements (``I_sec`` must be set for a meaningful check).
    results : list[dict[str, Any]]
        Element force results from :func:`calculate_element_forces`.
    tol : float, default 1e-12
        Numerical zero threshold for force, length and section values.

    Returns
    -------
    list[dict[str, Any]]
        One entry per element with keys ``id``, ``N``, ``length``, ``P_cr``,
        ``ratio`` (= ``-N / P_cr``), ``slenderness``, ``k_factor``, ``status``
        and ``safe``. ``status`` is one of ``"tension"`` (not checked),
        ``"zero_force"``, ``"checked"`` or ``"unknown"``.

    Warns
    -----
    BucklingCheckWarning
        When a compressed member cannot be assessed because ``I_sec`` or the
        member length is missing.
    """
    coords = {node.id: node for node in nodes}
    forces = {str(r.get("id")): float(r.get("N", 0.0)) for r in results}
    report: list[dict[str, Any]] = []
    unassessed: list[str] = []

    for e in elements:
        ni, nj = coords[e.node_i], coords[e.node_j]
        L = float(np.hypot(nj.x - ni.x, nj.y - ni.y))
        N = forces.get(str(e.id), 0.0)
        entry: dict[str, Any] = {
            "id": e.id,
            "N": N,
            "length": L,
            "P_cr": None,
            "ratio": 0.0,
            "slenderness": None,
            "k_factor": float(e.effective_length_factor),
            "status": "tension",
            # Conservative default: an unevaluated member is not a safe member.
            "safe": False,
        }

        if -tol <= N:
            # Nothing to buckle: tension or (numerically) zero force.
            entry["status"] = "zero_force" if abs(N) <= tol else "tension"
            entry["safe"] = True
            report.append(entry)
            continue

        if tol >= L or e.I_sec <= tol:
            # Compressed but unassessable. Report it loudly instead of
            # silently passing, and keep safe=False.
            entry["status"] = "unknown"
            unassessed.append(str(e.id))
            report.append(entry)
            continue

        p_cr = euler_buckling_load(e.I_sec, L, e.E, e.effective_length_factor)
        k_eff = e.effective_length_factor * L
        r_gyr = float(np.sqrt(e.I_sec / e.A)) if tol < e.A else 0.0
        entry["P_cr"] = p_cr
        entry["ratio"] = -N / p_cr
        # Slenderness on the *effective* length, consistent with P_cr.
        entry["slenderness"] = k_eff / r_gyr if r_gyr > tol else None
        entry["status"] = "checked"
        entry["safe"] = bool(entry["ratio"] < 1.0)
        report.append(entry)

    if unassessed:
        warnings.warn(
            f"buckling not assessed for {len(unassessed)} compressed member(s) "
            f"with missing I_sec or length: {', '.join(unassessed[:10])}"
            + (" ..." if len(unassessed) > 10 else "")
            + "; they are reported with status='unknown' and safe=False",
            BucklingCheckWarning,
            stacklevel=2,
        )

    return report


#: Largest nodal displacement as a fraction of the structure's characteristic
#: length beyond which the small-displacement assumption is considered
#: stretched and :class:`~truss_analysis.exceptions.LargeDisplacementWarning`
#: is emitted. 10 % is the round-5 audit's suggested physical-plausibility
#: threshold (C9 8.3): well above serviceability-relevant deflections, well
#: below "the geometry has visibly changed".
LARGE_DISPLACEMENT_RATIO = 0.1


def check_shallow_system(
    nodes: list[Node],
    ratio: float = 0.1,
) -> float | None:
    """Warn when the system geometry is shallow (rise/span < ratio).

    Linearised bifurcation analysis approximates the true snap-through limit
    point with an error that scales as O(theta_0^2) where theta_0 is the
    initial rise angle. For shallow systems (rise-to-span ratio below ~0.1),
    the linearised lambda_cr can be significantly optimistic compared to the
    actual collapse load. This warning flags such geometries so engineers can
    supplement with geometrically nonlinear analysis (round-5 audit C6#9,
    C7§4.2: member-specific or strain-based shallow detection is preferred).

    Parameters
    ----------
    nodes : list[Node]
        Model nodes (for computing rise and span from bounding box).
    ratio : float, default 0.1
        Threshold fraction; must be positive.

    Returns
    -------
    float or None
        ``rise / span`` (``None`` when span is zero, i.e. a degenerate model).
        The warning is emitted when the returned value is below ``ratio``.
    """
    if ratio <= 0.0:
        msg = f"ratio must be positive, got {ratio}"
        raise ValueError(msg)
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    span = max(xs) - min(xs) if nodes else 0.0
    rise = max(ys) - min(ys) if nodes else 0.0
    if span <= 0.0:
        return None
    shallow_ratio = rise / span
    if shallow_ratio < ratio:
        warnings.warn(
            f"system geometry is shallow: rise/span = {shallow_ratio:.3f} "
            f"(threshold {ratio:.1f}): linearised bifurcation analysis may be "
            f"optimistic compared to snap-through collapse. Consider a "
            f"geometrically nonlinear analysis.",
            ShallowSystemWarning,
            stacklevel=2,
        )
    return shallow_ratio


def check_displacement_magnitude(
    nodes: list[Node],
    U: np.ndarray,
    ratio: float = LARGE_DISPLACEMENT_RATIO,
) -> float | None:
    """Warn when max |u| exceeds ``ratio`` of the characteristic length.

    The solver is first-order: stiffness is assembled on the undeformed
    geometry and P-Delta effects are out of scope. This check makes that
    validity boundary *observable* instead of silent -- a model whose
    displacements are a sizeable fraction of its own dimensions has left
    the regime where the linear answer is trustworthy (round-5 audit,
    C9 8.3 / C2 / C5-1).

    Parameters
    ----------
    nodes : list[Node]
        Model nodes (for the characteristic length: the bounding-box
        diagonal, the same measure ``calculate_buckling`` reports as
        ``length_char``).
    U : np.ndarray
        Global displacement vector ``(2n,)`` [m].
    ratio : float, default LARGE_DISPLACEMENT_RATIO
        Threshold fraction; must be positive.

    Returns
    -------
    float or None
        ``max|u| / length_char`` (``None`` when the characteristic length
        is zero, i.e. a degenerate single-point model). The warning is
        emitted when the returned value exceeds ``ratio``.
    """
    if ratio <= 0.0:
        msg = f"ratio must be positive, got {ratio}"
        raise ValueError(msg)
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    length_char = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) if nodes else 0.0
    if length_char <= 0.0:
        return None
    u_max = float(np.max(np.abs(U))) if U.size else 0.0
    rel = u_max / length_char
    if rel > ratio:
        warnings.warn(
            f"largest nodal displacement {u_max:.4g} m is {rel:.1%} of the "
            f"characteristic length {length_char:.4g} m (threshold {ratio:.0%}): "
            "the small-displacement assumption of the linear solver is "
            "stretched; results may be non-conservative. Consider a "
            "geometrically nonlinear analysis.",
            LargeDisplacementWarning,
            stacklevel=2,
        )
    return rel
