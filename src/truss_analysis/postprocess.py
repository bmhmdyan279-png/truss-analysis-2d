"""Post-processing: element forces, reactions, equilibrium and buckling."""

from __future__ import annotations

from typing import Any

import numpy as np

from .model import Element, Node

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

    Parameters
    ----------
    nodes : list[Node]
        Model nodes (used for moment arms).
    reactions : dict[str, dict[str, float]]
        Support reactions as returned by :func:`calculate_reactions`.
    applied_loads : list[dict[str, Any]]
        Applied nodal loads with keys ``node_id``, ``Fx``, ``Fy``.
    tol : float, default 1e-6
        Relative tolerance scaled by the magnitudes present in the model.

    Returns
    -------
    dict[str, Any]
        ``{"sum_fx", "sum_fy", "sum_m", "is_valid"}`` with the raw residual
        sums and the scaled pass/fail verdict.
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

    # Tolerance scaling
    ref = 0.0
    for lf in applied_loads:
        ref += abs(lf["Fx"]) + abs(lf["Fy"])
    for rec in reactions.values():
        ref += abs(rec["Fx"]) + abs(rec["Fy"])
    limit = tol * max(1.0, ref)

    return {
        "sum_fx": float(sum_fx),
        "sum_fy": float(sum_fy),
        "sum_m": float(sum_m),
        "is_valid": bool(
            abs(sum_fx) <= limit
            and abs(sum_fy) <= limit
            and abs(sum_m) <= limit * max(1.0, ref)
        ),
    }


def calculate_buckling(
    nodes: list[Node],
    elements: list[Element],
    results: list[dict[str, Any]],
    tol: float = 1e-12,
) -> list[dict[str, Any]]:
    """Report Euler buckling utilisation for compressed members.

    Uses the pin-ended Euler load ``P_cr = pi^2 E I / L^2``; members in
    tension (or with negligible force, length or second moment of area)
    are reported with ``P_cr = None`` and ``safe = True``.

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
        One entry per element with keys ``id``, ``N``, ``length``,
        ``P_cr``, ``ratio`` (= ``-N / P_cr``), ``slenderness`` and ``safe``.
    """
    coords = {node.id: node for node in nodes}
    forces = {str(r.get("id")): float(r.get("N", 0.0)) for r in results}
    report: list[dict[str, Any]] = []

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
            "safe": True,
        }

        if -tol > N and tol < L and e.I_sec > tol:
            p_cr = float(np.pi**2 * e.E * e.I_sec / L**2)
            r_gyr = float(np.sqrt(e.I_sec / e.A)) if tol < e.A else 0.0
            entry["P_cr"] = p_cr
            entry["ratio"] = -N / p_cr
            entry["slenderness"] = L / r_gyr if r_gyr > tol else None
            entry["safe"] = bool(entry["ratio"] < 1.0)

        report.append(entry)

    return report
