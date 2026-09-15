"""Member stiffness-degradation operators and the geometric scaling rule.

Two degradation definitions share one interface (:class:`DegradationOperator`):

* ``mechanical`` - scale the cross-sectional area by ``alpha`` and the second
  moment of area by ``alpha**2`` (the geometric scaling rule);
* ``thermal`` - scale Young's modulus by the EN 1993-1-2 reduction factor
  ``k_E(T)`` from :mod:`truss_analysis.material`.

A registry guard (:func:`get_degradation_operator`) rejects any other,
ad-hoc definition.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, lapack

from .assembly import assemble_global_matrices
from .exceptions import SingularMatrixError
from .material.steel_eurocode import k_E as eurocode_k_E
from .model import Element, Node
from .postprocess import calculate_element_forces
from .reliability_adapter import NodalLoad
from .solver import solve

#: Reciprocal-condition threshold of the ``alpha = 1e-6`` key-element probe
#: (see :meth:`DamageOperator._check_mechanism`). Calibrated against the
#: probe magnitude, NOT a numerics-policy cutoff: removing an essential
#: member drives ``rcond`` to ``O(probe)``, a merely important one stays
#: orders of magnitude above this.
_KEY_ELEMENT_RCOND = 1e-5


@dataclass(frozen=True)
class DegradationPoint:
    """Result of a single alpha-degradation solve.

    Attributes
    ----------
    alpha : float
        Scaling factor applied to the target member.
    max_disp : float
        Maximum nodal displacement magnitude [m]; ``inf`` when the
        degraded model is a mechanism.
    axial_forces : dict[str, float]
        Member axial forces [N] keyed by element id (empty if singular).
    is_singular : bool
        ``True`` when the degraded stiffness matrix lost rank.
    error : str or None
        Solver error message when ``is_singular`` is set.
    """

    alpha: float
    max_disp: float
    axial_forces: dict[str, float]
    is_singular: bool
    error: str | None = None


@dataclass
class MemberSensitivityProfile:
    """Sensitivity profile of one member across a sequence of alpha values.

    Attributes
    ----------
    member_id : str
        Target element id.
    baseline_max_disp : float
        Maximum nodal displacement of the undegraded model [m].
    baseline_axial_forces : dict[str, float]
        Undegraded member axial forces [N].
    points : list[DegradationPoint]
        One :class:`DegradationPoint` per requested alpha.
    sensitivity_slope : float
        Least-squares slope of ``max_disp(alpha) / baseline`` over alpha
        on the non-singular points.
    scf_alpha_min : float
        Response ratio ``max_disp / baseline_max_disp`` at the smallest
        non-singular alpha (``inf`` when every point is singular).
    is_key_element : bool
        ``True`` when an near-zero-alpha probe turns the structure into a
        mechanism (the member is kinematically essential).
    mechanism_detected_at : float or None
        First alpha at which a mechanism was detected, if any.
    """

    member_id: str
    baseline_max_disp: float
    baseline_axial_forces: dict[str, float]
    points: list[DegradationPoint]
    sensitivity_slope: float
    scf_alpha_min: float
    is_key_element: bool
    mechanism_detected_at: float | None = None


class DamageOperator:
    """Apply alpha-degradation to truss members to assess system sensitivity.

    Parameters
    ----------
    nodes : list[Node]
        Model nodes.
    elements : list[Element]
        Model elements.
    loads : list[NodalLoad]
        External nodal loads.
    """

    def __init__(
        self,
        nodes: list[Node],
        elements: list[Element],
        loads: list[NodalLoad],
    ) -> None:
        self.nodes = nodes
        self.elements = elements
        self.loads = loads
        self.node_map = {node.id: i for i, node in enumerate(nodes)}

    def _solve(
        self,
        current_nodes: list[Node],
        current_elements: list[Element],
    ) -> tuple[np.ndarray, dict[str, float]]:
        K, F_ext, _, fixed_dofs = assemble_global_matrices(
            current_nodes, current_elements
        )
        for load in self.loads:
            idx = self.node_map[load.node_id]
            F_ext[2 * idx] += load.fx
            F_ext[2 * idx + 1] += load.fy

        U = solve(K, F_ext, fixed_dofs)

        # Delegate force recovery to the single canonical implementation.
        # An earlier revision re-derived delta_L_mech / prestress by hand
        # here -- the exact "two implementations of one physics" pattern
        # that produced the 2.6.0 demand-chain bugs (round-5 audit).
        results, _, _ = calculate_element_forces(current_nodes, current_elements, U)
        axial_forces: dict[str, float] = {str(r["id"]): float(r["N"]) for r in results}

        return U, axial_forces

    def _check_mechanism(
        self,
        current_nodes: list[Node],
        current_elements: list[Element],
    ) -> bool:
        """Near-mechanism probe consistent with the library's solver policy.

        ``K_ff`` of a stable truss is symmetric positive definite, so the
        probe Cholesky-factorises it (raising on indefiniteness == hard
        mechanism) and estimates the reciprocal condition number from the
        existing factors with LAPACK ``dpocon`` (``O(n^2)``, the same trick
        :func:`truss_analysis.criticality.engine._guard_tolerance` uses with
        ``dgecon``).  This replaces a full ``O(n^3)`` SVD per member *and*
        its hard-coded ``1e-5`` rank cutoff with a named, documented
        threshold.  ``_KEY_ELEMENT_RCOND`` is deliberately **not** one of the
        :class:`~truss_analysis.numerics.NumericalTolerances` policy cutoffs:
        it is calibrated against the ``alpha = 1e-6`` key-element probe --
        removing a kinematically essential member drops ``rcond`` to
        ``O(alpha_probe)``, while a merely important member leaves it orders
        of magnitude above the threshold.
        """
        K, _, _, fixed_dofs = assemble_global_matrices(current_nodes, current_elements)
        n = len(current_nodes)
        free_dofs = [i for i in range(2 * n) if i not in fixed_dofs]
        if not free_dofs:
            return False
        K_ff = K[np.ix_(free_dofs, free_dofs)]

        try:
            c, low = cho_factor(K_ff, lower=True, check_finite=False)
        except LinAlgError:
            return True
        anorm = float(np.abs(K_ff).sum(axis=0).max())
        if not np.isfinite(anorm) or anorm <= 0.0:
            return True
        try:
            rcond, info = lapack.dpocon(c, anorm, "L" if low else "U")
        except (ValueError, TypeError, lapack.LapackError):
            return True
        if info != 0 or not np.isfinite(rcond):
            return True
        return float(rcond) < _KEY_ELEMENT_RCOND

    def _apply_geometric_scaling(
        self,
        elements: list[Element],
        target_id: str,
        alpha: float,
    ) -> list[Element]:
        new_elements = []
        for elem in elements:
            if elem.id == target_id:
                new_elem = deepcopy(elem)
                new_elem.A = elem.A * alpha
                new_elem.I_sec = elem.I_sec * (alpha**2)
                new_elements.append(new_elem)
            else:
                new_elements.append(elem)
        return new_elements

    def analyze_member(
        self,
        target_id: str,
        alphas: Sequence[float],
        probe_near_zero: bool = True,
    ) -> MemberSensitivityProfile:
        """Degrade one member across ``alphas`` and profile the response.

        Parameters
        ----------
        target_id : str
            Element to degrade.
        alphas : Sequence[float]
            Scaling factors applied in order (area scales by ``alpha``,
            second moment of area by ``alpha**2``).
        probe_near_zero : bool, default True
            Additionally probe ``alpha = 1e-6`` to decide whether removing
            the member turns the structure into a mechanism.

        Returns
        -------
        MemberSensitivityProfile
            Baseline quantities, one point per alpha, the least-squares
            sensitivity slope and the kinematic-essentiality flag.
        """
        baseline_U, baseline_N = self._solve(self.nodes, self.elements)
        baseline_max_disp = float(np.max(np.hypot(baseline_U[0::2], baseline_U[1::2])))

        points: list[DegradationPoint] = []
        mechanism_detected_at: float | None = None

        for alpha in alphas:
            degraded_elements = self._apply_geometric_scaling(
                self.elements, target_id, alpha
            )
            try:
                U, N = self._solve(self.nodes, degraded_elements)
                max_disp = float(np.max(np.hypot(U[0::2], U[1::2])))
                points.append(
                    DegradationPoint(
                        alpha=alpha,
                        max_disp=max_disp,
                        axial_forces=N,
                        is_singular=False,
                    )
                )
            except SingularMatrixError as e:
                points.append(
                    DegradationPoint(
                        alpha=alpha,
                        max_disp=float("inf"),
                        axial_forces={},
                        is_singular=True,
                        error=str(e),
                    )
                )
                if mechanism_detected_at is None:
                    mechanism_detected_at = alpha

        is_key = False
        if probe_near_zero:
            probe_alpha = 1e-6
            degraded_elements = self._apply_geometric_scaling(
                self.elements, target_id, probe_alpha
            )
            if self._check_mechanism(self.nodes, degraded_elements):
                is_key = True
                if mechanism_detected_at is None:
                    mechanism_detected_at = probe_alpha

        valid_points = [p for p in points if not p.is_singular]
        if not valid_points or baseline_max_disp < 1e-15:
            slope = 0.0
            scf_min = float("inf")
        else:
            scfs = [p.max_disp / baseline_max_disp for p in valid_points]
            alphas_valid = [p.alpha for p in valid_points]
            x = np.array(alphas_valid)
            y = np.array(scfs)
            if len(x) > 1:
                A_mat = np.vstack([x, np.ones(len(x))]).T
                slope = float(np.linalg.lstsq(A_mat, y, rcond=None)[0][0])
            else:
                slope = 0.0
            # The documented contract is "SCF at the SMALLEST non-singular
            # alpha" -- select it explicitly instead of trusting the caller's
            # ordering (``scfs[-1]`` silently meant something else for any
            # non-descending ``alphas``; round-5 audit).
            scf_min = scfs[int(np.argmin(x))]

        return MemberSensitivityProfile(
            member_id=target_id,
            baseline_max_disp=baseline_max_disp,
            baseline_axial_forces=baseline_N,
            points=points,
            sensitivity_slope=float(slope),
            scf_alpha_min=scf_min,
            is_key_element=is_key,
            mechanism_detected_at=mechanism_detected_at,
        )

    def analyze_all(
        self,
        alphas: Sequence[float] = (1.0, 0.9, 0.8, 0.7),
    ) -> list[MemberSensitivityProfile]:
        """Profile every element in turn; returns one profile per element."""
        return [self.analyze_member(elem.id, alphas) for elem in self.elements]


# ----------------------------------------------------------------------
# One degradation interface, two registered implementations (mechanical
# area-scaling and thermal k_E(T)); the registry guard below rejects any
# third ad-hoc definition (covered by the test suite).
# ----------------------------------------------------------------------


class DegradationOperator(abc.ABC):
    """Common interface for member-stiffness degradation models."""

    kind: ClassVar[str] = "abstract"

    @abc.abstractmethod
    def apply(self, elements: list[Element]) -> list[Element]:
        """Return a new element list with the degradation applied."""


class MechanicalDegradation(DegradationOperator):
    """Mechanical definition: scale ``A`` by alpha and ``I`` by ``alpha**2``."""

    kind: ClassVar[str] = "mechanical"

    def __init__(self, target_id: str, alpha: float) -> None:
        self.target_id = target_id
        self.alpha = alpha

    def apply(self, elements: list[Element]) -> list[Element]:
        """Return a copy with ``A *= alpha`` and ``I *= alpha**2`` on target.

        Non-target elements are shared references (they are not mutated).
        """
        out = []
        for elem in elements:
            if elem.id == self.target_id:
                new = deepcopy(elem)
                new.A = elem.A * self.alpha
                new.I_sec = elem.I_sec * (self.alpha**2)
                out.append(new)
            else:
                out.append(elem)
        return out


class ThermalDegradation(DegradationOperator):
    """Thermal definition: scale ``E`` by the EN 1993-1-2 factor ``k_E(T)``.

    The reduction factors come from the single source of truth in
    :mod:`truss_analysis.material.steel_eurocode`.
    """

    kind: ClassVar[str] = "thermal"

    def __init__(self, temps: Mapping[str, float]) -> None:
        self.temps = dict(temps)

    def apply(self, elements: list[Element]) -> list[Element]:
        """Return copies of all elements with ``E *= k_E(T)`` per temperature.

        Every element is copied because any member may carry a temperature.

        Raises
        ------
        ValueError
            If any element has no entry in the temperature mapping. The
            pre-2.8 behaviour was a bare ``KeyError`` from the dict
            lookup, which did not say *which* members were missing nor what
            to do about them (round-5 audit).
        """
        missing = [elem.id for elem in elements if elem.id not in self.temps]
        if missing:
            msg = (
                "ThermalDegradation: no temperature supplied for member(s) "
                f"{', '.join(repr(m) for m in missing)}; every member needs "
                "an entry (use 20.0 degC for unaffected ones)"
            )
            raise ValueError(msg)
        out = []
        for elem in elements:
            new = deepcopy(elem)
            new.E = elem.E * float(eurocode_k_E(self.temps[elem.id]))
            out.append(new)
        return out


_REGISTRY: dict[str, type[DegradationOperator]] = {
    MechanicalDegradation.kind: MechanicalDegradation,
    ThermalDegradation.kind: ThermalDegradation,
}


def registered_degradation_kinds() -> tuple[str, ...]:
    """Return the degradation kinds registered in this library."""
    return tuple(_REGISTRY)


def get_degradation_operator(kind: str, **kwargs: object) -> DegradationOperator:
    """Instantiate a registered degradation operator by kind.

    Parameters
    ----------
    kind : str
        One of :func:`registered_degradation_kinds`.
    **kwargs : object
        Keyword arguments forwarded to the operator constructor.

    Returns
    -------
    DegradationOperator
        The constructed operator.

    Raises
    ------
    ValueError
        If ``kind`` is not registered (ad-hoc definitions are rejected).
    """
    try:
        cls = _REGISTRY[kind]
    except KeyError as exc:
        kinds = registered_degradation_kinds()
        msg = f"unknown degradation kind {kind!r}; registered: {kinds}"
        raise ValueError(msg) from exc
    return cls(**kwargs)
