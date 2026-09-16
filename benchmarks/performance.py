"""Performance regression guards for the criticality engine.

This module is **not** part of the verification matrix in ``docs/theory.md``
section 8, and it deliberately does not subclass
:class:`~benchmarks.reference_problems.ReferenceProblem`.  A reference problem
pairs a library quantity with an independent oracle for it; a timing
measurement has no oracle, because there is no independent source of truth for
how long this machine takes to do this work.  Presenting a wall-clock number as
verification would be exactly the category error section 8.1 exists to prevent,
so it is labelled as what it is: a *regression guard* on a claim the library
makes about itself.

The claim under guard is the criticality engine's reason to exist.  The rank-1
Sherman-Morrison path factorises once and serves every member, so the work is
``O(n^3)`` once plus ``O(n^2)`` per member instead of ``O(n^3)`` per member.
That is an architectural intention until it is measured, and an unmeasured
performance claim in a library whose whole culture is measured margins is a
claim without evidence.

Two disciplines keep the measurement honest:

* **Speed is never measured without agreement.**  Every measurement also
  compares the fast path's answer against the direct resolve it is being timed
  against, and refuses to report a speedup if the two disagree.  A path that
  returns garbage is very fast, and a guard that only timed would reward it.
* **The floor sits well below the measurement.**  Wall-clock ratios move with
  machine load, BLAS build and thread count, so a floor set at the measured
  value would flake and a flaky gate gets disabled.  The floor is set at the
  point where the architectural claim would be *false* -- below roughly ``m/4``
  on ``m`` members there is no ``O(n^3)``-once benefit left to observe -- and
  the measured value is recorded beside it so a regression short of the floor
  is still visible in the report.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from truss_analysis.criticality.engine import (
    T_AMBIENT,
    base_displacement,
    brute_force_ci,
    build_engine,
    ci_sweep,
    total_load_vector,
)
from truss_analysis.model import Element, Node
from truss_analysis.topology_generator import generate_topology

__all__ = [
    "PerformanceMeasurement",
    "measure_criticality_engine",
    "pratt_model",
]

#: Below this speedup the architectural claim is false, not merely diluted.
#: With ``m`` members the direct path pays ``m`` factorisations and the rank-1
#: path pays one plus ``m`` updates, so the ratio grows with ``m``; requiring
#: ``m / 4`` leaves a wide band for machine noise while still failing if the
#: engine ever degenerates to per-member factorisation.
SPEEDUP_FLOOR_DIVISOR = 4.0

#: Relative agreement required between the fast path and the direct resolve
#: before a speedup may be reported at all.
AGREEMENT_RTOL = 1e-9


@dataclass(frozen=True)
class PerformanceMeasurement:
    """One timed comparison of the rank-1 engine against a direct resolve.

    Attributes
    ----------
    n_members : int
        Members in the model.
    n_free_dof : int
        Unrestrained degrees of freedom, which is the size of the system that
        gets factorised.
    engine_s : float
        Wall-clock seconds for the rank-1 sweep over every member.
    direct_s : float
        Wall-clock seconds for one full rebuild and resolve per member.
    speedup : float
        ``direct_s / engine_s``.
    max_ci_deviation : float
        Largest absolute difference between the two paths' criticality indices.
        Reported alongside the speedup so the two cannot be read apart.
    agrees : bool
        Whether ``max_ci_deviation`` is within :data:`AGREEMENT_RTOL` of the
        direct path's scale.  A measurement that does not agree has no
        speedup worth reporting, and :attr:`speedup` should be ignored.
    floor : float
        The :data:`SPEEDUP_FLOOR_DIVISOR`-derived bound the claim has to clear.
    """

    n_members: int
    n_free_dof: int
    engine_s: float
    direct_s: float
    speedup: float
    max_ci_deviation: float
    agrees: bool
    floor: float

    @property
    def passed(self) -> bool:
        """The claim holds only if it is both fast and right."""
        return bool(self.agrees and self.speedup >= self.floor)

    def report(self) -> str:
        """One human-readable block, in the style of the benchmark summary."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] criticality_engine_speedup\n"
            f"    model        : {self.n_members} members, "
            f"{self.n_free_dof} free DOF\n"
            f"    rank-1 sweep : {self.engine_s * 1e3:9.2f} ms\n"
            f"    direct resolve: {self.direct_s * 1e3:8.2f} ms\n"
            f"    speedup      : {self.speedup:9.2f}x  "
            f"(floor {self.floor:.2f}x = m/{SPEEDUP_FLOOR_DIVISOR:g})\n"
            f"    agreement    : max |dCI| = {self.max_ci_deviation:.3e}  "
            f"({'agrees' if self.agrees else 'DISAGREES -- speedup is void'})"
        )


def pratt_model(n_panels: int, span: float = 0.0, depth: float = 0.0):
    """A Pratt girder sized for a performance run, as ``Node``/``Element``.

    Built through :func:`truss_analysis.topology_generator` and converted here,
    because this module times the library against itself rather than checking it
    against an oracle -- so sharing a model constructor is a convenience, not a
    threat to independence.  Span and depth default to a 4 m panel, which keeps
    the members stocky enough that the solve, not the conditioning, is what is
    being timed.
    """
    span = span or 4.0 * n_panels
    depth = depth or span / 8.0
    topo = generate_topology(
        "pratt", n_panels, span, depth, area=5.0e-3, total_load=2.0e5
    )
    nodes = [
        Node(
            id=str(nd["id"]),
            x=float(nd["x"]),
            y=float(nd["y"]),
            is_support=bool(nd.get("is_support", False)),
            support_dx=bool(nd.get("support_dx", False)),
            support_dy=bool(nd.get("support_dy", False)),
        )
        for nd in topo["nodes"]
    ]
    elements = [
        Element(
            id=str(el["id"]),
            node_i=str(el["node_i"]),
            node_j=str(el["node_j"]),
            E=float(el["E"]),
            A=float(el["A"]),
            I_sec=float(el.get("I_sec", 0.0)),
            alpha=float(el.get("alpha", 0.0)),
            effective_length_factor=float(el.get("effective_length_factor", 1.0)),
        )
        for el in topo["elements"]
    ]
    loads: dict[str, dict[str, float]] = {}
    for entry in topo["loads"]:
        loads[str(entry["node_id"])] = {
            "Fx": float(entry.get("Fx", 0.0)),
            "Fy": float(entry.get("Fy", 0.0)),
        }
    return nodes, elements, loads


def _n_free_dof(nodes: Sequence[Node]) -> int:
    return sum((0 if n.support_dx else 1) + (0 if n.support_dy else 1) for n in nodes)


def measure_criticality_engine(
    n_panels: int = 16, alpha: float = 0.7, repeats: int = 1
) -> PerformanceMeasurement:
    """Time the rank-1 sweep against a full resolve per member.

    Parameters
    ----------
    n_panels : int, default 16
        Pratt panel count.  16 panels gives on the order of 50 members, which is
        large enough for the ``O(n^3)``-once saving to dominate and small enough
        to run inside a test suite.
    alpha : float, default 0.7
        Stiffness multiplier defining the damaged state.
    repeats : int, default 1
        Timed repetitions; the *fastest* is kept, which is the standard way to
        discount scheduler noise rather than averaging it in.

    Returns
    -------
    PerformanceMeasurement
        The measurement, including the agreement between the two paths.
    """
    nodes, elements, loads = pratt_model(n_panels)
    n_free = _n_free_dof(nodes)

    setup = build_engine(nodes, elements, loads, None)
    u = base_displacement(setup, total_load_vector(nodes, loads, setup))

    engine_best = float("inf")
    engine_ci: dict[str, float] = {}
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        sweep = ci_sweep(setup, u, alpha)
        elapsed = time.perf_counter() - start
        if elapsed < engine_best:
            engine_best = elapsed
            engine_ci = dict(sweep.ci_values)

    # brute_force_ci takes an explicit temperature field rather than None, so
    # the ambient one is spelled out.  It is the same field the cold engine was
    # built with: k_E(T_AMBIENT) is exactly 1, so the two paths see identical
    # stiffnesses and the only difference between them is the algorithm.
    temps = {e.id: T_AMBIENT for e in elements}

    direct_best = float("inf")
    direct_ci: dict[str, float] = {}
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        direct, _, _ = brute_force_ci(nodes, elements, loads, temps, alpha)
        elapsed = time.perf_counter() - start
        if elapsed < direct_best:
            direct_best = elapsed
            direct_ci = direct

    # Both paths must describe the same physics before any ratio between their
    # timings means anything, so the agreement is measured on the same members
    # in the same order and refused if it fails.
    deviation = float(max(abs(engine_ci[e.id] - direct_ci[e.id]) for e in elements))
    scale = max(1.0, max(abs(v) for v in direct_ci.values()))
    agrees = bool(np.isfinite(deviation) and deviation <= AGREEMENT_RTOL * scale)

    speedup = direct_best / engine_best if engine_best > 0.0 else float("inf")
    return PerformanceMeasurement(
        n_members=len(elements),
        n_free_dof=n_free,
        engine_s=engine_best,
        direct_s=direct_best,
        speedup=float(speedup),
        max_ci_deviation=deviation,
        agrees=agrees,
        floor=len(elements) / SPEEDUP_FLOOR_DIVISOR,
    )
