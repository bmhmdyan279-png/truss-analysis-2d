"""Criticality package: rank-1 engine, scenarios, ranking, indices.

Public API:

* :func:`compute_ci_for_topology` — full CI pipeline on the exact rank-1
  engine (Sherman-Morrison), uniform scenario included with no special case.
* :mod:`.engine` — ``build_engine``, ``ci_sweep``, ``brute_force_ci``
  (reference), ``perturb_multi`` (Woodbury rank-r), ``MechanismError``.
* :mod:`.scenarios` — centroid-based partition ``left|mid|right`` plus
  ``uniform`` and ``linear_gradient``; ``get_scenario_temperatures``.
* :mod:`.ranking` — ``rank_members`` (natural-sort tie-break), ``tau_b``
  returning :class:`ranking.TauResult` (``tau=None`` + ``is_degenerate``
  instead of silent numbers).
* :mod:`.indices` — the single ``compute_nci`` (degenerate -> ``None``).
* :mod:`.criteria` — ``multi_criteria_ci``: displacement, force, strain-energy
  and reaction indices from one rank-1 sweep, for the cases where a single
  displacement-based scalar ranks the wrong member.
"""

from __future__ import annotations

from .criteria import (
    FORCE_BLOCK_SIZE,
    MultiCriteriaResult,
    ReactionInfluence,
    multi_criteria_ci,
    reaction_influence,
)
from .engine import (
    GUARD_TOL,
    CiSweep,
    EngineSetup,
    MechanismError,
    TopologyResult,
    base_displacement,
    brute_force_ci,
    build_engine,
    ci_sweep,
    compute_ci_for_topology,
    load_vector,
    member_matrices,
    perturb_multi,
)
from .indices import NciResult, compute_nci
from .ranking import TauResult, natural_sort_key, rank_members, tau_b
from .scenarios import (
    SCENARIOS,
    T_AMBIENT,
    get_scenario_temperatures,
    member_centroids,
    relative_eps,
    scenario_partition,
    span_bounds,
)

__all__ = [
    "FORCE_BLOCK_SIZE",
    "GUARD_TOL",
    "SCENARIOS",
    "T_AMBIENT",
    "CiSweep",
    "EngineSetup",
    "MechanismError",
    "MultiCriteriaResult",
    "NciResult",
    "ReactionInfluence",
    "TauResult",
    "TopologyResult",
    "base_displacement",
    "brute_force_ci",
    "build_engine",
    "ci_sweep",
    "compute_ci_for_topology",
    "compute_nci",
    "get_scenario_temperatures",
    "load_vector",
    "member_centroids",
    "member_matrices",
    "multi_criteria_ci",
    "natural_sort_key",
    "perturb_multi",
    "rank_members",
    "reaction_influence",
    "relative_eps",
    "scenario_partition",
    "span_bounds",
    "tau_b",
]
