"""Truss Analysis 2D - Public API."""

from __future__ import annotations

from .assembly import assemble_global_matrices
from .criticality import compute_ci_for_topology
from .graph_validation import structural_report, validate_topology
from .limitstates import ci_two_component, dcr_field, system_critical_temperature
from .main import AnalysisResult, run
from .model import Element, Node
from .postprocess import calculate_element_forces
from .sections import SquareHSS, euler_buckling_load, idealised_square_hss
from .solver import check_energy, solve
from .topology_generator import TrussConfig, TrussFamily, generate_topology

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "2.5.0"

solve_truss = solve

__all__ = [
    "AnalysisResult",
    "Element",
    "Node",
    "SquareHSS",
    "TrussConfig",
    "TrussFamily",
    "__version__",
    "assemble_global_matrices",
    "calculate_element_forces",
    "check_energy",
    "ci_two_component",
    "compute_ci_for_topology",
    "dcr_field",
    "euler_buckling_load",
    "generate_topology",
    "idealised_square_hss",
    "run",
    "solve",
    "solve_truss",
    "structural_report",
    "system_critical_temperature",
    "validate_topology",
]
