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
from .stability import (
    BucklingResult,
    geometric_stiffness,
    linearized_buckling_load_factor,
)
from .tangent_verification import (
    LinearizationCheck,
    TangentCheck,
    exact_tangent_stiffness,
    internal_force,
    linearized_tangent_stiffness,
    verify_linearization_convergence,
    verify_tangent_stiffness,
)
from .thermal.fire_curve import (
    SteelHeatingResult,
    iso_834_temperature,
    steel_temperature,
)
from .topology_generator import TrussConfig, TrussFamily, generate_topology

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover
    # Git-less source exports have no generated _version.py; fall back to
    # the installed distribution metadata before resorting to a literal.
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("truss-analysis")
    except (ImportError, PackageNotFoundError):
        __version__ = "2.7.0"

solve_truss = solve

__all__ = [
    "AnalysisResult",
    "BucklingResult",
    "Element",
    "LinearizationCheck",
    "Node",
    "SquareHSS",
    "SteelHeatingResult",
    "TangentCheck",
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
    "exact_tangent_stiffness",
    "generate_topology",
    "geometric_stiffness",
    "idealised_square_hss",
    "internal_force",
    "iso_834_temperature",
    "linearized_buckling_load_factor",
    "linearized_tangent_stiffness",
    "run",
    "solve",
    "solve_truss",
    "steel_temperature",
    "structural_report",
    "system_critical_temperature",
    "validate_topology",
    "verify_linearization_convergence",
    "verify_tangent_stiffness",
]
