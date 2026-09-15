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
    ImperfectionStudy,
    geometric_stiffness,
    imperfection_sensitivity,
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
    LUMPED_SECTION_FACTOR_LIMIT,
    ParametricFire,
    SteelHeatingResult,
    biot_critical_section_factor,
    iso_834_temperature,
    lumped_capacity_biot,
    parametric_fire_temperature,
    steel_temperature,
)
from .thermal.protection import (
    InsulationMaterial,
    insulation_catalogue,
    insulation_material,
    protected_steel_temperature,
    protected_temperatures_for_members,
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
    "LUMPED_SECTION_FACTOR_LIMIT",
    "AnalysisResult",
    "BucklingResult",
    "Element",
    "ImperfectionStudy",
    "InsulationMaterial",
    "LinearizationCheck",
    "Node",
    "ParametricFire",
    "SquareHSS",
    "SteelHeatingResult",
    "TangentCheck",
    "TrussConfig",
    "TrussFamily",
    "__version__",
    "assemble_global_matrices",
    "biot_critical_section_factor",
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
    "imperfection_sensitivity",
    "insulation_catalogue",
    "insulation_material",
    "internal_force",
    "iso_834_temperature",
    "linearized_buckling_load_factor",
    "linearized_tangent_stiffness",
    "lumped_capacity_biot",
    "parametric_fire_temperature",
    "protected_steel_temperature",
    "protected_temperatures_for_members",
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
