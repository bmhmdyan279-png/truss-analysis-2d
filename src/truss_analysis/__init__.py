"""Truss Analysis 2D - Public API."""

from __future__ import annotations

from .assembly import assemble_global_matrices
from .main import AnalysisResult, run
from .model import Element, Node
from .postprocess import calculate_element_forces
from .solver import check_energy, solve

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "2.1.6"

solve_truss = solve

__all__ = [
    "AnalysisResult",
    "Element",
    "Node",
    "__version__",
    "assemble_global_matrices",
    "calculate_element_forces",
    "check_energy",
    "run",
    "solve",
    "solve_truss",
]
