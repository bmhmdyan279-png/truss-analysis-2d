from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("truss_analysis")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .exceptions import TrussError, UnitConversionError
from .model import Element, Node
from .solver import solve_truss

__all__ = [
    "Node",
    "Element",
    "solve_truss",
    "TrussError",
    "UnitConversionError",
]
