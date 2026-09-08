"""Exception hierarchy for the truss analysis library.

All library-specific exceptions derive from :class:`TrussError`, so callers
can catch the whole family with a single ``except TrussError`` clause while
still being able to discriminate individual failure modes.
"""

from __future__ import annotations


class TrussError(Exception):
    """Base class for all errors raised by this library."""


class AssemblyError(TrussError):
    """Raised when global matrix assembly encounters inconsistent input."""


class InputValidationError(TrussError):
    """Raised when the input model (nodes, elements, loads) is invalid."""


class EnergyValidationError(TrussError):
    """Raised when the work-energy balance of a solve does not close."""


class SingularMatrixError(TrussError):
    """Raised when the free-free stiffness matrix is rank deficient."""


class UnitConversionError(TrussError):
    """Raised for unknown unit systems or unknown quantity keys."""


class IllConditionedWarning(UserWarning):
    """Warning issued when cond(K_ff) exceeds the screening threshold."""
