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


class BucklingCheckWarning(UserWarning):
    """Warning issued when a compressed member cannot be assessed for buckling.

    Emitted by :func:`truss_analysis.postprocess.calculate_buckling` when a
    member carrying compression has no usable ``I_sec`` or length. Such a
    member is reported with ``status="unknown"`` and ``safe=False``: missing
    data must never be interpreted as a pass.
    """


class LargeDisplacementWarning(UserWarning):
    """Warning issued when the linear kinematics assumption is being stretched.

    The whole library is a small-displacement, first-order solver: the
    stiffness is built on the undeformed geometry and second-order (P-Delta)
    effects are out of scope. Emitted by
    :func:`truss_analysis.postprocess.check_displacement_magnitude` (wired
    into :func:`truss_analysis.main.run`) when the largest nodal
    displacement exceeds a documented fraction of the structure's
    characteristic length -- past that point the reported forces and
    displacements are an *approximation of unknown conservatism*, and a
    geometrically nonlinear analysis is warranted. The linearised
    bifurcation check :func:`truss_analysis.stability.linearized_buckling_load_factor`
    is the in-scope companion diagnostic for stability concerns.
    """


class InputIgnoredWarning(UserWarning):
    """Warning issued when a recognised-but-unapplied input key is supplied.

    The model parser accepts several keys for compatibility with generated
    models. Silently swallowing a key that the user believes controls physics
    is worse than refusing it, so any key that is parsed but not applied
    raises this warning instead.
    """


class UnitAmbiguityWarning(UserWarning):
    """Warning issued when an input quantity's unit convention is ambiguous.

    Used for the Imperial mass-density trap: ``515.379`` converts
    slug/ft^3 to kg/m^3 whereas ``16.0185`` converts lbm/ft^3 (pcf). Both are
    in everyday use, and choosing the wrong one is a silent factor-of-16
    error in self-weight.
    """
