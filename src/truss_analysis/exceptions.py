from __future__ import annotations


class TrussError(Exception):
    pass


class AssemblyError(TrussError):
    pass


class InputValidationError(TrussError):
    pass


class EnergyValidationError(TrussError):
    pass


class SingularMatrixError(TrussError):
    pass


class UnitConversionError(TrussError):
    pass


class IllConditionedWarning(UserWarning):
    """Raised (as warning) when cond(K_ff) exceeds the screening threshold."""
