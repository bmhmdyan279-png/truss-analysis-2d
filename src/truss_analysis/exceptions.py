from __future__ import annotations


class TrussError(Exception):
    """Base exception for all truss analysis errors."""

    pass


class SingularMatrixError(TrussError):
    """TRUSS-3001: Stiffness matrix is singular (mechanism or unstable)."""

    def __init__(self, msg="ماتریس سختی منفرد است (مکانیزم یا سازه ناپایدار)."):
        super().__init__(f"[TRUSS-3001] {msg}")


class InputValidationError(TrussError):
    """TRUSS-3002: Invalid input data (NaN, Inf, Negative)."""

    pass


class UnitConversionError(TrussError):
    """TRUSS-3003: Unit conversion failed."""

    pass


class EnergyValidationError(TrussError):
    """TRUSS-3004: Clapeyron's Theorem energy balance failed."""

    pass
