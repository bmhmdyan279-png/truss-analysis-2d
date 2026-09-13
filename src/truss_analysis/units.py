"""Unit-system handling and conversion to SI.

The library solves every problem internally in SI base units (m, N, Pa, K).
Input files may declare ``"units": "Imperial"``; :func:`to_si` converts each
recognised quantity with fixed, exact-ratio factors.

Supported quantity keys
-----------------------
``L`` (length), ``L2`` (area-like length^2), ``L4`` (second moment of area),
``A`` (cross-section area), ``I_sec`` (second moment of area), ``E``
(Young's modulus), ``F`` (force), ``delta_T`` (temperature difference),
``alpha`` (thermal expansion coefficient), ``density`` (mass density).
"""

from __future__ import annotations

from enum import Enum
from typing import overload

from .exceptions import UnitConversionError

__all__ = ["UnitConversionError", "UnitSystem", "to_si"]


class UnitSystem(Enum):
    """Recognised input unit systems."""

    SI = "SI"
    IMPERIAL = "Imperial"


_CONVERSION_FACTORS: dict[UnitSystem, dict[str, float]] = {
    UnitSystem.SI: {
        "L": 1.0,
        "L2": 1.0,
        "L4": 1.0,
        "A": 1.0,
        "I_sec": 1.0,
        "E": 1.0,
        "F": 1.0,
        "delta_T": 1.0,
        "alpha": 1.0,
        "density": 1.0,
    },
    UnitSystem.IMPERIAL: {
        "L": 0.3048,
        "L2": 0.092903,
        "L4": 0.0086309,
        "A": 0.092903,
        "I_sec": 0.0086309,
        "E": 6894.757,
        "F": 4.44822,
        "delta_T": 5.0 / 9.0,
        "alpha": 1.8,
        "density": 515.379,
    },
}


@overload
def to_si(value: float, unit_system: str, quantity: str) -> float: ...


@overload
def to_si(value: None, unit_system: str, quantity: str) -> None: ...


def to_si(value: float | None, unit_system: str, quantity: str) -> float | None:
    """Convert a scalar input value to SI units.

    Parameters
    ----------
    value : float or None
        Numeric value to convert. ``None`` passes through unchanged, which
        keeps optional input fields ergonomic for callers.
    unit_system : str
        Name of the source unit system (``"SI"`` or ``"Imperial"``).
    quantity : str
        Quantity key identifying the conversion factor (see the module
        docstring for the supported keys).

    Returns
    -------
    float or None
        ``value`` multiplied by the conversion factor, or ``None`` when
        ``value`` is ``None``.

    Raises
    ------
    UnitConversionError
        If ``unit_system`` or ``quantity`` is not recognised.
    """
    if value is None:
        return None
    try:
        sys_enum = UnitSystem(unit_system)
    except ValueError as exc:
        raise UnitConversionError(f"Unknown unit system: {unit_system}") from exc
    factors = _CONVERSION_FACTORS.get(sys_enum)
    if not factors or quantity not in factors:
        raise UnitConversionError(f"Unknown quantity '{quantity}' for {unit_system}")
    return float(value) * factors[quantity]
